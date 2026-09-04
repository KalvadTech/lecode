"""Tests for the four memory tools through ToolRegistry.dispatch."""

from __future__ import annotations

import pytest

from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.config.models import Config
from lecode.memory.store import MemoryStore
from lecode.memory.tools import memory_tools
from lecode.permission import Decision, PermissionChecker


@pytest.fixture
def mem_ctx(tmp_path, monkeypatch):
    """A tool context with a live memory store (yolo, auto-approve)."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    ctx = ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)
    store = MemoryStore(tmp_path / "mem")
    ctx.extras["memory"] = store
    return ctx, store


@pytest.fixture
def registry():
    return ToolRegistry(memory_tools())


async def test_write_and_read_long_term(mem_ctx, registry):
    ctx, store = mem_ctx
    message, result = await registry.dispatch_result(
        "c1", "memory_write", '{"target": "long_term", "content": "the sky is blue"}', ctx
    )
    assert not result.is_error
    assert "the sky is blue" in store.read_long_term()

    message, result = await registry.dispatch_result("c2", "memory_read", "{}", ctx)
    assert "the sky is blue" in message["content"]


async def test_write_overwrite_mode(mem_ctx, registry):
    ctx, store = mem_ctx
    await registry.dispatch_result("c1", "memory_write", '{"content": "first"}', ctx)
    await registry.dispatch_result(
        "c2", "memory_write", '{"content": "second", "mode": "overwrite"}', ctx
    )
    content = store.read_long_term(capped=False)
    assert "second" in content
    assert "first" not in content


async def test_write_note_and_read(mem_ctx, registry):
    ctx, store = mem_ctx
    _, result = await registry.dispatch_result(
        "c1", "memory_write", '{"target": "note:todo", "content": "buy milk"}', ctx
    )
    assert not result.is_error
    assert store.read_note("todo") == "buy milk\n"
    message, _ = await registry.dispatch_result("c2", "memory_read", '{"target": "note:todo"}', ctx)
    assert "buy milk" in message["content"]


async def test_write_note_append(mem_ctx, registry):
    ctx, store = mem_ctx
    await registry.dispatch_result(
        "c1", "memory_write", '{"target": "note:n", "content": "one"}', ctx
    )
    await registry.dispatch_result(
        "c2", "memory_write", '{"target": "note:n", "content": "two"}', ctx
    )
    assert "one" in store.read_note("n")
    assert "two" in store.read_note("n")


async def test_write_daily_and_scratchpad(mem_ctx, registry):
    ctx, store = mem_ctx
    await registry.dispatch_result(
        "c1", "memory_write", '{"target": "daily", "content": "log"}', ctx
    )
    assert "log" in store.read_daily()
    await registry.dispatch_result(
        "c2", "memory_write", '{"target": "scratchpad", "content": "- [ ] x"}', ctx
    )
    await registry.dispatch_result(
        "c3", "memory_write", '{"target": "scratchpad", "content": "- [ ] y"}', ctx
    )
    scratch = store.read_scratchpad()
    assert "- [ ] x" in scratch and "- [ ] y" in scratch


async def test_write_errors(mem_ctx, registry):
    ctx, _ = mem_ctx
    _, r1 = await registry.dispatch_result("c1", "memory_write", '{"content": "  "}', ctx)
    assert r1.is_error and "empty" in r1.content
    _, r2 = await registry.dispatch_result(
        "c2", "memory_write", '{"content": "x", "mode": "sideways"}', ctx
    )
    assert r2.is_error and "invalid mode" in r2.content
    _, r3 = await registry.dispatch_result(
        "c3", "memory_write", '{"target": "nope", "content": "x"}', ctx
    )
    assert r3.is_error and "unknown target" in r3.content
    _, r4 = await registry.dispatch_result(
        "c4", "memory_write", '{"target": "note:BAD!", "content": "x"}', ctx
    )
    assert r4.is_error


async def test_edit_long_term(mem_ctx, registry):
    ctx, store = mem_ctx
    store.write_long_term("alpha beta")
    _, result = await registry.dispatch_result(
        "c1", "memory_edit", '{"old": "beta", "new": "BETA"}', ctx
    )
    assert not result.is_error
    assert "alpha BETA" in store.read_long_term()


async def test_edit_errors(mem_ctx, registry):
    ctx, store = mem_ctx
    store.write_long_term("dup dup")
    _, r1 = await registry.dispatch_result(
        "c1", "memory_edit", '{"old": "absent", "new": "x"}', ctx
    )
    assert r1.is_error and "not found" in r1.content
    _, r2 = await registry.dispatch_result("c2", "memory_edit", '{"old": "dup", "new": "x"}', ctx)
    assert r2.is_error and "ambiguous" in r2.content
    _, r3 = await registry.dispatch_result(
        "c3", "memory_edit", '{"old": "x", "new": "y", "target": "note:missing"}', ctx
    )
    assert r3.is_error and "no such note" in r3.content


async def test_edit_note(mem_ctx, registry):
    ctx, store = mem_ctx
    store.write_note("n", "old text")
    _, result = await registry.dispatch_result(
        "c1", "memory_edit", '{"old": "old", "new": "new", "target": "note:n"}', ctx
    )
    assert not result.is_error
    assert store.read_note("n") == "new text\n"


async def test_read_pagination(mem_ctx, registry):
    ctx, store = mem_ctx
    store.write_long_term("\n".join(f"line {i}" for i in range(1, 11)))
    message, _ = await registry.dispatch_result(
        "c1", "memory_read", '{"offset": 3, "limit": 2}', ctx
    )
    assert message["content"] == "line 3\nline 4"


async def test_read_daily_and_missing(mem_ctx, registry):
    ctx, _ = mem_ctx
    message, _ = await registry.dispatch_result(
        "c1", "memory_read", '{"target": "daily", "date": "1999-01-01"}', ctx
    )
    assert "empty" in message["content"]
    _, result = await registry.dispatch_result(
        "c2", "memory_read", '{"target": "note:missing"}', ctx
    )
    assert result.is_error and "no such note" in result.content


async def test_search_tool_groups_by_file(mem_ctx, registry):
    ctx, store = mem_ctx
    store.write_long_term("keyword here\nand keyword again")
    store.write_note("n", "keyword in note")
    message, result = await registry.dispatch_result(
        "c1", "memory_search", '{"pattern": "keyword"}', ctx
    )
    assert result.metadata["hits"] == 3
    assert "MEMORY.md (2 hits):" in message["content"]
    assert "notes/n.md (1 hits):" in message["content"]
    assert message["content"].index("MEMORY.md") < message["content"].index("notes/n.md")


async def test_search_no_match_and_invalid_regex(mem_ctx, registry):
    ctx, _ = mem_ctx
    message, _ = await registry.dispatch_result("c1", "memory_search", '{"pattern": "zzz"}', ctx)
    assert message["content"] == "(no matches)"
    _, result = await registry.dispatch_result("c2", "memory_search", '{"pattern": "["}', ctx)
    assert result.is_error and "invalid regex" in result.content


async def test_readonly_mode_gating(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    checker = PermissionChecker(config, mode="readonly", cwd=tmp_path)
    assert checker.check("memory_read", {}).decision == Decision.ALLOW
    assert checker.check("memory_search", {"pattern": "x"}).decision == Decision.ALLOW
    assert checker.check("memory_write", {"content": "x"}).decision == Decision.DENY
    assert checker.check("memory_edit", {"old": "a", "new": "b"}).decision == Decision.DENY


async def test_disabled_memory_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.memory.enabled = False
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    ctx = ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)
    ctx.extras["memory"] = MemoryStore(tmp_path / "mem")  # store present but disabled
    registry = ToolRegistry(memory_tools())
    for tool in ("memory_write", "memory_edit", "memory_read", "memory_search"):
        _, result = await registry.dispatch_result("c1", tool, "{}", ctx)
        assert result.is_error
        assert "disabled" in result.content


async def test_missing_store_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    checker = PermissionChecker(config, mode="yolo", cwd=tmp_path)
    ctx = ToolContext(cwd=tmp_path, config=config, permission_checker=checker, auto_approve=True)
    registry = ToolRegistry(memory_tools())
    _, result = await registry.dispatch_result("c1", "memory_read", "{}", ctx)
    assert result.is_error
    assert "not available" in result.content
