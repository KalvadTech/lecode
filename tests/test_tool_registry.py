"""Tests for the tool registry: specs, dispatch error paths, permission gating."""

from __future__ import annotations

import pytest

from lecode.agent.tools import core_tools
from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, grant_always
from lecode.config.models import Config
from lecode.permission import PermissionChecker, SessionPermissions
from lecode.session import SessionStore


@pytest.fixture
def registry():
    return ToolRegistry(core_tools())


def test_openai_tool_specs_shape(registry):
    specs = registry.openai_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {
        "read",
        "write",
        "edit",
        "bash",
        "grep",
        "find_files",
        "list_dir",
        "todo_write",
    }
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


async def test_dispatch_unknown_tool(registry, tool_ctx):
    message = await registry.dispatch("c1", "nope", "{}", tool_ctx)
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "c1"
    assert "unknown tool" in message["content"]


async def test_dispatch_bad_json(registry, tool_ctx):
    message = await registry.dispatch("c1", "read", "{not json", tool_ctx)
    assert "invalid tool arguments" in message["content"]


async def test_dispatch_non_object_args(registry, tool_ctx):
    message = await registry.dispatch("c1", "read", "[1, 2]", tool_ctx)
    assert "must be a JSON object" in message["content"]


async def test_dispatch_tool_exception_becomes_error_result(registry, tool_ctx):
    class BoomTool(Tool):
        def __init__(self):
            super().__init__(name="boom", description="x", parameters={"type": "object"})

        async def run(self, args, ctx):
            raise RuntimeError("kaboom")

    registry.register(BoomTool())
    message = await registry.dispatch("c1", "boom", "{}", tool_ctx)
    assert "kaboom" in message["content"]


async def test_dispatch_success(registry, tool_ctx, tmp_path):
    (tmp_path / "f.txt").write_text("data\n")
    message = await registry.dispatch("c1", "read", '{"path": "f.txt"}', tool_ctx)
    assert "1\tdata" in message["content"]


# -- permission gating ---------------------------------------------------------------


def _ctx(tmp_path, mode, auto_approve, monkeypatch, perms=None, rules=None):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    permissions = {"mode": mode}
    if rules:
        permissions["rules"] = rules
    config = Config.model_validate({"permissions": permissions})
    checker = PermissionChecker(config, session_perms=perms, cwd=tmp_path)
    return ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=checker,
        auto_approve=auto_approve,
        session_perms=perms,
    )


async def test_ask_without_approver_denied(registry, tmp_path, monkeypatch):
    # yolo allows everything, so the Ask is driven by an ask rule
    ctx = _ctx(
        tmp_path,
        "yolo",
        auto_approve=False,
        monkeypatch=monkeypatch,
        rules={"ask": {"bash": [{"pattern": "*"}]}},
    )
    message = await registry.dispatch("c1", "bash", '{"command": "ls"}', ctx)
    assert "requires approval" in message["content"]
    assert message["content"].startswith("denied")


async def test_ask_with_auto_approve_runs(registry, tmp_path, monkeypatch):
    ctx = _ctx(
        tmp_path,
        "yolo",
        auto_approve=True,
        monkeypatch=monkeypatch,
        rules={"ask": {"bash": [{"pattern": "*"}]}},
    )
    message = await registry.dispatch("c1", "bash", '{"command": "echo ran"}', ctx)
    assert "ran" in message["content"]


async def test_deny_never_converted(registry, tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, "readonly", auto_approve=True, monkeypatch=monkeypatch)
    message = await registry.dispatch("c1", "write", '{"path": "x", "content": "y"}', ctx)
    assert message["content"].startswith("denied")
    assert not (tmp_path / "x").exists()


async def test_doom_loop_applies_through_dispatch(registry, tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, "yolo", auto_approve=False, monkeypatch=monkeypatch)
    for _ in range(2):
        message = await registry.dispatch("c1", "bash", '{"command": "ls"}', ctx)
        assert "requires approval" not in message["content"]
    third = await registry.dispatch("c1", "bash", '{"command": "ls"}', ctx)
    assert "requires approval" in third["content"]
    fourth = await registry.dispatch("c1", "bash", '{"command": "ls"}', ctx)
    assert fourth["content"].startswith("denied: doom loop")


async def test_grant_always_persists_when_session_present(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    store = SessionStore()
    session = store.create("s", cwd=str(tmp_path))
    perms = SessionPermissions()
    config = Config.model_validate({"permissions": {"mode": "yolo"}})
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=PermissionChecker(config, session_perms=perms, cwd=tmp_path),
        session=session,
        session_store=store,
        session_perms=perms,
    )
    grant_always(ctx, "bash", "git *")
    assert perms.grants == [("bash", "git *")]
    assert store.load_grants(session) == [("bash", "git *")]
    # and the checker now allows matching calls
    assert ctx.permission_checker.check("bash", {"command": "git status"}).decision.value == "allow"
