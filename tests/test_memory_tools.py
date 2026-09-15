"""Tests for the four memory tools through ToolRegistry.dispatch."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

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


async def test_correct_and_forget_dispatch_require_selected_id_revision_and_real_snapshot(
    mem_ctx, registry
):
    from lecode.memory.facts import FactStore
    from lecode.session.storage import SessionStore

    ctx, markdown = mem_ctx
    sessions = SessionStore(ctx.cwd / "cfg")
    session = sessions.create("source", ctx.cwd)
    ctx.session, ctx.session_store = session, sessions
    facts = FactStore(markdown.root / "facts.sqlite3")
    ctx.extras["facts"] = facts
    sessions.bind_facts(ctx.cwd, facts)
    sessions.append_message(session, {"role": "user", "content": "old claim"})
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=ctx.cwd).ref
    fact = facts.remember("old claim", ref, sessions=sessions, project_root=ctx.cwd)
    request = sessions.append_message(session, {"role": "user", "content": "Correct it: new claim"})
    ref = sessions.source_snapshot(session.id, request.seq, request.seq, project_root=ctx.cwd).ref
    args = {
        "fact_id": fact.id,
        "expected_revision": 1,
        "text": "new claim",
        "source_snapshot": asdict(ref),
    }
    _, result = await registry.dispatch_result("correct", "memory_correct", json.dumps(args), ctx)
    assert not result.is_error, result.content
    assert facts.get(fact.id).text == "new claim"
    assert facts.source(fact.id, 2) == ref
    _, result = await registry.dispatch_result("conflict", "memory_correct", json.dumps(args), ctx)
    assert result.is_error and "revision" in result.content
    markdown.write_long_term("independently authored note")
    _, result = await registry.dispatch_result(
        "forget", "memory_forget", json.dumps({"fact_id": fact.id}), ctx
    )
    assert not result.is_error, result.content
    assert facts.get(fact.id) is None
    assert markdown.read_long_term() == "independently authored note\n"
    facts.close()


@pytest.mark.parametrize(
    "case",
    ["readonly", "child", "unknown", "bulk", "traversal", "tamper", "bool_revision", "tool_only"],
)
async def test_memory_mutations_reject_unsafe_requests_without_changing_facts(
    mem_ctx, registry, case
):
    from lecode.memory.facts import FactStore
    from lecode.session.storage import SessionStore

    ctx, markdown = mem_ctx
    sessions = SessionStore(ctx.cwd / "cfg")
    session = sessions.create("source", ctx.cwd)
    ctx.session, ctx.session_store = session, sessions
    facts = FactStore(markdown.root / "facts.sqlite3")
    ctx.extras["facts"] = facts
    sessions.bind_facts(ctx.cwd, facts)
    sessions.append_message(session, {"role": "user", "content": "old"})
    fact = facts.add("old", source_id=session.id, source_seq=1)
    sessions.append_message(
        session, {"role": "tool" if case == "tool_only" else "user", "content": "new"}
    )
    ref = sessions.source_snapshot(session.id, 2, 2, project_root=ctx.cwd).ref
    args = {
        "fact_id": fact.id,
        "text": "new",
        "expected_revision": 1,
        "source_snapshot": asdict(ref),
    }
    if case == "readonly":
        ctx.permission_checker.set_mode("readonly")
    elif case == "child":
        ctx.session = None
    elif case == "unknown":
        args["fact_id"] = "f" * 64
    elif case == "bulk":
        args["pattern"] = ".*"
    elif case == "traversal":
        args["source_snapshot"]["session_id"] = "../" + session.id
    elif case == "tamper":
        args["source_snapshot"]["digest"] = "0" * 64
    elif case == "bool_revision":
        args["expected_revision"] = True
    _, result = await registry.dispatch_result("c", "memory_correct", json.dumps(args), ctx)
    assert result.is_error, result.content
    assert facts.get(fact.id) == fact
    assert len(facts.provenance(fact.id)) == 1
    assert facts.generation() == 0
    facts.close()


@pytest.mark.parametrize("action", ["correct", "forget"])
async def test_fact_mutations_preserve_pre_tool_hooks(tmp_path, monkeypatch, action):
    from lecode.agent.builder import build_runtime
    from lecode.session.storage import SessionStore

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    config = Config(hooks={"PreToolUse": ['echo \'{"verdict":"deny","reason":"blocked"}\'']})
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    runtime = build_runtime(config, tmp_path, session=session, store=sessions, auto_approve=True)
    facts = runtime.ctx.extras["facts"]
    sessions.append_message(session, {"role": "user", "content": "evidence"})
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("claim", ref, sessions=sessions, project_root=tmp_path)
    args = {"fact_id": fact.id}
    if action == "correct":
        args.update(text="new claim", expected_revision=1, source_snapshot=asdict(ref))
    _, result = await runtime.registry.dispatch_result(
        "c", f"memory_{action}", json.dumps(args), runtime.ctx
    )
    assert result.is_error and "denied by hook" in result.content
    assert facts.get(fact.id) == fact
    assert facts.generation() == 0
    facts.close()


async def test_correction_from_old_provider_epoch_cannot_promote_fresh_provenance(
    mem_ctx, registry
):
    from lecode.memory.facts import FactStore
    from lecode.session.storage import SessionStore

    ctx, markdown = mem_ctx
    sessions = SessionStore(ctx.cwd / "cfg")
    ctx.session_store = sessions
    ctx.session = sessions.create("source", ctx.cwd)
    facts = FactStore(markdown.root / "facts.sqlite3")
    ctx.extras["facts"] = facts
    sessions.append_message(ctx.session, {"role": "user", "content": "fresh independent evidence"})
    ref = sessions.source_snapshot(ctx.session.id, 1, 1, project_root=ctx.cwd).ref
    fact = facts.remember("independent", ref, sessions=sessions, project_root=ctx.cwd)
    other = facts.add("forgotten", source_id="other", source_seq=1)
    ctx.memory_generation = facts.generation()
    facts.forget(other.id)
    _, result = await registry.dispatch_result(
        "c",
        "memory_correct",
        json.dumps(
            {
                "fact_id": fact.id,
                "expected_revision": 1,
                "text": "old recalled claim",
                "source_snapshot": asdict(ref),
            }
        ),
        ctx,
    )
    assert result.is_error and "exclusions changed" in result.content
    assert facts.get(fact.id) == fact
    facts.close()


@pytest.mark.parametrize(
    "tool,args",
    [
        ("memory_write", {"content": "forgotten content"}),
        ("memory_edit", {"old": "independent note", "new": "forgotten content"}),
    ],
)
@pytest.mark.parametrize("race", ["siblings", "approval", "hook"])
async def test_generated_markdown_write_cannot_race_forget(mem_ctx, registry, tool, args, race):
    from tests.fakes import FakeProvider

    from lecode.agent.runner import AgentRunner
    from lecode.memory.facts import FactStore
    from lecode.permission import AllowOnce
    from lecode.session.storage import SessionStore

    ctx, markdown = mem_ctx
    sessions = SessionStore(ctx.cwd / "cfg")
    ctx.session_store = sessions
    ctx.session = sessions.create("parent", ctx.cwd)
    facts = FactStore(markdown.root / "facts.sqlite3")
    ctx.extras["facts"] = facts
    sessions.bind_facts(ctx.cwd, facts)
    fact = facts.add("forgotten content", source_id=ctx.session.id, source_seq=1)
    markdown.write_long_term("independent note")
    if race == "siblings":
        provider = FakeProvider(
            [
                {
                    "tool_calls": [
                        {
                            "id": "forget",
                            "name": "memory_forget",
                            "arguments": json.dumps({"fact_id": fact.id}),
                        },
                        {"id": "write", "name": tool, "arguments": json.dumps(args)},
                    ]
                },
                {"text": "done"},
            ]
        )
        await AgentRunner(provider, registry, ctx, session=ctx.session, store=sessions).run(
            [{"role": "user", "content": "update memory"}]
        )
        seqs = [r.seq for r in sessions.read_records(ctx.session) if hasattr(r, "seq")]
        assert len(seqs) == len(set(seqs))
    elif race == "approval":
        from lecode.config.models import PermissionRule

        ctx.config.permissions.rules.ask[tool] = [PermissionRule(pattern="*")]
        ctx.auto_approve = False
        ctx.memory_generation = facts.generation()
        entered, resume = asyncio.Event(), asyncio.Event()

        async def approve(*_):
            entered.set()
            await resume.wait()
            return AllowOnce()

        ctx.approval_callback = approve
        task = asyncio.create_task(registry.dispatch_result("write", tool, json.dumps(args), ctx))
        await entered.wait()
        other = FactStore(facts.path)
        other.forget(fact.id)
        other.close()
        resume.set()
        _, result = await task
        assert result.is_error and "exclusions changed" in result.content
    else:
        import shlex
        import sys

        from lecode.hooks import apply_hooks, dispatcher_from_config

        ctx.memory_generation = facts.generation()
        script = (
            f"from lecode.memory.facts import FactStore; f=FactStore({str(facts.path)!r}); "
            f'f.forget({fact.id!r}); f.close(); print(\'{{"verdict":"allow"}}\')'
        )
        ctx.config.hooks = {
            "PreToolUse": [f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"]
        }
        hooks, _ = dispatcher_from_config(ctx.config, ctx.cwd, session=ctx.session)
        apply_hooks(registry, hooks)
        _, result = await registry.dispatch_result("write", tool, json.dumps(args), ctx)
        assert result.is_error and "exclusions changed" in result.content
    assert markdown.read_long_term() == "independent note\n"
    # A genuinely new human request uses the current epoch and may edit notes.
    ctx.memory_generation = facts.generation()
    ctx.auto_approve = True
    fresh_args = (
        {"content": "fresh note"}
        if tool == "memory_write"
        else {"old": "independent note", "new": "fresh note"}
    )
    _, result = await registry.dispatch_result("fresh", tool, json.dumps(fresh_args), ctx)
    assert not result.is_error
    assert "fresh note" in markdown.read_long_term()
    facts.close()
