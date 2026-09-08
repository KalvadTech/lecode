"""Integration tests for the lifecycle hook fire sites (runner, compaction, TUI)."""

from __future__ import annotations

import asyncio
import json
from io import StringIO

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.agent.runner import AgentRunner
from lecode.agent.tools.base import ToolRegistry
from lecode.config.models import Config
from lecode.hooks.runner import HookDispatcher, HookHandler
from lecode.providers.openai_compat import ProviderError
from lecode.session.compaction import compact_session
from lecode.session.storage import SessionStore
from lecode.tui.app import TuiApp


def log_dispatcher(tmp_path, *events: str) -> tuple[HookDispatcher, object]:
    """A dispatcher whose handlers append each envelope as a JSON line."""
    log = tmp_path / "hooks.jsonl"
    handlers = {event: [HookHandler(f"cat >> {log}; echo >> {log}")] for event in events}
    return HookDispatcher(handlers, tmp_path), log


def logged(log) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


# -- Stop (end of AgentRunner.run) -------------------------------------------------


async def test_stop_hook_fires_with_done_reason(tool_ctx, tmp_path):
    hooks, log = log_dispatcher(tmp_path, "Stop")
    tool_ctx.extras["hooks"] = hooks
    runner = AgentRunner(FakeProvider([{"text": "hi"}]), ToolRegistry([]), tool_ctx)

    result = await runner.run([{"role": "user", "content": "hi"}])

    assert result.stop_reason == "done"
    events = logged(log)
    assert [e["event"] for e in events] == ["Stop"]
    assert events[0]["reason"] == "done"


async def test_stop_hook_fires_with_max_turns_reason(tool_ctx, tmp_path):
    hooks, log = log_dispatcher(tmp_path, "Stop")
    tool_ctx.extras["hooks"] = hooks
    tool_ctx.config.agent.max_turns = 1
    script = [{"tool_calls": [{"name": "echo", "arguments": "{}"}]}] * 3
    runner = AgentRunner(FakeProvider(script), ToolRegistry([]), tool_ctx)

    result = await runner.run([{"role": "user", "content": "hi"}])

    assert result.stop_reason == "max_turns"
    events = logged(log)
    assert [e["event"] for e in events] == ["Stop"]
    assert events[0]["reason"] == "max_turns"


async def test_stop_hook_absent_dispatcher_still_runs(tool_ctx):
    runner = AgentRunner(FakeProvider([{"text": "hi"}]), ToolRegistry([]), tool_ctx)
    result = await runner.run([{"role": "user", "content": "hi"}])
    assert result.stop_reason == "done"


# -- PreCompact / PostCompact -------------------------------------------------------


def make_store(tmp_path, monkeypatch) -> SessionStore:
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


def fill(store, session, pairs: int = 5) -> None:
    for index in range(pairs):
        store.append_message(session, {"role": "user", "content": f"q{index}"})
        store.append_message(session, {"role": "assistant", "content": f"a{index}"})


async def test_compact_session_fires_pre_and_post_compact(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    session = store.create("demo", tmp_path)
    fill(store, session)
    hooks, log = log_dispatcher(tmp_path, "PreCompact", "PostCompact")
    hooks.session = session

    summary = await compact_session(
        FakeProvider([{"text": "the summary"}]), store, session, "test-model", hooks=hooks
    )

    assert summary == "the summary"
    events = logged(log)
    assert [e["event"] for e in events] == ["PreCompact", "PostCompact"]
    assert events[0]["session"]["name"] == "demo"


async def test_compact_session_provider_failure_fires_pre_only(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    session = store.create("demo", tmp_path)
    fill(store, session)
    hooks, log = log_dispatcher(tmp_path, "PreCompact", "PostCompact")
    provider = FakeProvider([{"error": ProviderError("down", status=500)}])

    summary = await compact_session(provider, store, session, "test-model", hooks=hooks)

    assert summary is None
    assert [e["event"] for e in logged(log)] == ["PreCompact"]


async def test_compact_session_without_hooks_still_works(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    session = store.create("demo", tmp_path)
    fill(store, session)
    summary = await compact_session(
        FakeProvider([{"text": "the summary"}]), store, session, "test-model"
    )
    assert summary == "the summary"


# -- TUI fire sites (pipe input) -----------------------------------------------------


def make_app(tmp_path, monkeypatch, script, hooks: dict[str, list[str]] | None = None):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False  # never play sounds in tests
    config.hooks = hooks or {}
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    out = StringIO()
    app = TuiApp(
        config,
        runtime,
        FakeProvider(script),
        session,
        store,
        console=Console(record=True, file=out, width=200),
    )
    return app, out


async def wait_for(cond, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def test_tui_user_prompt_submit_deny_blocks(tmp_path, monkeypatch):
    deny = f"echo '{json.dumps({'verdict': 'deny', 'reason': 'prompts closed'})}'"
    app, out = make_app(tmp_path, monkeypatch, [{"text": "no"}], {"UserPromptSubmit": [deny]})
    with create_pipe_input() as inp:
        inp.send_text("hello\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "prompt blocked by hook" in out.getvalue())
        assert "prompts closed" in out.getvalue()
        inp.send_text("/quit\r")
        assert await task == 0
    assert app.runner.provider.requests == []  # the turn never started


async def test_tui_session_and_notification_hooks(tmp_path, monkeypatch):
    log = tmp_path / "hooks.jsonl"
    app, _out = make_app(
        tmp_path,
        monkeypatch,
        [{"text": "hello back"}],
        {
            "SessionStart": [f"cat >> {log}; echo >> {log}"],
            "SessionEnd": [f"cat >> {log}; echo >> {log}"],
            "Notification": [f"cat >> {log}; echo >> {log}"],
        },
    )
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        inp.send_text("hi there\r")
        await wait_for(lambda: "Notification" in (log.read_text() if log.exists() else ""))
        inp.send_text("/quit\r")
        assert await task == 0
    events = logged(log)
    names = [e["event"] for e in events]
    assert names[0] == "SessionStart"
    assert names[-1] == "SessionEnd"
    notification = next(e for e in events if e["event"] == "Notification")
    assert notification["kind"] == "finish"
    start = events[0]
    assert start["session"]["name"] == "s"
