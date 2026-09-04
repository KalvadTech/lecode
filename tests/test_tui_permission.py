"""Tests for the inline permission prompt (y/a/n/ESC) and approval callback."""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from lecode.config.models import Config, PermissionRule
from lecode.permission import (
    AllowAlways,
    AllowOnce,
    Deny,
    PermissionChecker,
    SessionPermissions,
)
from lecode.session.storage import SessionStore
from lecode.tui.app import TuiApp
from lecode.tui.permission import ApprovalPrompt, approval_prompt_text
from lecode.tui.statusline import StatusLineState


class FakeTool(Tool):
    """Records runs; returns the command/path it was given."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, description="fake", parameters={})
        self.runs: list[dict[str, Any]] = []

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs.append(args)
        return ToolResult(f"ran: {args}")


def make_ctx(tmp_path, monkeypatch, callback=None, mode="yolo", tool_name="bash", ask=True):
    """A ToolContext with session + store so grants can persist.

    The two modes never Ask by themselves, so ``ask=True`` installs an ask
    rule for ``tool_name`` to drive the approval-prompt flows.
    """
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False  # never play sounds in tests
    if ask:
        config.permissions.rules.ask[tool_name] = [PermissionRule(pattern="*")]
    store = SessionStore()
    session = store.create("s", tmp_path)
    perms = SessionPermissions()
    checker = PermissionChecker(config, session_perms=perms, mode=mode, cwd=tmp_path)
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=checker,
        session=session,
        session_store=store,
        session_perms=perms,
        approval_callback=callback,
    )
    return ctx, store, session, perms


async def test_allow_once_runs_without_grant(tmp_path, monkeypatch):
    calls = []

    async def approve(name, args, reason):
        calls.append((name, args, reason))
        return AllowOnce()

    ctx, store, session, perms = make_ctx(tmp_path, monkeypatch, approve)
    registry = ToolRegistry([FakeTool("bash")])
    _, result = await registry.dispatch_result("1", "bash", '{"command": "ls"}', ctx)
    assert result.content == "ran: {'command': 'ls'}"
    assert calls == [("bash", {"command": "ls"}, "ask rule matched: *")]
    assert perms.grants == []
    assert store.load_grants(session) == []


async def test_deny_blocks_the_tool(tmp_path, monkeypatch):
    async def deny(name, args, reason):
        return Deny()

    ctx, _, _, _ = make_ctx(tmp_path, monkeypatch, deny)
    tool = FakeTool("bash")
    registry = ToolRegistry([tool])
    _, result = await registry.dispatch_result("1", "bash", '{"command": "ls"}', ctx)
    assert result.is_error
    assert "denied by user" in result.content
    assert tool.runs == []


async def test_allow_always_persists_grant(tmp_path, monkeypatch):
    async def always(name, args, reason):
        return AllowAlways(pattern="ls")

    ctx, store, session, perms = make_ctx(tmp_path, monkeypatch, always)
    registry = ToolRegistry([FakeTool("bash")])
    await registry.dispatch_result("1", "bash", '{"command": "ls"}', ctx)
    assert perms.matching_grant("bash", "ls") == "ls"
    assert store.load_grants(session) == [("bash", "ls")]


async def test_allow_always_still_prompts_for_ask_rules(tmp_path, monkeypatch):
    """Ask rules are evaluated before session grants, so (a)lways records the
    grant but the next rule-matched call still prompts."""
    calls = []

    async def always(name, args, reason):
        calls.append(name)
        return AllowAlways(pattern="ls")

    ctx, _, _, perms = make_ctx(tmp_path, monkeypatch, always)
    registry = ToolRegistry([FakeTool("bash")])
    await registry.dispatch_result("1", "bash", '{"command": "ls"}', ctx)
    _, result = await registry.dispatch_result("2", "bash", '{"command": "ls"}', ctx)
    assert result.content.startswith("ran:")
    assert calls == ["bash", "bash"]  # ask rules outrank session grants
    assert perms.matching_grant("bash", "ls") == "ls"


async def test_no_callback_keeps_needs_approval(tmp_path, monkeypatch):
    ctx, _, _, _ = make_ctx(tmp_path, monkeypatch, callback=None)
    registry = ToolRegistry([FakeTool("bash")])
    _, result = await registry.dispatch_result("1", "bash", '{"command": "ls"}', ctx)
    assert result.is_error
    assert result.metadata.get("needs_approval") is True


async def test_checker_deny_never_reaches_callback(tmp_path, monkeypatch):
    calls = []

    async def approve(name, args, reason):
        calls.append(name)
        return AllowOnce()

    ctx, _, _, _ = make_ctx(tmp_path, monkeypatch, approve, mode="readonly", ask=False)
    registry = ToolRegistry([FakeTool("bash")])
    _, result = await registry.dispatch_result("1", "bash", '{"command": "sudo ls"}', ctx)
    assert result.is_error
    assert result.content.startswith("denied:")
    assert calls == []


async def test_doom_loop_ask_flows_through_callback(tmp_path, monkeypatch):
    """The 3rd identical read call is doom-Ask; the 4th is Deny, no callback."""
    calls = []

    async def approve(name, args, reason):
        calls.append(reason)
        return AllowOnce()

    ctx, _, _, _ = make_ctx(tmp_path, monkeypatch, approve)
    registry = ToolRegistry([FakeTool("read")])
    args = '{"path": "same.txt"}'
    await registry.dispatch_result("1", "read", args, ctx)
    await registry.dispatch_result("2", "read", args, ctx)
    _, third = await registry.dispatch_result("3", "read", args, ctx)
    assert not third.is_error  # approved through the prompt
    assert len(calls) == 1
    assert "doom loop" in calls[0]
    _, fourth = await registry.dispatch_result("4", "read", args, ctx)
    assert fourth.is_error
    assert "doom loop" in fourth.content
    assert len(calls) == 1  # deny never asks


# -- prompt state + rendering ----------------------------------------------------


def test_approval_prompt_text():
    text = approval_prompt_text("bash", "ls -la")
    assert "allow bash 'ls -la'?" in text
    assert "(y)once" in text and "(a)lways" in text and "(n)deny" in text
    long_target = "x" * 100
    assert "…" in approval_prompt_text("bash", long_target)


async def test_approval_prompt_request_resolve_cancel():
    prompt = ApprovalPrompt()
    future = prompt.request("bash", "ls", "reason")
    assert prompt.is_pending
    prompt.resolve(AllowOnce())
    assert await future == AllowOnce()
    assert not prompt.is_pending
    future2 = prompt.request("bash", "rm x", "reason")
    prompt.cancel()
    assert not prompt.is_pending
    with pytest.raises(asyncio.CancelledError):
        await future2


# -- app integration ---------------------------------------------------------------


def make_app(tmp_path, monkeypatch, script):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False  # never play sounds in tests
    # the two modes never Ask by themselves; gate bash with an ask rule
    config.permissions.rules.ask["bash"] = [PermissionRule(pattern="*")]
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


async def test_request_approval_renders_and_restores_state(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch, [])
    task = asyncio.ensure_future(app._request_approval("bash", {"command": "ls"}, ""))
    await wait_for(lambda: "allow bash 'ls'?" in out.getvalue())
    assert app._status.state is StatusLineState.AWAITING_APPROVAL
    app._approval.resolve(AllowAlways(pattern="ls"))
    assert await task == AllowAlways(pattern="ls")
    assert app._status.state is StatusLineState.RUNNING
    assert not app._approval.is_pending


async def test_request_approval_shows_doom_reason(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch, [])
    reason = "doom loop: same call"
    task = asyncio.ensure_future(app._request_approval("read", {"path": "x"}, reason))
    await wait_for(lambda: "doom loop: same call" in out.getvalue())
    app._approval.resolve(Deny())
    assert await task == Deny()
    task.result()  # consume


async def test_pipe_approval_y_runs_asked_tool(tmp_path, monkeypatch):
    """Full flow: model calls bash, user answers 'y', output appears."""
    script = [
        {"tool_calls": [{"name": "bash", "arguments": '{"command": "echo approved-output"}'}]},
        {"text": "tool ran"},
    ]
    app, out = make_app(tmp_path, monkeypatch, script)
    with create_pipe_input() as inp:
        inp.send_text("run it\n")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "allow bash 'echo approved-output'?" in out.getvalue())
        inp.send_text("y")
        await wait_for(lambda: "tool ran" in out.getvalue())  # turn fully done
        inp.send_text("/quit\n")
        assert await task == 0
    rendered = out.getvalue()
    assert "approved-output" in rendered
    assert "tool ran" in rendered


async def test_pipe_approval_escape_denies(tmp_path, monkeypatch):
    script = [
        {"tool_calls": [{"name": "bash", "arguments": '{"command": "echo should-not-run"}'}]},
        {"text": "after denial"},
    ]
    app, out = make_app(tmp_path, monkeypatch, script)
    with create_pipe_input() as inp:
        inp.send_text("run it\n")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "allow bash" in out.getvalue())
        inp.send_text("\x1b")
        await wait_for(lambda: "denied by user" in out.getvalue())
        inp.send_text("/quit\n")
        assert await task == 0
    # the command string only ever appears in the tool-call echo and the ask
    tool_lines = [line for line in out.getvalue().splitlines() if "should-not-run" in line]
    assert tool_lines
    assert all(line.startswith(("⚙", "allow bash")) for line in tool_lines)
