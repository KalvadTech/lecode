"""Tests for background tasks: manager lifecycle, tasks_* tools, runner/TUI wiring."""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest
from tests.fakes import FakeProvider
from tests.test_subagents import make_runtime
from tests.test_tui_app import make_app, wait_for
from typer.testing import CliRunner

from lecode.agent.runner import AgentRunner
from lecode.agent.tools import bash
from lecode.agent.tools.base import Tool, ToolRegistry
from lecode.agent.tools.base import ToolResult as ToolExecResult
from lecode.extras import background
from lecode.extras.background import BACKGROUND_EXTRA, BackgroundTaskManager


def ctx_with_manager(tool_ctx) -> tuple[object, BackgroundTaskManager]:
    manager = BackgroundTaskManager()
    tool_ctx.extras[BACKGROUND_EXTRA] = manager
    return tool_ctx, manager


class BlockTool(Tool):
    """Blocks forever — stands in for a long in-flight tool call."""

    def __init__(self) -> None:
        super().__init__(name="block", description="Block forever.", parameters={})

    async def run(self, args, ctx) -> ToolExecResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class WaitNoteTool(Tool):
    """Returns once the manager has a completion notification queued."""

    def __init__(self) -> None:
        super().__init__(name="wait_note", description="Wait for a notification.", parameters={})

    async def run(self, args, ctx) -> ToolExecResult:
        manager = ctx.extras[BACKGROUND_EXTRA]
        for _ in range(500):
            if manager._pending:
                return ToolExecResult("noted")
            await asyncio.sleep(0.01)
        return ToolExecResult("no notification", is_error=True)


# -- bash background lifecycle ----------------------------------------------------


async def test_bash_background_lifecycle(tool_ctx, tmp_path):
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    result = await bash.make_tool().run(
        {"command": "echo bg-hello", "run_in_background": True}, tool_ctx
    )
    assert not result.is_error
    assert result.content == "background task bg-1 started: echo bg-hello"

    record = await manager.wait("bg-1", 5.0)
    assert record.status == "done"
    assert record.exit_code == 0
    assert "bg-hello" in record.output
    log = tmp_path / "cfg" / "tasks" / "bg-1.log"
    assert "bg-hello" in log.read_text(encoding="utf-8")

    notes = manager.drain_notifications()
    assert len(notes) == 1
    assert notes[0].startswith("[background bg-1 done, exit 0]")
    assert "bg-hello" in notes[0]
    assert manager.drain_notifications() == []  # drained once


async def test_bash_background_nonzero_exit(tool_ctx):
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    await bash.make_tool().run({"command": "exit 3", "run_in_background": True}, tool_ctx)
    record = await manager.wait("bg-1", 5.0)
    assert record.status == "failed"
    assert record.exit_code == 3
    assert "[background bg-1 failed, exit 3]" in manager.drain_notifications()[0]


async def test_notify_callback_invoked(tool_ctx):
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    seen: list[str] = []
    manager.notify = seen.append
    await bash.make_tool().run({"command": "echo hi", "run_in_background": True}, tool_ctx)
    await manager.wait("bg-1", 5.0)
    assert seen and seen[0].startswith("[background bg-1 done, exit 0]")


async def test_manager_absent_errors(tool_ctx):
    result = await bash.make_tool().run({"command": "echo hi", "run_in_background": True}, tool_ctx)
    assert result.is_error
    assert "unavailable" in result.content


async def test_tasks_tools_manager_absent(tool_ctx):
    from lecode.agent.tools.background import make_tools

    registry = ToolRegistry(make_tools())
    for name, args in [
        ("tasks_list", "{}"),
        ("tasks_output", '{"id": "bg-1"}'),
        ("tasks_stop", '{"id": "bg-1"}'),
        ("tasks_wait", '{"id": "bg-1"}'),
    ]:
        _, result = await registry.dispatch_result("c1", name, args, tool_ctx)
        assert result.is_error, name
        assert "unavailable" in result.content


# -- tasks_* tools --------------------------------------------------------------------


async def test_tasks_list_and_output(tool_ctx):
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    from lecode.agent.tools.background import make_tools

    registry = ToolRegistry(make_tools())
    _, result = await registry.dispatch_result("c1", "tasks_list", "{}", tool_ctx)
    assert result.content == "(no background tasks)"

    await bash.make_tool().run(
        {"command": "printf 0123456789", "run_in_background": True}, tool_ctx
    )
    await manager.wait("bg-1", 5.0)

    _, result = await registry.dispatch_result("c2", "tasks_list", "{}", tool_ctx)
    assert "bg-1" in result.content
    assert "bash" in result.content
    assert "done" in result.content

    _, result = await registry.dispatch_result("c3", "tasks_output", '{"id": "bg-1"}', tool_ctx)
    assert "0123456789" in result.content
    _, result = await registry.dispatch_result(
        "c4", "tasks_output", '{"id": "bg-1", "tail": 4}', tool_ctx
    )
    assert result.content == "6789"


async def test_tasks_tools_unknown_id(tool_ctx):
    ctx_with_manager(tool_ctx)
    from lecode.agent.tools.background import make_tools

    registry = ToolRegistry(make_tools())
    for name in ("tasks_output", "tasks_stop", "tasks_wait"):
        _, result = await registry.dispatch_result("c1", name, '{"id": "bg-99"}', tool_ctx)
        assert result.is_error, name
        assert "unknown background task: bg-99" in result.content


async def test_tasks_stop_kills_process(tool_ctx):
    tool_ctx, _manager = ctx_with_manager(tool_ctx)
    from lecode.agent.tools.background import make_tools

    registry = ToolRegistry(make_tools())
    await bash.make_tool().run({"command": "sleep 44", "run_in_background": True}, tool_ctx)
    _, result = await registry.dispatch_result("c1", "tasks_stop", '{"id": "bg-1"}', tool_ctx)
    assert not result.is_error
    assert "bg-1 stopped" in result.content
    assert subprocess.run(["pgrep", "-f", "sleep 44"], capture_output=True).stdout == b""


async def test_tasks_stop_escalates_to_sigkill(tool_ctx, monkeypatch):
    """A SIGTERM-ignoring process is SIGKILLed after the grace period."""
    monkeypatch.setattr(background, "STOP_GRACE_S", 0.3)
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    await bash.make_tool().run(
        {"command": "trap '' TERM; sleep 45", "run_in_background": True}, tool_ctx
    )
    record = await manager.stop("bg-1")
    assert record.status == "stopped"
    assert subprocess.run(["pgrep", "-f", "sleep 45"], capture_output=True).stdout == b""


async def test_tasks_wait_returns_final_output(tool_ctx):
    tool_ctx, _manager = ctx_with_manager(tool_ctx)
    from lecode.agent.tools.background import make_tools

    registry = ToolRegistry(make_tools())
    await bash.make_tool().run(
        {"command": "sleep 0.2; echo late-result", "run_in_background": True}, tool_ctx
    )
    _, result = await registry.dispatch_result(
        "c1", "tasks_wait", '{"id": "bg-1", "timeout": 5}', tool_ctx
    )
    assert "bg-1 done, exit 0" in result.content
    assert "late-result" in result.content


async def test_tasks_wait_timeout(tool_ctx):
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    from lecode.agent.tools.background import make_tools

    registry = ToolRegistry(make_tools())
    await bash.make_tool().run({"command": "sleep 46", "run_in_background": True}, tool_ctx)
    try:
        _, result = await registry.dispatch_result(
            "c1", "tasks_wait", '{"id": "bg-1", "timeout": 0.2}', tool_ctx
        )
        assert not result.is_error
        assert "bg-1 still running after 0.2s" in result.content
    finally:
        await manager.shutdown()


async def test_live_task_cap(tool_ctx, monkeypatch):
    monkeypatch.setattr(background, "MAX_LIVE_TASKS", 2)
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    tool = bash.make_tool()
    try:
        for _ in range(2):
            result = await tool.run({"command": "sleep 47", "run_in_background": True}, tool_ctx)
            assert not result.is_error
        result = await tool.run({"command": "sleep 47", "run_in_background": True}, tool_ctx)
        assert result.is_error
        assert "limit reached (2 running)" in result.content
    finally:
        await manager.shutdown()


async def test_shutdown_stops_everything(tool_ctx):
    tool_ctx, manager = ctx_with_manager(tool_ctx)
    tool = bash.make_tool()
    await tool.run({"command": "sleep 48", "run_in_background": True}, tool_ctx)
    await tool.run({"command": "sleep 48", "run_in_background": True}, tool_ctx)
    await manager.shutdown()
    assert all(r.status == "stopped" for r in manager.tasks())
    assert subprocess.run(["pgrep", "-f", "sleep 48"], capture_output=True).stdout == b""


# -- subagent background -------------------------------------------------------------


async def test_task_tool_background(tmp_path, monkeypatch):
    provider = FakeProvider([{"text": "child finished"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    manager = runtime.ctx.extras[BACKGROUND_EXTRA]
    conversation = [{"role": "user", "content": "parent"}]
    runtime.ctx.extras["conversation"] = conversation

    _, result = await runtime.registry.dispatch_result(
        "c1",
        "task",
        '{"prompt": "scan the repo", "run_in_background": true}',
        runtime.ctx,
    )
    assert not result.is_error
    assert result.content.startswith("background task bg-1 started (explore)")

    record = await manager.wait("bg-1", 5.0)
    assert record.status == "done"
    assert record.kind == "agent"
    assert "child finished" in record.output
    # The parent's conversation seam survived the detached child run.
    assert runtime.ctx.extras["conversation"] is conversation


async def test_task_tool_background_manager_absent(tmp_path, monkeypatch):
    provider = FakeProvider([])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    del runtime.ctx.extras[BACKGROUND_EXTRA]
    _, result = await runtime.registry.dispatch_result(
        "c1", "task", '{"prompt": "x", "run_in_background": true}', runtime.ctx
    )
    assert result.is_error
    assert "unavailable" in result.content


async def test_task_tool_background_unknown_agent(tmp_path, monkeypatch):
    provider = FakeProvider([])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    manager = runtime.ctx.extras[BACKGROUND_EXTRA]
    _, result = await runtime.registry.dispatch_result(
        "c1", "task", '{"prompt": "x", "agent": "nope", "run_in_background": true}', runtime.ctx
    )
    assert not result.is_error  # the start succeeds; the failure lands async
    record = await manager.wait("bg-1", 5.0)
    assert record.status == "failed"
    assert "unknown subagent: nope" in record.output


# -- runner integration ----------------------------------------------------------------


async def test_completion_notification_injected_next_turn(tmp_path, monkeypatch):
    script = [
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "bash",
                    "arguments": json.dumps({"command": "echo bg-done", "run_in_background": True}),
                }
            ]
        },
        {"tool_calls": [{"id": "c2", "name": "wait_note", "arguments": "{}"}]},
        {"text": "all done"},
    ]
    provider = FakeProvider(script)
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    runtime.registry.register(WaitNoteTool())
    runner = AgentRunner(provider, runtime.registry, runtime.ctx)

    result = await runner.run([{"role": "user", "content": "go"}])

    assert result.final_text == "all done"
    injected = [
        m["content"]
        for request in provider.requests[1:]
        for m in request["messages"]
        if m["role"] == "user" and str(m["content"]).startswith("[background bg-1")
    ]
    assert injected, "no background notification reached the model"
    assert "[background bg-1 done, exit 0]" in injected[0]
    assert "bg-done" in injected[0]


async def test_notification_drained_at_run_start(tmp_path, monkeypatch):
    provider = FakeProvider([{"text": "hi"}])
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    manager = runtime.ctx.extras[BACKGROUND_EXTRA]
    await bash.make_tool().run({"command": "echo earlier", "run_in_background": True}, runtime.ctx)
    await manager.wait("bg-1", 5.0)  # finishes before the run starts

    runner = AgentRunner(provider, runtime.registry, runtime.ctx)
    await runner.run([{"role": "user", "content": "go"}])

    contents = [m["content"] for m in provider.requests[0]["messages"]]
    assert any(str(c).startswith("[background bg-1 done, exit 0]") for c in contents)


async def test_background_survives_turn_cancellation(tmp_path, monkeypatch):
    """Ctrl-C cancels the turn; the manager-owned task keeps running."""
    script = [
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "bash",
                    "arguments": json.dumps({"command": "sleep 49", "run_in_background": True}),
                }
            ]
        },
        {"tool_calls": [{"id": "c2", "name": "block", "arguments": "{}"}]},
    ]
    provider = FakeProvider(script)
    runtime = make_runtime(tmp_path, monkeypatch, provider)
    runtime.registry.register(BlockTool())
    manager = runtime.ctx.extras[BACKGROUND_EXTRA]
    runner = AgentRunner(provider, runtime.registry, runtime.ctx)

    run_task = asyncio.ensure_future(runner.run([{"role": "user", "content": "go"}]))
    await wait_for(lambda: manager.get("bg-1") is not None)
    await wait_for(lambda: len(provider.requests) == 2)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    record = manager.get("bg-1")
    assert record.status == "running"  # not cancelled with the turn
    await manager.shutdown()
    assert record.status == "stopped"
    assert subprocess.run(["pgrep", "-f", "sleep 49"], capture_output=True).stdout == b""


# -- headless teardown -----------------------------------------------------------------


def test_headless_teardown_stops_background_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.setattr("lecode.cli.find_missing_binaries", lambda: [])
    script = [
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "bash",
                    "arguments": json.dumps({"command": "sleep 43", "run_in_background": True}),
                }
            ]
        },
        {"text": "started it"},
    ]
    provider = FakeProvider(script)
    monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: provider)
    from lecode.agent.builder import build_runtime as real_build_runtime

    captured: dict = {}

    def capturing_build(*args, **kwargs):
        runtime = real_build_runtime(*args, **kwargs)
        captured["runtime"] = runtime
        return runtime

    monkeypatch.setattr("lecode.cli.build_runtime", capturing_build)

    from lecode.cli import app

    result = CliRunner().invoke(app, ["-p", "start something slow"])
    assert result.exit_code == 0, result.stderr
    manager = captured["runtime"].ctx.extras[BACKGROUND_EXTRA]
    records = manager.tasks()
    assert records and all(r.status == "stopped" for r in records)
    assert subprocess.run(["pgrep", "-f", "sleep 43"], capture_output=True).stdout == b""


# -- /tasks slash command ----------------------------------------------------------------


async def test_tasks_command_empty(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tasks")
    assert "(no background tasks)" in out.getvalue()


async def test_tasks_command_lists_and_notifies(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    manager = app.runtime.ctx.extras[BACKGROUND_EXTRA]
    assert manager.notify is not None  # the TUI installed its feed reporter
    result = await bash.make_tool().run(
        {"command": "echo tui-done", "run_in_background": True}, app.runtime.ctx
    )
    assert not result.is_error
    await manager.wait("bg-1", 5.0)
    await wait_for(lambda: "[background bg-1 done, exit 0]" in out.getvalue())

    await app.handle_command("/tasks")
    rendered = out.getvalue()
    assert "bg-1" in rendered
    assert "bash" in rendered
    assert "done, exit 0" in rendered
