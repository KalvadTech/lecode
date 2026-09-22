"""WorkerManager's public persistence and scheduling contract."""

import asyncio
import json
import shlex
import subprocess
import sys
from dataclasses import replace

import pytest
from tests.fakes import FakeProvider
from tests.test_worker_controls import workflow as workflow

from lecode.agent.builder import build_runtime
from lecode.agent.runner import (
    AgentRunner,
    LlmCall,
    LlmResponse,
    Token,
)
from lecode.agent.runner import Done as RunnerDone
from lecode.config.models import Config
from lecode.context.agents import AgentDefinition, AgentRegistry
from lecode.extras.subagents import SubagentError
from lecode.extras.workers import WORKER_CURRENT_EXTRA, WorkerManager
from lecode.extras.worktree import WorktreeError, WorktreeManager
from lecode.hooks import dispatcher_from_config
from lecode.permission.checker import AgentOverlay
from lecode.session.stats import session_stats
from lecode.session.storage import SessionInUseError, SessionStore


class GatedProvider(FakeProvider):
    def __init__(self):
        super().__init__([])
        self.release = asyncio.Event()
        self.started = asyncio.Queue()
        self.active = 0
        self.peak = 0

    async def _stream(self, entry):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.put_nowait(None)
        try:
            await self.release.wait()
            async for event in super()._stream({"text": "done"}):
                yield event
        finally:
            self.active -= 1


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = Config()
    config.memory.enabled = False
    config.lsp.enabled = False
    config.pierre.enabled = False
    store = SessionStore(tmp_path / "config")
    session = store.create("main", cwd)
    runtime = build_runtime(config, cwd, session=session, store=store)
    provider = FakeProvider([{"text": "first"}, {"text": "second"}])
    runtime.ctx.extras["provider"] = provider
    manager = WorkerManager(config, cwd=cwd, root_ctx=runtime.ctx)
    runtime.ctx.extras["workers"] = manager
    return manager, runtime.ctx, provider, store, session


@pytest.mark.asyncio
@pytest.mark.parametrize("readonly", [False, True])
async def test_default_general_task_inherits_permissions_and_isolates_writes(setup, readonly):
    from tests.test_worker_controls import commit, git

    manager, ctx, _, _, _ = setup
    git(ctx.cwd, "init", "-b", "main")
    commit(ctx.cwd, "base\n")
    ctx.auto_approve = True
    if readonly:
        ctx.permission_checker = ctx.permission_checker.for_agent(AgentOverlay(mode="readonly"))
    ctx.extras["provider"] = FakeProvider(
        [
            {
                "tool_calls": [
                    {
                        "id": "write",
                        "name": "write",
                        "arguments": json.dumps({"path": "new.txt", "content": "implemented"}),
                    }
                ]
            },
            {"text": "done"},
        ]
    )
    try:
        _, result = await ctx.extras["registry"].dispatch_result(
            "create",
            "task",
            '{"agent":"general","prompt":"Implement","run_in_background":true}',
            ctx,
        )
        assert not result.is_error, result.content
        worker = manager.get(result.metadata["worker_id"])
        await manager.wait(worker.id)
        assert worker.agent == "general"
        assert (worker.worktree is None) == readonly
        assert (worker.cwd == ctx.cwd) == readonly
        assert not (ctx.cwd / "new.txt").exists()
        if not readonly:
            assert (worker.cwd / "new.txt").read_text() == "implemented"
        else:
            messages = manager.store.load_for_model(worker.session)
            assert any(m["role"] == "tool" and "denied" in m["content"].lower() for m in messages)
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_worker_creation_discovery_and_control_errors(setup):
    manager, ctx, _, _, _ = setup
    registry = ctx.extras["registry"]
    try:
        for name in ("missing", "build", "plan"):
            _, result = await registry.dispatch_result(
                "bad-agent", "task", json.dumps({"agent": name, "prompt": "work"}), ctx
            )
            assert result.is_error
            assert "available: explore, general" in result.content
        _, result = await registry.dispatch_result(
            "bad-id", "workers", '{"action":"send","id":"general","text":"work"}', ctx
        )
        assert result.is_error
        assert "unknown worker id" in result.content
        assert "task(agent='general'" in result.content
        assert "returned worker_id" in result.content
        _, result = await registry.dispatch_result("empty", "workers", '{"action":"list"}', ctx)
        assert "Create one with task" in result.content
        _, result = await registry.dispatch_result(
            "no-git", "task", '{"agent":"general","prompt":"work"}', ctx
        )
        assert result.is_error
        assert "Restart the session from a Git repository with a committed HEAD" in result.content
        assert "shell cd does not change" in result.content
        assert manager.list() == []
        assert "does not create workers" in registry.get("workers").description
        assert "actual worker_id" in registry.get("task").description
        assert (
            "workers"
            in registry.get("task").parameters["properties"]["run_in_background"]["description"]
        )
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_followup_is_persisted_and_replayed(setup):
    manager, ctx, provider, store, _ = setup
    worker = await manager.start(ctx, agent="explore", prompt="question")
    try:
        assert (await manager.wait(worker.id)).final_text == "first"
        await manager.send(worker.id, "follow up")
        assert (await manager.wait(worker.id)).final_text == "second"
        messages = store.load_for_model(worker.session)
        assert [(m["role"], m["content"]) for m in messages] == [
            ("user", "question"),
            ("assistant", "first"),
            ("user", "follow up"),
            ("assistant", "second"),
        ]
        assert provider.requests[1]["messages"][1:] == messages[:-1]
        assert manager.pending(worker.id) == []
        assert worker.usage_incomplete  # Provider omitted usage, not a known zero cost.
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_task_tool_creates_persisted_workers_and_nests(setup):
    manager, ctx, _, _, _ = setup
    ctx.extras["provider"] = FakeProvider(
        [
            {"tool_calls": [{"id": "nested", "name": "task", "arguments": '{"prompt":"child"}'}]},
            {"text": "child result"},
            {"text": "parent result"},
        ]
    )
    try:
        outer = await manager.start(ctx, agent="explore", prompt="parent")
        assert (await manager.wait(outer.id)).final_text == "parent result"
        child = manager.children(outer.id)[0]
        assert child.depth == 2
        assert child.result is not None and child.result.final_text == "child result"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_background_task_tool_delivers_worker_notification(setup):
    manager, ctx, _, _, _ = setup
    try:
        _, result = await ctx.extras["registry"].dispatch_result(
            "background", "task", '{"prompt":"scan","run_in_background":true}', ctx
        )
        assert not result.is_error
        worker_id = result.metadata["worker_id"]
        await manager.wait(worker_id)
        history = []
        assert manager.consume(None, history)
        assert worker_id in history[0]["content"]
        assert "first" in history[0]["content"]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_task_uses_worker_manager_when_tui_events_are_installed(setup):
    manager, ctx, _, _, _ = setup
    ctx.extras["subagent_events"] = lambda event: None
    try:
        _, result = await ctx.extras["registry"].dispatch_result(
            "task", "task", '{"prompt":"scan"}', ctx
        )
        assert not result.is_error
        assert result.metadata["worker_id"] == manager.list()[0].id
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_worker_forwards_all_runner_events_to_tui_callback(setup):
    manager, ctx, _, _, _ = setup
    seen = []
    ctx.extras["subagent_events"] = seen.append
    try:
        worker = await manager.start(ctx, agent="explore", prompt="scan")
        await manager.wait(worker.id)
        assert {type(progress.event) for progress in seen} == {
            LlmCall,
            Token,
            LlmResponse,
            RunnerDone,
        }
        assert {progress.run_id for progress in seen} == {worker.id}
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_parent_cancellation_does_not_cancel_managed_worker(setup):
    manager, ctx, _, _, _ = setup
    provider = GatedProvider()
    ctx.extras["provider"] = provider
    call = asyncio.create_task(
        ctx.extras["registry"].dispatch_result("task", "task", '{"prompt":"scan"}', ctx)
    )
    try:
        async with asyncio.timeout(2):
            await provider.started.get()
        worker = manager.list()[0]
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert worker.state == "running"
        provider.release.set()
        await manager.wait(worker.id)
    finally:
        if not call.done():
            call.cancel()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_worker_controls_enforce_descendant_hierarchy_and_questions(setup):
    from lecode.agent.tools.workers import make_tool

    manager, ctx, _, _, _ = setup
    parent = await manager.start(ctx, agent="explore", prompt="parent")
    nested = replace(ctx, extras={**ctx.extras, WORKER_CURRENT_EXTRA: parent.id})
    child = await manager.start(nested, agent="explore", prompt="child")
    child_ctx = replace(ctx, extras={**ctx.extras, WORKER_CURRENT_EXTRA: child.id})
    tool = make_tool()
    try:
        listed = await tool.run({"action": "list"}, nested)
        assert child.id in listed.content and parent.id not in listed.content
        denied = await tool.run({"action": "stop", "id": parent.id}, nested)
        assert denied.is_error and "descendants" in denied.content
        invalid = await tool.run({"action": "send", "id": child.id, "text": 3}, nested)
        assert invalid.is_error
        asked = await tool.run({"action": "question", "text": "Need a choice"}, child_ctx)
        assert asked.content == "question sent to parent"
        history = []
        assert manager.consume(parent.id, history)
        assert any("Need a choice" in message["content"] for message in history)
    finally:
        await manager.shutdown()


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.asyncio
async def test_write_worktree_pins_parent_head_and_readonly_shares_parent_cwd(setup):
    manager, ctx, _, _, _ = setup
    git(ctx.cwd, "init", "-b", "main")
    git(
        ctx.cwd,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    agents = ctx.extras["agents"]
    ctx.extras["agents"] = AgentRegistry(
        {
            "explore": agents.get("explore"),
            "writer": AgentDefinition("writer", "write", "", mode="subagent"),
        }
    )
    try:
        writer = await manager.start(ctx, agent="writer", prompt="write")
        await manager.wait(writer.id)
        assert writer.cwd != ctx.cwd
        assert writer.worktree.path == writer.cwd
        wm = await WorktreeManager.discover(ctx.cwd)
        sidecar = wm.read_sidecar(writer.worktree.name)
        assert sidecar["dest_path"] == str(ctx.cwd)
        assert sidecar["dest_branch"] == "main"
        assert sidecar["base_commit"] == git(ctx.cwd, "rev-parse", "HEAD")
        nested = replace(
            ctx, cwd=writer.cwd, extras={**ctx.extras, WORKER_CURRENT_EXTRA: writer.id}
        )
        reader = await manager.start(nested, agent="explore", prompt="read")
        await manager.wait(reader.id)
        assert reader.cwd == writer.cwd
        assert reader.worktree is None
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_write_worker_refuses_non_git(setup):
    manager, ctx, _, _, _ = setup
    ctx.extras["agents"] = AgentRegistry(
        {
            "writer": AgentDefinition("writer", "write", "", mode="subagent"),
        }
    )
    try:
        with pytest.raises(WorktreeError, match="git"):
            await manager.start(ctx, agent="writer", prompt="write")
        assert manager.list() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_restart_keeps_inbox_and_stopped_intent_without_executing(setup):
    manager, ctx, provider, store, session = setup
    worker = await manager.start(ctx, agent="explore", prompt="original")
    await manager.stop(worker.id)
    await manager.send(worker.id, "later")
    await manager.shutdown()
    restored = WorkerManager(ctx.config, cwd=ctx.cwd, root_ctx=ctx, store=store, session=session)
    try:
        assert restored.load()[0].id == worker.id
        assert restored.get(worker.id).state == "stopped"
        assert [m["text"] for m in restored.pending(worker.id)] == ["original", "later"]
        assert provider.requests == []
        assert restored.load() == restored.list()
    finally:
        await restored.shutdown()


@pytest.mark.asyncio
async def test_depth_and_agent_eligibility_fail_before_creating_sessions(setup):
    manager, ctx, _, store, _ = setup
    try:
        first = await manager.start(ctx, agent="explore", prompt="one")
        nested = replace(ctx, extras={**ctx.extras, WORKER_CURRENT_EXTRA: first.id})
        second = await manager.start(nested, agent="explore", prompt="two")
        nested = replace(nested, extras={**nested.extras, WORKER_CURRENT_EXTRA: second.id})
        before = len(store.list_sessions())
        with pytest.raises(SubagentError, match="depth"):
            await manager.start(nested, agent="explore", prompt="three")
        for name in ("missing", "build"):
            with pytest.raises(SubagentError, match="ineligible"):
                await manager.start(ctx, agent=name, prompt="no")
        assert len(store.list_sessions()) == before
        assert manager.children(first.id) == [second]
        assert [first.depth, second.depth] == [1, 2]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_strict_cap_and_shielded_wait(setup):
    manager, ctx, _, _, _ = setup
    provider = GatedProvider()
    ctx.extras["provider"] = provider
    try:
        workers = [await manager.start(ctx, agent="explore", prompt=str(i)) for i in range(12)]
        async with asyncio.timeout(2):
            for _ in range(10):
                await provider.started.get()
        await asyncio.sleep(0)
        assert provider.active == 10
        assert sum(w.state == "queued" for w in workers) == 2
        waiter = asyncio.create_task(manager.wait(workers[0].id))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert workers[0].state == "running"
        provider.release.set()
        await asyncio.gather(*(manager.wait(w.id) for w in workers))
        assert provider.peak == 10
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_send_does_not_wake_stopped_worker_even_with_interrupt(setup):
    manager, ctx, provider, _, _ = setup
    try:
        worker = await manager.start(ctx, agent="explore", prompt="initial")
        await manager.stop(worker.id)
        await manager.send(worker.id, "steer", interrupt=True)
        await asyncio.sleep(0)
        assert worker.state == "stopped"
        assert provider.requests == []
        assert [m["text"] for m in manager.pending(worker.id)] == ["initial", "steer"]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_failed_usage_survives_restart_and_child_locks_last_until_shutdown(setup):
    manager, ctx, _, store, session = setup
    ctx.extras["provider"] = FakeProvider(
        [
            {
                "tool_calls": [{"id": "call", "name": "read", "arguments": '{"path":"missing"}'}],
                "usage": {"input_tokens": 7, "output_tokens": 3, "cost_usd": 0.25},
            },
            {"error": RuntimeError("provider broke")},
        ]
    )
    worker = await manager.start(ctx, agent="explore", prompt="question")
    try:
        with pytest.raises(SubagentError, match="provider broke"):
            await manager.wait(worker.id)
        assert worker.usage_totals.input_tokens == 7
        assert worker.usage_totals.cost_usd == 0.25
        assert worker.usage_incomplete
        assert [m["role"] for m in store.load_for_model(worker.session)] == [
            "user",
            "assistant",
            "tool",
        ]
        assert store.load_events(worker.session, "worker_usage_checkpoint")[-1]["input_tokens"] == 7
        with pytest.raises(SessionInUseError):
            store.acquire_lock(worker.session)
    finally:
        await manager.shutdown()
    lock = store.acquire_lock(worker.session)
    lock.release()
    assert worker.session.path.with_suffix(".lock").exists()
    restored = WorkerManager(ctx.config, cwd=ctx.cwd, root_ctx=ctx, store=store, session=session)
    try:
        restored.load()
        assert restored.get(worker.id).usage_totals == worker.usage_totals
        assert restored.get(worker.id).state == "failed"
    finally:
        await restored.shutdown()


@pytest.mark.asyncio
async def test_load_interrupted_uses_child_usage_if_root_snapshot_lagged(setup):
    manager, ctx, provider, store, session = setup
    worker = await manager.start(ctx, agent="explore", prompt="original")
    snapshot = store.load_events(session, "worker")[-1]
    await manager.shutdown()
    # Simulate a process dying after child usage persisted, before root update.
    store.append_event(
        worker.session,
        "worker_usage_checkpoint",
        {
            "dispatch_id": worker.dispatch_id,
            "input_tokens": 9,
            "output_tokens": 2,
            "cost_usd": 0.5,
            "context_tokens": 9,
        },
    )
    store.append_event(session, "worker", snapshot)
    restored = WorkerManager(ctx.config, cwd=ctx.cwd, root_ctx=ctx)
    try:
        loaded = restored.load()[0]
        assert loaded.state == "interrupted"
        assert loaded.usage_incomplete
        assert loaded.usage_totals.input_tokens == 9
        assert provider.requests == []
    finally:
        await restored.shutdown()


@pytest.mark.asyncio
async def test_interrupt_repairs_unanswered_calls_without_replaying_inputs(setup):
    manager, ctx, provider, store, _ = setup
    worker = await manager.start(ctx, agent="explore", prompt="original")
    await manager.stop(worker.id)
    manager.consume(worker.id, [])
    store.append_message(
        worker.session,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "unanswered",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
    )
    try:
        message_id = await manager.send(worker.id, "new direction", interrupt=True)
        assert manager.pending(worker.id)[0]["id"] == message_id
        with pytest.raises(RuntimeError, match="outstanding"):
            manager.consume(worker.id, store.load_for_model(worker.session))
        assert [m["role"] for m in store.load_for_model(worker.session)] == ["user", "assistant"]
        await manager.resume(worker.id)
        await manager.wait(worker.id)
        history = provider.requests[0]["messages"][1:]
        assert [m["role"] for m in history] == ["user", "assistant", "tool", "user"]
        assert history[2]["tool_call_id"] == "unanswered"
        assert "unknown" in history[2]["content"]
        assert history[3]["content"] == "new direction"
        assert manager.pending(worker.id) == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_running_inbox_is_durable_but_only_consumed_after_safe_boundary(setup):
    manager, ctx, _, store, _ = setup
    provider = GatedProvider()
    ctx.extras["provider"] = provider
    try:
        worker = await manager.start(ctx, agent="explore", prompt="initial")
        async with asyncio.timeout(2):
            await provider.started.get()
        await manager.send(worker.id, "follow up")
        assert [m["content"] for m in store.load_for_model(worker.session)] == ["initial"]
        assert [m["text"] for m in manager.pending(worker.id)] == ["follow up"]
        provider.release.set()
        await manager.wait(worker.id)
        assert len(provider.requests) == 2
        assert manager.pending(worker.id) == []
        assert [m["role"] for m in store.load_for_model(worker.session)] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_completion_delivery_and_idempotent_human_submit(setup):
    manager, ctx, _, _, _ = setup
    notifications = []
    manager.notify = notifications.append
    try:
        foreground = await manager.start(ctx, agent="explore", prompt="foreground")
        await manager.wait(foreground.id)
        assert [n["worker_id"] for n in manager.drain_notifications()] == [foreground.id]
        assert notifications == []
        background = await manager.start(ctx, agent="explore", prompt="background", background=True)
        await manager.wait(background.id)
        notes = manager.drain_notifications()
        assert [n["worker_id"] for n in notes] == [background.id]
        assert manager.drain_notifications() == []
        human = await manager.start(
            ctx, agent="explore", prompt="human", origin="human", background=True
        )
        await manager.wait(human.id)
        assert notifications[-1]["worker_id"] == human.id
        assert manager.drain_notifications() == []
        assert (await manager.submit(human.id))["new"]
        assert [n["worker_id"] for n in manager.drain_notifications()] == [human.id]
        assert not (await manager.submit(human.id))["new"]
        assert manager.drain_notifications() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_submit_rejects_delegated_worker(setup):
    manager, ctx, _, _, _ = setup
    try:
        worker = await manager.start(ctx, agent="explore", prompt="delegated")
        await manager.wait(worker.id)
        with pytest.raises(SubagentError, match="only human workers"):
            await manager.submit(worker.id)
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_completed_worker_followup_after_restart(setup):
    manager, ctx, provider, store, _ = setup
    worker = await manager.start(ctx, agent="explore", prompt="first question")
    await manager.wait(worker.id)
    await manager.shutdown()
    restored = WorkerManager(ctx.config, cwd=ctx.cwd, root_ctx=ctx)
    try:
        restored.load()
        assert (await restored.wait(worker.id)).final_text == "first"
        await restored.send(worker.id, "second question")
        assert (await restored.wait(worker.id)).final_text == "second"
        assert provider.requests[1]["messages"][1:] == store.load_for_model(worker.session)[:-1]
    finally:
        await restored.shutdown()


@pytest.mark.asyncio
async def test_stop_is_individual_unless_tree_requested(setup):
    manager, ctx, _, _, _ = setup
    provider = GatedProvider()
    ctx.extras["provider"] = provider
    try:
        parent = await manager.start(ctx, agent="explore", prompt="parent")
        nested = replace(ctx, extras={**ctx.extras, WORKER_CURRENT_EXTRA: parent.id})
        child = await manager.start(nested, agent="explore", prompt="child")
        other = await manager.start(ctx, agent="explore", prompt="other")
        async with asyncio.timeout(2):
            for _ in range(3):
                await provider.started.get()
        await manager.stop(parent.id)
        assert parent.state == "stopped"
        assert child.state == other.state == "running"
        await manager.stop(parent.id, tree=True)
        assert child.state == "stopped"
        assert other.state == "running"
        provider.release.set()
        await manager.wait(other.id)
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_runtime_is_rebuilt_with_fresh_grants_and_cwd_bound_extras(setup, monkeypatch):
    manager, ctx, _, _, _ = setup
    ctx.session_perms.grant("bash", "*")
    ctx.extras["registry"].unregister("bash")
    captured = []
    original_run = AgentRunner.run

    async def run(runner, *args, **kwargs):
        captured.append(runner.ctx)
        return await original_run(runner, *args, **kwargs)

    monkeypatch.setattr(AgentRunner, "run", run)
    try:
        worker = await manager.start(ctx, agent="explore", prompt="read")
        await manager.wait(worker.id)
        child = captured[0]
        assert child.cwd == ctx.cwd
        assert child.session is worker.session
        assert child.session_perms is not ctx.session_perms
        assert child.session_perms.grants == []
        assert child.extras["background"] is not ctx.extras["background"]
        assert child.extras["registry"] is not ctx.extras["registry"]
        assert "bash" not in child.extras["registry"].names()
        assert child.permission_checker.read_only
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_real_parent_checker_restrictions_reach_child_dispatch(setup):
    manager, ctx, _, _, _ = setup
    ctx.permission_checker = ctx.permission_checker.for_agent(AgentOverlay(denied_tools=("read",)))
    ctx.extras["provider"] = FakeProvider(
        [
            {"tool_calls": [{"id": "call", "name": "read", "arguments": '{"path":"file"}'}]},
            {"text": "denied as expected"},
        ]
    )
    try:
        worker = await manager.start(ctx, agent="explore", prompt="read")
        await manager.wait(worker.id)
        tool = ctx.extras["provider"].requests[1]["messages"][-1]
        assert tool["role"] == "tool"
        assert "denied" in tool["content"]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_usage_records_do_not_double_count_child_transcript(setup):
    manager, ctx, _, store, session = setup
    ctx.extras["provider"] = FakeProvider(
        [
            {
                "tool_calls": [{"id": "call", "name": "read", "arguments": '{"path":"missing"}'}],
                "usage": {"input_tokens": 7, "output_tokens": 3, "cost_usd": 0.25},
            },
            {"text": "answer", "usage": {"input_tokens": 11, "output_tokens": 5, "cost_usd": 0.5}},
        ]
    )
    try:
        worker = await manager.start(ctx, agent="explore", prompt="question")
        await manager.wait(worker.id)
        assert worker.usage_totals.input_tokens == 18
        assert session_stats(store, session).input_tokens == 18
        assert session_stats(store, worker.session).input_tokens == 18
        assert session_stats(store, session).cost_usd == 0.75
        assert not worker.usage_incomplete
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_stopped_worker_does_not_return_a_stale_result(setup):
    manager, ctx, _, _, _ = setup
    try:
        worker = await manager.start(ctx, agent="explore", prompt="question")
        await manager.wait(worker.id)
        await manager.stop(worker.id)
        with pytest.raises(SubagentError, match="stopped"):
            await manager.wait(worker.id)
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_rejects_followup_before_writing_without_child_lock(setup):
    manager, ctx, _, _, _ = setup
    worker = await manager.start(ctx, agent="explore", prompt="question")
    await manager.shutdown()
    before = worker.session.path.read_text()
    with pytest.raises(RuntimeError, match="shut down"):
        await manager.send(worker.id, "late")
    with pytest.raises(RuntimeError, match="shut down"):
        await manager.resume(worker.id, "late")
    assert worker.session.path.read_text() == before


@pytest.mark.asyncio
async def test_dirty_root_requires_human_confirmation_and_detached_is_refused(setup):
    manager, ctx, _, _, _ = setup
    git(ctx.cwd, "init", "-b", "main")
    git(
        ctx.cwd,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    ctx.extras["agents"] = AgentRegistry(
        {
            "writer": AgentDefinition("writer", "write", "", mode="subagent"),
        }
    )
    ctx.auto_approve = True
    dirty = ctx.cwd / "uncommitted.txt"
    dirty.write_text("not committed")
    try:
        with pytest.raises(WorktreeError, match="human confirmation"):
            await manager.start(ctx, agent="writer", prompt="write")
        confirmations = []

        async def confirm(question):
            confirmations.append(question)
            return True

        manager.confirm = confirm
        worker = await manager.start(ctx, agent="writer", prompt="write")
        assert len(confirmations) == 1
        assert isinstance(confirmations[0], str)
        assert worker.id in confirmations[0] and str(ctx.cwd) in confirmations[0]
        assert not (worker.cwd / dirty.name).exists()
        assert dirty.read_text() == "not committed"
        git(ctx.cwd, "checkout", "--detach")
        with pytest.raises(WorktreeError, match="detached"):
            await manager.start(ctx, agent="writer", prompt="write")
    finally:
        await manager.shutdown()


@pytest.mark.parametrize(
    "agent_model,subagent_model,expected",
    [
        ("agent-model", "subagent-model", "agent-model"),
        (None, "subagent-model", "subagent-model"),
        (None, None, "main-model"),
    ],
)
@pytest.mark.asyncio
async def test_worker_model_precedence_is_recorded(setup, agent_model, subagent_model, expected):
    manager, ctx, provider, _, _ = setup
    ctx.config.llm.model = "main-model"
    ctx.config.agent.subagent_model = subagent_model
    ctx.extras["agents"] = AgentRegistry(
        {
            "explore": replace(ctx.extras["agents"].get("explore"), model=agent_model),
        }
    )
    try:
        worker = await manager.start(ctx, agent="explore", prompt="question")
        await manager.wait(worker.id)
        assert provider.requests[0]["model"] == expected
        assert worker.session.meta.model == expected
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_followup_pins_session_model_in_provider_context_and_events(setup, restart):
    manager, ctx, provider, store, session = setup
    ctx.config.agent.subagent_model = "pinned-model"
    events = []
    ctx.extras["subagent_events"] = events.append
    worker = await manager.start(ctx, agent="explore", prompt="initial")
    await manager.wait(worker.id)
    ctx.config.agent.subagent_model = "new-default"
    ctx.extras["agents"] = AgentRegistry(
        {"explore": replace(ctx.extras["agents"].get("explore"), model="new-agent-model")}
    )
    if restart:
        await manager.shutdown()
        manager = WorkerManager(ctx.config, cwd=ctx.cwd, root_ctx=ctx, store=store, session=session)
        worker = manager.load()[0]
    try:
        await manager.send(worker.id, "follow up")
        await manager.wait(worker.id)
        assert [r["model"] for r in provider.requests] == ["pinned-model"] * 2
        assert manager._runtime(worker).ctx.config.llm.model == worker.session.meta.model
        assert {p.event.model for p in events if isinstance(p.event, (LlmCall, LlmResponse))} == {
            "pinned-model"
        }
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["max_turns", "empty", "context_overflow"])
async def test_non_success_worker_stop_preserves_result_and_allows_resume(setup, reason):
    manager, ctx, _, store, session = setup
    tool = {
        "tool_calls": [{"id": "read", "name": "read", "arguments": '{"file_path":"missing"}'}],
        "usage": {"input_tokens": 900, "cost_usd": 0.25},
    }
    if reason == "max_turns":
        ctx.config.agent.max_turns = 1
        script = [tool]
    elif reason == "empty":
        script = [{"usage": {"input_tokens": 7, "cost_usd": 0.25}}] * 4
    else:
        ctx.config.agent.context_window = 3300
        ctx.config.compaction.buffer_tokens = 200
        ctx.config.compaction.on_overflow = "pause"
        script = [tool, tool, {"text": "summary"}, tool]
    provider = FakeProvider(script)
    ctx.extras["provider"] = provider
    try:
        worker = await manager.start(ctx, agent="explore", prompt="work")
        with pytest.raises(SubagentError, match=reason):
            await manager.wait(worker.id)
        assert worker.state == "failed"
        assert worker.result.stop_reason == reason
        assert worker.result.usage_totals.cost_usd > 0
        assert worker.usage_totals.cost_usd == worker.result.usage_totals.cost_usd
        note = store.load_events(session, "worker_notification")[-1]
        assert note["state"] == "failed" and reason in note["content"]
        ctx.config.agent.max_turns = 10
        ctx.config.agent.context_window = 10000
        ctx.config.compaction.enabled = False
        await manager.resume(worker.id, "continue explicitly")
        assert (await manager.wait(worker.id)).stop_reason == "done"
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_mixed_tool_batches_release_slots_only_after_ordinary_work(setup, cancel):
    manager, ctx, _, store, _ = setup
    parents_ready = asyncio.Event()
    children_ready = asyncio.Event()
    release_children = asyncio.Event()
    parent_calls = child_calls = 0

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            nonlocal parent_calls, child_calls
            assert sum(w.state == "running" for w in manager.list()) <= 10
            prompt = messages[1]["content"]
            if prompt == "parent" and len(messages) == 2:
                parent_calls += 1
                if parent_calls == 10:
                    parents_ready.set()
                await parents_ready.wait()
                entry = {
                    "tool_calls": [
                        {"id": "child", "name": "task", "arguments": '{"prompt":"child"}'},
                        {
                            "id": "ordinary",
                            "name": "read",
                            "arguments": '{"file_path":"sample.txt"}',
                        },
                    ]
                }
            elif prompt == "child":
                child_calls += 1
                if child_calls == 10:
                    children_ready.set()
                await release_children.wait()
                entry = {"text": "child result"}
            else:
                assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
                    "child",
                    "ordinary",
                ]
                entry = {"text": "reviewed"}
            async for event in self._stream(entry):
                yield event

    ctx.cwd.joinpath("sample.txt").write_text("ordinary result")
    ctx.extras["provider"] = Provider([])
    try:
        parents = [await manager.start(ctx, agent="explore", prompt="parent") for _ in range(10)]
        async with asyncio.timeout(3):
            await children_ready.wait()
        assert sum(w.state == "waiting" for w in manager.list()) == 10
        assert sum(w.state == "running" for w in manager.list()) == 10
        if cancel:
            async with asyncio.timeout(2):
                await asyncio.gather(*(manager.stop(w.id) for w in parents))
            assert all(w.state == "stopped" and not w.is_active for w in parents)
        release_children.set()
        if cancel:
            async with asyncio.timeout(2):
                await asyncio.gather(
                    *(
                        manager.wait(child.id)
                        for parent in parents
                        for child in manager.children(parent.id)
                    )
                )
            assert all(w.state == "stopped" for w in parents)
            for worker in parents:
                assert [
                    m["tool_call_id"]
                    for m in store.load_for_model(worker.session)
                    if m["role"] == "tool"
                ] == ["ordinary"]
            return
        async with asyncio.timeout(3):
            assert all(
                r.final_text == "reviewed"
                for r in await asyncio.gather(*(manager.wait(w.id) for w in parents))
            )
        for worker in parents:
            assert [
                m["tool_call_id"]
                for m in store.load_for_model(worker.session)
                if m["role"] == "tool"
            ] == ["child", "ordinary"]
    finally:
        release_children.set()
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("child_error", [False, True])
async def test_root_reviews_unresolved_delegation_but_not_human_workers(
    setup, background, child_error
):
    manager, ctx, _, store, session = setup
    child_started = asyncio.Event()
    root_finished = asyncio.Event()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            if prompt == "unrelated human":
                await root_finished.wait()
                entry = {"text": "human result"}
            elif prompt == "child":
                child_started.set()
                await asyncio.sleep(0.02)
                entry = {"text": "delegated result", "usage": {"input_tokens": 7}}
                if child_error:
                    entry = {"error": RuntimeError("delegated result")}
            elif len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {
                            "id": "delegate",
                            "name": "task",
                            "arguments": json.dumps(
                                {"prompt": "child", "run_in_background": background}
                            ),
                        }
                    ]
                }
            elif any("delegated result" in str(m.get("content")) for m in messages):
                entry = {"text": "reviewed delegated result"}
            else:
                await child_started.wait()
                entry = {"text": "premature final"}
            async for event in self._stream(entry):
                yield event

    provider = Provider([])
    ctx.extras["provider"] = provider
    try:
        human = await manager.start(ctx, agent="explore", prompt="unrelated human", origin="human")
        store.append_message(session, {"role": "user", "content": "root"})
        runner = AgentRunner(provider, ctx.extras["registry"], ctx, session=session, store=store)
        async with asyncio.timeout(2):
            result = await runner.run(
                [{"role": "system", "content": "root"}, *store.load_for_model(session)]
            )
        assert result.final_text == "reviewed delegated result"
        assert human.state == "running"
        history = store.load_for_model(session)
        deliveries = [
            m
            for m in history
            if m["role"] in {"user", "tool"} and "delegated result" in str(m.get("content"))
        ]
        assert len(deliveries) == 1
        assert session_stats(store, session).input_tokens == (0 if child_error else 7)
    finally:
        root_finished.set()
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("supervisor", ["root", "readonly", "writable"])
@pytest.mark.parametrize("background", [False, True])
async def test_child_question_is_answered_at_safe_parent_boundary(setup, supervisor, background):
    manager, ctx, _, store, session = setup
    nested = supervisor != "root"
    if supervisor == "writable":
        git(ctx.cwd, "init", "-b", "main")
        git(
            ctx.cwd,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "initial",
        )
        ctx.extras["agents"] = AgentRegistry(
            {
                "writer": AgentDefinition("writer", "write", "", mode="subagent"),
                "explore": ctx.extras["agents"].get("explore"),
            }
        )
    approvals = []
    ctx.approval_callback = lambda *args: approvals.append(args)
    seen_questions = []
    denials = []

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            denials.extend(
                m["content"]
                for m in messages
                if m.get("name") == "workers"
                and m["role"] == "tool"
                and "denied" in m.get("content", "")
            )
            assert not denials, denials
            outstanding = set()
            for message in messages:
                if message["role"] == "tool":
                    outstanding.remove(message["tool_call_id"])
                else:
                    assert not outstanding, "question delivery broke the tool protocol"
                    outstanding.update(c["id"] for c in message.get("tool_calls", []))
            assert not outstanding
            prompt = messages[1]["content"]
            questions = [
                m["content"]
                for m in messages
                if m["role"] == "user" and " asks] " in str(m["content"])
            ]
            answered = any(
                m.get("name") == "workers" and "queued" in m.get("content", "")
                for m in messages
                if m["role"] == "tool"
            )
            if questions and not answered:
                seen_questions.append(prompt)
                worker_id = questions[-1].split()[1]
                entry = {
                    "tool_calls": [
                        {
                            "id": "answer",
                            "name": "workers",
                            "arguments": json.dumps(
                                {"action": "send", "id": worker_id, "text": "use blue"}
                            ),
                        }
                    ]
                }
            elif prompt == "leaf":
                if len(messages) == 2:
                    entry = {
                        "tool_calls": [
                            {
                                "id": "ask",
                                "name": "workers",
                                "arguments": '{"action":"question","text":"which color?"}',
                            }
                        ]
                    }
                else:
                    assert any(m["content"] == "use blue" for m in messages)
                    entry = {"text": "blue result"}
            elif len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {
                            "id": "delegate",
                            "name": "task",
                            "arguments": json.dumps(
                                {
                                    "prompt": "parent" if prompt == "root" and nested else "leaf",
                                    "agent": "writer"
                                    if prompt == "root" and supervisor == "writable"
                                    else "explore",
                                    "run_in_background": background,
                                }
                            ),
                        }
                    ]
                }
            elif any("blue result" in str(m.get("content")) for m in messages):
                entry = {"text": "reviewed blue result"}
            else:
                entry = {"text": "premature final"}
            async for event in self._stream(entry):
                yield event

    provider = Provider([])
    store.append_message(session, {"role": "user", "content": "root"})
    runner = AgentRunner(provider, ctx.extras["registry"], ctx, session=session, store=store)
    try:
        async with asyncio.timeout(3):
            result = await runner.run(
                [{"role": "system", "content": "root"}, *store.load_for_model(session)]
            )
        assert result.final_text == "reviewed blue result"
        assert seen_questions == ["parent" if nested else "root"]
        assert approvals == []
        assert all(worker.state == "completed" for worker in manager.list())
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "stopped", "queued_stop"])
async def test_worker_dispatch_lifecycle_hooks_use_parent_context_once(setup, outcome):
    manager, ctx, _, _, session = setup
    log = ctx.cwd / "hooks.jsonl"
    script = ctx.cwd / "hook.py"
    script.write_text(
        "import json, sys\n"
        f"with open({str(log)!r}, 'a') as out:\n"
        "    out.write(json.dumps(json.load(sys.stdin)) + '\\n')\n"
        "print(json.dumps({'verdict': 'deny'}))\n"
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    ctx.config.hooks = {event: [command] for event in ("SubagentStart", "SubagentEnd")}
    ctx.extras["hooks"], _ = dispatcher_from_config(ctx.config, ctx.cwd, session=session)
    provider = (
        GatedProvider()
        if outcome == "stopped"
        else FakeProvider(
            [{"error": RuntimeError("broken")} if outcome == "failed" else {"text": "done"}]
        )
    )
    ctx.extras["provider"] = provider
    try:
        if outcome == "completed":
            _, tool_result = await ctx.extras["registry"].dispatch_result(
                "delegate", "task", '{"prompt":"work"}', ctx
            )
            assert not tool_result.is_error
            worker = manager.get(tool_result.metadata["worker_id"])
        else:
            worker = await manager.start(ctx, agent="explore", prompt="work")
        if outcome == "stopped":
            async with asyncio.timeout(2):
                await provider.started.get()
            await manager.stop(worker.id)
        elif outcome == "queued_stop":
            await manager.stop(worker.id)
        elif outcome == "failed":
            with pytest.raises(SubagentError, match="broken"):
                await manager.wait(worker.id)
        else:
            await manager.wait(worker.id)
        assert log.exists()
        events = [json.loads(line) for line in log.read_text().splitlines()]
        assert [event["event"] for event in events] == ["SubagentStart", "SubagentEnd"]
        for event in events:
            assert event["agent"] == "explore"
            assert event["session"]["id"] == session.id
            assert event["cwd"] == str(ctx.cwd)
            assert event["worker"]["id"] == worker.id
            assert event["worker"]["dispatch_id"] == worker.dispatch_id
            assert event["worker"]["parent_id"] is None
        assert events[-1]["result"]["is_error"] == (outcome != "completed")
        if outcome == "queued_stop":
            assert provider.requests == []
        if outcome == "completed":
            first_dispatch = worker.dispatch_id
            await manager.send(worker.id, "follow up")
            await manager.wait(worker.id)
            events = [json.loads(line) for line in log.read_text().splitlines()]
            assert [event["event"] for event in events] == ["SubagentStart", "SubagentEnd"] * 2
            assert [event["worker"]["dispatch_id"] for event in events] == [
                first_dispatch,
                first_dispatch,
                worker.dispatch_id,
                worker.dispatch_id,
            ]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_notification_ack_failure_cannot_lose_or_duplicate_delivery(setup, monkeypatch):
    manager, ctx, _, store, session = setup
    try:
        worker = await manager.start(ctx, agent="explore", prompt="work", background=True)
        await manager.wait(worker.id)
        append = store.append_event

        def fail_ack(session, kind, data):
            if kind == "worker_notification_ack":
                raise OSError("ack disk failure")
            return append(session, kind, data)

        monkeypatch.setattr(store, "append_event", fail_ack)
        history = []
        with pytest.raises(OSError, match="ack disk failure"):
            manager.consume(None, history)
        assert history == store.load_for_model(session)
        assert len(history) == 1
        monkeypatch.setattr(store, "append_event", append)
        assert manager.consume(None, history) == []
        assert manager.drain_notifications() == []
        assert len(store.load_for_model(session)) == 1
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["send", "resume"])
async def test_write_followup_requires_workspace_guard_before_provider(setup, action):
    manager, ctx, provider, _, _ = setup
    # Missing guard must still fail closed even though production supplies one.
    manager.workspace_guard = None
    git(ctx.cwd, "init", "-b", "main")
    git(
        ctx.cwd,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    ctx.extras["agents"] = AgentRegistry(
        {
            "writer": AgentDefinition("writer", "write", "", mode="subagent"),
        }
    )
    try:
        worker = await manager.start(ctx, agent="writer", prompt="initial")
        await manager.wait(worker.id)
        if action == "send":
            await manager.send(worker.id, "follow up")
        else:
            await manager.stop(worker.id)
            await manager.resume(worker.id, "follow up")
        with pytest.raises(SubagentError, match="workspace guard"):
            await manager.wait(worker.id)
        assert len(provider.requests) == 1
        assert [item["text"] for item in manager.pending(worker.id)] == ["follow up"]
        guarded = []

        async def guard(candidate):
            guarded.append(candidate.id)
            wm = await WorktreeManager.discover(ctx.cwd)
            inspection = await wm.reconcile(candidate.worktree.name, recreate=False)
            assert inspection.present and not inspection.merge_in_progress

        manager.workspace_guard = guard
        await manager.resume(worker.id)
        assert (await manager.wait(worker.id)).final_text == "second"
        assert guarded == [worker.id]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_waiting_worker_control_yields_after_read_and_reacquires_at_cap(setup):
    manager, ctx, _, store, _ = setup
    git(ctx.cwd, "init", "-b", "main")
    git(
        ctx.cwd,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    end_started = ctx.cwd / "end-started"
    release_end = ctx.cwd / "release-end"
    script = ctx.cwd / "end.py"
    script.write_text(
        "import json, sys, time\nfrom pathlib import Path\n"
        "event = json.load(sys.stdin)\n"
        "if event['agent'] == 'explore':\n"
        f"    Path({str(end_started)!r}).touch()\n"
        f"    while not Path({str(release_end)!r}).exists(): time.sleep(0.01)\n"
    )
    ctx.config.hooks = {
        "SubagentEnd": [f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"]
    }
    ctx.extras["hooks"], _ = dispatcher_from_config(ctx.config, ctx.cwd, session=ctx.session)
    ctx.extras["agents"] = AgentRegistry(
        {
            "writer": AgentDefinition("writer", "write", "", mode="subagent"),
            "explore": ctx.extras["agents"].get("explore"),
        }
    )
    manager.confirm = lambda _: True
    blockers_ready = asyncio.Event()
    release_blockers = asyncio.Event()
    blockers = 0

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            nonlocal blockers
            assert sum(w.state == "running" for w in manager.list()) <= 10
            prompt = messages[1]["content"]
            if prompt == "blocker":
                blockers += 1
                if blockers == 9:
                    blockers_ready.set()
                await release_blockers.wait()
                entry = {"text": "done"}
            elif prompt == "parent" and len(messages) == 2:
                await blockers_ready.wait()
                entry = {
                    "tool_calls": [
                        {
                            "id": "child",
                            "name": "task",
                            "arguments": '{"prompt":"never runs","run_in_background":true}',
                        }
                    ]
                }
            elif prompt == "parent" and not any(m.get("tool_call_id") == "stop" for m in messages):
                parent = next(w for w in manager.list() if w.agent == "writer")
                child = manager.children(parent.id)[0]
                entry = {
                    "tool_calls": [
                        {
                            "id": "stop",
                            "name": "workers",
                            "arguments": json.dumps({"action": "stop", "id": child.id}),
                        },
                        {"id": "read", "name": "read", "arguments": '{"file_path":"missing"}'},
                    ]
                }
            else:
                assert prompt == "parent", "queued child must be stopped before its model runs"
                entry = {"text": "reviewed"}
            async for event in self._stream(entry):
                yield event

    ctx.extras["provider"] = Provider([])
    try:
        for _ in range(9):
            await manager.start(ctx, agent="explore", prompt="blocker", origin="human")
        parent = await manager.start(ctx, agent="writer", prompt="parent")
        async with asyncio.timeout(3):
            while not end_started.exists() or parent.state != "waiting":
                await asyncio.sleep(0.01)
        assert sum(w.state == "running" for w in manager.list()) == 9
        release_end.touch()
        async with asyncio.timeout(3):
            assert (await manager.wait(parent.id)).final_text == "reviewed"
        assert [
            m["tool_call_id"] for m in store.load_for_model(parent.session) if m["role"] == "tool"
        ] == ["child", "stop", "read"]
    finally:
        release_end.touch()
        release_blockers.set()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_root_reviews_orphaned_foreground_descendant_without_waking_stopped_parent(setup):
    manager, ctx, _, store, session = setup
    child_started = asyncio.Event()
    root_called = asyncio.Event()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            if prompt == "parent":
                entry = {"text": "parent done"}
            elif prompt == "descendant":
                child_started.set()
                await root_called.wait()
                entry = {"text": "orphan result"}
            else:
                root_called.set()
                entry = {
                    "text": "reviewed"
                    if any("orphan result" in str(m.get("content")) for m in messages)
                    else "premature"
                }
            async for event in self._stream(entry):
                yield event

    provider = Provider([])
    ctx.extras["provider"] = provider
    try:
        parent = await manager.start(ctx, agent="explore", prompt="parent")
        await manager.wait(parent.id)
        await manager.stop(parent.id)
        nested = replace(ctx, extras={**ctx.extras, WORKER_CURRENT_EXTRA: parent.id})
        await manager.start(nested, agent="explore", prompt="descendant")
        await child_started.wait()
        store.append_message(session, {"role": "user", "content": "root"})
        runner = AgentRunner(provider, ctx.extras["registry"], ctx, session=session, store=store)
        async with asyncio.timeout(2):
            result = await runner.run(
                [{"role": "system", "content": "root"}, *store.load_for_model(session)]
            )
        assert result.final_text == "reviewed"
        assert parent.state == "stopped"
        assert manager.pending_notifications(parent.id) == []
    finally:
        root_called.set()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_worker_remains_active_until_lifecycle_cleanup_finishes(setup):
    manager, ctx, _, _, session = setup
    started = ctx.cwd / "end-started"
    release = ctx.cwd / "end-release"
    script = ctx.cwd / "end.py"
    script.write_text(
        "import time\nfrom pathlib import Path\n"
        f"Path({str(started)!r}).touch()\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
    )
    ctx.config.hooks = {
        "SubagentEnd": [f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"]
    }
    ctx.extras["hooks"], _ = dispatcher_from_config(ctx.config, ctx.cwd, session=session)
    try:
        worker = await manager.start(ctx, agent="explore", prompt="work")
        async with asyncio.timeout(2):
            while not started.exists():
                await asyncio.sleep(0.01)
        assert worker.is_active
        with pytest.raises(RuntimeError, match="active"):
            manager.attach(session)
        release.touch()
        await manager.wait(worker.id)
        assert not worker.is_active
    finally:
        release.touch()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_unanswered_question_escalates_when_its_parent_is_stopped(setup):
    manager, ctx, _, _, _ = setup
    ctx.extras["provider"] = GatedProvider()
    try:
        parent = await manager.start(ctx, agent="explore", prompt="parent")
        nested = replace(ctx, extras={**ctx.extras, WORKER_CURRENT_EXTRA: parent.id})
        child = await manager.start(nested, agent="explore", prompt="child")
        manager.ask_parent(child.id, "which color?")
        parent_history = []
        manager.consume(parent.id, parent_history)
        assert "which color?" in parent_history[-1]["content"]
        await manager.stop(parent.id)
        root_history = []
        manager.consume(None, root_history)
        assert any("which color?" in m["content"] for m in root_history)
        assert manager.consume(None, root_history) == []
        await manager.send(child.id, "use blue")
        assert parent.state == "stopped"
        assert manager.questions(child.id) == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_parent_questions_yield_all_ten_child_leases(setup):
    manager, ctx, _, _, _ = setup
    spare_started = asyncio.Event()
    release_spare = asyncio.Event()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            if messages[1]["content"] == "spare":
                spare_started.set()
                await release_spare.wait()
                entry = {"text": "done"}
            elif len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {
                            "id": "question",
                            "name": "workers",
                            "arguments": '{"action":"question","text":"need an answer"}',
                        }
                    ]
                }
            else:
                assert messages[-1]["content"] == "answer"
                entry = {"text": "done"}
            assert sum(w.state == "running" for w in manager.list()) <= 10
            async for event in self._stream(entry):
                yield event

    ctx.extras["provider"] = Provider([])
    try:
        children = [await manager.start(ctx, agent="explore", prompt="ask") for _ in range(10)]
        spare = await manager.start(ctx, agent="explore", prompt="spare", origin="human")
        async with asyncio.timeout(2):
            await spare_started.wait()
            while any(w.state != "waiting" for w in children):
                await asyncio.sleep(0)
        for child in children:
            await manager.send(child.id, "answer")
        release_spare.set()
        async with asyncio.timeout(2):
            await asyncio.gather(*(manager.wait(w.id) for w in [*children, spare]))
        assert all(w.state == "completed" for w in children)
    finally:
        release_spare.set()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_batch_persists_completed_delegation_once_across_resume(setup):
    manager, ctx, _, store, _ = setup
    slow_started = asyncio.Event()
    release = asyncio.Event()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            prompt = messages[1]["content"]
            if prompt == "parent" and len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {"id": "fast", "name": "task", "arguments": '{"prompt":"fast"}'},
                        {"id": "slow", "name": "task", "arguments": '{"prompt":"slow"}'},
                    ]
                }
            elif prompt == "slow":
                slow_started.set()
                await release.wait()
                entry = {"text": "slow result"}
            elif prompt == "fast":
                entry = {"text": "fast result"}
            else:
                assert sum("fast result" in str(m.get("content")) for m in messages) == 1
                entry = {"text": "reviewed"}
            async for event in self._stream(entry):
                yield event

    ctx.extras["provider"] = Provider([])
    try:
        parent = await manager.start(ctx, agent="explore", prompt="parent")
        async with asyncio.timeout(2):
            await slow_started.wait()
            while not any(
                w.state == "completed" and not w.is_active for w in manager.children(parent.id)
            ):
                await asyncio.sleep(0)
            # The fast tool can return through its supervisor's reacquisition.
            await asyncio.sleep(0.01)
            await manager.stop(parent.id)
        assert [
            m["tool_call_id"] for m in store.load_for_model(parent.session) if m["role"] == "tool"
        ] == ["fast"]
        release.set()
        await manager.resume(parent.id)
        async with asyncio.timeout(2):
            assert (await manager.wait(parent.id)).final_text == "reviewed"
    finally:
        release.set()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_wait_includes_followup_queued_during_completion_notification(setup):
    manager, ctx, _, _, _ = setup

    async def notify(note):
        if note["content"] == "first":
            await manager.send(note["worker_id"], "follow up")

    manager.notify = notify
    try:
        worker = await manager.start(ctx, agent="explore", prompt="work", background=True)
        assert (await manager.wait(worker.id)).final_text == "second"
        assert not worker.is_active
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_depth_two_approvals_keep_requester_identity_fifo_and_child_grants(setup):
    from lecode.config.models import PermissionRule
    from lecode.permission import AllowAlways, AllowOnce
    from lecode.tui.permission import ApprovalPrompt

    manager, ctx, _, store, _ = setup
    ctx.config.permissions.rules.ask["read"] = [PermissionRule(pattern="*")]
    fifo = ApprovalPrompt()
    requested = asyncio.Queue()

    async def approve(name, args, reason, *, worker, conversation):
        future = fifo.request(
            name, args["file_path"], reason, worker=worker, conversation=conversation
        )
        requested.put_nowait((worker, conversation))
        return await future

    ctx.approval_callback = approve

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            if len(messages) > 2:
                entry = {"text": "done"}
            elif messages[1]["content"] == "parent":
                entry = {
                    "tool_calls": [
                        {"id": "read", "name": "read", "arguments": '{"file_path":"parent.txt"}'},
                        {"id": "spawn", "name": "task", "arguments": '{"prompt":"child"}'},
                    ]
                }
            else:
                entry = {
                    "tool_calls": [
                        {"id": "one", "name": "read", "arguments": '{"file_path":"one.txt"}'},
                        {"id": "two", "name": "read", "arguments": '{"file_path":"two.txt"}'},
                    ]
                }
            async for event in self._stream(entry):
                yield event

    ctx.extras["provider"] = Provider([])
    try:
        parent = await manager.start(ctx, agent="explore", prompt="parent")
        async with asyncio.timeout(2):
            identities = [await requested.get() for _ in range(3)]
            child = manager.children(parent.id)[0]
            assert child.depth == 2
            assert identities == [
                (parent.id, parent.session.name),
                (child.id, child.session.name),
                (child.id, child.session.name),
            ]
            for identity, decision in zip(
                identities, [AllowOnce(), AllowAlways("one.txt"), AllowOnce()], strict=True
            ):
                assert (fifo.pending.worker, fifo.pending.conversation) == identity
                fifo.resolve(decision)
            await manager.wait(parent.id)
        assert store.load_grants(child.session) == [("read", "one.txt")]
        assert store.load_grants(parent.session) == []
        assert ctx.session_perms.grants == []
        assert not fifo.is_pending
    finally:
        fifo.cancel()
        await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("running_supervisor", [False, True])
@pytest.mark.parametrize("validation", [False, True])
async def test_nested_integration_reserves_destination_execution(
    workflow, monkeypatch, running_supervisor, validation
):
    from tests.test_worker_controls import commit, dispatch, start, supervisor_runtime

    from lecode.agent.tools.bash import BashTool

    ctx = workflow.ctx
    manager = ctx.extras["workers"]
    entered, release = asyncio.Event(), asyncio.Event()
    questions = []

    async def confirm(question):
        assert isinstance(question, str)
        questions.append(question)
        entered.set()
        await release.wait()
        return True

    manager.confirm = confirm
    if validation:
        manager.config.worktree.validation = ["true"]
        original = BashTool.run

        async def validate(tool, args, context):
            entered.set()
            await release.wait()
            return await original(tool, args, context)

        monkeypatch.setattr(BashTool, "run", validate)

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            if messages[1]["content"] == "child":
                entry = {"text": "child done"}
            elif len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {
                            "id": "spawn",
                            "name": "task",
                            "arguments": '{"agent":"writer","prompt":"child"}',
                        }
                    ]
                }
            elif not any(m.get("tool_call_id") == "integrate" for m in messages):
                parent = manager.children(None)[0]
                child = manager.children(parent.id)[0]
                entry = {
                    "tool_calls": [
                        {
                            "id": "integrate",
                            "name": "workers",
                            "arguments": json.dumps(
                                {
                                    "action": "integrate",
                                    "id": child.id,
                                    "reviewed_head": commit(child.cwd, "nested change\n"),
                                    "reviewed_parent_head": git(parent.cwd, "rev-parse", "HEAD"),
                                }
                            ),
                        }
                    ]
                }
            else:
                result = next(m for m in messages if m.get("tool_call_id") == "integrate")
                assert "error" not in result["content"], result["content"]
                entry = {"text": "integrated"}
            async for event in self._stream(entry):
                yield event

    integration = None
    if running_supervisor:
        ctx.extras["provider"] = Provider([])
        parent = await manager.start(ctx, agent="writer", prompt="parent")
    else:
        parent = await start(workflow)
        nested = supervisor_runtime(workflow, parent)
        child = await start(nested)
        head = commit(child.cwd, "nested change\n")
        integration = asyncio.create_task(
            dispatch(
                nested,
                "integrate",
                id=child.id,
                reviewed_head=head,
                reviewed_parent_head=git(parent.cwd, "rev-parse", "HEAD"),
            )
        )
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            child = manager.children(parent.id)[0]
            for worker in (parent, child):
                with pytest.raises(WorktreeError, match="maintenance"):
                    await manager.send(worker.id, "race")
                with pytest.raises(WorktreeError, match="maintenance"):
                    await manager.resume(worker.id, "race")
                with pytest.raises(WorktreeError, match="maintenance"):
                    await manager.reconcile_workspace(worker)
                assert manager.pending(worker.id) == []
            if not validation:
                assert child.id in questions[0] and str(child.cwd) in questions[0]
            release.set()
            if integration:
                result = await integration
                assert not result.is_error, result.content
            else:
                assert (await manager.wait(parent.id)).final_text == "integrated"
        assert (parent.cwd / "change.txt").read_text() == "nested change\n"
        assert (ctx.cwd / "change.txt").read_text() == "base\n"
        assert not manager._maintenance
    finally:
        release.set()
        if integration:
            await integration


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cleanup", "recover"])
async def test_parent_workspace_maintenance_reserves_human_origin_shared_subtree(
    workflow, monkeypatch, action
):
    import shutil

    from tests.test_worker_controls import dispatch, start, supervisor_runtime

    ctx = workflow.ctx
    manager = ctx.extras["workers"]
    ctx.extras["agents"] = AgentRegistry(
        {
            "writer": ctx.extras["agents"].get("writer"),
            "reader": AgentDefinition(
                "reader", "read", "", mode="subagent", overlay=AgentOverlay(mode="readonly")
            ),
        }
    )
    parent = await start(workflow)
    nested = supervisor_runtime(workflow, parent)
    child = await manager.start(nested.ctx, agent="reader", prompt="read", origin="human")
    await manager.wait(child.id)
    assert child.cwd == parent.cwd and child.worktree is None
    entered, release = asyncio.Event(), asyncio.Event()

    async def confirm(question):
        assert isinstance(question, str)
        assert parent.id in question and str(parent.cwd) in question
        entered.set()
        await release.wait()
        return True

    if action == "recover":
        shutil.rmtree(parent.cwd)
        manager.confirm = confirm
    else:
        original = WorktreeManager.cleanup_worker

        async def cleanup(worktrees, name):
            entered.set()
            await release.wait()
            return await original(worktrees, name)

        monkeypatch.setattr(WorktreeManager, "cleanup_worker", cleanup)

    maintenance = asyncio.create_task(dispatch(workflow, action, id=parent.id))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            for worker in (parent, child):
                with pytest.raises(WorktreeError, match="maintenance"):
                    await manager.send(worker.id, "race")
                with pytest.raises(WorktreeError, match="maintenance"):
                    await manager.resume(worker.id, "race")
            with pytest.raises(WorktreeError, match="maintenance"):
                await manager.start(nested.ctx, agent="reader", prompt="race")
            release.set()
            result = await maintenance
        assert not result.is_error, result.content
        assert parent.cwd.exists() is (action == "recover")
        assert manager.pending(parent.id) == manager.pending(child.id) == []
        assert not manager._maintenance
    finally:
        release.set()
        await maintenance


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True], ids=["root", "parent-worker"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["integrate", "rewritten-inspect"])
@pytest.mark.parametrize("validation", [False, True], ids=["confirmation", "validation"])
async def test_integration_refuses_unfinished_sibling_bash(
    workflow, monkeypatch, nested, rewrite, validation
):
    from tests.test_worker_controls import commit, start, supervisor_runtime

    from lecode.agent.tools.base import ToolRegistry
    from lecode.agent.tools.bash import BashTool
    from lecode.hooks import apply_hooks
    from lecode.hooks.runner import HookDispatcher, HookHandler

    manager = workflow.ctx.extras["workers"]
    parent = await start(workflow) if nested else None
    runtime = supervisor_runtime(workflow, parent) if nested else workflow
    child = await start(runtime)
    args = {
        "action": "integrate",
        "id": child.id,
        "reviewed_head": commit(child.cwd),
        "reviewed_parent_head": git(runtime.ctx.cwd, "rev-parse", "HEAD"),
    }
    release_bash, bash_finished = asyncio.Event(), asyncio.Event()
    mutation_reservations, checks, controls, bash_results = [], [], [], []
    owner = parent.id if parent else None

    def assert_lease():
        assert len(manager._leases) <= 10
        if parent:
            assert parent.id in manager._leases
            assert parent.state == "running"

    inspect = WorktreeManager.inspect

    async def inspect_workspace(worktrees, name):
        if name == child.worktree.name:
            checks.append("inspect")
            assert_lease()
            # On the buggy path, let the sibling mutate the reserved destination.
            release_bash.set()
            await bash_finished.wait()
        return await inspect(worktrees, name)

    monkeypatch.setattr(WorktreeManager, "inspect", inspect_workspace)
    control = runtime.registry.get("workers")
    run_control = control.run

    async def observed_control(arguments, context):
        assert asyncio.current_task() in manager._tool_owners
        assert manager._tool_owners[asyncio.current_task()] == owner
        try:
            result = await run_control(arguments, context)
            controls.append((result, list(checks)))
            return result
        finally:
            release_bash.set()

    monkeypatch.setattr(control, "run", observed_control)
    if rewrite:
        output = json.dumps({"verdict": "allow", "rewritten_input": args})
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(f'print({output!r})')}"
        apply_hooks(
            ToolRegistry([control]),
            HookDispatcher({"PreToolUse": [HookHandler(command)]}, runtime.ctx.cwd),
        )

    bash = runtime.registry.get("bash")
    run_bash = bash.run

    async def mutate_destination(arguments, context):
        await release_bash.wait()
        assert_lease()
        mutation_reservations.append(bool(manager._maintenance))
        try:
            result = await run_bash(arguments, context)
            bash_results.append(result)
            return result
        finally:
            bash_finished.set()

    monkeypatch.setattr(bash, "run", mutate_destination)

    async def confirm(question):
        checks.append("confirmation")
        assert bash_finished.is_set()
        assert manager._maintenance
        return True

    manager.confirm = confirm
    if validation:
        manager.config.worktree.validation = ["true"]
        original = BashTool.run

        async def validate(tool, arguments, context):
            checks.append("validation")
            assert bash_finished.is_set()
            assert manager._maintenance
            assert_lease()
            return await original(tool, arguments, context)

        monkeypatch.setattr(BashTool, "run", validate)

    integration = {
        "id": "integrate",
        "name": "workers",
        "arguments": json.dumps({"action": "inspect", "id": child.id} if rewrite else args),
    }
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    integration,
                    {
                        "id": "mutate",
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "rm change.txt && git restore -- change.txt"}
                        ),
                    },
                ]
            },
            {"tool_calls": [{**integration, "id": "retry"}]},
            {"text": "integrated"},
        ]
    )
    runtime.ctx.extras["provider"] = provider
    async with asyncio.timeout(5):
        if parent:
            await manager.resume(parent.id, "integrate child")
            result = await manager.wait(parent.id)
        else:
            runner = AgentRunner(
                provider,
                runtime.registry,
                runtime.ctx,
                session=runtime.ctx.session,
                store=manager.store,
            )
            result = await runner.run([{"role": "user", "content": "integrate child"}])
    assert result.final_text == "integrated"
    assert len(bash_results) == 1
    assert not bash_results[0].is_error, bash_results[0].content
    assert mutation_reservations == [False], "sibling mutated a reserved destination"
    refused, early_checks = controls[0]
    assert refused.is_error
    assert (
        "workspace maintenance must run separately after sibling tools complete" in refused.content
    )
    assert early_checks == [], "mixed batch reached inspection, confirmation or validation"
    assert not controls[1][0].is_error, controls[1][0].content
    assert ("validation" if validation else "confirmation") in checks
    assert (runtime.ctx.cwd / "change.txt").read_text() == "worker change\n"
    assert not manager._maintenance
    assert not manager._tool_owners


@pytest.mark.asyncio
async def test_independent_maintenance_cannot_borrow_running_supervisor(workflow):
    from tests.test_worker_controls import start, supervisor_runtime

    manager = workflow.ctx.extras["workers"]
    started, release = asyncio.Event(), asyncio.Event()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            if any(m.get("content") == "active" for m in messages):
                started.set()
                await release.wait()
            async for event in self._stream({"text": "done"}):
                yield event

    workflow.ctx.extras["provider"] = Provider([])
    parent = await start(workflow)
    child = await start(supervisor_runtime(workflow, parent))
    try:
        await manager.resume(parent.id, "active")
        async with asyncio.timeout(2):
            await started.wait()
            with pytest.raises(WorktreeError, match="idle"):
                async with manager.maintain_workspace(child.id):
                    pytest.fail("independent execution entered the running parent's workspace")
            release.set()
            await manager.wait(parent.id)
        assert not manager._maintenance
    finally:
        release.set()
