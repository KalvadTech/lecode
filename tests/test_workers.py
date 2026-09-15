"""WorkerManager's public persistence and scheduling contract."""

import asyncio
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import pytest
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.agent.runner import AgentRunner, RunResult, UsageTotals
from lecode.config.models import Config
from lecode.context.agents import AgentDefinition, AgentRegistry
from lecode.extras.subagents import SubagentError
from lecode.extras.workers import WORKER_CURRENT_EXTRA, WorkerManager
from lecode.extras.worktree import WorktreeError, WorktreeManager
from lecode.permission import PermissionChecker
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


@pytest.fixture
def checker_contract(setup):
    """Isolate the sibling-owned checker seam in runner/scheduler tests.

    The real checker integration is tested separately, once for_child lands.
    This double does not purport to verify inherited permission restrictions.
    """
    _, ctx, _, _, _ = setup
    parent = ctx.permission_checker
    checker = Mock(wraps=parent)
    checker.mode = parent.mode
    checker.read_only = parent.read_only

    def derive(overlay=None, **kw):
        child = PermissionChecker(ctx.config, mode=parent.mode, _overlay=overlay, **kw)
        derived = Mock(wraps=child)
        derived.mode = child.mode
        derived.read_only = child.read_only
        derived.for_child = Mock(side_effect=derive)
        return derived

    checker.for_child = Mock(side_effect=derive)
    ctx.permission_checker = checker
    return checker


@pytest.mark.asyncio
async def test_followup_is_persisted_and_replayed(setup, checker_contract):
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
async def test_task_tool_creates_persisted_workers_and_nests(setup, checker_contract):
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
async def test_background_task_tool_delivers_worker_notification(setup, checker_contract):
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
async def test_parent_cancellation_does_not_cancel_managed_worker(setup, checker_contract):
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
async def test_worker_controls_enforce_descendant_hierarchy_and_questions(setup, checker_contract):
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
async def test_write_worktree_pins_parent_head_and_readonly_shares_parent_cwd(
    setup, checker_contract
):
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
async def test_strict_cap_and_shielded_wait(setup, checker_contract):
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
async def test_failed_usage_survives_restart_and_child_locks_last_until_shutdown(
    setup, checker_contract
):
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
async def test_interrupt_repairs_unanswered_calls_without_replaying_inputs(setup, checker_contract):
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
async def test_running_inbox_is_durable_but_only_consumed_after_safe_boundary(
    setup, checker_contract
):
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
async def test_completion_delivery_background_only_and_human_submit(setup, checker_contract):
    manager, ctx, _, _, _ = setup
    notifications = []
    manager.notify = notifications.append
    try:
        foreground = await manager.start(ctx, agent="explore", prompt="foreground")
        await manager.wait(foreground.id)
        assert manager.drain_notifications() == []
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
        await manager.submit(human.id)
        assert [n["worker_id"] for n in manager.drain_notifications()] == [human.id]
        await manager.submit(human.id)
        assert manager.drain_notifications() == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_completed_worker_followup_after_restart(setup, checker_contract):
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
async def test_suspended_supervisors_release_capacity_and_reacquire(
    setup, checker_contract, monkeypatch
):
    manager, ctx, _, _, _ = setup
    children_started = asyncio.Queue()
    release = asyncio.Event()

    async def run(runner, messages, on_event=None):
        id = runner.ctx.extras[WORKER_CURRENT_EXTRA]
        assert sum(w.state == "running" for w in manager.list()) <= 10
        if manager.get(id).depth == 1:
            child = await manager.start(runner.ctx, agent="explore", prompt="child")
            async with manager.suspend(id):
                assert manager.get(id).state == "waiting"
                await manager.wait(child.id)
            assert manager.get(id).state == "running"
            assert sum(w.state == "running" for w in manager.list()) <= 10
        else:
            children_started.put_nowait(id)
            await release.wait()
        return RunResult("done", 1, "done", UsageTotals())

    monkeypatch.setattr(AgentRunner, "run", run)
    try:
        parents = [await manager.start(ctx, agent="explore", prompt="parent") for _ in range(10)]
        async with asyncio.timeout(2):
            for _ in range(10):
                await children_started.get()
        assert sum(w.state == "waiting" for w in manager.list()) == 10
        assert sum(w.state == "running" for w in manager.list()) == 10
        with pytest.raises(RuntimeError, match="supervisor"):
            async with manager.suspend(manager.children(parents[0].id)[0].id):
                pass
        release.set()
        async with asyncio.timeout(2):
            await asyncio.gather(*(manager.wait(w.id) for w in parents))
        assert all(w.state == "completed" for w in manager.list())
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_stop_is_individual_unless_tree_requested(setup, checker_contract):
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
async def test_runtime_is_rebuilt_with_fresh_grants_and_cwd_bound_extras(
    setup, checker_contract, monkeypatch
):
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
        assert checker_contract.for_child.call_args_list[0].args == (
            ctx.extras["agents"].get("explore").overlay,
        )
        assert checker_contract.for_child.call_args_list[0].kwargs == {"cwd": ctx.cwd}
        assert checker_contract.for_child.call_args_list[-1].args == (
            ctx.extras["agents"].get("explore").overlay,
        )
        assert checker_contract.for_child.call_args_list[-1].kwargs == {
            "cwd": child.cwd,
            "session_perms": child.session_perms,
            "read_only": True,
        }
    finally:
        await manager.shutdown()


@pytest.mark.skipif(
    not hasattr(PermissionChecker, "for_child"),
    reason="sibling-owned PermissionChecker.for_child has not landed",
)
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
async def test_usage_records_do_not_double_count_child_transcript(setup, checker_contract):
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
async def test_stopped_worker_does_not_return_a_stale_result(setup, checker_contract):
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
async def test_worker_model_precedence_is_recorded(
    setup, checker_contract, agent_model, subagent_model, expected
):
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
