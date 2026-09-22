"""Tests for the multi-turn agent runner against the scripted FakeProvider."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.fakes import FakeProvider, sample_catalog
from tests.test_workers import setup as setup

from lecode.agent.runner import (
    CONTINUE_PROMPT,
    EMPTY_NUDGE,
    AgentRunner,
    CompactionFinished,
    CompactionStarted,
    Done,
    Error,
    LlmCall,
    LlmResponse,
    QueuedMessage,
    Retrying,
    Token,
    ToolCall,
    ToolResult,
)
from lecode.agent.tools.base import Tool, ToolRegistry
from lecode.agent.tools.base import ToolResult as ToolExecResult
from lecode.extras.background import BACKGROUND_EXTRA
from lecode.providers.openai_compat import ProviderError
from lecode.providers.types import TokenDelta
from lecode.session.model import EventRecord, MessageRecord
from lecode.session.storage import SessionStore


class EchoTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Echo the given text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    async def run(self, args, ctx) -> ToolExecResult:
        return ToolExecResult(content=str(args.get("text", "")))


class SleepTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="sleep", description="Block forever.", parameters={})

    async def run(self, args, ctx) -> ToolExecResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def make_runner(tool_ctx, script, **kwargs) -> tuple[AgentRunner, FakeProvider]:
    provider = FakeProvider(script)
    kwargs.setdefault("catalog", sample_catalog())
    runner = AgentRunner(provider, ToolRegistry([EchoTool()]), tool_ctx, **kwargs)
    return runner, provider


def collect_events() -> tuple[list, object]:
    events: list = []
    return events, events.append


async def test_single_turn_done(tool_ctx):
    runner, provider = make_runner(tool_ctx, [{"text": ["Hello", " world"]}])
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "hi"}], on_event)

    assert result.final_text == "Hello world"
    assert result.stop_reason == "done"
    assert result.turns == 1
    assert [e.text for e in events if isinstance(e, Token)] == ["Hello", " world"]
    assert isinstance(events[-1], Done)
    # the tool specs are offered to the provider
    assert provider.requests[0]["tools"][0]["function"]["name"] == "echo"


async def test_llm_call_event_per_round(tool_ctx):
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}],
            "usage": {"input_tokens": 30, "output_tokens": 4},
        },
        {"text": "done", "usage": {"input_tokens": 50, "output_tokens": 6}},
    ]
    runner, _ = make_runner(tool_ctx, script)
    events, on_event = collect_events()

    await runner.run([{"role": "user", "content": "echo hi"}], on_event)

    calls = [e for e in events if isinstance(e, LlmCall)]
    assert [(e.model, e.turn) for e in calls] == [(runner.model, 1), (runner.model, 2)]
    # each LlmCall precedes its LlmResponse, which precedes the round's tools
    first_call = events.index(calls[0])
    assert isinstance(events[first_call + 1], LlmResponse)
    # each finished call logs its own usage
    responses = [e for e in events if isinstance(e, LlmResponse)]
    assert [(r.turn, r.input_tokens, r.output_tokens) for r in responses] == [
        (1, 30, 4),
        (2, 50, 6),
    ]
    assert all(r.cost_usd >= 0 for r in responses)


async def test_response_timing_excludes_tools(tool_ctx, monkeypatch):
    runner, _ = make_runner(
        tool_ctx,
        [
            {"tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}]},
            {"text": "done", "usage": {"input_tokens": 10, "output_tokens": 15}},
        ],
    )
    now = 0.0
    monkeypatch.setattr("lecode.agent.runner.time", SimpleNamespace(monotonic=lambda: now))
    stream_turn = runner._stream_turn

    async def timed_stream(history, on_event):
        nonlocal now
        completed = await stream_turn(history, on_event)
        now += 2.0
        return completed

    monkeypatch.setattr(runner, "_stream_turn", timed_stream)
    responses = []

    def on_event(event):
        nonlocal now
        if isinstance(event, ToolResult):
            now += 100.0
        elif isinstance(event, LlmResponse):
            responses.append(event)

    result = await runner.run([{"role": "user", "content": "echo hi"}], on_event)
    assert [event.elapsed_s for event in responses] == [2.0, 2.0]
    assert result.elapsed_s == 104.0


async def test_tool_round_trip(tool_ctx):
    script = [
        {"tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "hi"}'}]},
        {"text": "tool said hi"},
    ]
    runner, provider = make_runner(tool_ctx, script)
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "echo hi"}], on_event)

    assert result.stop_reason == "done"
    assert result.final_text == "tool said hi"
    assert result.turns == 2
    # second request carries the assistant tool call and the paired result
    messages = provider.requests[1]["messages"]
    assistant = messages[-2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "c1"
    tool_message = messages[-1]
    assert tool_message == {"role": "tool", "tool_call_id": "c1", "name": "echo", "content": "hi"}
    assert [e.id for e in events if isinstance(e, ToolCall)] == ["c1"]
    result_events = [e for e in events if isinstance(e, ToolResult)]
    assert result_events[0].content == "hi"
    assert result_events[0].is_error is False


async def test_parallel_tool_calls_paired_by_id(tool_ctx):
    script = [
        {
            "tool_calls": [
                {"id": "c1", "name": "echo", "arguments": '{"text": "one"}'},
                {"id": "c2", "name": "echo", "arguments": '{"text": "two"}'},
            ]
        },
        {"text": "done"},
    ]
    runner, provider = make_runner(tool_ctx, script)

    result = await runner.run([{"role": "user", "content": "go"}])

    assert result.stop_reason == "done"
    messages = provider.requests[1]["messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_messages] == [
        ("c1", "one"),
        ("c2", "two"),
    ]


async def test_empty_response_guard_gives_up(tool_ctx):
    runner, provider = make_runner(tool_ctx, [{}, {}, {}, {}])
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "hi"}], on_event)

    assert result.stop_reason == "empty"
    assert result.turns == 4
    # three nudges injected, one per retry
    for index, expected_nudges in ((1, 1), (2, 2), (3, 3)):
        messages = provider.requests[index]["messages"]
        nudges = [m for m in messages if m.get("content") == EMPTY_NUDGE]
        assert len(nudges) == expected_nudges
    assert isinstance(events[-1], Done)


async def test_empty_response_guard_recovers(tool_ctx):
    runner, _ = make_runner(tool_ctx, [{}, {"text": "fine now"}])

    result = await runner.run([{"role": "user", "content": "hi"}])

    assert result.stop_reason == "done"
    assert result.final_text == "fine now"
    assert result.turns == 2


async def test_please_continue_on_length(tool_ctx):
    script = [
        {"text": "part one ", "finish_reason": "length"},
        {"text": "part two"},
    ]
    runner, provider = make_runner(tool_ctx, script)

    result = await runner.run([{"role": "user", "content": "write"}])

    assert result.stop_reason == "done"
    assert result.final_text == "part one part two"
    messages = provider.requests[1]["messages"]
    assert messages[-1] == {"role": "user", "content": CONTINUE_PROMPT}
    assert messages[-2]["content"] == "part one "


async def test_max_turns(tool_ctx):
    tool_ctx.config.agent.max_turns = 2
    script = [
        {"tool_calls": [{"id": f"c{i}", "name": "echo", "arguments": "{}"}]} for i in range(5)
    ]
    runner, _ = make_runner(tool_ctx, script)

    result = await runner.run([{"role": "user", "content": "loop"}])

    assert result.stop_reason == "max_turns"
    assert result.turns == 2


class BlockingProvider:
    """Streams one token, then blocks forever (cancellation mid-stream)."""

    def __init__(self, started: asyncio.Event) -> None:
        self.started = started

    def stream_chat(self, messages, model, tools=None, **kwargs):
        return self._stream()

    async def _stream(self):
        yield TokenDelta(text="partial")
        self.started.set()
        await asyncio.Event().wait()


async def test_cancel_mid_stream_persists_partial(tool_ctx, tmp_path):
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("cancel-stream", tmp_path)
    started = asyncio.Event()
    runner = AgentRunner(
        BlockingProvider(started),
        ToolRegistry([EchoTool()]),
        tool_ctx,
        session=session,
        store=store,
    )

    task = asyncio.create_task(runner.run([{"role": "user", "content": "hi"}]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    records = [r for r in store.read_records(session) if isinstance(r, MessageRecord)]
    assert any(r.message.get("content") == "partial" for r in records)


async def test_cancel_mid_tool_cancels_in_flight(tool_ctx):
    provider = FakeProvider(
        [{"tool_calls": [{"id": "c1", "name": "sleep", "arguments": "{}"}]}, {"text": "late"}]
    )
    runner = AgentRunner(provider, ToolRegistry([SleepTool()]), tool_ctx)
    called = asyncio.Event()

    original_run = SleepTool.run

    async def run_and_mark(self, args, ctx):
        called.set()
        return await original_run(self, args, ctx)

    runner.registry.get("sleep").run = run_and_mark.__get__(runner.registry.get("sleep"))

    task = asyncio.create_task(runner.run([{"role": "user", "content": "hi"}]))
    await called.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.requests and len(provider.requests) == 1  # never reached turn 2


async def test_retrying_then_success(tool_ctx):
    script = [
        {"error": ProviderError("overloaded", status=503, retryable=True)},
        {"text": "recovered"},
    ]
    runner, _ = make_runner(tool_ctx, script)
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "hi"}], on_event)

    assert result.stop_reason == "done"
    assert result.final_text == "recovered"
    retries = [e for e in events if isinstance(e, Retrying)]
    assert len(retries) == 1
    assert retries[0].attempt == 1
    assert "overloaded" in retries[0].error


async def test_non_retryable_error_propagates(tool_ctx):
    runner, _ = make_runner(tool_ctx, [{"error": ProviderError("bad key", status=401)}])
    events, on_event = collect_events()

    with pytest.raises(ProviderError):
        await runner.run([{"role": "user", "content": "hi"}], on_event)

    errors = [e for e in events if isinstance(e, Error)]
    assert len(errors) == 1
    assert "bad key" in errors[0].message
    assert not [e for e in events if isinstance(e, Retrying)]


async def test_usage_and_cost_totals(tool_ctx):
    # catalog pricing for deepseek/deepseek-v4-flash: $0.09 / $0.18 per million tokens
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
        {"text": "done", "usage": {"input_tokens": 2000, "output_tokens": 100, "cost_usd": 0.5}},
    ]
    runner, _ = make_runner(tool_ctx, script)

    result = await runner.run([{"role": "user", "content": "go"}])

    totals = result.usage_totals
    assert totals.input_tokens == 3000
    assert totals.output_tokens == 600
    # context fill = last call's prompt size, not the accumulated sum
    assert totals.context_tokens == 2000
    expected = (1000 * 0.09 + 500 * 0.18) / 1_000_000 + 0.5  # catalog + provider-reported
    assert totals.cost_usd == pytest.approx(expected)


async def test_session_records_include_usage(tool_ctx, tmp_path):
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("usage", tmp_path, model=tool_ctx.config.llm.model)
    script = [
        {"tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "x"}'}]},
        {"text": "done", "usage": {"input_tokens": 42, "output_tokens": 7}},
    ]
    runner = AgentRunner(
        FakeProvider(script),
        ToolRegistry([EchoTool()]),
        tool_ctx,
        session=session,
        store=store,
        catalog=sample_catalog(),
    )

    result = await runner.run([{"role": "user", "content": "go"}])
    assert result.stop_reason == "done"

    records = [r for r in store.read_records(session) if isinstance(r, MessageRecord)]
    roles = [r.role for r in records]
    assert roles == ["assistant", "tool", "assistant"]
    final = records[-1]
    assert final.usage == {
        "input_tokens": 42,
        "output_tokens": 7,
        "cost_usd": (42 * 0.09 + 7 * 0.18) / 1_000_000,
    }
    assert records[1].usage is None  # tool messages carry no usage


async def test_steer_queue_drained_before_input_queue(tool_ctx):
    script = [
        {"tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}]},
        {"text": "done"},
    ]
    steer: asyncio.Queue[str] = asyncio.Queue()
    input_q: asyncio.Queue[str] = asyncio.Queue()
    steer.put_nowait("steer me")
    input_q.put_nowait("regular input")
    runner, provider = make_runner(tool_ctx, script, steer_queue=steer, input_queue=input_q)
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)
    assert result.stop_reason == "done"

    # queues drain between turns: turn 1 saw only the original messages
    assert [m["content"] for m in provider.requests[0]["messages"]] == ["go"]
    contents = [m["content"] for m in provider.requests[1]["messages"] if m["role"] == "user"]
    assert contents == ["go", "steer me", "regular input"]
    assert steer.empty() and input_q.empty()
    # each drained message is reported, steer first
    drained = [e.content for e in events if isinstance(e, QueuedMessage)]
    assert drained == ["steer me", "regular input"]


async def test_root_resume_repairs_interrupted_tool_before_worker_notification(setup):
    manager, ctx, _, store, session = setup

    class Background:
        def drain_notifications(self):
            return ["background finished"]

    ctx.extras[BACKGROUND_EXTRA] = Background()
    store.append_message(session, {"role": "user", "content": "resume"})
    store.append_message(
        session,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "interrupted",
                    "type": "function",
                    "function": {"name": "task", "arguments": "{}"},
                }
            ],
        },
    )
    store.append_event(
        session,
        "worker_notification",
        {
            "id": "finished",
            "worker_id": "child",
            "parent_id": None,
            "state": "completed",
            "kind": "completed",
            "content": "finished work",
            "deliver": True,
        },
    )
    provider = FakeProvider([{"text": "recovered"}])
    runner = AgentRunner(provider, ctx.extras["registry"], ctx, session=session, store=store)

    result = await runner.run(
        [{"role": "system", "content": "root"}, *store.load_for_model(session)]
    )

    assert result.final_text == "recovered"
    messages = provider.requests[0]["messages"]
    repaired = next(message for message in messages if message["role"] == "tool")
    assert repaired["tool_call_id"] == "interrupted"
    assert "outcome unknown" in repaired["content"]
    assert messages.index(repaired) < next(
        i for i, message in enumerate(messages) if message.get("content") == "background finished"
    )
    assert messages[-1]["content"] == "[worker child completed] finished work"
    assert not manager._outstanding(store.load_for_model(session))


@pytest.mark.parametrize("queue_name", ["input_queue", "steer_queue"])
async def test_child_question_root_human_reply_wakes_completion(setup, queue_name):
    manager, ctx, _, store, session = setup
    asked_human = asyncio.Event()
    queue = asyncio.Queue()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            assert not manager._outstanding(messages)
            outstanding = set()
            for message in messages:
                if message["role"] == "tool":
                    outstanding.remove(message["tool_call_id"])
                else:
                    assert not outstanding, "human reply preceded tool results"
                    outstanding.update(c["id"] for c in message.get("tool_calls", []))
            if messages[1]["content"] == "child":
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
                    assert messages[-1]["content"] == "use blue"
                    entry = {"text": "blue result"}
            elif len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {"id": "spawn", "name": "task", "arguments": '{"prompt":"child"}'}
                    ]
                }
            elif any("blue result" in str(m.get("content")) for m in messages):
                entry = {"text": "reviewed blue result"}
            elif any(m.get("tool_call_id") == "reply" for m in messages):
                entry = {"text": "waiting for child"}
            elif any(m.get("content") == "use blue" for m in messages):
                entry = {
                    "tool_calls": [
                        {
                            "id": "reply",
                            "name": "workers",
                            "arguments": json.dumps(
                                {"action": "send", "id": manager.list()[0].id, "text": "use blue"}
                            ),
                        }
                    ]
                }
            else:
                assert any("which color?" in str(m.get("content")) for m in messages)
                entry = {"text": "Human, which color?"}
            async for event in self._stream(entry):
                yield event

    def on_event(event):
        if isinstance(event, Token) and event.text == "Human, which color?":
            asked_human.set()

    store.append_message(session, {"role": "user", "content": "root"})
    runner = AgentRunner(
        Provider([]),
        ctx.extras["registry"],
        ctx,
        session=session,
        store=store,
        **{queue_name: queue},
    )
    task = asyncio.create_task(
        runner.run(
            [{"role": "system", "content": "root"}, *store.load_for_model(session)], on_event
        )
    )
    try:
        async with asyncio.timeout(2):
            await asked_human.wait()
            assert not task.done()
            queue.put_nowait("use blue")
            result = await task
        assert result.final_text == "reviewed blue result"
        assert manager.list()[0].result.final_text == "blue result"
        assert manager.list()[0].state == "completed"
        assert [m["content"] for m in store.load_for_model(session)].count("use blue") == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.parametrize("supervisor", ["root", "worker"])
@pytest.mark.parametrize("max_turns", [1, 2])
async def test_last_turn_background_spawn_reports_limit_with_unresolved_child(
    setup, supervisor, max_turns
):
    from lecode.extras.subagents import SubagentError

    manager, ctx, _, store, session = setup
    ctx.config.agent.max_turns = max_turns
    release = asyncio.Event()

    class Provider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            if messages[1]["content"] == "child":
                await release.wait()
                entry = {"text": "child done"}
            elif len(messages) == 2:
                entry = {
                    "tool_calls": [
                        {
                            "id": "spawn",
                            "name": "task",
                            "arguments": '{"prompt":"child","run_in_background":true}',
                        }
                    ],
                    "usage": {"input_tokens": 5, "cost_usd": 0.25},
                }
            else:
                entry = {"text": "premature final"}
            async for event in self._stream(entry):
                yield event

    provider = Provider([])
    ctx.extras["provider"] = provider
    events = []
    try:
        async with asyncio.timeout(2):
            if supervisor == "worker":
                parent = await manager.start(ctx, agent="explore", prompt="parent")
                with pytest.raises(SubagentError, match="max_turns"):
                    await manager.wait(parent.id)
                result = parent.result
                assert parent.state == "failed"
                child = manager.children(parent.id)[0]
                assert child.id in parent.error
                stopped_session = parent.session
            else:
                store.append_message(session, {"role": "user", "content": "root"})
                runner = AgentRunner(
                    provider, ctx.extras["registry"], ctx, session=session, store=store
                )
                result = await runner.run(
                    [{"role": "system", "content": "root"}, *store.load_for_model(session)],
                    events.append,
                )
                child = manager.children(None)[0]
                assert any(isinstance(e, Error) and child.id in e.message for e in events)
                assert events[-1] == Done("max_turns", max_turns)
                stopped_session = session
            assert result.stop_reason == "max_turns"
            assert result.turns == max_turns
            assert result.usage_totals.cost_usd == 0.25
            assert result.final_text == ("premature final" if max_turns == 2 else "")
            assert child.is_active
            stopped = store.load_events(stopped_session, "run_stopped")[-1]
            assert stopped["reason"] == "max_turns" and child.id in stopped["message"]
            release.set()
            await manager.wait(child.id)
    finally:
        release.set()
        await manager.shutdown()


@pytest.mark.parametrize("cancel", [False, True])
async def test_completion_queue_race_preserves_priority_inputs_and_cleans_waiters(setup, cancel):
    from tests.test_workers import GatedProvider

    manager, ctx, _, store, session = setup
    child_provider = GatedProvider()
    ctx.extras["provider"] = child_provider

    class Queue(asyncio.Queue):
        def __init__(self):
            super().__init__()
            self.waiting = asyncio.Event()
            self.received = asyncio.Event()
            self.waiters = []

        async def get(self):
            self.waiters.append(asyncio.current_task())
            self.waiting.set()
            item = await super().get()
            self.received.set()
            return item

    steer, inputs = Queue(), Queue()
    child = await manager.start(ctx, agent="explore", prompt="child")
    await child_provider.started.get()
    provider = FakeProvider([{"text": "waiting"}, {"text": "reviewed"}])
    runner = AgentRunner(
        provider,
        ctx.extras["registry"],
        ctx,
        session=session,
        store=store,
        steer_queue=steer,
        input_queue=inputs,
    )
    task = asyncio.create_task(runner.run([{"role": "user", "content": "root"}]))
    try:
        async with asyncio.timeout(2):
            await steer.waiting.wait()
            await inputs.waiting.wait()
            for queue, prefix in ((inputs, "input"), (steer, "steer")):
                queue.put_nowait(f"{prefix} one")
                queue.put_nowait(f"{prefix} two")
            await inputs.received.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                child_provider.release.set()
                await task
        texts = [
            m["content"]
            for m in store.load_for_model(session)
            if m["role"] == "user" and not m["content"].startswith("[worker")
        ]
        assert texts == ["steer one", "steer two", "input one", "input two"]
        assert all(waiter.done() for queue in (steer, inputs) for waiter in queue.waiters)
        assert steer.empty() and inputs.empty()
        if not cancel:
            assert [
                m["content"]
                for m in provider.requests[1]["messages"]
                if m["role"] == "user" and not m["content"].startswith("[worker")
            ] == ["root", *texts]
        child_provider.release.set()
        await manager.wait(child.id)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        child_provider.release.set()
        await manager.shutdown()


# -- automatic compaction ---------------------------------------------------------


def _compaction_setup(tool_ctx, tmp_path, pairs: int = 5):
    """A runner with a session seeded with enough history to compact."""
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("auto-compact", tmp_path)
    for index in range(pairs):
        store.append_message(session, {"role": "user", "content": f"q{index}"})
        store.append_message(session, {"role": "assistant", "content": f"a{index}"})
    # no catalog: the window comes from config.agent.context_window
    return store, session


def _compact_events(store, session):
    return [
        r for r in store.read_records(session) if isinstance(r, EventRecord) and r.kind == "compact"
    ]


async def test_auto_compaction_triggers_near_window(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200  # trigger at 800
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 900, "output_tokens": 5},
        },
        {"text": "the summary"},  # consumed by the compaction provider call
        {"text": "done", "usage": {"input_tokens": 100, "output_tokens": 4}},
    ]
    provider = FakeProvider(script)
    runner = AgentRunner(
        provider, ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
    )
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    assert result.stop_reason == "done"
    assert result.turns == 2
    compacts = _compact_events(store, session)
    assert compacts and compacts[-1].data["summary"] == "the summary"
    started = [e for e in events if isinstance(e, CompactionStarted)]
    assert [(e.context_tokens, e.threshold) for e in started] == [(900, 800)]
    finished = [e for e in events if isinstance(e, CompactionFinished)]
    assert [e.summary_chars for e in finished] == [len("the summary")]
    # the next call carries the summary plus the kept tail, old turns gone
    messages = provider.requests[-1]["messages"]
    assert messages[0] == {"role": "system", "content": "the summary"}
    contents = [m.get("content") for m in messages]
    assert "q4" in contents and "q0" not in contents


async def test_auto_compaction_disabled(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200
    tool_ctx.config.compaction.enabled = False
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 900, "output_tokens": 5},
        },
        {"text": "done", "usage": {"input_tokens": 950, "output_tokens": 4}},
    ]
    provider = FakeProvider(script)
    runner = AgentRunner(
        provider, ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
    )
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    assert result.stop_reason == "done"
    assert not _compact_events(store, session)
    assert not [e for e in events if isinstance(e, CompactionStarted)]


async def test_no_compaction_below_threshold(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200  # trigger at 800
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 700, "output_tokens": 5},
        },
        {"text": "done", "usage": {"input_tokens": 750, "output_tokens": 4}},
    ]
    runner = AgentRunner(
        FakeProvider(script), ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
    )
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    assert result.stop_reason == "done"
    assert not _compact_events(store, session)
    assert not [e for e in events if isinstance(e, CompactionStarted)]


async def test_pause_on_overflow_stops_run(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200
    tool_ctx.config.compaction.on_overflow = "pause"
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 900, "output_tokens": 5},
        },
        {"text": "the summary"},  # compaction call
        # still over the hard threshold after compacting → stop the run
        {
            "tool_calls": [{"id": "c2", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 850, "output_tokens": 4},
        },
        {"text": "never reached"},
    ]
    provider = FakeProvider(script)
    runner = AgentRunner(
        provider, ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
    )
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    assert result.stop_reason == "context_overflow"
    assert result.turns == 2
    assert _compact_events(store, session)
    assert isinstance(events[-1], Done) and events[-1].stop_reason == "context_overflow"
    assert len(provider.requests) == 3  # the run stopped before another call


async def test_compaction_skipped_without_session(tool_ctx):
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 900, "output_tokens": 5},
        },
        {"text": "done", "usage": {"input_tokens": 950, "output_tokens": 4}},
    ]
    runner, provider = make_runner(tool_ctx, script, catalog=None)
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    assert result.stop_reason == "done"
    assert not [e for e in events if isinstance(e, CompactionStarted)]
    assert len(provider.requests) == 2  # no compaction call consumed a script entry


async def test_compaction_failure_continues(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 900, "output_tokens": 5},
        },
        {"error": ProviderError("boom", retryable=False)},  # compaction call fails
        {"text": "done", "usage": {"input_tokens": 100, "output_tokens": 4}},
    ]
    runner = AgentRunner(
        FakeProvider(script), ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
    )
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    # fail-open: the run continues uncompacted
    assert result.stop_reason == "done"
    assert not _compact_events(store, session)
    assert [e for e in events if isinstance(e, CompactionStarted)]
    assert not [e for e in events if isinstance(e, CompactionFinished)]


async def test_mid_turn_threshold_triggers_earlier(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    # hard threshold far away (200000 - 20000); mid-turn threshold at 500
    tool_ctx.config.compaction.mid_turn_threshold = 500
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 600, "output_tokens": 5},
        },
        {"text": "the summary"},  # compaction call
        {"text": "done", "usage": {"input_tokens": 100, "output_tokens": 4}},
    ]
    provider = FakeProvider(script)
    runner = AgentRunner(
        provider, ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
    )
    events, on_event = collect_events()

    result = await runner.run([{"role": "user", "content": "go"}], on_event)

    assert result.stop_reason == "done"
    assert _compact_events(store, session)
    started = [e for e in events if isinstance(e, CompactionStarted)]
    assert [(e.context_tokens, e.threshold) for e in started] == [(600, 500)]


@pytest.mark.parametrize(
    ("model", "usage", "incomplete"),
    [
        ("missing/model", {"input_tokens": 7, "output_tokens": 2}, True),
        ("missing/model", {"input_tokens": 0, "output_tokens": 0}, True),
        ("missing/model", {"input_tokens": 7, "output_tokens": 2, "cost_usd": 0}, False),
        ("missing/model", {"cost_usd": 0}, False),
        ("openai/gpt-5-", {"input_tokens": 7}, True),
        ("openai/gpt-5-", {"cost_usd": 0}, False),
        ("free/model", {"input_tokens": 7, "output_tokens": 2}, False),
        ("free/model", {"input_tokens": 0, "output_tokens": 0}, False),
        ("free/model", None, True),
        ("free/model", {"cost_usd": 0, "incomplete": True}, True),
    ],
)
async def test_usage_completeness_reaches_events_totals_and_storage(
    tool_ctx, tmp_path, monkeypatch, model, usage, incomplete
):
    from lecode.providers.catalog import Pricing
    from lecode.session.stats import session_stats

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    catalog = sample_catalog()
    free = catalog.all()[0].model_copy(
        update={"id": "free/model", "pricing": Pricing(prompt=0, completion=0)}
    )
    tool_ctx.catalog = catalog.merge([free])
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("usage-completeness", tmp_path, model=model)
    runner, _ = make_runner(
        tool_ctx, [{"text": "done", "usage": usage}], catalog=None, store=store, session=session
    )
    runner.model = model
    events = []
    result = await runner.run([{"role": "user", "content": "hi"}], events.append)
    response = next(event for event in events if isinstance(event, LlmResponse))
    recorded = store.load_messages(session)[0].usage
    stats = session_stats(store, session, catalog=tool_ctx.catalog)

    assert response.usage_incomplete is incomplete
    assert result.usage_totals.usage_incomplete is incomplete
    assert bool(recorded.get("incomplete")) is incomplete
    assert stats.usage_incomplete is incomplete
    assert response.cost_usd == result.usage_totals.cost_usd == stats.cost_usd == 0
    assert response.input_tokens == stats.input_tokens == (usage or {}).get("input_tokens", 0)
    assert response.output_tokens == stats.output_tokens == (usage or {}).get("output_tokens", 0)


@pytest.mark.parametrize("usage", [None, {"input_tokens": 9, "output_tokens": 2, "cost_usd": 0.25}])
@pytest.mark.parametrize("text", ["partial", ""])
async def test_cancel_preserves_known_usage_or_unknown_marker(
    tool_ctx, tmp_path, monkeypatch, usage, text
):
    from lecode.providers.types import Usage
    from lecode.session.stats import session_stats

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    ready = asyncio.Event()

    class PartialProvider:
        async def stream_chat(self, *args, **kwargs):
            if text:
                yield TokenDelta(text=text)
            if usage is not None:
                yield Usage(usage=usage)
            ready.set()
            await asyncio.Event().wait()

    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("partial-usage", tmp_path, model=tool_ctx.config.llm.model)
    runner = AgentRunner(PartialProvider(), ToolRegistry(), tool_ctx, store=store, session=session)
    task = asyncio.create_task(runner.run([{"role": "user", "content": "hi"}]))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    recorded = store.load_messages(session)
    assert len(recorded) == 1
    assert recorded[0].usage == (
        usage or {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "incomplete": True}
    )
    stats = session_stats(store, session)
    assert stats.usage_incomplete is (usage is None)
    assert stats.cost_usd == (usage or {}).get("cost_usd", 0)


@pytest.mark.parametrize("review_usage", [None, {"input_tokens": 3}, {"cost_usd": 0}])
async def test_pierre_usage_completeness_is_persisted(
    tool_ctx, tmp_path, monkeypatch, review_usage
):
    from lecode.session.stats import session_stats

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    tool_ctx.config.pierre.enabled = True
    tool_ctx.config.pierre.model = "missing/reviewer"
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("review-usage", tmp_path, model=tool_ctx.config.llm.model)
    runner, _ = make_runner(
        tool_ctx,
        [{"text": "done", "usage": {"cost_usd": 0.25}}, {"text": "review", "usage": review_usage}],
        store=store,
        session=session,
    )
    result = await runner.run([{"role": "user", "content": "hi"}])
    incomplete = review_usage != {"cost_usd": 0}
    assert result.review == "review"
    assert result.usage_totals.usage_incomplete is incomplete
    assert store.load_events(session, "pierre")[0]["usage"]["incomplete"] is incomplete
    stats = session_stats(store, session)
    assert stats.usage_incomplete is incomplete
    assert stats.cost_usd == result.usage_totals.cost_usd == 0.25
