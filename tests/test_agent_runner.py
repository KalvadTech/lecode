"""Tests for the multi-turn agent runner against the scripted FakeProvider."""

from __future__ import annotations

import asyncio

import pytest
from tests.fakes import FakeProvider, sample_catalog

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


async def test_run_refreshes_system_prompt_in_place(tool_ctx):
    provider = FakeProvider([{"text": "done"}])
    runner = AgentRunner(
        provider,
        ToolRegistry([]),
        tool_ctx,
        refresh_prompt=lambda: "REFRESHED PROMPT",
    )

    await runner.run(
        [
            {"role": "system", "content": "stale prompt"},
            {"role": "user", "content": "hi"},
        ]
    )

    messages = provider.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]  # replaced, not appended
    assert messages[0]["content"] == "REFRESHED PROMPT"


async def test_memory_write_refreshes_next_request_preserving_turn_overlay(tool_ctx):
    state = {"memory": "old memory"}

    class WriteMemory(EchoTool):
        async def run(self, args, ctx):
            state["memory"] = "new memory"
            return ToolExecResult(content="written")

    provider = FakeProvider(
        [
            {"tool_calls": [{"id": "c", "name": "echo", "arguments": "{}"}]},
            {"text": "done"},
        ]
    )
    runner = AgentRunner(
        provider,
        ToolRegistry([WriteMemory()]),
        tool_ctx,
        refresh_prompt=lambda: "AGENTS skills custom " + state["memory"],
    )
    await runner.run(
        [
            {"role": "system", "content": "stale base"},
            {"role": "system", "content": "turn persona"},
            {"role": "user", "content": "remember"},
        ]
    )
    second = provider.requests[1]["messages"]
    assert second[0]["content"] == "AGENTS skills custom new memory"
    assert second[1]["content"] == "turn persona"
    assert str(second).count("AGENTS skills custom") == 1


async def test_refresh_keeps_valid_summary_when_no_base_prompt_supplied(tool_ctx, tmp_path):
    from lecode.session.compaction import compact_session

    store, session = _compaction_setup(tool_ctx, tmp_path)
    await compact_session(FakeProvider([{"text": "valid summary"}]), store, session, "test-model")
    provider = FakeProvider([{"text": "done"}])
    runner = AgentRunner(
        provider,
        ToolRegistry([]),
        tool_ctx,
        session=session,
        store=store,
        refresh_prompt=lambda: "base instructions",
    )
    await runner.run(store.load_for_model(session))
    assert [m["content"] for m in provider.requests[0]["messages"][:2]] == [
        "base instructions",
        "valid summary",
    ]


async def test_source_clear_during_auto_compaction_pauses_without_sending_stale_history(
    tool_ctx, tmp_path
):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.compaction.mid_turn_threshold = 100

    class ClearProvider(FakeProvider):
        async def complete(self, *args, **kwargs):
            store.append_event(session, "clear")
            await asyncio.sleep(0)
            return await super().complete(*args, **kwargs)

    provider = ClearProvider(
        [
            {"tool_calls": [{"id": "c", "name": "echo", "arguments": "{}"}]},
            {"text": "stale"},
            {"text": "must not send"},
        ]
    )
    runner = AgentRunner(
        provider, ToolRegistry([EchoTool()]), tool_ctx, store=store, session=session
    )
    result = await runner.run(store.load_for_model(session))
    assert result.stop_reason == "context_overflow"
    assert len(provider.requests) == 2
    assert store.load_for_model(session) == []


async def test_last_request_boundary_rechecks_refreshed_prompt(tool_ctx):
    state = {"prompt": "short"}
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200
    provider = FakeProvider([{"text": "must not send"}])
    runner = AgentRunner(
        provider, ToolRegistry([]), tool_ctx, refresh_prompt=lambda: state["prompt"]
    )

    def on_event(event):
        if isinstance(event, LlmCall):
            state["prompt"] = "new memory " * 1000

    result = await runner.run([{"role": "user", "content": "go"}], on_event)
    assert result.stop_reason == "context_overflow"
    assert not provider.requests


async def test_last_request_boundary_rechecks_source_visibility(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    provider = FakeProvider([{"text": "must not send hidden source"}])
    runner = AgentRunner(provider, ToolRegistry([]), tool_ctx, store=store, session=session)

    async def on_event(event):
        if isinstance(event, LlmCall):
            store.append_event(session, "clear")
            await asyncio.sleep(0)

    result = await runner.run(store.load_for_model(session), on_event)
    assert result.stop_reason == "context_overflow"
    assert not provider.requests


async def test_session_bound_chain_reports_unsafe_request_and_stops(tool_ctx, tmp_path):
    from lecode.extras.chain import run_chain

    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200
    provider = FakeProvider([])
    with pytest.raises(ProviderError, match="chain paused"):
        await run_chain(
            lambda: AgentRunner(provider, ToolRegistry([]), tool_ctx, store=store, session=session),
            "huge " * 2000,
        )
    # A session summary may be attempted, but no oversized phase request is sent.
    assert len(provider.requests) <= 1
    assert all("inert" in r["messages"][0]["content"] for r in provider.requests)


async def test_run_rebuilds_stale_summary_from_visible_raw_sources(tool_ctx, tmp_path):
    from lecode.session.compaction import compact_session

    store, session = _compaction_setup(tool_ctx, tmp_path)
    await compact_session(FakeProvider([{"text": "now invalid"}]), store, session, "test-model")
    stale = [{"role": "system", "content": "base"}, *store.load_for_model(session)]
    store.rewind_to(session, 2)
    provider = FakeProvider([{"text": "done"}])
    runner = AgentRunner(
        provider,
        ToolRegistry([]),
        tool_ctx,
        store=store,
        session=session,
        refresh_prompt=lambda: "fresh base",
    )
    await runner.run([*stale, {"role": "user", "content": "new transient request"}])
    sent = provider.requests[0]["messages"]
    assert [m["content"] for m in sent] == ["fresh base", "q0", "a0", "new transient request"]


@pytest.mark.parametrize("oversize", ["system", "tools", "user"])
async def test_current_catalog_model_limits_preflight_even_with_continue(tool_ctx, oversize):
    catalog = sample_catalog()
    small = catalog.get(tool_ctx.config.llm.model).model_copy(
        update={"context_window": 1000, "max_output": 100}
    )
    catalog = catalog.merge([small])
    tool_ctx.config.agent.context_window = 1000000  # not the current model's limit
    tool_ctx.config.compaction.on_overflow = "continue"
    tool = EchoTool()
    if oversize == "tools":
        tool.description = "large tool schema " * 1000
    provider = FakeProvider([{"text": "must not send"}])
    runner = AgentRunner(provider, ToolRegistry([tool]), tool_ctx, catalog=catalog)
    result = await runner.run(
        [
            {"role": "system", "content": "s" * (4000 if oversize == "system" else 1)},
            {"role": "user", "content": "q" * (4000 if oversize == "user" else 1)},
        ]
    )
    assert result.stop_reason == "context_overflow"
    assert not provider.requests


async def test_huge_new_tool_result_pauses_before_next_request_without_losing_raw(
    tool_ctx, tmp_path
):
    class HugeTool(EchoTool):
        async def run(self, args, ctx):
            return ToolExecResult(content="exact huge output " * 1000)

    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 2000
    tool_ctx.config.compaction.buffer_tokens = 200
    provider = FakeProvider(
        [
            {"tool_calls": [{"id": "huge", "name": "echo", "arguments": "{}"}]},
            {"text": "summary"},
        ]
    )
    runner = AgentRunner(
        provider, ToolRegistry([HugeTool()]), tool_ctx, store=store, session=session
    )
    result = await runner.run(store.load_for_model(session))
    assert result.stop_reason == "context_overflow" and result.turns == 1
    assert len(provider.requests) == 2
    tail = store.load_for_model(session)
    assert tail[-1]["content"] == "exact huge output " * 1000
    assert tail[-2]["tool_calls"][0]["id"] == "huge"


async def test_six_automatic_compactions_keep_one_summary_and_transient_overlay(tool_ctx, tmp_path):
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("six-runs", tmp_path)
    tool_ctx.config.agent.context_window = 15000
    tool_ctx.config.compaction.buffer_tokens = 4000
    provider = FakeProvider(
        [entry for i in range(6) for entry in ({"text": f"summary-{i}"}, {"text": f"done-{i}"})]
    )
    runner = AgentRunner(
        provider,
        ToolRegistry([]),
        tool_ctx,
        store=store,
        session=session,
        refresh_prompt=lambda: "fresh base",
    )
    for i in range(6):
        for j in range(6):
            store.append_message(session, {"role": "user", "content": f"raw-{i}-{j}" + "x" * 3000})
            store.append_message(session, {"role": "assistant", "content": "y" * 3000})
        result = await runner.run(
            [
                {"role": "system", "content": "old base"},
                {"role": "system", "content": "persona"},
                *store.load_for_model(session),
                {"role": "user", "content": "current request"},
            ]
        )
        assert result.final_text == f"done-{i}"
        assert result.usage_totals.unknown_usage_calls == 1
        request = provider.requests[-1]["messages"]
        assert [m["content"] for m in request if m["role"] == "system"] == [
            "fresh base",
            "persona",
            f"summary-{i}",
        ]
        assert request[-1]["content"] == "current request"
        if i:
            summary_input = str(provider.requests[-2]["messages"])
            assert f"summary-{i - 1}" in summary_input
            assert "raw-0-0" not in summary_input


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
    assert records[-1].message.get("incomplete") is True


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


# -- automatic compaction ---------------------------------------------------------


async def test_resumed_first_request_compacts_and_accounts_memory_once(tool_ctx, tmp_path):
    from lecode.session.stats import session_stats

    assert tool_ctx.config.memory.auto_learn is False
    assert 0 <= tool_ctx.config.memory.facts_max_bytes <= 65536

    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("resumed", tmp_path, model=tool_ctx.config.llm.model)
    for i in range(6):
        store.append_message(session, {"role": "user", "content": f"old-{i}" + "x" * 3000})
        store.append_message(session, {"role": "assistant", "content": "y" * 3000})
    store.append_message(session, {"role": "user", "content": "finish now"})
    tool_ctx.config.agent.context_window = 15000
    tool_ctx.config.compaction.buffer_tokens = 4000
    provider = FakeProvider(
        [
            {
                "text": "working summary",
                "usage": {"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.1},
            },
            {"text": "done", "usage": {"input_tokens": 50, "output_tokens": 5, "cost_usd": 0.02}},
        ]
    )
    runner = AgentRunner(
        provider,
        ToolRegistry([]),
        tool_ctx,
        store=store,
        session=session,
        refresh_prompt=lambda: "fresh instructions",
    )
    result = await runner.run(
        [{"role": "system", "content": "old instructions"}, *store.load_for_model(session)]
    )
    assert result.final_text == "done" and result.turns == 1
    assert len(provider.requests) == 2
    messages = provider.requests[-1]["messages"]
    assert messages[0]["content"] == "fresh instructions"
    assert messages[1]["content"] == "working summary"
    assert messages[-1]["content"] == "finish now"
    assert (result.usage_totals.input_tokens, result.usage_totals.output_tokens) == (150, 25)
    assert result.usage_totals.cost_usd == pytest.approx(0.12)
    stats = session_stats(store, store.open(session.id))
    assert (stats.input_tokens, stats.output_tokens, stats.cost_usd) == (
        150,
        25,
        pytest.approx(0.12),
    )


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
    tool_ctx.config.agent.context_window = 1200
    tool_ctx.config.compaction.buffer_tokens = 200  # trigger at 1000
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

    result = await runner.run(
        [*store.load_for_model(session), {"role": "user", "content": "go"}], on_event
    )

    assert result.stop_reason == "done"
    assert result.turns == 2
    compacts = _compact_events(store, session)
    assert compacts and compacts[-1].data["summary"] == "the summary"
    started = [e for e in events if isinstance(e, CompactionStarted)]
    assert started and all(e.context_tokens >= e.threshold == 1000 for e in started)
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

    assert result.stop_reason == "context_overflow"  # disabled cannot bypass a known bound
    assert not _compact_events(store, session)
    assert not [e for e in events if isinstance(e, CompactionStarted)]


async def test_no_compaction_below_threshold(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 1000
    tool_ctx.config.compaction.buffer_tokens = 200  # trigger at 800
    script = [
        {
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": "{}"}],
            "usage": {"input_tokens": 100, "output_tokens": 5},
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
    assert result.turns == 1
    assert _compact_events(store, session)
    assert isinstance(events[-1], Done) and events[-1].stop_reason == "context_overflow"
    assert len(provider.requests) == 2  # pause before the already unsafe next request


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

    assert result.stop_reason == "context_overflow"
    assert not [e for e in events if isinstance(e, CompactionStarted)]
    assert len(provider.requests) == 1


async def test_compaction_failure_continues(tool_ctx, tmp_path):
    store, session = _compaction_setup(tool_ctx, tmp_path)
    tool_ctx.config.agent.context_window = 200000
    tool_ctx.config.compaction.buffer_tokens = 200
    tool_ctx.config.compaction.mid_turn_threshold = 800
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
    assert started and all(e.context_tokens >= e.threshold == 500 for e in started)


async def test_live_runner_reloads_after_forget_and_cannot_launder_recalled_output(
    tool_ctx, tmp_path
):
    import json

    from lecode.memory.facts import FactStore
    from lecode.memory.tools import memory_tools

    store = SessionStore(tmp_path / "cfg")
    source = store.create("source", tmp_path)
    session = store.create("consumer", tmp_path)
    facts = FactStore(tmp_path / "facts.sqlite3")
    store.bind_facts(tmp_path, facts)
    tool_ctx.session, tool_ctx.session_store = session, store
    tool_ctx.extras["facts"] = facts
    store.append_message(source, {"role": "user", "content": "forgotten evidence"})
    ref = store.source_snapshot(source.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("claim", ref, sessions=store, project_root=tmp_path)
    store.append_message(session, {"role": "user", "content": "recall the fact"})
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {
                        "id": "r",
                        "name": "memory_recall",
                        "arguments": json.dumps({"fact_id": fact.id}),
                    }
                ]
            },
            {"text": "forgotten evidence rephrased"},
            {"text": "clean answer"},
        ]
    )
    runner = AgentRunner(
        provider, ToolRegistry(memory_tools()), tool_ctx, store=store, session=session
    )
    await runner.run(store.load_for_model(session))
    cached = list(tool_ctx.extras["conversation"])
    derived = store.load_messages(session)[-1]
    derived_ref = store.source_snapshot(
        session.id, derived.seq, derived.seq, project_root=tmp_path
    ).ref
    descendant = facts.remember(
        "laundered claim", derived_ref, sessions=store, project_root=tmp_path
    )
    other = FactStore(facts.path)
    other.forget(fact.id)
    other.close()
    request = {"role": "user", "content": "independent new request"}
    store.append_message(session, request)
    await runner.run([*cached, request])
    assert "forgotten evidence" not in str(provider.requests[-1]["messages"])
    assert "independent new request" in str(provider.requests[-1]["messages"])
    assert "clean answer" in str(store.load_for_model(session))
    assert store.validate_source(derived_ref, project_root=tmp_path).status == "hidden"
    _, recalled = await runner.registry.dispatch_result(
        "r2", "memory_recall", json.dumps({"fact_id": descendant.id}), tool_ctx
    )
    assert "laundered claim" not in recalled.content
    with pytest.raises(ValueError, match="hidden"):
        facts.remember("new laundering", derived_ref, sessions=store, project_root=tmp_path)
    facts.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_forget_during_provider_await_does_not_promote_stale_response(
    tool_ctx, tmp_path, cancel
):
    from lecode.memory.facts import FactStore

    store = SessionStore(tmp_path / "cfg")
    session = store.create("waiting", tmp_path)
    facts = FactStore(tmp_path / "facts.sqlite3")
    store.bind_facts(tmp_path, facts)
    tool_ctx.extras["facts"] = facts
    store.append_message(session, {"role": "user", "content": "old evidence"})
    ref = store.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("old claim", ref, sessions=store, project_root=tmp_path)
    entered, resume = asyncio.Event(), asyncio.Event()

    class Delayed(FakeProvider):
        async def _stream(self, entry):
            yield TokenDelta(text="stale partial")
            entered.set()
            await resume.wait()
            async for event in super()._stream(entry):
                yield event

    provider = Delayed([{"text": "stale response"}])
    runner = AgentRunner(provider, ToolRegistry([]), tool_ctx, store=store, session=session)
    task = asyncio.create_task(runner.run(store.load_for_model(session)))
    await entered.wait()
    other = FactStore(facts.path)
    other.forget(fact.id)
    other.close()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        resume.set()
        result = await task
        assert result.stop_reason == "context_overflow"
        assert result.final_text == ""
    assert len(provider.requests) == 1
    assert store.load_for_model(session) == []
    assert all(r.role == "user" for r in store.load_messages(session))
    facts.close()


async def test_chain_does_not_reintroduce_forgotten_prior_phase_output(tool_ctx, tmp_path):
    from lecode.extras.chain import run_chain
    from lecode.memory.facts import FactStore

    facts = FactStore(tmp_path / "facts.sqlite3")
    tool_ctx.extras["facts"] = facts
    fact = facts.add("evidence", source_id="s", source_seq=1)
    provider = FakeProvider([{"text": "derived evidence"}, {"text": "must not send"}])
    with pytest.raises(ProviderError, match="chain paused"):
        await run_chain(
            lambda: AgentRunner(provider, ToolRegistry([]), tool_ctx),
            "go",
            on_phase=lambda phase, text: facts.forget(fact.id),
        )
    assert len(provider.requests) == 1
    facts.close()


async def test_completed_background_recall_cannot_launder_into_new_request(
    tool_ctx, tmp_path, monkeypatch
):
    from lecode.agent.builder import build_runtime
    from lecode.extras.background import BACKGROUND_EXTRA

    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    runtime = build_runtime(tool_ctx.config, tmp_path)
    facts = runtime.ctx.extras["facts"]
    fact = facts.add("evidence", source_id="s", source_seq=1)
    manager = runtime.ctx.extras[BACKGROUND_EXTRA]

    async def body(emit):
        emit(b"old recalled evidence")
        return "old recalled evidence", 0

    record = manager.start("agent", "derived description", body)
    await record.task
    facts.forget(fact.id)
    provider = FakeProvider(
        [
            {"tool_calls": [{"id": "bg", "name": "tasks_output", "arguments": '{"id":"bg-1"}'}]},
            {"text": "clean"},
        ]
    )
    runner = AgentRunner(provider, runtime.registry, runtime.ctx)
    await runner.run([{"role": "user", "content": "new independent request"}])
    assert "old recalled evidence" not in str(provider.requests)
    assert "derived description" not in str(provider.requests)
    facts.close()


@pytest.mark.parametrize("change", ["clear", "undo"])
async def test_cached_raw_runner_history_respects_working_visibility(tool_ctx, tmp_path, change):
    store = SessionStore(tmp_path / "cfg")
    session = store.create("source", tmp_path)
    store.append_message(session, {"role": "user", "content": "old request"})
    store.append_message(session, {"role": "assistant", "content": "old answer"})
    cached = store.load_for_model(session)
    runner, provider = make_runner(tool_ctx, [{"text": "clean"}], store=store, session=session)
    if change == "clear":
        store.append_event(session, "clear")
    else:
        store.undo(session)
    fresh = {"role": "user", "content": "fresh independent request"}
    store.append_message(session, fresh)
    await runner.run([*cached, fresh])
    assert provider.requests[0]["messages"] == [fresh]
