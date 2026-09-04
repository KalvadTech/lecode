"""Tests for the multi-turn agent runner against the scripted FakeProvider."""

from __future__ import annotations

import asyncio

import pytest
from tests.fakes import FakeProvider

from lecode.agent.runner import (
    CONTINUE_PROMPT,
    EMPTY_NUDGE,
    AgentRunner,
    Done,
    Error,
    Retrying,
    Token,
    ToolCall,
    ToolResult,
)
from lecode.agent.tools.base import Tool, ToolRegistry
from lecode.agent.tools.base import ToolResult as ToolExecResult
from lecode.providers.openai_compat import ProviderError
from lecode.providers.types import TokenDelta
from lecode.session.model import MessageRecord
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
    # catalog pricing for openai/gpt-5-mini: $0.25 / $2.00 per million tokens
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
    expected = (1000 * 0.25 + 500 * 2.0) / 1_000_000 + 0.5  # catalog + provider-reported
    assert totals.cost_usd == pytest.approx(expected)


async def test_session_records_include_usage(tool_ctx, tmp_path):
    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("usage", tmp_path, model=tool_ctx.config.llm.model)
    script = [
        {"tool_calls": [{"id": "c1", "name": "echo", "arguments": '{"text": "x"}'}]},
        {"text": "done", "usage": {"input_tokens": 42, "output_tokens": 7}},
    ]
    runner = AgentRunner(
        FakeProvider(script), ToolRegistry([EchoTool()]), tool_ctx, session=session, store=store
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
        "cost_usd": (42 * 0.25 + 7 * 2.0) / 1_000_000,
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

    result = await runner.run([{"role": "user", "content": "go"}])
    assert result.stop_reason == "done"

    # queues drain between turns: turn 1 saw only the original messages
    assert [m["content"] for m in provider.requests[0]["messages"]] == ["go"]
    contents = [m["content"] for m in provider.requests[1]["messages"] if m["role"] == "user"]
    assert contents == ["go", "steer me", "regular input"]
    assert steer.empty() and input_q.empty()
