"""Tests for the scripted FakeProvider."""

from __future__ import annotations

import pytest
from tests.fakes import FakeProvider

from lecode.providers.openai_compat import ProviderError
from lecode.providers.types import Done, ReasoningDelta, TokenDelta, ToolCallDelta, Usage, collect


async def test_text_and_reasoning_replay():
    provider = FakeProvider([{"reasoning": "thinking ", "text": ["hello", " world"]}])
    events = [e async for e in provider.stream_chat([], "m")]
    assert events == [
        ReasoningDelta(text="thinking "),
        TokenDelta(text="hello"),
        TokenDelta(text=" world"),
        Done(finish_reason="stop"),
    ]


async def test_tool_calls_chunked_arguments():
    provider = FakeProvider(
        [{"tool_calls": [{"id": "c1", "name": "read", "arguments": '{"path": "a.py"}'}]}]
    )
    message = await collect(provider.stream_chat([], "m"))
    assert message.finish_reason == "tool_calls"
    assert message.tool_calls[0]["function"]["name"] == "read"
    assert message.tool_calls[0]["function"]["arguments"] == '{"path": "a.py"}'


async def test_error_entry_raises():
    provider = FakeProvider([{"error": ProviderError("down", status=503, retryable=True)}])
    with pytest.raises(ProviderError, match="down"):
        await collect(provider.stream_chat([], "m"))


async def test_usage_entry():
    provider = FakeProvider([{"text": "x", "usage": {"prompt_tokens": 1}}])
    events = [e async for e in provider.stream_chat([], "m")]
    assert Usage(usage={"prompt_tokens": 1}) in events


async def test_request_capture():
    provider = FakeProvider([{"text": "a"}, {"text": "b"}])
    messages = [{"role": "user", "content": "hi"}]
    await collect(provider.stream_chat(messages, "m1", tools=[{"type": "function"}]))
    await collect(provider.stream_chat(messages, "m2"))
    assert len(provider.requests) == 2
    assert provider.requests[0]["model"] == "m1"
    assert provider.requests[0]["messages"] == messages
    assert provider.requests[0]["tools"] == [{"type": "function"}]
    assert provider.requests[1]["model"] == "m2"


async def test_script_exhaustion_fallback():
    provider = FakeProvider([])
    message = await collect(provider.stream_chat([], "m"))
    assert "no scripted response" in message.content


async def test_tool_call_delta_indices():
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {"id": "c1", "name": "read", "arguments": "{}"},
                    {"id": "c2", "name": "grep", "arguments": "{}"},
                ]
            }
        ]
    )
    events = [e async for e in provider.stream_chat([], "m")]
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert {d.index for d in deltas} == {0, 1}
