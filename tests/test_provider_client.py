"""Contract tests for the OpenAI-compatible streaming client (SSE, errors)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from lecode.providers.openai_compat import ChatClient, ProviderError
from lecode.providers.types import Done, TokenDelta, ToolCallDelta, Usage, collect

BASE = "https://api.test/v1"
CHAT_URL = f"{BASE}/chat/completions"


def _sse(*events: str) -> bytes:
    """Encode SSE events, blank-line separated."""
    return ("\n\n".join(events) + "\n\n").encode()


def _chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> str:
    chunk: dict = {"id": "chatcmpl-1", "choices": []}
    if delta is not None or finish is not None:
        chunk["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        chunk["usage"] = usage
    return "data: " + json.dumps(chunk)


async def _aiter(body: bytes):
    yield body


def _sse_route(body: bytes) -> respx.Route:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_aiter(body), headers={"content-type": "text/event-stream"}
        )

    return respx.post(CHAT_URL).mock(side_effect=handler)


async def _stream_events(client: ChatClient) -> list:
    """One full pass over a simple prompt stream."""
    messages = [{"role": "user", "content": "hi"}]
    return [e async for e in client.stream_chat(messages, model="m")]


async def _collect_once(client: ChatClient):
    """One collected turn over a simple prompt stream."""
    messages = [{"role": "user", "content": "hi"}]
    return await collect(client.stream_chat(messages, model="m"))


@respx.mock
async def test_token_stream_and_done():
    _sse_route(
        _sse(_chunk({"content": "Hello"}), _chunk({"content": " world"}, "stop"), "data: [DONE]")
    )
    async with ChatClient(BASE, api_key="sk-test") as client:
        events = await _stream_events(client)
    assert events == [
        TokenDelta(text="Hello"),
        TokenDelta(text=" world"),
        Done(finish_reason="stop"),
    ]


@respx.mock
async def test_request_shape_and_auth_header():
    route = _sse_route(_sse(_chunk({"content": "ok"}, "stop"), "data: [DONE]"))
    async with ChatClient(BASE, api_key="sk-test") as client:
        await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-test"
    payload = json.loads(request.content)
    assert payload["model"] == "m"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


@respx.mock
async def test_keyless_client_sends_no_authorization():
    route = _sse_route(_sse(_chunk({"content": "ok"}, "stop"), "data: [DONE]"))
    async with ChatClient(BASE) as client:
        await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
async def test_multiline_data_and_comments_ignored():
    # One JSON payload split across two data: lines, plus a keep-alive comment.
    body = (
        b": keep-alive\n"
        b"\n"
        b'data: {"choices": [{"delta": {"content":\n'
        b'data:  "Hi"}, "finish_reason": "stop"}]}\n'
        b"\n"
        b"data: [DONE]\n"
        b"\n"
    )
    _sse_route(body)
    async with ChatClient(BASE) as client:
        events = await _stream_events(client)
    assert events == [TokenDelta(text="Hi"), Done(finish_reason="stop")]


@respx.mock
async def test_reasoning_deltas_both_keys():
    body = _sse(
        _chunk({"reasoning_content": "let me "}),
        _chunk({"reasoning": "think"}),
        _chunk({"content": "answer"}, "stop"),
        "data: [DONE]",
    )
    _sse_route(body)
    async with ChatClient(BASE) as client:
        message = await _collect_once(client)
    assert message.reasoning == "let me think"
    assert message.content == "answer"


@respx.mock
async def test_interleaved_tool_calls_accumulate_by_index():
    body = _sse(
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "read", "arguments": '{"path":'},
                    }
                ]
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call_2",
                        "function": {"name": "grep", "arguments": '{"pat":'},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "a.py"}'}}]}),
        _chunk({"tool_calls": [{"index": 1, "function": {"arguments": ' "x"}'}}]}, "tool_calls"),
        "data: [DONE]",
    )
    _sse_route(body)
    async with ChatClient(BASE) as client:
        events = await _stream_events(client)
        message = await _collect_once(client)
    assert any(isinstance(e, ToolCallDelta) and e.index == 1 for e in events)
    assert message.finish_reason == "tool_calls"
    assert message.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path": "a.py"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "grep", "arguments": '{"pat": "x"}'},
        },
    ]


@respx.mock
async def test_usage_captured_from_final_chunk():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    body = _sse(_chunk({"content": "ok"}, "stop"), _chunk(usage=usage), "data: [DONE]")
    _sse_route(body)
    async with ChatClient(BASE) as client:
        events = await _stream_events(client)
    assert Usage(usage=usage) in events


@respx.mock
async def test_multimodal_content_parts_serialized_verbatim():
    route = _sse_route(_sse(_chunk({"content": "ok"}, "stop"), "data: [DONE]"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {
                    "type": "file",
                    "file": {"file_data": "data:application/pdf;base64,BBB", "filename": "x.pdf"},
                },
            ],
        }
    ]
    async with ChatClient(BASE) as client:
        await collect(client.stream_chat(messages, model="m"))
    payload = json.loads(route.calls.last.request.content)
    assert payload["messages"][0]["content"] == messages[0]["content"]


@respx.mock
async def test_cache_control_passes_through_verbatim():
    route = _sse_route(_sse(_chunk({"content": "ok"}, "stop"), "data: [DONE]"))
    messages = [
        {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}},
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
        },
    ]
    async with ChatClient(BASE) as client:
        await collect(client.stream_chat(messages, model="m"))
    payload = json.loads(route.calls.last.request.content)
    assert payload["messages"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
@respx.mock
async def test_error_mapping_per_status(status: int, retryable: bool):
    body = json.dumps({"error": {"message": f"boom {status}", "code": status}})
    respx.post(CHAT_URL).mock(return_value=httpx.Response(status, text=body))
    async with ChatClient(BASE) as client:
        with pytest.raises(ProviderError) as excinfo:
            await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    error = excinfo.value
    assert error.status == status
    assert error.retryable is retryable
    assert f"boom {status}" in str(error)


@respx.mock
async def test_midstream_error_event_raises():
    body = _sse(
        _chunk({"content": "partial"}),
        "data: " + json.dumps({"error": {"message": "upstream exploded", "code": 502}}),
        "data: [DONE]",
    )
    _sse_route(body)
    async with ChatClient(BASE) as client:
        with pytest.raises(ProviderError) as excinfo:
            await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    assert excinfo.value.status == 502
    assert excinfo.value.retryable is True
    assert "upstream exploded" in str(excinfo.value)


@respx.mock
async def test_transport_error_is_retryable():
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    async with ChatClient(BASE) as client:
        with pytest.raises(ProviderError) as excinfo:
            await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    assert excinfo.value.retryable is True


@respx.mock
async def test_list_models():
    respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )
    async with ChatClient(BASE) as client:
        models = await client.list_models()
    assert models == [{"id": "a"}, {"id": "b"}]


@respx.mock
async def test_complete_helper_collects():
    _sse_route(_sse(_chunk({"content": "full answer"}, "stop"), "data: [DONE]"))
    async with ChatClient(BASE) as client:
        message = await client.complete([{"role": "user", "content": "hi"}], model="m")
    assert message.content == "full answer"
    assert message.finish_reason == "stop"
