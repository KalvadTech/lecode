"""Contract tests for the OpenRouter preset (headers, catalog, routing)."""

from __future__ import annotations

import json

import httpx
import respx

from lecode.providers.openrouter import (
    OPENROUTER_BASE_URL,
    fetch_remote_catalog,
    openrouter_client,
    routing_extra_body,
)
from lecode.providers.types import collect

CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
MODELS_URL = f"{OPENROUTER_BASE_URL}/models"


def _ok_stream() -> httpx.Response:
    chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

    async def aiter():
        yield body

    return httpx.Response(200, content=aiter())


@respx.mock
async def test_app_identity_headers_sent():
    route = respx.post(CHAT_URL).mock(return_value=_ok_stream())
    async with openrouter_client(api_key="sk-or-test") as client:
        await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    headers = route.calls.last.request.headers
    assert headers["HTTP-Referer"] == "https://github.com/KalvadTech/lecode"
    assert headers["X-Title"] == "lecode"
    assert headers["Authorization"] == "Bearer sk-or-test"


@respx.mock
async def test_keyless_openrouter_client_sends_no_auth():
    route = respx.post(CHAT_URL).mock(return_value=_ok_stream())
    async with openrouter_client() as client:
        await collect(client.stream_chat([{"role": "user", "content": "hi"}], model="m"))
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
async def test_fetch_remote_catalog_maps_fields():
    payload = {
        "data": [
            {
                "id": "openai/gpt-5",
                "name": "OpenAI: GPT-5",
                "created": 1750000000,
                "context_length": 400000,
                "pricing": {"prompt": "0.00000125", "completion": "0.00001"},
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["tools", "reasoning"],
                "top_provider": {"max_completion_tokens": 128000},
            },
            {
                "id": "some/text-only",
                "context_length": 8192,
                # No pricing, architecture, or supported_parameters: tolerated.
            },
            {
                "id": "sparse/no-context",  # no context_length: default window
            },
            {
                # No id at all: skipped.
                "context_length": 8192,
            },
        ]
    }
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=payload))
    async with openrouter_client() as client:
        entries = await fetch_remote_catalog(client)

    assert [e.id for e in entries] == ["openai/gpt-5", "some/text-only", "sparse/no-context"]

    gpt5 = entries[0]
    assert gpt5.name == "OpenAI: GPT-5"
    assert gpt5.created == 1750000000
    assert gpt5.context_window == 400000
    assert gpt5.max_output == 128000
    # per-token -> per-million conversion
    assert gpt5.pricing.prompt == 1.25
    assert gpt5.pricing.completion == 10.0
    assert gpt5.modalities.input == ["text", "image"]
    assert gpt5.supports_tools is True
    assert gpt5.supports_reasoning is True

    sparse = entries[1]
    assert sparse.name == "some/text-only"
    assert sparse.created is None
    assert sparse.pricing.prompt == 0.0
    assert sparse.modalities.input == ["text"]
    assert sparse.supports_tools is True
    assert sparse.supports_reasoning is False

    defaulted = entries[2]
    assert defaulted.context_window == 128_000
    assert defaulted.pricing.prompt == 0.0


@respx.mock
async def test_routing_extra_body_reaches_request():
    route = respx.post(CHAT_URL).mock(return_value=_ok_stream())
    extra = routing_extra_body(["openai", "anthropic"], allow_fallbacks=False, sort="price")
    async with openrouter_client() as client:
        await collect(
            client.stream_chat([{"role": "user", "content": "hi"}], model="m", extra_body=extra)
        )
    payload = json.loads(route.calls.last.request.content)
    assert payload["provider"] == {
        "order": ["openai", "anthropic"],
        "allow_fallbacks": False,
        "sort": "price",
    }


def test_routing_extra_body_minimal():
    assert routing_extra_body() == {"provider": {}}
    assert routing_extra_body(["openai"]) == {"provider": {"order": ["openai"]}}


@respx.mock
async def test_usage_include_requested_and_billed_cost_normalized():
    """usage.include asks OpenRouter for the real cost; usage.cost → cost_usd."""
    chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.000123},
    }
    body = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode()

    async def aiter():
        yield body

    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, content=aiter()))
    async with openrouter_client() as client:
        completed = await collect(
            client.stream_chat([{"role": "user", "content": "hi"}], model="m")
        )

    payload = json.loads(route.calls.last.request.content)
    assert payload["usage"] == {"include": True}
    assert completed.usage["cost_usd"] == 0.000123
