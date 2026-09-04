"""Tests for the live model catalog: fetch → cache → bundled fallback."""

from __future__ import annotations

import json

import httpx
import respx

from lecode.providers.live import CACHE_FILENAME, load_catalog
from lecode.providers.openrouter import OPENROUTER_BASE_URL, openrouter_client

MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

LIVE_MODEL = {
    "id": "acme/live-1",
    "name": "Live One",
    "context_length": 64000,
    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    "supported_parameters": ["tools"],
}


def _payload(models: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"data": models})


@respx.mock
async def test_live_fetch_merges_over_bundled_and_writes_cache(tmp_path):
    respx.get(MODELS_URL).mock(return_value=_payload([LIVE_MODEL]))
    async with openrouter_client() as client:
        loaded = await load_catalog(client, tmp_path)

    assert loaded.origin == "live"
    assert loaded.remote_count == 1
    entry = loaded.catalog.get("acme/live-1")
    assert entry.context_window == 64000
    assert entry.pricing.prompt == 1.0
    # bundled entries survive the merge
    assert loaded.catalog.get("deepseek/deepseek-v4-flash").context_window == 1048576

    cache = json.loads((tmp_path / CACHE_FILENAME).read_text("utf-8"))
    assert cache["base_url"] == OPENROUTER_BASE_URL
    assert cache["models"][0]["id"] == "acme/live-1"


@respx.mock
async def test_failed_fetch_without_cache_falls_back_to_bundled(tmp_path):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    async with openrouter_client() as client:
        loaded = await load_catalog(client, tmp_path)

    assert loaded.origin == "bundled"
    assert loaded.catalog.get("deepseek/deepseek-v4-flash") is not None
    assert not (tmp_path / CACHE_FILENAME).exists()


@respx.mock
async def test_failed_fetch_uses_matching_cache(tmp_path):
    # Seed the cache via a successful fetch, then fail the next one.
    responses = [_payload([LIVE_MODEL])]
    respx.get(MODELS_URL).mock(side_effect=lambda request: responses[0])
    async with openrouter_client() as client:
        await load_catalog(client, tmp_path)

    responses[0] = httpx.Response(500)
    async with openrouter_client() as client:
        loaded = await load_catalog(client, tmp_path)

    assert loaded.origin == "cache"
    assert loaded.remote_count == 1
    assert loaded.catalog.get("acme/live-1").context_window == 64000


@respx.mock
async def test_cache_from_a_different_base_url_is_ignored(tmp_path):
    responses = [_payload([LIVE_MODEL])]
    respx.get(MODELS_URL).mock(side_effect=lambda request: responses[0])
    async with openrouter_client() as client:
        await load_catalog(client, tmp_path)

    responses[0] = httpx.Response(500)
    # Same cache dir, but the client now reports another endpoint.
    async with openrouter_client() as client:
        client.base_url = "https://other.example/v1"
        loaded = await load_catalog(client, tmp_path)

    assert loaded.origin == "bundled"


@respx.mock
async def test_corrupt_cache_falls_back_to_bundled(tmp_path):
    (tmp_path / CACHE_FILENAME).write_text("{not json", encoding="utf-8")
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500))
    async with openrouter_client() as client:
        loaded = await load_catalog(client, tmp_path)

    assert loaded.origin == "bundled"


@respx.mock
async def test_openai_shaped_models_endpoint_still_yields_entries(tmp_path):
    respx.get(MODELS_URL).mock(return_value=_payload([{"id": "local-model"}]))
    async with openrouter_client() as client:
        loaded = await load_catalog(client, tmp_path)

    assert loaded.origin == "live"
    entry = loaded.catalog.get("local-model")
    assert entry.context_window == 128_000
    assert entry.pricing.prompt == 0.0
