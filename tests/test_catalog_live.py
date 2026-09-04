"""Tests for the live model catalog: fetch → empty fallback (no disk cache)."""

from __future__ import annotations

import httpx
import respx

from lecode.providers.live import load_catalog
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
async def test_live_fetch_builds_the_catalog():
    respx.get(MODELS_URL).mock(return_value=_payload([LIVE_MODEL]))
    async with openrouter_client() as client:
        loaded = await load_catalog(client)

    assert loaded.origin == "live"
    assert loaded.remote_count == 1
    entry = loaded.catalog.get("acme/live-1")
    assert entry.context_window == 64000
    assert entry.pricing.prompt == 1.0


@respx.mock
async def test_failed_fetch_yields_an_empty_catalog():
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    async with openrouter_client() as client:
        loaded = await load_catalog(client)

    assert loaded.origin == "empty"
    assert loaded.catalog.all() == []


@respx.mock
async def test_no_cache_file_is_ever_written(tmp_path, monkeypatch):
    """The config dir must not grow a models-cache.json."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path))
    respx.get(MODELS_URL).mock(return_value=_payload([LIVE_MODEL]))
    async with openrouter_client() as client:
        await load_catalog(client)
    assert not (tmp_path / "models-cache.json").exists()


@respx.mock
async def test_openai_shaped_models_endpoint_still_yields_entries():
    respx.get(MODELS_URL).mock(return_value=_payload([{"id": "local-model"}]))
    async with openrouter_client() as client:
        loaded = await load_catalog(client)

    assert loaded.origin == "live"
    entry = loaded.catalog.get("local-model")
    assert entry.context_window == 128_000
    assert entry.pricing.prompt == 0.0
