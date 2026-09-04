"""Tests for the static model catalog."""

from __future__ import annotations

import pytest

from lecode.providers.catalog import (
    AmbiguousModelError,
    Catalog,
    ModelInfo,
    ModelNotFoundError,
)


def test_bundled_catalog_loads_and_validates():
    catalog = Catalog.default()
    entries = catalog.all()
    assert len(entries) >= 25
    for entry in entries:
        assert entry.context_window > 0
        assert entry.pricing.prompt >= 0
        assert "text" in entry.modalities.input
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))
    for expected in (
        "openai/gpt-5",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-r1",
        "qwen/qwen3-coder",
        "meta-llama/llama-4-maverick",
        "mistralai/codestral-2501",
        "x-ai/grok-4",
    ):
        assert expected in ids


def test_lookup_by_exact_id():
    info = Catalog.default().get("openai/gpt-5-mini")
    assert info.name == "GPT-5 Mini"


def test_lookup_by_unique_prefix():
    info = Catalog.default().get("anthropic/claude-sonnet")
    assert info.id == "anthropic/claude-sonnet-4"


def test_ambiguous_prefix_raises():
    with pytest.raises(AmbiguousModelError) as excinfo:
        Catalog.default().get("openai/gpt")
    assert "openai/gpt-5" in excinfo.value.matches
    assert "openai/gpt-4o" in excinfo.value.matches


def test_lookup_by_case_insensitive_name():
    info = Catalog.default().get("gpt-4o")
    assert info.id == "openai/gpt-4o"


def test_unknown_model_raises():
    with pytest.raises(ModelNotFoundError):
        Catalog.default().get("no/such-model")


def test_modalities_for():
    catalog = Catalog.default()
    assert "image" in catalog.modalities_for("openai/gpt-4o").input
    assert catalog.modalities_for("deepseek/deepseek-r1").input == ["text"]


def _entry(model_id: str, **overrides) -> ModelInfo:
    payload = {
        "id": model_id,
        "name": model_id,
        "context_window": 1000,
        "pricing": {"prompt": 1.0, "completion": 2.0},
        "modalities": {"input": ["text"], "output": ["text"]},
    }
    payload.update(overrides)
    return ModelInfo.model_validate(payload)


def test_merge_overrides_and_adds():
    base = Catalog([_entry("a/1", context_window=1000), _entry("b/2")])
    merged = base.merge([_entry("a/1", context_window=9999), _entry("c/3")])
    assert merged.get("a/1").context_window == 9999
    assert merged.get("c/3").id == "c/3"
    assert merged.get("b/2").id == "b/2"
    # Original catalog is untouched.
    assert base.get("a/1").context_window == 1000
    with pytest.raises(ModelNotFoundError):
        base.get("c/3")
