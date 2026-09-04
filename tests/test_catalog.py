"""Tests for the model catalog (lookup, merge; there is no bundled snapshot)."""

from __future__ import annotations

import pytest
from tests.fakes import sample_catalog

from lecode.providers.catalog import (
    AmbiguousModelError,
    Catalog,
    ModelInfo,
    ModelNotFoundError,
)


def test_default_catalog_is_empty():
    """No bundled snapshot: the default catalog holds nothing, fail-open."""
    catalog = Catalog.default()
    assert catalog.all() == []
    with pytest.raises(ModelNotFoundError):
        catalog.get("openai/gpt-5")


def test_lookup_by_exact_id():
    info = sample_catalog().get("openai/gpt-5-mini")
    assert info.name == "GPT-5 Mini"


def test_lookup_by_unique_prefix():
    info = sample_catalog().get("anthropic/claude-sonnet")
    assert info.id == "anthropic/claude-sonnet-4"


def test_ambiguous_prefix_raises():
    with pytest.raises(AmbiguousModelError) as excinfo:
        sample_catalog().get("openai/gpt")
    assert "openai/gpt-5" in excinfo.value.matches
    assert "openai/gpt-4o" in excinfo.value.matches


def test_lookup_by_case_insensitive_name():
    info = sample_catalog().get("gpt-4o")
    assert info.id == "openai/gpt-4o"


def test_unknown_model_raises():
    with pytest.raises(ModelNotFoundError):
        sample_catalog().get("no/such-model")


def test_modalities_for():
    catalog = sample_catalog()
    assert "image" in catalog.modalities_for("openai/gpt-4o").input
    assert catalog.modalities_for("deepseek/deepseek-v4-flash").input == ["text"]


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
