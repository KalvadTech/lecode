"""Tests for session stats: counts, tokens, cost reporting."""

from __future__ import annotations

import pytest
from tests.fakes import sample_catalog

from lecode.session import SessionStore
from lecode.session.stats import session_stats

GPT5_MINI = "openai/gpt-5-mini"  # catalog pricing: 0.25 prompt / 2.0 completion per 1M


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


@pytest.fixture
def session(store):
    s = store.create("stats", cwd="/tmp", model=GPT5_MINI)
    store.append_message(s, {"role": "user", "content": "hi"})
    store.append_message(
        s,
        {"role": "assistant", "content": "hello"},
        usage={"input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.01},
    )
    store.append_message(
        s,
        {"role": "assistant", "content": "more"},
        usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},  # no recorded cost
    )
    return s


def test_counts_and_roles(store, session):
    stats = session_stats(store, session, catalog=sample_catalog())
    assert stats.message_count == 3
    assert stats.role_counts == {"user": 1, "assistant": 2}
    assert stats.input_tokens == 1_001_000
    assert stats.output_tokens == 1_000_500


def test_cost_combines_recorded_and_catalog_pricing(store, session):
    stats = session_stats(store, session, catalog=sample_catalog())
    # recorded 0.01 + catalog: (1M * 0.25 + 1M * 2.0) / 1M = 2.25
    assert stats.cost_usd == pytest.approx(2.26)


def test_cost_unknown_model_skipped(store):
    s = store.create("mystery", cwd="/tmp", model="no/such-model")
    store.append_message(
        s, {"role": "assistant", "content": "x"}, usage={"input_tokens": 5, "output_tokens": 5}
    )
    stats = session_stats(store, s, catalog=sample_catalog())
    assert stats.cost_usd == 0.0
    assert stats.input_tokens == 5


def test_openai_style_usage_keys(store):
    s = store.create("oa", cwd="/tmp", model=GPT5_MINI)
    store.append_message(
        s,
        {"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 0},
    )
    stats = session_stats(store, s, catalog=sample_catalog())
    assert stats.cost_usd == pytest.approx(0.25)


def test_timestamps_and_tombstones(store, session):
    assert session_stats(store, session).tombstone_count == 0
    store.undo(session)
    stats = session_stats(store, session)
    assert stats.tombstone_count == 1
    assert stats.created_at == session.meta.created_at
    assert stats.last_active is not None
    assert stats.last_active >= stats.created_at


def test_empty_session(store):
    s = store.create("empty", cwd="/tmp")
    stats = session_stats(store, s)
    assert stats.message_count == 0
    assert stats.last_active is None
    assert stats.cost_usd == 0.0


def test_context_tokens_from_last_assistant_usage(store, session):
    stats = session_stats(store, session)
    assert stats.context_tokens == 1_000_000  # last assistant turn's input


def test_context_tokens_fall_back_after_undo(store, session):
    store.append_message(session, {"role": "user", "content": "again"})
    store.append_message(
        session,
        {"role": "assistant", "content": "big"},
        usage={"input_tokens": 50_000, "output_tokens": 10},
    )
    assert session_stats(store, session).context_tokens == 50_000
    store.undo(session)  # hides the last user turn and its assistant reply
    assert session_stats(store, session).context_tokens == 1_000_000


def test_context_tokens_zero_without_usage(store):
    s = store.create("plain", cwd="/tmp")
    store.append_message(s, {"role": "assistant", "content": "no usage"})
    assert session_stats(store, s).context_tokens == 0
