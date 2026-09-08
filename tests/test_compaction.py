"""Tests for the compaction helper shared by ``/compact`` and the runner."""

from __future__ import annotations

import pytest
from tests.fakes import FakeProvider

from lecode.providers.openai_compat import ProviderError
from lecode.session.compaction import compact_session
from lecode.session.model import EventRecord
from lecode.session.storage import SessionStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


def _fill(store, session, pairs: int = 5) -> None:
    for index in range(pairs):
        store.append_message(session, {"role": "user", "content": f"q{index}"})
        store.append_message(session, {"role": "assistant", "content": f"a{index}"})


def _compacts(store, session):
    return [
        r for r in store.read_records(session) if isinstance(r, EventRecord) and r.kind == "compact"
    ]


async def test_compact_session_records_summary(store, tmp_path):
    session = store.create("demo", tmp_path)
    _fill(store, session)
    provider = FakeProvider([{"text": "the summary"}])

    summary = await compact_session(provider, store, session, "test-model")

    assert summary == "the summary"
    compacts = _compacts(store, session)
    assert compacts and compacts[-1].data["summary"] == "the summary"
    loaded = store.load_for_model(session)
    assert loaded[0] == {"role": "system", "content": "the summary"}
    # the recent tail stays raw
    assert {"role": "assistant", "content": "a4"} in loaded
    assert {"role": "user", "content": "q0"} not in loaded
    # the summarizer saw the old transcript, with the model passed through
    request = provider.requests[-1]
    assert request["model"] == "test-model"
    assert "q0" in request["messages"][-1]["content"]


async def test_compact_session_too_little_history(store, tmp_path):
    session = store.create("demo", tmp_path)
    store.append_message(session, {"role": "user", "content": "only one"})
    provider = FakeProvider([{"text": "unused"}])

    assert await compact_session(provider, store, session, "test-model") is None
    assert not provider.requests
    assert not _compacts(store, session)


async def test_compact_session_provider_failure(store, tmp_path):
    session = store.create("demo", tmp_path)
    _fill(store, session)
    provider = FakeProvider([{"error": ProviderError("boom", retryable=False)}])

    assert await compact_session(provider, store, session, "test-model") is None
    assert not _compacts(store, session)


async def test_compact_session_empty_summary(store, tmp_path):
    session = store.create("demo", tmp_path)
    _fill(store, session)
    provider = FakeProvider([{"text": "   "}])

    assert await compact_session(provider, store, session, "test-model") is None
    assert not _compacts(store, session)
