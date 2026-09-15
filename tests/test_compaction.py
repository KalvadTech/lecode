"""Tests for the compaction helper shared by ``/compact`` and the runner."""

from __future__ import annotations

import pytest
from tests.fakes import FakeProvider

import lecode.session.compaction as compaction_module
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


# -- prefix coverage -----------------------------------------------------------------


async def test_compact_transcript_covers_exactly_the_removed_range(store, tmp_path, monkeypatch):
    """The summarizer transcript is a prefix of visible messages; the recorded
    range is the complement of the replayed tail, so nothing is omitted."""
    session = store.create("demo", tmp_path)
    _fill(store, session)
    monkeypatch.setattr(
        compaction_module, "COMPACT_TRANSCRIPT_CAP", len("user: q0\nassistant: a0\nuser: q1")
    )
    provider = FakeProvider([{"text": "the summary"}])

    summary = await compact_session(provider, store, session, "test-model")

    assert summary == "the summary"
    transcript = provider.requests[-1]["messages"][-1]["content"]
    assert transcript == "user: q0\nassistant: a0\nuser: q1"
    compact = _compacts(store, session)[-1]
    assert compact.data["keep_from_seq"] == 4  # a1 is the first kept message
    assert compact.data["source_start_seq"] == 1
    assert compact.data["source_end_seq"] == 3
    loaded = store.load_for_model(session)
    assert loaded[0] == {"role": "system", "content": "the summary"}
    assert [m["content"] for m in loaded[1:]] == [
        "a1",
        "q2",
        "a2",
        "q3",
        "a3",
        "q4",
        "a4",
    ]


async def test_compact_never_starts_the_tail_on_tool_results(store, tmp_path):
    """A kept tail that would begin with tool output walks back until the
    assistant message carrying the matching tool_calls is kept too."""
    session = store.create("demo", tmp_path)
    store.append_message(session, {"role": "user", "content": "q0"})
    store.append_message(
        session,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "echo", "arguments": "{}"}},
            ],
        },
    )
    store.append_message(session, {"role": "tool", "content": "r1", "tool_call_id": "c1"})
    store.append_message(session, {"role": "tool", "content": "r2", "tool_call_id": "c2"})
    store.append_message(session, {"role": "assistant", "content": "a1"})
    store.append_message(session, {"role": "user", "content": "q1"})
    store.append_message(session, {"role": "assistant", "content": "a2"})
    provider = FakeProvider([{"text": "the summary"}])

    summary = await compact_session(provider, store, session, "test-model")

    assert summary == "the summary"
    compact = _compacts(store, session)[-1]
    assert compact.data["keep_from_seq"] == 2  # the assistant that issued the calls
    transcript = provider.requests[-1]["messages"][-1]["content"]
    assert transcript == "user: q0"
    loaded = store.load_for_model(session)
    assert loaded[1]["tool_calls"][0]["id"] == "c1"
    assert [m.get("content") for m in loaded[2:4]] == ["r1", "r2"]


async def test_compact_returns_none_when_first_message_exceeds_cap(store, tmp_path, monkeypatch):
    session = store.create("demo", tmp_path)
    store.append_message(session, {"role": "user", "content": "x" * 200})
    _fill(store, session, pairs=2)
    monkeypatch.setattr(compaction_module, "COMPACT_TRANSCRIPT_CAP", 32)
    provider = FakeProvider([{"text": "unused"}])

    assert await compact_session(provider, store, session, "test-model") is None
    assert not provider.requests
    assert not _compacts(store, session)


async def test_compact_returns_none_when_tail_cannot_avoid_tool(store, tmp_path):
    session = store.create("demo", tmp_path)
    for index in range(5):
        store.append_message(
            session, {"role": "tool", "content": f"r{index}", "tool_call_id": f"c{index}"}
        )
    provider = FakeProvider([{"text": "unused"}])

    assert await compact_session(provider, store, session, "test-model") is None
    assert not provider.requests
    assert not _compacts(store, session)


async def test_compact_covers_only_messages_visible_after_clear(store, tmp_path):
    session = store.create("demo", tmp_path)
    for index in range(5):
        store.append_message(session, {"role": "user", "content": f"old{index}"})
        store.append_message(session, {"role": "assistant", "content": f"old-a{index}"})
    store.append_event(session, "clear")
    for index in range(3):
        store.append_message(session, {"role": "user", "content": f"new{index}"})
        store.append_message(session, {"role": "assistant", "content": f"new-a{index}"})
    provider = FakeProvider([{"text": "the summary"}])

    await compact_session(provider, store, session, "test-model")

    transcript = provider.requests[-1]["messages"][-1]["content"]
    assert "old" not in transcript
    compact = _compacts(store, session)[-1]
    assert compact.data["source_start_seq"] == 12
    assert compact.data["keep_from_seq"] == 14


async def test_compact_skips_tombstoned_messages(store, tmp_path):
    session = store.create("demo", tmp_path)
    _fill(store, session)  # seqs 1-10
    store.undo(session)  # hides the last user turn (seqs 9, 10)
    store.append_message(session, {"role": "user", "content": "fresh"})  # seq 12
    provider = FakeProvider([{"text": "the summary"}])

    await compact_session(provider, store, session, "test-model")

    transcript = provider.requests[-1]["messages"][-1]["content"]
    assert "q4" not in transcript
    assert "fresh" not in transcript
    compact = _compacts(store, session)[-1]
    assert compact.data["keep_from_seq"] == 6
