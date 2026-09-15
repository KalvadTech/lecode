"""Tests for the compaction helper shared by ``/compact`` and the runner."""

from __future__ import annotations

import asyncio

import pytest
from tests.fakes import FakeProvider

import lecode.session.compaction as compaction_module
from lecode.providers.openai_compat import ProviderError
from lecode.session.compaction import compact_session
from lecode.session.model import EventRecord
from lecode.session.storage import SessionStore


@pytest.mark.parametrize("unfinished", [False, True])
async def test_compaction_bridges_filtered_completed_turns_not_unresolved_tools(
    store, tmp_path, unfinished
):
    from lecode.memory.facts import FactStore

    session = store.create("filtered", tmp_path)
    for i in range(2):
        store.append_message(session, {"role": "user", "content": f"retained note {i}"})
        store.append_message(
            session,
            {
                "role": "assistant",
                "content": "old derived text",
                **(
                    {
                        "tool_calls": [
                            {"id": "pending", "function": {"name": "read", "arguments": "{}"}}
                        ]
                    }
                    if unfinished and i == 0
                    else {}
                ),
            },
        )
    facts = FactStore(tmp_path / "facts.sqlite3")
    store.bind_facts(tmp_path, facts)
    fact = facts.add("forgotten", source_id="elsewhere", source_seq=1)
    facts.forget(fact.id)
    for i in range(4):
        store.append_message(session, {"role": "user", "content": f"new request {i}"})
        store.append_message(
            session,
            {"role": "assistant", "content": f"new answer {i}"},
            memory_generation=facts.generation(),
        )
    before = session.path.read_bytes()
    provider = FakeProvider([{"text": "safe rebuilt summary"}])
    result = await compact_session(provider, store, session, "test-model")
    if unfinished:
        assert result is None and not provider.requests
    else:
        assert result == "safe rebuilt summary"
        transcript = provider.requests[0]["messages"][-1]["content"]
        assert "retained note 0" in transcript and "retained note 1" in transcript
        assert "old derived text" not in transcript
        assert store.working_summary(session) is not None
        assert "new request 3" in str(store.load_for_model(session))
    assert session.path.read_bytes().startswith(before)
    facts.close()


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


async def test_six_compactions_chain_only_previous_summary_and_new_sources(store, tmp_path):
    session = store.create("chain", tmp_path)
    _fill(store, session)
    provider = FakeProvider([{"text": f"summary-{i}"} for i in range(6)])
    for i in range(6):
        assert await compact_session(provider, store, session, "test-model") == f"summary-{i}"
        request = provider.requests[-1]["messages"]
        if i:
            assert f"summary-{i - 1}" in str(request)
            assert "q0" not in str(request)
            assert f"summary-{i - 2}" not in str(request)
        event = _compacts(store, session)[-1]
        covered = [seq for ref in event.data["source_refs"] for seq in ref["seqs"]]
        assert covered == [
            m.seq for m in store.visible_messages(session) if m.seq < event.data["keep_from_seq"]
        ]
        assert len([m for m in store.load_for_model(session) if m["role"] == "system"]) == 1
        store.append_message(session, {"role": "user", "content": f"next-{i}"})
        store.append_message(session, {"role": "assistant", "content": f"answer-{i}"})


async def test_compaction_preserves_tool_exchange_and_omits_media_payload(store, tmp_path):
    session = store.create("tools", tmp_path)
    store.append_message(
        session,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + "A" * 200000},
                },
            ],
        },
    )
    store.append_message(
        session,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"important.py"}'},
                }
            ],
        },
    )
    store.append_message(
        session, {"role": "tool", "tool_call_id": "read-1", "content": "exact result"}
    )
    store.append_message(session, {"role": "assistant", "content": "finished"})
    _fill(store, session, pairs=2)
    provider = FakeProvider([{"text": "summary"}])
    assert await compact_session(provider, store, session, "test-model") == "summary"
    request = provider.requests[0]
    text = str(request["messages"])
    assert all(value in text for value in ("read_file", "important.py", "read-1", "exact result"))
    assert "payload omitted" in text and "AAAA" not in text
    assert "inert" in text.lower()
    assert _compacts(store, session)[-1].data["keep_from_seq"] == 5


async def test_compaction_does_not_consume_abandoned_user_exchange(store, tmp_path):
    session = store.create("cancelled", tmp_path)
    store.append_message(session, {"role": "user", "content": "cancelled before response"})
    _fill(store, session)
    provider = FakeProvider([{"text": "unsafe"}])
    assert await compact_session(provider, store, session, "test-model") is None
    assert not provider.requests


@pytest.mark.parametrize(
    "change", ["clear", "undo", "tamper", "append", "delete", "exclude", "forget"]
)
async def test_source_change_during_summary_declines_stale_commit(store, tmp_path, change):
    session = store.create("race", tmp_path)
    _fill(store, session)
    if change == "forget":
        from lecode.memory.facts import FactStore

        facts = FactStore(tmp_path / "facts.sqlite3")
        ref = store.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
        fact = facts.remember("claim", ref, sessions=store, project_root=tmp_path)
    entered, resume = asyncio.Event(), asyncio.Event()

    class WaitingProvider(FakeProvider):
        async def complete(self, *args, **kwargs):
            entered.set()
            await resume.wait()
            return await super().complete(*args, **kwargs)

    provider = WaitingProvider([{"text": "stale summary"}])
    task = asyncio.create_task(compact_session(provider, store, session, "test-model"))
    await entered.wait()
    if change == "clear":
        store.append_event(session, "clear")
    elif change == "undo":
        store.undo(session)
    elif change == "append":
        store.append_message(session, {"role": "user", "content": "concurrent"})
    elif change == "delete":
        store.delete(session.id)
    elif change == "exclude":
        store.exclusion_reader = lambda session_id: frozenset({1, 2})
    elif change == "forget":
        facts.forget(fact.id)
        facts.close()
    else:
        session.path.write_text(session.path.read_text().replace("q1", "CHANGED"))
    resume.set()
    assert await task is None
    if change != "delete":
        assert not _compacts(store, session)


@pytest.mark.parametrize(
    "result",
    [
        {"text": "x" * 9000},
        {"text": " "},
        {"text": "cut off", "finish_reason": "length"},
        {"text": "tool instead", "tool_calls": [{"id": "x", "name": "bad"}]},
    ],
)
async def test_invalid_summary_preserves_previous_cutoff_and_caps_output(store, tmp_path, result):
    session = store.create("bounded", tmp_path)
    _fill(store, session)
    provider = FakeProvider([{"text": "valid"}, result])
    assert await compact_session(provider, store, session, "test-model") == "valid"
    _fill(store, session, pairs=2)
    before = store.load_for_model(session)
    assert await compact_session(provider, store, session, "test-model") is None
    assert store.load_for_model(session) == before
    assert len(_compacts(store, session)) == 1
    assert 0 < provider.requests[-1]["kwargs"]["max_tokens"] <= 2048


async def test_full_lineage_over_200_positions_revalidates_exclusions_and_rebuilds(store, tmp_path):
    session = store.create("long", tmp_path)
    _fill(store, session, pairs=110)
    provider = FakeProvider([{"text": "old-summary"}, {"text": "rebuilt"}])
    assert await compact_session(provider, store, session, "test-model") == "old-summary"
    event = _compacts(store, session)[-1]
    assert len(event.data["source_refs"]) == 2
    assert sum(len(ref["seqs"]) for ref in event.data["source_refs"]) == 216
    store.exclusion_reader = lambda session_id: frozenset({101, 102})
    replay = store.load_for_model(session)
    assert all(m["role"] != "system" for m in replay)
    assert all(m["content"] not in {"q50", "a50"} for m in replay)
    assert await compact_session(provider, store, session, "test-model") == "rebuilt"
    text = str(provider.requests[-1]["messages"])
    assert "old-summary" not in text and "q50" not in text and "q0" in text


async def test_memory_usage_is_durable_once_including_rejected_output(store, tmp_path):
    from lecode.session.stats import session_stats

    session = store.create("usage", tmp_path, model="test-model")
    _fill(store, session)
    provider = FakeProvider(
        [
            {
                "text": "summary",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.1},
            },
            {"text": " ", "usage": {"input_tokens": 50, "output_tokens": 2, "cost_usd": 0.02}},
            {"text": "no usage"},
        ]
    )
    assert await compact_session(provider, store, session, "test-model") == "summary"
    _fill(store, session, pairs=2)
    assert await compact_session(provider, store, session, "test-model") is None
    assert await compact_session(provider, store, session, "test-model") == "no usage"
    stats = session_stats(store, store.open(session.id))
    assert (stats.input_tokens, stats.output_tokens) == (150, 22)
    assert stats.cost_usd == pytest.approx(0.12)
    assert stats.unknown_usage_calls == 1
    assert _compacts(store, session)[0].data["usage"]["prompt_tokens"] == 100


async def test_summary_input_and_output_respect_current_model_catalog(store, tmp_path):
    from lecode.config.models import Config
    from lecode.providers.catalog import Catalog, ModelInfo

    session = store.create("small", tmp_path, model="old-large-model")
    _fill(store, session, pairs=100)
    config = Config()
    config.compaction.buffer_tokens = 100
    catalog = Catalog(
        [
            ModelInfo.model_validate(
                dict(
                    id="current-small",
                    name="Small",
                    context_window=600,
                    max_output=64,
                    pricing={"prompt": 0, "completion": 0},
                    modalities={"input": ["text"], "output": ["text"]},
                )
            )
        ]
    )
    provider = FakeProvider([{"text": "bounded"}, {"text": "next"}])
    assert (
        await compact_session(
            provider, store, session, "current-small", config=config, catalog=catalog
        )
        == "bounded"
    )
    assert (
        await compact_session(
            provider, store, session, "current-small", config=config, catalog=catalog
        )
        == "next"
    )
    for request in provider.requests:
        assert request["model"] == "current-small"
        assert request["kwargs"]["max_tokens"] == 64
        assert len(str(request["messages"]).encode()) < 1500
    assert "bounded" in str(provider.requests[1]["messages"])
    assert _compacts(store, session)[-1].data["keep_from_seq"] < 200


@pytest.mark.parametrize("change", ["middle", "parent", "undo", "clear"])
async def test_chained_lineage_invalidates_on_raw_or_prior_revision_change(store, tmp_path, change):
    session = store.create("lineage", tmp_path)
    _fill(store, session)
    provider = FakeProvider([{"text": "first-summary"}, {"text": "second-summary"}])
    await compact_session(provider, store, session, "test-model")
    _fill(store, session, pairs=3)
    await compact_session(provider, store, session, "test-model")
    if change == "middle":
        session.path.write_text(
            session.path.read_text().replace('"content":"q1"', '"content":"changed"', 1)
        )
    elif change == "parent":
        session.path.write_text(
            session.path.read_text().replace("first-summary", "altered-summary", 1)
        )
    elif change == "undo":
        store.rewind_to(session, 2)
    else:
        store.append_event(session, "clear")
    assert not any(m["role"] == "system" for m in store.load_for_model(session))


async def test_source_is_fsynced_before_provider_and_compact_append_is_fsynced(
    store, tmp_path, monkeypatch
):
    import os

    session = store.create("durable", tmp_path)
    _fill(store, session)
    synced = []
    original_fsync = os.fsync

    def fsync(fd):
        original_fsync(fd)
        synced.append(os.fstat(fd).st_size)

    monkeypatch.setattr(os, "fsync", fsync)

    class CheckedProvider(FakeProvider):
        async def complete(self, *args, **kwargs):
            assert synced == [session.path.stat().st_size]
            return await super().complete(*args, **kwargs)

    assert (
        await compact_session(
            CheckedProvider([{"text": "durable summary"}]), store, session, "test-model"
        )
        == "durable summary"
    )
    assert len(synced) == 2 and synced[1] > synced[0]


async def test_malformed_provider_completion_leaves_prior_summary_intact(store, tmp_path):
    session = store.create("malformed", tmp_path)
    _fill(store, session)
    await compact_session(FakeProvider([{"text": "valid"}]), store, session, "test-model")
    _fill(store, session, pairs=2)
    before = store.load_for_model(session)

    class MalformedProvider(FakeProvider):
        async def complete(self, *args, **kwargs):
            await super().complete(*args, **kwargs)
            return {"content": "wrong protocol"}

    assert await compact_session(MalformedProvider([]), store, session, "test-model") is None
    assert store.load_for_model(session) == before


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
    monkeypatch.setattr(compaction_module, "COMPACT_TRANSCRIPT_CAP", 120)
    provider = FakeProvider([{"text": "the summary"}])

    summary = await compact_session(provider, store, session, "test-model")

    assert summary == "the summary"
    transcript = provider.requests[-1]["messages"][-1]["content"]
    assert '"content": "q0"' in transcript and '"content": "a0"' in transcript
    assert '"q1"' not in transcript  # never consume half an exchange
    compact = _compacts(store, session)[-1]
    assert compact.data["keep_from_seq"] == 3
    assert compact.data["source_start_seq"] == 1
    assert compact.data["source_end_seq"] == 2
    loaded = store.load_for_model(session)
    assert loaded[0] == {"role": "system", "content": "the summary"}
    assert [m["content"] for m in loaded[1:]] == [
        "q1",
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

    assert summary is None  # the whole exchange must remain raw
    assert not provider.requests
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
    assert compact.data["keep_from_seq"] == 5
