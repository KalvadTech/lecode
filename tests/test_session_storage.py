"""Tests for JSONL session storage: append, replay, undo/redo/rewind, compaction."""

from __future__ import annotations

import json

import pytest

from lecode.session import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SessionStore,
)
from lecode.session.model import EventRecord, MessageRecord, MetaRecord, TombstoneRecord


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


@pytest.fixture
def session(store):
    s = store.create("demo", cwd="/tmp/project", model="openai/gpt-5-mini")
    store.append_message(s, {"role": "user", "content": "first"})  # seq 1
    store.append_message(s, {"role": "assistant", "content": "answer one"})  # seq 2
    store.append_message(s, {"role": "user", "content": "second"})  # seq 3
    store.append_message(s, {"role": "assistant", "content": "answer two"})  # seq 4
    return s


def _texts(messages) -> list[str]:
    return [m.message["content"] for m in messages]


def test_create_writes_meta_line(store):
    s = store.create("demo", cwd="/tmp/p", model="m")
    line = s.path.read_text().splitlines()[0]
    meta = MetaRecord.model_validate(json.loads(line))
    assert meta.name == "demo"
    assert meta.agent == "build"
    assert meta.model == "m"
    assert meta.schema_version == 1
    assert s.path.parent == store.sessions_dir


def test_open_round_trip(store, session):
    reopened = store.open(session.id)
    assert reopened.meta.name == "demo"
    assert reopened.next_seq == session.next_seq == 5


def test_open_missing_raises(store):
    with pytest.raises(SessionNotFoundError):
        store.open("nope")


def test_append_assigns_increasing_seq(store, session):
    record = store.append_message(session, {"role": "user", "content": "third"})
    assert record.seq == 5
    assert _texts(store.load_messages(session)) == [
        "first",
        "answer one",
        "second",
        "answer two",
        "third",
    ]


def test_usage_stored(store, session):
    usage = {"input_tokens": 10, "output_tokens": 4, "cost_usd": 0.001}
    record = store.append_message(session, {"role": "assistant", "content": "x"}, usage=usage)
    assert record.usage == usage


def test_corrupt_lines_skipped_and_counted(store, session):
    with session.path.open("a") as f:
        f.write("{not json}\n")
        f.write('{"type": "message", "seq": "bad"}\n')
    assert len(store.load_messages(session)) == 4
    assert store.corrupt_lines == 2


def test_list_sessions_parses_meta_only(store):
    store.create("one", cwd="/tmp")
    store.create("two", cwd="/tmp")
    metas = store.list_sessions()
    assert {m.name for m in metas} == {"one", "two"}
    # most recent first
    assert metas[0].created_at >= metas[1].created_at


def test_resolve_by_id_prefix_name_latest(store):
    a = store.create("alpha", cwd="/tmp")
    b = store.create("beta", cwd="/tmp")
    assert store.resolve(a.id).name == "alpha"
    assert store.resolve(a.id[:-1]).name == "alpha"  # unique prefix (hex suffix differs)
    assert store.resolve("beta").id == b.id
    assert store.resolve("latest").id == b.id
    assert store.resolve(None).id == b.id


def test_resolve_ambiguous_prefix_raises(store):
    a = store.create("alpha", cwd="/tmp")
    b = store.create("beta", cwd="/tmp")
    common = a.id[:8]  # timestamp portion is shared
    with pytest.raises(AmbiguousSessionError):
        store.resolve(common)
    with pytest.raises(SessionNotFoundError):
        store.resolve("zz-no-such-ref")
    assert b.id != a.id


def test_delete(store, session):
    store.delete(session.id)
    assert store.list_sessions() == []
    with pytest.raises(SessionNotFoundError):
        store.delete(session.id)


def test_undo_hides_last_user_turn(store, session):
    tombstone = store.undo(session)
    assert isinstance(tombstone, TombstoneRecord)
    assert tombstone.up_to_seq == 2  # last user message is seq 3
    assert _texts(store.load_messages(session)) == ["first", "answer one"]


def test_undo_twice_then_nothing_left(store, session):
    store.undo(session)
    store.undo(session)
    assert store.load_messages(session) == []
    assert store.undo(session) is None


def test_redo_restores_tombstoned_turn(store, session):
    store.undo(session)
    assert store.redo(session) is True
    assert _texts(store.load_messages(session)) == [
        "first",
        "answer one",
        "second",
        "answer two",
    ]


def test_redo_only_while_tombstone_is_last_record(store, session):
    store.undo(session)
    store.append_message(session, {"role": "user", "content": "new turn"})
    assert store.redo(session) is False
    # the old turn stays hidden
    assert _texts(store.load_messages(session)) == ["first", "answer one", "new turn"]


def test_double_redo_not_possible(store, session):
    store.undo(session)
    assert store.redo(session) is True
    assert store.redo(session) is False


def test_rewind_to_writes_restore_point_then_tombstone(store, session):
    store.rewind_to(session, 2)
    records = store.read_records(session)
    kinds = [r.kind for r in records if isinstance(r, EventRecord)]
    assert kinds == ["restore_point"]
    assert isinstance(records[-1], TombstoneRecord)
    assert _texts(store.load_messages(session)) == ["first", "answer one"]


def test_history_never_rewritten(store, session):
    before = session.path.read_text()
    store.undo(session)
    store.redo(session)
    store.compact(session, "summary", keep_from_seq=3)
    after = session.path.read_text()
    assert after.startswith(before)  # only appends happened


def test_compact_replay_for_model(store, session):
    store.compact(session, "SUMMARY: discussed first topic", keep_from_seq=3)
    replayed = store.load_for_model(session)
    assert replayed[0] == {"role": "system", "content": "SUMMARY: discussed first topic"}
    assert [m["content"] for m in replayed[1:]] == ["second", "answer two"]
    # full history still on disk and logically loadable
    assert len(store.load_messages(session)) == 4


def test_load_for_model_without_compaction(store, session):
    replayed = store.load_for_model(session)
    assert [m["content"] for m in replayed] == ["first", "answer one", "second", "answer two"]


def test_compact_then_tombstone_both_apply(store, session):
    store.compact(session, "S", keep_from_seq=3)
    store.undo(session)  # hides seq 3+ -> nothing visible from the kept tail
    replayed = store.load_for_model(session)
    assert [m["content"] for m in replayed] == ["S"]


def test_permission_grant_round_trip(store, session):
    store.grant_permission(session, "bash", "git *")
    store.grant_permission(session, "write", "src/**")
    reopened = store.open(session.id)
    assert store.load_grants(reopened) == [("bash", "git *"), ("write", "src/**")]


def test_import_valid_session(store, tmp_path):
    source = tmp_path / "foreign.jsonl"
    records = [
        MetaRecord(id="imported-id", name="imported", cwd="/x", created_at="2026-01-01"),
        MessageRecord(
            seq=1, ts="2026-01-01", role="user", message={"role": "user", "content": "hi"}
        ),
    ]
    source.write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    session = store.import_session(source)
    assert session.name == "imported"
    assert store.load_messages(session)[0].message["content"] == "hi"


def test_import_id_and_name_collisions(store, tmp_path):
    existing = store.create("imported", cwd="/tmp")
    source = tmp_path / "dup.jsonl"
    meta = MetaRecord(id=existing.id, name="imported", cwd="/x", created_at="2026-01-01")
    source.write_text(meta.model_dump_json() + "\n")
    session = store.import_session(source)
    assert session.id != existing.id  # fresh id on collision
    assert session.name == "imported-2"  # suffixed on collision


def test_import_rejects_non_session_file(store, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type": "message", "seq": 1, "ts": "t", "role": "user", "message": {}}\n')
    with pytest.raises(ValueError, match="meta"):
        store.import_session(bad)
