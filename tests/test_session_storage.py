"""Tests for JSONL session storage: append, replay, undo/redo/rewind, compaction."""

from __future__ import annotations

import json
import os

import pytest

from lecode.session import (
    AmbiguousSessionError,
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
)
from lecode.session.model import EventRecord, MessageRecord, MetaRecord, TombstoneRecord


def test_forget_marker_retries_after_commit_and_reopen_without_duplicates(tmp_path):
    from lecode.memory.facts import FactStore
    from lecode.memory.store import memory_root

    sessions = SessionStore(tmp_path / "cfg")
    source = sessions.create("source", tmp_path)
    sessions.append_message(source, {"role": "user", "content": "evidence"})
    facts = FactStore(memory_root(tmp_path, sessions.config_dir) / "facts.sqlite3")
    ref = sessions.source_snapshot(source.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("claim", ref, sessions=sessions, project_root=tmp_path)
    correction = sessions.create("correction", tmp_path)
    sessions.append_message(correction, {"role": "user", "content": "corrected evidence"})
    corrected_ref = sessions.source_snapshot(correction.id, 1, 1, project_root=tmp_path).ref
    facts.correct(
        fact.id,
        "corrected claim",
        expected_revision=1,
        ref=corrected_ref,
        sessions=sessions,
        project_root=tmp_path,
    )
    before = source.path.read_bytes()
    facts.forget(fact.id)  # Process dies before it can append any session marker.
    assert facts.pending_forgets(source.id) == [fact.id]
    sessions.close()
    sessions = SessionStore(tmp_path / "cfg")
    for _ in range(2):
        reopened = sessions.open(source.id)
        lock = sessions.acquire_lock(reopened)
        assert lock is not None
        records = sessions.read_records(reopened)
        markers = [r for r in records if isinstance(r, EventRecord) and r.kind == "forget"]
        assert len(markers) == 1
        assert markers[0].data == {"fact_id": fact.id}
        assert reopened.next_seq == markers[0].seq + 1
        lock.release()
    assert facts.pending_forgets(source.id) == []
    assert facts.pending_forgets(correction.id) == [fact.id]
    sessions.flush_forgets(correction.id)
    assert facts.pending_forget_sessions(fact.id) == []
    assert facts.forget(fact.id) is False
    assert source.path.read_bytes().startswith(before)
    assert sessions.load_for_model(source) == []
    sessions.close()
    facts.close()


def test_forget_marker_fsync_failure_keeps_retry_and_attached_sequence(tmp_path, monkeypatch):
    from lecode.memory.facts import FactStore
    from lecode.memory.store import memory_root

    sessions = SessionStore(tmp_path / "cfg")
    source = sessions.create("source", tmp_path)
    lock = sessions.acquire_lock(source)
    sessions.append_message(source, {"role": "user", "content": "evidence"})
    facts = FactStore(memory_root(tmp_path, sessions.config_dir) / "facts.sqlite3")
    ref = sessions.source_snapshot(source.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("claim", ref, sessions=sessions, project_root=tmp_path)
    facts.forget(fact.id)

    def failed_fsync(fd):
        raise OSError("interrupted marker flush")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", failed_fsync)
        sessions.flush_forgets(source.id)
    assert facts.pending_forgets(source.id) == [fact.id]
    assert sessions.load_for_model(source) == []
    marker = sessions.read_records(source)[-1]
    assert marker.kind == "forget"
    assert source.next_seq == marker.seq + 1
    sessions.append_message(source, {"role": "user", "content": "independent note"})
    lock.release()
    sessions.close()
    sessions = SessionStore(tmp_path / "cfg")
    reopened = sessions.open(source.id)
    lock = sessions.acquire_lock(reopened)
    records = sessions.read_records(reopened)
    assert sum(isinstance(r, EventRecord) and r.kind == "forget" for r in records) == 1
    assert facts.pending_forgets(source.id) == []
    assert sessions.load_for_model(reopened) == [{"role": "user", "content": "independent note"}]
    lock.release()
    sessions.close()
    facts.close()


def test_forget_marker_defers_to_cross_process_attach(tmp_path):
    import subprocess
    import sys

    from lecode.memory.facts import FactStore
    from lecode.memory.store import memory_root

    sessions = SessionStore(tmp_path / "cfg")
    source = sessions.create("source", tmp_path)
    sessions.append_message(source, {"role": "user", "content": "evidence"})
    facts = FactStore(memory_root(tmp_path, sessions.config_dir) / "facts.sqlite3")
    ref = sessions.source_snapshot(source.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("claim", ref, sessions=sessions, project_root=tmp_path)
    script = (
        "import sys; from lecode.session.storage import SessionStore; "
        "s=SessionStore(sys.argv[1]); lock=s.acquire_lock(s.open(sys.argv[2])); "
        "print('attached', flush=True); sys.stdin.readline(); lock.release(); s.close()"
    )
    with subprocess.Popen(
        [sys.executable, "-c", script, str(sessions.config_dir), source.id],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "attached"
            inode = source.path.with_suffix(".lock").stat().st_ino
            facts.forget(fact.id)
            sessions.flush_forgets(source.id)
            assert facts.pending_forgets(source.id) == [fact.id]
            assert not any(
                isinstance(r, EventRecord) and r.kind == "forget"
                for r in sessions.read_records(source)
            )
            assert sessions.load_for_model(source) == []
            child.communicate("\n", timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
    sessions.flush_forgets(source.id)
    assert facts.pending_forgets(source.id) == []
    assert source.path.with_suffix(".lock").stat().st_ino == inode
    sessions.close()
    facts.close()


def test_partial_marker_append_does_not_swallow_retry_or_next_user_message(tmp_path):
    from lecode.memory.facts import FactStore

    sessions = SessionStore(tmp_path / "cfg")
    source = sessions.create("source", tmp_path)
    facts = FactStore(tmp_path / "facts.sqlite3")
    sessions.bind_facts(tmp_path, facts)
    fact = facts.add("claim", source_id=source.id, source_seq=1)
    facts.forget(fact.id)
    with source.path.open("ab") as stream:
        stream.write(b'{"type":"event","kind":"forget"')  # Interrupted JSONL write.
    sessions.flush_forgets(source)
    sessions.append_message(source, {"role": "user", "content": "fresh note"})
    records = sessions.read_records(source)
    assert sum(isinstance(r, EventRecord) and r.kind == "forget" for r in records) == 1
    assert records[-1].message["content"] == "fresh note"
    assert facts.pending_forgets(source.id) == []
    sessions.close()


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


def test_list_sessions_scoped_to_folder(store):
    store.create("here", cwd="/tmp/here")
    store.create("there", cwd="/tmp/there")
    assert [m.name for m in store.list_sessions(cwd="/tmp/here")] == ["here"]
    assert {m.name for m in store.list_sessions()} == {"here", "there"}  # unscoped


def test_resolve_scoped_to_folder(store):
    store.create("same-name", cwd="/tmp/a")
    other = store.create("same-name", cwd="/tmp/b")
    # name resolution only sees the folder's own session
    assert store.resolve("same-name", cwd="/tmp/b").id == other.id
    # latest / None are folder-relative too
    assert store.resolve(None, cwd="/tmp/b").id == other.id
    # a foreign session is invisible, even by exact id
    with pytest.raises(SessionNotFoundError):
        store.resolve(other.id, cwd="/tmp/a")
    with pytest.raises(SessionNotFoundError):
        store.resolve(None, cwd="/tmp/empty")


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


def test_unvalidated_legacy_summary_cannot_hide_raw_sources(store, session):
    store.append_event(
        session,
        "compact",
        {"summary": "unverified", "keep_from_seq": 3, "source_start_seq": 1, "source_end_seq": 2},
    )
    assert [m["content"] for m in store.load_for_model(session)] == [
        "first",
        "answer one",
        "second",
        "answer two",
    ]


def test_compact_then_tombstone_drops_undone_summary(store, session):
    store.compact(session, "S", keep_from_seq=3)
    store.undo(session)  # hides seq 3+ -> the compact event is undone too
    replayed = store.load_for_model(session)
    assert [m["content"] for m in replayed] == ["first", "answer one"]


def test_summary_intersecting_tombstone_drops_then_redo_restores(store, session):
    store.append_message(session, {"role": "user", "content": "third"})  # seq 5
    store.append_message(session, {"role": "assistant", "content": "answer three"})  # seq 6
    store.compact(session, "S", keep_from_seq=5, source_start_seq=1, source_end_seq=4)  # seq 7
    store.rewind_to(session, 2)  # hides seq 3+ including part of the covered range

    assert [m["content"] for m in store.load_for_model(session)] == ["first", "answer one"]
    assert store.redo(session) is True
    # restored by cancelling the tombstone, no second compaction event
    assert [m["content"] for m in store.load_for_model(session)] == [
        "S",
        "third",
        "answer three",
    ]
    compacts = [
        r for r in store.read_records(session) if isinstance(r, EventRecord) and r.kind == "compact"
    ]
    assert len(compacts) == 1


def test_summary_survives_tombstone_outside_covered_range(store, session):
    store.compact(session, "S", keep_from_seq=3, source_start_seq=1, source_end_seq=2)
    store.append_message(session, {"role": "user", "content": "third"})  # seq 6
    store.append_message(session, {"role": "assistant", "content": "answer three"})  # seq 7
    store.undo(session)  # hides the last turn (seqs 6, 7), not the covered range

    assert [m["content"] for m in store.load_for_model(session)] == [
        "S",
        "second",
        "answer two",
    ]


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


# -- agent-run activity ---------------------------------------------------------


def test_agent_run_round_trip(store, session):
    store.record_agent_run(
        session,
        {
            "run_id": "ab12cd34",
            "agent": "explore",
            "description": "Explore src",
            "prompt": "scan src",
            "status": "ok",
            "answer": "found 3 files",
            "turns": 2,
            "input_tokens": 10,
            "output_tokens": 4,
            "cost_usd": 0.001,
            "duration_s": 1.5,
            "tool_calls": [
                {"name": "read", "args": '{"path": "x"}', "result": "contents", "is_error": False}
            ],
        },
    )
    runs = store.load_agent_runs(session)
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == "ab12cd34"
    assert run["agent"] == "explore"
    assert run["description"] == "Explore src"
    assert run["prompt"] == "scan src"
    assert run["status"] == "ok"
    assert run["answer"] == "found 3 files"
    assert run["turns"] == 2
    assert run["tool_calls"] == [
        {"name": "read", "args": '{"path": "x"}', "result": "contents", "is_error": False}
    ]


def test_agent_runs_keep_append_order(store, session):
    store.record_agent_run(session, {"run_id": "one", "agent": "explore", "status": "ok"})
    store.record_agent_run(session, {"run_id": "two", "agent": "explore", "status": "error"})
    assert [run["run_id"] for run in store.load_agent_runs(session)] == ["one", "two"]


def test_agent_run_bounds_large_payloads(store, session):
    store.record_agent_run(
        session,
        {
            "run_id": "big",
            "agent": "explore",
            "status": "ok",
            "prompt": "p" * 5000,
            "answer": "a" * 40000,
            "tool_calls": [
                {"name": "read", "args": "g" * 5000, "result": "r" * 10000, "is_error": False}
            ],
        },
    )
    run = store.load_agent_runs(session)[0]
    assert run["truncated"] is True
    assert run["answer"].endswith("… (truncated)")
    assert len(run["answer"]) <= 32 * 1024 + len("\n… (truncated)")
    assert len(run["prompt"]) <= 2000 + len("\n… (truncated)")
    tool = run["tool_calls"][0]
    assert tool["result"].endswith("… (truncated)")
    assert len(tool["result"]) <= 2000 + len("\n… (truncated)")
    assert len(tool["args"]) <= 500 + len("\n… (truncated)")


def test_agent_runs_respect_tombstones(store, session):
    record = store.record_agent_run(session, {"run_id": "gone", "agent": "explore", "status": "ok"})
    store.append_tombstone(session, up_to_seq=record.seq - 1)
    assert store.load_agent_runs(session) == []


def test_load_agent_runs_matches_generic_events(store, session):
    store.record_agent_run(session, {"run_id": "one", "agent": "explore", "status": "ok"})
    store.append_event(session, "worker_usage", {"input_tokens": 3})
    assert store.load_agent_runs(session) == store.load_events(session, "agent_run")
    assert [run["run_id"] for run in store.load_agent_runs(session)] == ["one"]


# -- generic event loading ------------------------------------------------------


def test_load_events_in_append_order(store, session):
    store.append_event(session, "worker", {"id": "one"})
    store.append_event(session, "worker_state", {"id": "one", "state": "done"})
    store.append_event(session, "worker", {"id": "two"})
    assert store.load_events(session, "worker") == [{"id": "one"}, {"id": "two"}]
    assert store.load_events(session, "worker_state") == [{"id": "one", "state": "done"}]


def test_load_events_respects_tombstones(store, session):
    first = store.append_event(session, "worker_usage", {"dispatch": 1})
    hidden = store.append_event(session, "worker_usage", {"dispatch": 2})
    store.append_tombstone(session, up_to_seq=hidden.seq - 1)
    last = store.append_event(session, "worker_usage", {"dispatch": 3})
    assert store.load_events(session, "worker_usage") == [{"dispatch": 1}, {"dispatch": 3}]
    assert first.seq < hidden.seq < last.seq


def test_load_events_unknown_kind(store, session):
    store.record_agent_run(session, {"run_id": "r", "agent": "explore", "status": "ok"})
    assert store.load_events(session, "no_such_kind") == []


# -- attach locking -------------------------------------------------------------


def test_lock_blocks_second_attach(store):
    s = store.create("locked", cwd="/tmp/p")
    lock = store.acquire_lock(s)
    assert lock is not None
    with pytest.raises(SessionInUseError) as exc_info:
        store.acquire_lock(store.open(s.id))
    assert exc_info.value.holder_pid == os.getpid()
    assert "locked" in str(exc_info.value)


def test_lock_release_allows_reattach(store):
    s = store.create("locked", cwd="/tmp/p")
    lock = store.acquire_lock(s)
    assert lock is not None
    lock.release()
    again = store.acquire_lock(store.open(s.id))
    assert again is not None
    again.release()


def test_delete_preserves_lock_inode_and_refuses_active_writer(store):
    s = store.create("locked", cwd="/tmp/p")
    lock = store.acquire_lock(s)
    assert lock is not None
    before = s.path.read_bytes()
    inode = s.path.with_suffix(".lock").stat().st_ino
    with pytest.raises(SessionInUseError):
        store.delete(s.id)
    assert s.path.read_bytes() == before
    lock.release()
    assert s.path.with_suffix(".lock").is_file()
    store.delete(s.id)
    assert s.path.with_suffix(".lock").stat().st_ino == inode
    with pytest.raises(SessionNotFoundError):
        store.acquire_lock(s)


def test_forget_filters_shared_sessions_and_invalidates_summary_without_marker(tmp_path):
    from lecode.memory.facts import FactStore
    from lecode.memory.store import memory_root

    cfg = tmp_path / "cfg"
    sessions = SessionStore(cfg)
    project = tmp_path / "project"
    other = tmp_path / "other"
    source = sessions.create("source", project)
    independent = sessions.create("other", other)
    for session in (source, independent):
        sessions.append_message(session, {"role": "user", "content": "evidence"})
        sessions.append_message(session, {"role": "assistant", "content": "derived"})
        sessions.append_message(session, {"role": "user", "content": "independent note"})
    sessions.compact(source, "summary of evidence", keep_from_seq=3)
    facts = FactStore(memory_root(project, cfg) / "facts.sqlite3")
    ref = sessions.source_snapshot(source.id, 1, 1, project_root=project).ref
    fact = facts.remember("claim", ref, sessions=sessions, project_root=project)
    before = source.path.read_bytes()
    assert sessions.working_summary(source) is not None
    # Commit without a JSONL marker, as if interrupted immediately after the commit.
    facts.forget(fact.id)
    sessions = SessionStore(cfg)
    assert sessions.working_summary(source) is None
    assert [m["content"] for m in sessions.load_for_model(source)] == ["independent note"]
    assert [m.message["content"] for m in sessions.visible_messages(source)] == ["independent note"]
    assert sessions.validate_source(ref, project_root=project).status == "hidden"
    assert sessions.source_snapshot(source.id, 2, 2, project_root=project).status == "hidden"
    assert len(sessions.load_for_model(independent)) == 3
    assert source.path.read_bytes() == before
    assert len(sessions.load_messages(source)) == 3
    facts.close()


def test_handoff_seed_is_filtered_and_cannot_become_fresh_evidence_after_forget(tmp_path):
    from lecode.memory.facts import FactStore
    from lecode.session.handoff import handoff

    sessions = SessionStore(tmp_path / "cfg")
    source = sessions.create("source", tmp_path)
    sessions.append_message(source, {"role": "user", "content": "forget this evidence"})
    ref = sessions.source_snapshot(source.id, 1, 1, project_root=tmp_path).ref
    facts = FactStore(tmp_path / "facts.sqlite3")
    fact = facts.remember("claim", ref, sessions=sessions, project_root=tmp_path)
    seeded = handoff(source, sessions, "before")
    facts.forget(fact.id)
    assert sessions.load_for_model(seeded) == []
    assert sessions.source_snapshot(seeded.id, 1, 1, project_root=tmp_path).status == "hidden"
    after = handoff(source, sessions, "after")
    assert "forget this evidence" not in str(sessions.load_for_model(after))
    facts.close()


def test_lock_holder_reports_pid_while_held(store):
    s = store.create("locked", cwd="/tmp/p")
    assert store.lock_holder(s.id) is None  # no lock file yet
    lock = store.acquire_lock(s)
    assert lock is not None
    assert store.lock_holder(s.id) == os.getpid()
    lock.release()
    assert store.lock_holder(s.id) is None
