"""Tests for the SQLite fact store: facts, revisions, provenance."""

from __future__ import annotations

import threading

import pytest

from lecode.memory.facts import FactStore, RevisionConflict


@pytest.fixture
def store(tmp_path):
    store = FactStore(tmp_path / "facts.sqlite3")
    yield store
    store.close()


def test_add_is_idempotent_for_same_text_and_source(store):
    first = store.add("The sky is blue", source_id="session-1", source_seq=3)
    again = store.add("  The sky is blue  ", source_id="session-1", source_seq=3)
    assert again == first
    assert store.get(first.id) == first
    assert store.get(first.id).revision == 1
    assert len(store.provenance(first.id)) == 1


def test_automatic_remember_deduplicates_revision_text_and_checks_generation(store, tmp_path):
    from lecode.session.storage import SessionStore

    sessions = SessionStore(tmp_path / "sessions-config")
    session = sessions.create("evidence", tmp_path)
    for text in ("I prefer tabs", "I prefer spaces", "I prefer tabs"):
        sessions.append_message(session, {"role": "user", "content": text})
    refs = [
        sessions.source_snapshot(session.id, seq, seq, project_root=tmp_path).ref
        for seq in (1, 2, 3)
    ]
    first = store.remember("I prefer tabs", refs[0], sessions=sessions, project_root=tmp_path)
    revised = store.correct(
        first.id,
        "I prefer spaces",
        expected_revision=1,
        ref=refs[1],
        sessions=sessions,
        project_root=tmp_path,
    )
    assert (
        store.remember(
            "  I  prefer TABS ",
            refs[2],
            sessions=sessions,
            project_root=tmp_path,
            expected_generation=store.generation(),
            deduplicate=True,
        )
        == revised
    )
    assert len(store.search("prefer")) == 1
    generation = store.generation()
    store.forget(first.id)
    with pytest.raises(ValueError, match="exclusions changed"):
        store.remember(
            "new claim",
            refs[2],
            sessions=sessions,
            project_root=tmp_path,
            expected_generation=generation,
            deduplicate=True,
        )
    assert store.search("claim") == []


def test_provenance_round_trip(store):
    fact = store.add("Alpacas are camelids", source_id="s-1", source_seq=2)
    refs = store.provenance(fact.id)
    assert len(refs) == 1
    assert refs[0].fact_id == fact.id
    assert refs[0].source_id == "s-1"
    assert refs[0].source_seq == 2
    assert refs[0].created_at


def test_remember_rechecks_source_epoch_inside_transaction(store, tmp_path):
    from lecode.session.storage import SessionStore

    sessions = SessionStore(tmp_path / "config")
    session = sessions.create("source", tmp_path)
    sessions.append_message(session, {"role": "user", "content": "evidence"})
    sessions.bind_facts(tmp_path, store)
    version = sessions.source_version(session, sync=True)
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    sessions.append_event(session, "clear")
    with pytest.raises(ValueError, match="sources changed"):
        store.remember(
            "fact",
            ref,
            sessions=sessions,
            project_root=tmp_path,
            expected_version=version,
            expected_generation=store.generation(),
        )
    assert store.search("fact") == []


def test_revise_bumps_revision_and_text(store):
    fact = store.add("v1", source_id="s", source_seq=1)
    revised = store.revise(fact.id, "v2", expected_revision=1, source_id="s", source_seq=2)
    assert revised.id == fact.id
    assert revised.text == "v2"
    assert revised.revision == 2
    assert store.get(fact.id).text == "v2"
    assert len(store.provenance(fact.id)) == 2


def test_revise_with_stale_expected_revision_conflicts(store):
    fact = store.add("v1", source_id="s", source_seq=1)
    store.revise(fact.id, "v2", expected_revision=1, source_id="s", source_seq=2)
    with pytest.raises(RevisionConflict):
        store.revise(fact.id, "v3", expected_revision=1, source_id="s", source_seq=3)
    assert store.get(fact.id).text == "v2"
    assert store.get(fact.id).revision == 2
    assert [ref.source_seq for ref in store.provenance(fact.id)] == [1, 2]
    revised = store.revise(fact.id, "v3", expected_revision=2, source_id="s", source_seq=3)
    assert revised.revision == 3
    assert [ref.source_seq for ref in store.provenance(fact.id)] == [1, 2, 3]


def test_concurrent_connections_write_without_locking(tmp_path):
    path = tmp_path / "facts.sqlite3"
    errors: list[BaseException] = []

    def writer(tag: str) -> None:
        try:
            store = FactStore(path)
            for seq in range(25):
                store.add(f"{tag} fact {seq}", source_id=tag, source_seq=seq)
            store.close()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    store = FactStore(path)
    for tag in ("w0", "w1"):
        fact = store.get(store.add(f"{tag} fact 0", source_id=tag, source_seq=0).id)
        assert fact is not None
        assert fact.text == f"{tag} fact 0"
    store.close()


def test_facts_persist_across_instances(tmp_path):
    path = tmp_path / "facts.sqlite3"
    first = FactStore(path)
    fact = first.add("persisted", source_id="s", source_seq=1)
    first.close()
    second = FactStore(path)
    assert second.get(fact.id) == fact
    assert second.provenance(fact.id)[0].source_id == "s"
    second.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", " \n\t"),
        ("text", None),
        ("text", "bad\x00text"),
        ("text", "\ud800"),
        ("source_id", " "),
        ("source_id", 123),
        ("source_id", "s\x00"),
        ("source_seq", -1),
        ("source_seq", True),
        ("source_seq", 1.5),
        ("source_seq", "1"),
        ("source_seq", 2**63),
    ],
)
def test_invalid_fact_input_rejected_before_creating_database(tmp_path, field, value):
    path = tmp_path / "memory" / "facts.sqlite3"
    store = FactStore(path)
    args = {"text": "valid", "source_id": "s", "source_seq": 1, field: value}
    with pytest.raises(ValueError):
        store.add(**args)
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "overrides",
    [{"text": " "}, {"source_id": ""}, {"source_seq": True}, {"expected_revision": True}],
)
def test_invalid_revision_leaves_fact_and_provenance_unchanged(store, overrides):
    fact = store.add("original", source_id="s", source_seq=0)
    refs = store.provenance(fact.id)
    args = {"text": "new", "source_id": "s", "source_seq": 1, "expected_revision": 1}
    args.update(overrides)
    with pytest.raises(ValueError):
        store.revise(fact.id, **args)
    assert store.get(fact.id) == fact
    assert store.provenance(fact.id) == refs


def test_missing_reads_do_not_create_database(tmp_path):
    path = tmp_path / "memory" / "facts.sqlite3"
    store = FactStore(path)
    assert store.get("missing") is None
    assert store.provenance("missing") == []
    store.close()
    assert not path.parent.exists()


def test_two_independent_writers_cannot_commit_same_revision(tmp_path):
    path = tmp_path / "facts.sqlite3"
    store = FactStore(path)
    fact = store.add("original", source_id="s", source_seq=0)
    barrier = threading.Barrier(2)
    results = []

    def writer(seq):
        other = FactStore(path)
        try:
            assert other.get(fact.id) == fact
            barrier.wait(timeout=5)
            results.append(
                other.revise(
                    fact.id, f"writer {seq}", expected_revision=1, source_id="s", source_seq=seq
                )
            )
        except BaseException as e:
            results.append(e)
        finally:
            other.close()

    threads = [threading.Thread(target=writer, args=(seq,)) for seq in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sum(isinstance(result, RevisionConflict) for result in results) == 1
    winner = store.get(fact.id)
    assert winner is not None
    assert winner in results
    assert winner.revision == 2
    assert [ref.source_seq for ref in store.provenance(fact.id)] == [0, int(winner.text[-1])]
    assert store.add("original", source_id="s", source_seq=0) == winner
    assert store.add("original", source_id="other", source_seq=0).id != fact.id
    assert len(store.provenance(fact.id)) == 2
    store.close()


def test_forget_purges_all_revisions_and_blocks_relearning_after_reopen(tmp_path):
    from lecode.session.storage import SessionStore

    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    for text in ("original evidence", "support", "correction evidence", "unrelated"):
        sessions.append_message(session, {"role": "user", "content": text})
    ref = sessions.source_snapshot(session.id, 1, 2, project_root=tmp_path).ref
    correction = sessions.source_snapshot(session.id, 3, 3, project_root=tmp_path).ref
    facts = FactStore(tmp_path / "facts.sqlite3")
    fact = facts.remember("original", ref, sessions=sessions, project_root=tmp_path)
    facts.correct(
        fact.id,
        "corrected",
        expected_revision=1,
        ref=correction,
        sessions=sessions,
        project_root=tmp_path,
    )
    unrelated = facts.add("unrelated", source_id=session.id, source_seq=4)
    before = session.path.read_bytes()
    generation = facts.generation()
    assert facts.forget(fact.id) is True
    assert facts.generation() > generation
    generation = facts.generation()
    facts.close()
    facts = FactStore(tmp_path / "facts.sqlite3")
    assert facts.get(fact.id) is None
    assert facts.provenance(fact.id) == []
    assert facts.source(fact.id, 1) is None
    assert facts.source(fact.id, 2) is None
    assert facts.search("corrected") == []
    assert facts.excluded_seqs(session.id) == frozenset({1, 2, 3})
    assert facts.forget(fact.id) is False
    assert facts.generation() == generation
    assert facts.get(unrelated.id) == unrelated
    assert session.path.read_bytes() == before
    with pytest.raises(ValueError, match="excluded"):
        facts.add("relearned", source_id=session.id, source_seq=2)
    with pytest.raises(ValueError, match="excluded"):
        facts.revise(
            unrelated.id, "relearned", expected_revision=1, source_id=session.id, source_seq=3
        )
    with pytest.raises(ValueError, match=r"hidden|excluded"):
        facts.remember("relearned", ref, sessions=sessions, project_root=tmp_path)
    with pytest.raises(KeyError, match="unknown fact"):
        facts.forget("f" * 64)
    facts.close()


def test_competing_forget_connections_are_idempotent_and_exclusions_only_grow(tmp_path):
    path = tmp_path / "facts.sqlite3"
    facts = FactStore(path)
    first = facts.add("first", source_id="s", source_seq=1)
    second = facts.add("second", source_id="s", source_seq=2)
    barrier = threading.Barrier(2)
    results = []

    def forget():
        connection = FactStore(path)
        try:
            assert connection.get(first.id) == first
            barrier.wait(timeout=5)
            results.append(connection.forget(first.id))
        except BaseException as e:
            results.append(e)
        finally:
            connection.close()

    threads = [threading.Thread(target=forget) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sorted(results) == [False, True]
    epoch = facts.generation()
    assert facts.excluded_seqs("s") == frozenset({1})
    facts.forget(second.id)
    assert facts.excluded_seqs("s") == frozenset({1, 2})
    assert facts.generation() > epoch
    with pytest.raises(ValueError, match="excluded"):
        facts.add("new text", source_id="s", source_seq=1)
    facts.close()


def test_guarded_external_write_serializes_against_forget(tmp_path):
    facts = FactStore(tmp_path / "facts.sqlite3")
    fact = facts.add("claim", source_id="s", source_seq=1)
    attempted, finished = threading.Event(), threading.Event()
    errors = []

    def other_writer():
        other = FactStore(facts.path)
        try:
            assert other.get(fact.id) == fact
            attempted.set()
            other.forget(fact.id)
        except BaseException as exc:
            errors.append(exc)
        finally:
            other.close()
            finished.set()

    with facts.guard_generation(facts.generation()):
        writer = threading.Thread(target=other_writer)
        writer.start()
        assert attempted.wait(5)
        assert not finished.wait(0.05)
        (tmp_path / "note.md").write_text("independent note")
    writer.join(timeout=5)
    assert not writer.is_alive() and not errors
    assert facts.get(fact.id) is None
    assert (tmp_path / "note.md").read_text() == "independent note"
    with pytest.raises(ValueError, match="exclusions changed"), facts.guard_generation(0):
        pytest.fail("stale external mutation must never run")
    facts.close()
