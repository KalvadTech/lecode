"""Source-linked recall through public stores and tool dispatch."""

import json

import pytest

from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.config.models import Config
from lecode.memory.facts import FactStore
from lecode.memory.tools import memory_tools
from lecode.permission import PermissionChecker
from lecode.session.storage import SessionStore


@pytest.mark.parametrize("change", ["undo", "delete"])
def test_recall_checks_superseded_revision_sources(tmp_path, change):
    from lecode.memory.recall import RecallContext

    sessions = SessionStore(tmp_path / "cfg")
    a, b = (sessions.create(name, tmp_path) for name in ("a", "b"))
    for session in (a, b):
        sessions.append_message(session, {"role": "user", "content": "evidence"})
    facts = FactStore(tmp_path / "facts.sqlite3")
    first = sessions.source_snapshot(a.id, 1, 1, project_root=tmp_path).ref
    second = sessions.source_snapshot(b.id, 1, 1, project_root=tmp_path).ref
    fact = facts.remember("original", first, sessions=sessions, project_root=tmp_path)
    facts.correct(
        fact.id,
        "corrected",
        expected_revision=1,
        ref=second,
        sessions=sessions,
        project_root=tmp_path,
    )
    recall = RecallContext(sessions, facts, tmp_path)
    if change == "undo":
        sessions.undo(a)
    else:
        sessions.delete(a.id)
    result = recall.recall({"fact_id": fact.id})
    assert json.loads(result)["status"] == ("hidden" if change == "undo" else "missing")
    assert "corrected" not in result and "evidence" not in result
    if change == "undo":
        assert sessions.redo(a)
        assert json.loads(recall.recall({"fact_id": fact.id}))["fact_text"] == "corrected"
    facts.close()


def test_source_snapshot_covers_messages_across_event_gaps(tmp_path):
    store = SessionStore(tmp_path / "cfg")
    session = store.create("source", tmp_path)
    store.append_message(session, {"role": "user", "content": "run it"})
    store.append_event(session, "checkpoint")
    store.append_message(session, {"role": "tool", "name": "bash", "content": "exact output"})
    snapshot = store.source_snapshot(session.id, 1, 3, project_root=tmp_path)
    assert snapshot.status == "valid"
    assert snapshot.ref.seqs == (1, 3)
    assert (
        store.validate_source(snapshot.ref, project_root=tmp_path).messages[1]["content"]
        == "exact output"
    )


def test_fact_source_attachment_persists_and_search_is_bounded(tmp_path):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    sessions.append_message(session, {"role": "user", "content": "evidence"})
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    path = tmp_path / "facts.sqlite3"
    facts = FactStore(path)
    fact = facts.add("claim", source_id=session.id, source_seq=1)
    assert facts.source(fact.id) is None
    facts.attach_source(fact.id, 1, ref, sessions=sessions, project_root=tmp_path)
    facts.close()
    facts = FactStore(path)
    assert facts.source(fact.id) == ref
    assert facts.search("claim", limit=1) == [fact]
    assert facts.excluded_seqs(session.id) == frozenset()
    facts.close()


async def test_recall_pages_with_hard_byte_cap_and_omits_binary_payloads(tmp_path):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    sessions.append_message(
        session,
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + "AAAA" * 10000},
                },
                {"type": "text", "text": '漢字"\\' * 10000},
            ],
        },
    )
    config = Config()
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=PermissionChecker(config, mode="readonly", cwd=tmp_path),
        session_store=sessions,
    )
    registry = ToolRegistry(memory_tools())
    offset = 0
    pages = []
    while True:
        _, result = await registry.dispatch_result(
            "c",
            "memory_recall",
            json.dumps(
                {
                    "session_id": session.id,
                    "start_seq": 1,
                    "end_seq": 1,
                    "offset": offset,
                    "limit": 999999,
                }
            ),
            ctx,
        )
        assert not result.is_error
        assert len(result.content.encode()) <= 16384
        assert "AAAA" not in result.content
        data = json.loads(result.content)
        pages.append(data["source_text"])
        if data["next_offset"] is None:
            break
        assert data["next_offset"] > offset
        offset = data["next_offset"]
    source = json.loads("".join(pages))
    assert source[0]["message"]["content"][1]["text"] == '漢字"\\' * 10000
    assert "omitted" in source[0]["message"]["content"][0]["image_url"]["url"]


async def test_dispatch_recovers_exact_original_tool_call_and_output(tmp_path):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    call = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }
        ],
    }
    output = {
        "role": "tool",
        "name": "bash",
        "tool_call_id": "call-1",
        "content": "/exact/project\n",
        "data": {"exit_code": 0},
    }
    sessions.append_message(session, call)
    sessions.append_message(session, output)
    config = Config()
    ctx = ToolContext(
        cwd=tmp_path,
        project_root=tmp_path,
        config=config,
        permission_checker=PermissionChecker(config, mode="readonly", cwd=tmp_path),
        session_store=sessions,
    )
    registry = ToolRegistry(memory_tools())
    _, result = await registry.dispatch_result(
        "c",
        "memory_recall",
        json.dumps({"session_id": session.id, "start_seq": 1, "end_seq": 2}),
        ctx,
    )
    assert not result.is_error
    data = json.loads(result.content)
    assert data["status"] == "valid"
    assert json.loads(data["source_text"]) == [
        {"seq": 1, "message": call},
        {"seq": 2, "message": output},
    ]
    assert data["next_offset"] is None


@pytest.mark.parametrize(
    "change,status",
    [
        ("none", "valid"),
        ("middle", "stale"),
        ("delete", "missing"),
        ("undo", "hidden"),
        ("redo", "valid"),
        ("clear", "valid"),
        ("legacy", "unverified"),
        ("corrupt", "stale"),
        ("missing_record", "missing"),
    ],
)
async def test_fact_recall_revalidates_source_and_never_leaks_invalid_evidence(
    tmp_path, change, status
):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    for role in ("user", "assistant", "tool"):
        sessions.append_message(session, {"role": role, "content": "evidence", "name": "original"})
    facts = FactStore(tmp_path / "facts.sqlite3")
    fact = facts.add("authoritative secret claim", source_id=session.id, source_seq=1)
    if change != "legacy":
        ref = sessions.source_snapshot(session.id, 1, 3, project_root=tmp_path).ref
        facts.attach_source(fact.id, 1, ref, sessions=sessions, project_root=tmp_path)
    if change in ("middle", "missing_record"):
        records = [json.loads(line) for line in session.path.read_text().splitlines()]
        if change == "middle":
            records[2]["message"]["name"] = "changed only tool metadata"
        else:
            del records[2]
        session.path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    elif change == "delete":
        sessions.delete(session.id)
    elif change in ("undo", "redo"):
        sessions.undo(session)
        if change == "redo":
            assert sessions.redo(session)
    elif change == "clear":
        sessions.append_event(session, "clear")
    elif change == "corrupt":
        with session.path.open("a") as stream:
            stream.write("broken record\n")
    config = Config()
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=PermissionChecker(config, mode="readonly", cwd=tmp_path),
        session_store=sessions,
        extras={"facts": facts},
    )
    _, result = await ToolRegistry(memory_tools()).dispatch_result(
        "c", "memory_recall", json.dumps({"fact_id": fact.id}), ctx
    )
    assert not result.is_error
    data = json.loads(result.content)
    assert data["status"] == status
    assert data["fact_id"] == fact.id
    if status == "valid":
        assert data["fact_text"] == fact.text
        assert "evidence" in data["source_text"]
    else:
        assert "authoritative secret claim" not in result.content
        assert "evidence" not in result.content
    facts.close()


async def test_child_recalls_parent_source_without_persisting_child_history(tmp_path, monkeypatch):
    from tests.fakes import FakeProvider

    from lecode.agent.builder import build_runtime
    from lecode.extras.subagents import run_subagent

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("parent", tmp_path)
    sessions.append_message(session, {"role": "user", "content": "parent source"})
    before = sessions.read_records(session)
    provider = FakeProvider(
        [
            {
                "tool_calls": [
                    {
                        "name": "memory_recall",
                        "arguments": json.dumps(
                            {"session_id": session.id, "start_seq": 1, "end_seq": 1}
                        ),
                    }
                ]
            },
            {"text": "done"},
        ]
    )
    runtime = build_runtime(Config(), tmp_path, session=session, store=sessions)
    runtime.ctx.extras["provider"] = provider
    extras = dict(runtime.ctx.extras)
    await run_subagent(
        runtime.ctx, runtime.registry, runtime.agents, name="explore", prompt="recall"
    )
    response = next(
        message for message in provider.requests[1]["messages"] if message["role"] == "tool"
    )
    assert json.loads(response["content"])["status"] == "valid"
    assert "parent source" in response["content"]
    records = sessions.read_records(session)
    assert records[:-1] == before
    assert records[-1].kind == "agent_run"
    assert runtime.ctx.extras == extras


@pytest.mark.parametrize(
    "case",
    [
        "cross_project",
        "traversal",
        "absolute",
        "symlink",
        "disabled",
        "oversized_range",
        "missing_range",
        "bool_range",
    ],
)
async def test_recall_rejects_unsafe_or_disabled_requests(tmp_path, case):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path / "other" if case == "cross_project" else tmp_path)
    sessions.append_message(session, {"role": "user", "content": "must not leak"})
    args = {"session_id": session.id, "start_seq": 1, "end_seq": 1}
    if case == "traversal":
        args["session_id"] = "../" + session.id
    elif case == "absolute":
        args["session_id"] = str(session.path)
    elif case == "symlink":
        (sessions.sessions_dir / "alias.jsonl").symlink_to(session.path)
        args["session_id"] = "alias"
    elif case == "oversized_range":
        args["end_seq"] = 201
    elif case == "missing_range":
        del args["end_seq"]
    elif case == "bool_range":
        args["start_seq"] = True
    config = Config()
    config.memory.enabled = case != "disabled"
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=PermissionChecker(config, mode="readonly", cwd=tmp_path),
        session_store=sessions,
    )
    _, result = await ToolRegistry(memory_tools()).dispatch_result(
        "c", "memory_recall", json.dumps(args), ctx
    )
    assert result.is_error
    assert "must not leak" not in result.content


async def test_existing_exclusions_suppress_direct_and_fact_recall_and_attachment(tmp_path):
    import sqlite3

    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    sessions.append_message(session, {"role": "user", "content": "excluded evidence"})
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    path = tmp_path / "facts.sqlite3"
    facts = FactStore(path)
    fact = facts.add("excluded claim", source_id=session.id, source_seq=1)
    facts.attach_source(fact.id, 1, ref, sessions=sessions, project_root=tmp_path)
    facts.close()
    # Seed the existing on-disk exclusion format, without adding a forget API.
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO exclusions VALUES (?, ?)", (session.id, 1))
    facts = FactStore(path)
    assert facts.excluded_seqs(session.id) == frozenset({1})
    with pytest.raises(ValueError, match="hidden"):
        facts.attach_source(fact.id, 1, ref, sessions=sessions, project_root=tmp_path)
    config = Config()
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=PermissionChecker(config, mode="readonly", cwd=tmp_path),
        session_store=sessions,
        extras={"facts": facts},
    )
    for args in ({"fact_id": fact.id}, {"session_id": session.id, "start_seq": 1, "end_seq": 1}):
        _, result = await ToolRegistry(memory_tools()).dispatch_result(
            "c", "memory_recall", json.dumps(args), ctx
        )
        assert json.loads(result.content)["status"] == "hidden"
        assert "excluded evidence" not in result.content
        assert "excluded claim" not in result.content
    facts.close()


def test_source_attachment_rejects_changed_or_cross_project_refs(tmp_path):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("source", tmp_path)
    sessions.append_message(session, {"role": "user", "content": "old"})
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    facts = FactStore(tmp_path / "facts.sqlite3")
    fact = facts.add("claim", source_id=session.id, source_seq=1)
    with pytest.raises(ValueError, match="another project"):
        facts.attach_source(fact.id, 1, ref, sessions=sessions, project_root=tmp_path / "other")
    session.path.write_text(session.path.read_text().replace('"old"', '"new"'))
    with pytest.raises(ValueError, match="stale"):
        facts.attach_source(fact.id, 1, ref, sessions=sessions, project_root=tmp_path)
    assert facts.source(fact.id) is None
    facts.close()


def test_legacy_database_migrates_without_verifying_old_provenance(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE facts (id TEXT PRIMARY KEY, text TEXT NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE revisions (fact_id TEXT, revision INTEGER, text TEXT,
                PRIMARY KEY (fact_id, revision));
            CREATE TABLE provenance (fact_id TEXT, revision INTEGER, source_id TEXT,
                source_seq INTEGER, created_at TEXT DEFAULT 'legacy',
                PRIMARY KEY (fact_id, revision));
            CREATE TABLE exclusions (source_id TEXT, source_seq INTEGER,
                PRIMARY KEY (source_id, source_seq));
            INSERT INTO facts VALUES ('old', 'preserved', 1);
            INSERT INTO revisions VALUES ('old', 1, 'preserved');
            INSERT INTO provenance VALUES ('old', 1, 'original-session', 5, 'legacy');
            INSERT INTO exclusions VALUES ('original-session', 9);
        """)
    facts = FactStore(path)
    assert facts.get("old").text == "preserved"
    assert facts.source("old") is None
    assert facts.provenance("old")[0].source_id == "original-session"
    assert facts.excluded_seqs("original-session") == frozenset({9})
    facts.revise("old", "new revision", expected_revision=1, source_id="new-session", source_seq=6)
    assert facts.source("old") is None
    assert len(facts.provenance("old")) == 2
    facts.close()


def test_explicit_source_ids_work_across_linked_worktrees(tmp_path):
    from tests.test_worktree import git_sync, make_repo_sync

    repo = make_repo_sync(tmp_path / "main")
    linked = tmp_path / "linked"
    git_sync(repo, "worktree", "add", "-b", "feature", str(linked))
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("linked source", linked)
    sessions.append_message(session, {"role": "user", "content": "linked evidence"})
    snapshot = sessions.source_snapshot(session.id, 1, 1, project_root=repo)
    assert snapshot.status == "valid"
    assert snapshot.messages[0]["content"] == "linked evidence"


def test_submodule_sources_use_runtime_identity_and_reject_parent_project(tmp_path, monkeypatch):
    from tests.test_worktree import git_sync, make_repo_sync

    from lecode.agent.builder import build_runtime

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "runtime-config"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    parent = make_repo_sync(tmp_path / "parent")
    source = make_repo_sync(tmp_path / "source")
    git_sync(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "sub")
    sub = parent / "sub"
    linked = tmp_path / "linked-sub"
    git_sync(sub, "worktree", "add", "-b", "feature", str(linked))
    # A separate explicit session directory also exercises the runtime's fact binding.
    sessions = SessionStore(tmp_path / "session-config")
    runtime = build_runtime(Config(), sub, store=sessions)
    identity = runtime.ctx.project_root
    assert identity is not None
    facts = runtime.ctx.extras["facts"]
    try:
        for cwd in (sub, linked):
            session = sessions.create(cwd.name, cwd)
            sessions.append_message(session, {"role": "user", "content": "submodule evidence"})
            snapshot = sessions.source_snapshot(session.id, 1, 1, project_root=identity)
            assert snapshot.status == "valid"
            assert snapshot.ref is not None
            fact = facts.remember(
                "submodule fact", snapshot.ref, sessions=sessions, project_root=identity
            )
            assert facts.get(fact.id) == fact
            with pytest.raises(ValueError, match="another project"):
                sessions.source_snapshot(session.id, 1, 1, project_root=parent)
            with pytest.raises(ValueError, match="another project"):
                facts.remember(
                    "wrong project", snapshot.ref, sessions=sessions, project_root=parent
                )
            assert facts.forget(fact.id)
            assert sessions.memory_generation(session.id) == facts.generation() > 0
            assert (
                sessions.source_snapshot(session.id, 1, 1, project_root=identity).status == "hidden"
            )

        parent_session = sessions.create("parent", parent)
        sessions.append_message(parent_session, {"role": "user", "content": "parent evidence"})
        with pytest.raises(ValueError, match="another project"):
            sessions.source_snapshot(parent_session.id, 1, 1, project_root=identity)
    finally:
        sessions.close()


def test_import_collision_does_not_alias_original_source_identity(tmp_path):
    sessions = SessionStore(tmp_path / "cfg")
    session = sessions.create("original", tmp_path)
    sessions.append_message(session, {"role": "user", "content": "original evidence"})
    ref = sessions.source_snapshot(session.id, 1, 1, project_root=tmp_path).ref
    imported = sessions.import_session(session.path)
    assert imported.id != session.id
    sessions.delete(session.id)
    assert sessions.validate_source(ref, project_root=tmp_path).status == "missing"
    assert sessions.source_snapshot(imported.id, 1, 1, project_root=tmp_path).status == "valid"


async def test_forget_suppresses_overlapping_revision_sources_but_keeps_unrelated_facts(tmp_path):
    from lecode.memory.recall import RecallContext

    sessions = SessionStore(tmp_path / "cfg")
    source = sessions.create("source", tmp_path)
    for content in ("shared evidence", "separate evidence"):
        sessions.append_message(source, {"role": "user", "content": content})
    facts = FactStore(tmp_path / "facts.sqlite3")
    shared = sessions.source_snapshot(source.id, 1, 1, project_root=tmp_path).ref
    separate = sessions.source_snapshot(source.id, 2, 2, project_root=tmp_path).ref
    selected = facts.remember("selected", shared, sessions=sessions, project_root=tmp_path)
    overlap = facts.remember("overlap", shared, sessions=sessions, project_root=tmp_path)
    facts.correct(
        overlap.id,
        "changed overlap",
        expected_revision=1,
        ref=separate,
        sessions=sessions,
        project_root=tmp_path,
    )
    unrelated = facts.remember("unrelated", separate, sessions=sessions, project_root=tmp_path)
    facts.forget(selected.id)
    recall = RecallContext(sessions, facts, tmp_path)
    assert json.loads(recall.recall({"fact_id": selected.id}))["status"] == "missing"
    hidden = recall.recall({"fact_id": overlap.id})
    assert json.loads(hidden)["status"] == "hidden"
    assert "changed overlap" not in hidden
    assert json.loads(recall.recall({"fact_id": unrelated.id}))["fact_text"] == "unrelated"
    assert facts.get(overlap.id) is not None  # suppressed, not bulk-purged
    with pytest.raises(ValueError, match="excluded"):
        facts.revise(
            overlap.id,
            "cannot wash lineage",
            expected_revision=2,
            source_id=source.id,
            source_seq=2,
        )
    # Clear affects working visibility and new extraction, not durable fact evidence.
    sessions.append_event(source, "clear")
    assert sessions.validate_source(separate, project_root=tmp_path).status == "hidden"
    assert json.loads(recall.recall({"fact_id": unrelated.id}))["status"] == "valid"
    with pytest.raises(ValueError, match="hidden"):
        facts.remember("new extraction", separate, sessions=sessions, project_root=tmp_path)
    sessions.delete(source.id)
    assert facts.get(unrelated.id) == unrelated
    assert json.loads(recall.recall({"fact_id": unrelated.id}))["status"] == "missing"
    facts.close()
