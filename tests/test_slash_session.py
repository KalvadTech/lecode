"""Tests for session slash commands (new/clear/resume/undo/rewind/…)."""

from __future__ import annotations

from tests.test_tui_app import make_app, make_blocking_app, wait_for

from lecode.session.model import EventRecord
from lecode.session.storage import SessionStore


def _events(app, kind: str) -> list[EventRecord]:
    return [
        r
        for r in app.store.read_records(app.session)
        if isinstance(r, EventRecord) and r.kind == kind
    ]


def _fake_name_prompt(value):
    async def _prompt(store, **kwargs):
        return value

    return _prompt


# -- /new ------------------------------------------------------------------------


async def test_new_with_name_switches_session(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    old_id = app.session.id
    await app.handle_command("/new second-session")
    assert app.session.name == "second-session"
    assert app.session.id != old_id
    assert app.runner.session is app.session
    assert app.runtime.ctx.session is app.session
    assert app.status.session_name == "second-session"
    assert "new session: second-session" in out.getvalue()
    assert SessionStore().resolve("second-session").name == "second-session"


async def test_new_duplicate_name_is_suffixed(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/new test-session")
    assert app.session.name == "test-session-2"


async def test_new_invalid_name_errors(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/new bad/name")
    assert "must not contain path separators" in out.getvalue()
    assert app.session.name == "test-session"


async def test_new_without_name_prompts(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    monkeypatch.setattr("lecode.tui.name_prompt.prompt_session_name", _fake_name_prompt("prompted"))
    await app.handle_command("/new")
    assert app.session.name == "prompted"


async def test_new_prompt_aborted_keeps_session(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    monkeypatch.setattr("lecode.tui.name_prompt.prompt_session_name", _fake_name_prompt(None))
    await app.handle_command("/new")
    assert app.session.name == "test-session"
    assert "cancelled" in out.getvalue()


async def test_new_refused_while_turn_runs(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    await app._submit("go")
    await wait_for(lambda: len(provider.requests) == 1)
    await app.handle_command("/new nope")
    assert "finish or cancel the current turn first" in out.getvalue()
    assert app.session.name == "test-session"
    provider.blocked = False
    provider.release.set()
    await app._turn_task


# -- /clear ------------------------------------------------------------------------


async def test_clear_resets_history_keeps_file(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "answer"}])
    await app._submit("hello")
    await app._turn_task
    assert len(app.store.load_messages(app.session)) == 2
    await app.handle_command("/clear")
    assert _events(app, "clear")
    assert app.store.load_for_model(app.session) == []
    assert len(app._history) == 1  # the system prompt only
    assert "conversation cleared" in out.getvalue()
    assert app.session.path.is_file()


async def test_clear_then_new_turn_sees_nothing_old(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "a"}, {"text": "b"}])
    await app._submit("first prompt")
    await app._turn_task
    await app.handle_command("/clear")
    await app._submit("second prompt")
    await app._turn_task
    contents = [m["content"] for m in provider.requests[-1]["messages"]]
    assert "first prompt" not in contents
    assert "second prompt" in contents


# -- /resume ------------------------------------------------------------------------


async def test_resume_by_name_switches(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    other = app.store.create("other-session", tmp_path)
    app.store.append_message(other, {"role": "user", "content": "old message"})
    await app.handle_command("/resume other-session")
    assert app.session.id == other.id
    assert "resumed session: other-session" in out.getvalue()
    contents = [m.get("content") for m in app._history]
    assert "old message" in contents


async def test_resume_unknown_ref_errors(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/resume nope")
    assert "✗" in out.getvalue() and "nope" in out.getvalue()
    assert app.session.name == "test-session"


async def test_resume_current_session_is_noop(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/resume test-session")
    assert "already in this session" in out.getvalue()


async def test_resume_picker_lists_sessions(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    app.store.create("another", tmp_path)
    await app.handle_command("/resume")
    rendered = out.getvalue()
    assert "1. another" in rendered
    assert "test-session (current)" in rendered
    assert "/resume --delete" in rendered


async def test_resume_delete_removes_session(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    app.store.create("doomed", tmp_path)
    await app.handle_command("/resume --delete doomed")
    assert "deleted session: doomed" in out.getvalue()
    assert [m.name for m in app.store.list_sessions()] == ["test-session"]


async def test_resume_delete_current_refused(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/resume --delete test-session")
    assert "cannot delete the current session" in out.getvalue()
    assert app.session.path.is_file()


# -- /session ------------------------------------------------------------------------


async def test_session_shows_metadata_and_stats(tmp_path, monkeypatch):
    script = [{"text": "hi", "usage": {"input_tokens": 10, "output_tokens": 5}}]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("hello")
    await app._turn_task
    await app.handle_command("/session")
    rendered = out.getvalue()
    assert "session: test-session" in rendered
    assert "agent: build" in rendered
    assert "tokens: 10 in / 5 out" in rendered
    assert f"cwd: {tmp_path}" in rendered


# -- /undo /redo /rewind /retry -------------------------------------------------------


async def test_undo_and_redo_through_commands(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "answer"}, {"text": "again"}])
    await app._submit("hello")
    await app._turn_task
    await app.handle_command("/undo")
    assert "undid the last user turn" in out.getvalue()
    assert [m["role"] for m in app._history] == ["system"]
    await app.handle_command("/redo")
    assert "restored the undone turn" in out.getvalue()
    assert [m["role"] for m in app._history] == ["system", "user", "assistant"]


async def test_undo_empty_session(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/undo")
    assert "nothing to undo" in out.getvalue()


async def test_redo_without_undo(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/redo")
    assert "nothing to redo" in out.getvalue()


async def test_rewind_lists_recent_turns(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "a"}, {"text": "b"}])
    await app._submit("first")
    await app._turn_task
    await app._submit("second")
    await app._turn_task
    await app.handle_command("/rewind")
    rendered = out.getvalue()
    assert "recent turns:" in rendered
    assert "first" in rendered and "second" in rendered
    assert "/rewind <seq>" in rendered


async def test_rewind_to_seq_hides_suffix(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "a"}, {"text": "b"}])
    await app._submit("first")
    await app._turn_task
    await app._submit("second")
    await app._turn_task
    seq = next(m.seq for m in app.store.load_messages(app.session) if m.role == "user")
    await app.handle_command(f"/rewind {seq}")
    rendered = out.getvalue()
    assert f"rewound to seq {seq}" in rendered
    assert "restore point recorded" in rendered
    contents = [m.get("content") for m in app._history]
    assert "first" in contents and "second" not in contents


async def test_rewind_bad_seq_errors(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "a"}])
    await app._submit("hello")
    await app._turn_task
    await app.handle_command("/rewind 999")
    assert "no visible message with seq 999" in out.getvalue()
    await app.handle_command("/rewind abc")
    assert "usage: /rewind <seq>" in out.getvalue()


async def test_retry_reruns_last_prompt(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "one"}, {"text": "two"}])
    await app._submit("do the thing")
    await app._turn_task
    await app.handle_command("/retry")
    await app._turn_task
    assert provider.requests[-1]["messages"][-1] == {"role": "user", "content": "do the thing"}
    # the original turn was tombstoned: one visible user message only
    users = [m for m in app.store.load_messages(app.session) if m.role == "user"]
    assert len(users) == 1


async def test_retry_empty_session(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/retry")
    assert "nothing to retry" in out.getvalue()


# -- /rename /history ------------------------------------------------------------------


async def test_rename_updates_session_and_statusline(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/rename better-name")
    assert app.session.name == "better-name"
    assert app.status.session_name == "better-name"
    assert "renamed to: better-name" in out.getvalue()
    assert _events(app, "rename")[-1].data["name"] == "better-name"
    # the rename survives reopening (latest rename event wins)
    assert app.store.open(app.session.id).name == "better-name"


async def test_rename_invalid_and_same_name(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/rename")
    assert "usage: /rename <name>" in out.getvalue()
    await app.handle_command("/rename test-session")
    assert "already named: test-session" in out.getvalue()


async def test_history_lists_messages(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "the answer"}])
    await app._submit("the question")
    await app._turn_task
    await app.handle_command("/history")
    rendered = out.getvalue()
    assert "user" in rendered and "the question" in rendered
    assert "assistant" in rendered and "the answer" in rendered


async def test_history_empty_session(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/history")
    assert "(empty session)" in out.getvalue()


# -- /handoff -------------------------------------------------------------------------


async def test_handoff_creates_and_switches(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "progress so far"}])
    await app._submit("work on the widget")
    await app._turn_task
    await app.handle_command("/handoff widget-part-two")
    assert app.session.name == "widget-part-two"
    assert "handed off to: widget-part-two" in out.getvalue()
    messages = app.store.load_messages(app.session)
    assert len(messages) == 1 and messages[0].role == "user"
    assert "Session handoff" in messages[0].message["content"]
    assert "work on the widget" in messages[0].message["content"]


async def test_handoff_without_name_prompts(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    monkeypatch.setattr(
        "lecode.tui.name_prompt.prompt_session_name", _fake_name_prompt("handed-off")
    )
    await app.handle_command("/handoff")
    assert app.session.name == "handed-off"


# -- /compact --------------------------------------------------------------------------


async def _fill_session(app, turns: int) -> None:
    for index in range(turns):
        app.store.append_message(app.session, {"role": "user", "content": f"question {index}"})
        app.store.append_message(app.session, {"role": "assistant", "content": f"answer {index}"})


async def test_compact_summarizes_and_records(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "the summary"}])
    await _fill_session(app, 5)
    await app.handle_command("/compact")
    assert "compacted 6 messages" in out.getvalue()
    compacted = _events(app, "compact")
    assert compacted and compacted[-1].data["summary"] == "the summary"
    loaded = app.store.load_for_model(app.session)
    assert loaded[0] == {"role": "system", "content": "the summary"}
    # the recent tail stays raw
    assert {"role": "assistant", "content": "answer 4"} in loaded
    assert {"role": "user", "content": "question 0"} not in loaded
    # the summarizer saw the old transcript
    request = provider.requests[-1]
    assert "question 0" in request["messages"][-1]["content"]


async def test_compact_too_little_history(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/compact")
    assert "not enough history to compact" in out.getvalue()


async def test_compact_provider_failure(tmp_path, monkeypatch):
    from lecode.providers.openai_compat import ProviderError

    app, _, out = make_app(
        tmp_path, monkeypatch, [{"error": ProviderError("boom", retryable=False)}]
    )
    await _fill_session(app, 5)
    await app.handle_command("/compact")
    assert "compaction failed" in out.getvalue()
    assert not _events(app, "compact")
