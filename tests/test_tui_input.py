"""Tests for input-editor extras: history, kill ring, $EDITOR, path completion."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from lecode.extras.proc import ProcResult
from lecode.session import SessionStore
from lecode.tui.input import (
    PATH_COMPLETION_LIMIT,
    KillRing,
    PathCompleter,
    SessionHistory,
    _path_token_before_cursor,
    kill_to_end_of_line,
    kill_to_start_of_line,
    kill_word_back,
    open_in_editor,
)

# -- SessionHistory -------------------------------------------------------------


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    store = SessionStore()
    return store, store.create("hist", cwd="/tmp")


def test_history_round_trip(session):
    store, s = session
    history = SessionHistory(store, s)
    history.store_string("first")
    history.store_string("second")
    reloaded = SessionHistory(store, s)
    assert list(reloaded.load_history_strings()) == ["first", "second"]


def test_history_lives_in_the_session_file(session):
    store, s = session
    SessionHistory(store, s).store_string("hello")
    kinds = [r.kind for r in store.read_records(s) if hasattr(r, "kind")]
    assert "input" in kinds


def test_history_cap_keeps_recent_in_memory(session):
    store, s = session
    history = SessionHistory(store, s, cap=3)
    for i in range(5):
        history.store_string(f"entry-{i}")
    assert list(history.load_history_strings()) == ["entry-2", "entry-3", "entry-4"]


def test_history_dedupes_consecutive(session):
    store, s = session
    history = SessionHistory(store, s)
    history.store_string("same")
    history.store_string("same")
    history.store_string("other")
    history.store_string("same")  # non-consecutive repeat is kept
    assert list(history.load_history_strings()) == ["same", "other", "same"]


def test_history_ignores_blank_entries(session):
    store, s = session
    history = SessionHistory(store, s)
    history.store_string("")
    history.store_string("   ")
    assert list(history.load_history_strings()) == []


def test_draft_persist_load_clear(session):
    store, s = session
    history = SessionHistory(store, s)
    history.store_string("submitted")
    history.save_draft("half-typed")
    reloaded = SessionHistory(store, s)
    assert reloaded.load_draft() == "half-typed"
    # consumed (tombstoned), submitted entries survive
    assert SessionHistory(store, s).load_draft() == ""
    assert list(SessionHistory(store, s).load_history_strings()) == ["submitted"]


def test_load_draft_without_draft(session):
    store, s = session
    assert SessionHistory(store, s).load_draft() == ""


def test_rebind_switches_session(session):
    store, s = session
    history = SessionHistory(store, s)
    history.store_string("in-first")
    other = store.create("other", cwd="/tmp")
    history.rebind(other)
    assert list(history.load_history_strings()) == []
    history.store_string("in-second")
    history.rebind(s)
    assert list(history.load_history_strings()) == ["in-first"]


# -- KillRing ------------------------------------------------------------------


def test_kill_ring_order():
    ring = KillRing()
    ring.kill("first")
    ring.kill("second")
    assert ring.yank() == "second"
    assert len(ring) == 2


def test_kill_ring_ignores_empty_kills():
    ring = KillRing()
    ring.kill("")
    assert ring.yank() is None
    assert len(ring) == 0


def test_kill_to_end_of_line():
    assert kill_to_end_of_line("foo bar", 0) == "foo bar"
    assert kill_to_end_of_line("foo\nbar", 3) == "\n"  # at the newline: kill it
    assert kill_to_end_of_line("foo\nbar", 4) == "bar"
    assert kill_to_end_of_line("foo", 3) == ""


def test_kill_to_start_of_line():
    assert kill_to_start_of_line("foo bar", 3) == "foo"
    assert kill_to_start_of_line("foo\nbar", 7) == "bar"
    assert kill_to_start_of_line("foo\nbar", 4) == ""


def test_kill_word_back():
    assert kill_word_back("foo bar baz", 11) == "baz"
    assert kill_word_back("foo bar   ", 10) == "bar   "
    assert kill_word_back("foo", 0) == ""


# -- $EDITOR --------------------------------------------------------------------


def _fake_editor_script(tmp_path: Path, body: str) -> str:
    script = tmp_path / "editor.sh"
    script.write_text(f"#!/bin/sh\n{body}\n")
    return f"/bin/sh {script}"


async def _run_in_terminal_directly(func, **kwargs):
    func()


async def test_open_in_editor_returns_edited_text(tmp_path, monkeypatch):
    monkeypatch.setattr("lecode.tui.input.run_in_terminal", _run_in_terminal_directly)
    monkeypatch.setenv("EDITOR", _fake_editor_script(tmp_path, "printf 'edited content' > \"$1\""))
    assert await open_in_editor("original") == "edited content"


async def test_open_in_editor_unset_editor_returns_none(monkeypatch):
    monkeypatch.delenv("EDITOR", raising=False)
    assert await open_in_editor("original") is None


async def test_open_in_editor_unchanged_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("lecode.tui.input.run_in_terminal", _run_in_terminal_directly)
    monkeypatch.setenv("EDITOR", _fake_editor_script(tmp_path, "true"))
    assert await open_in_editor("original") is None


async def test_open_in_editor_empty_save_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("lecode.tui.input.run_in_terminal", _run_in_terminal_directly)
    monkeypatch.setenv("EDITOR", _fake_editor_script(tmp_path, "printf '' > \"$1\""))
    assert await open_in_editor("original") is None


# -- PathCompleter --------------------------------------------------------------


def test_path_token_detection():
    assert _path_token_before_cursor(Document("edit src/le", 11)) == "src/le"
    assert _path_token_before_cursor(Document("open ./rea", 10)) == "./rea"
    assert _path_token_before_cursor(Document("open ~/.con", 11)) == "~/.con"
    assert _path_token_before_cursor(Document("plain word", 10)) is None
    assert _path_token_before_cursor(Document("", 0)) is None
    # "/..." at buffer start is the slash-command trigger and "...." the
    # personas trigger (their pickers own them); mid-message both stay path
    # tokens, and absolute paths complete anywhere after a space.
    assert _path_token_before_cursor(Document("/mod", 4)) is None
    assert _path_token_before_cursor(Document("/", 1)) is None
    assert _path_token_before_cursor(Document(".per", 4)) is None
    assert _path_token_before_cursor(Document("read /etc/ho", 12)) == "/etc/ho"
    assert _path_token_before_cursor(Document("read .env", 9)) == ".env"


def _fd_result(stdout: str) -> ProcResult:
    return ProcResult(exit_code=0, stdout=stdout, stderr="")


async def _complete(completer, text: str) -> list[str]:
    doc = Document(text, len(text))
    gen = completer.get_completions_async(doc, CompleteEvent())
    return [c.text async for c in gen]


async def test_path_completer_nested_match(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lecode.tui.input.run_proc",
        lambda *a, **k: _async(_fd_result("src/\nsrc/lecode/\nsrc/lecode/app.py\nREADME.md\n")),
    )
    completer = PathCompleter(tmp_path)
    assert await _complete(completer, "edit src/le") == ["src/lecode/", "src/lecode/app.py"]


async def test_path_completer_plain_word_yields_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lecode.tui.input.run_proc", lambda *a, **k: _async(_fd_result("word.txt\n"))
    )
    completer = PathCompleter(tmp_path)
    assert await _complete(completer, "just a word") == []


async def test_path_completer_caps_file_completions(tmp_path, monkeypatch):
    """The path completer yields at most PATH_COMPLETION_LIMIT matches."""
    listing = "\n".join(f"logs/file{i:02d}.txt" for i in range(30))
    monkeypatch.setattr("lecode.tui.input.run_proc", lambda *a, **k: _async(_fd_result(listing)))
    completer = PathCompleter(tmp_path)
    completions = await _complete(completer, "logs/")
    assert len(completions) == PATH_COMPLETION_LIMIT
    assert completions[0] == "logs/file00.txt"


async def test_path_completer_scandir_fallback(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("x")
    (tmp_path / ".git").mkdir()

    async def _fd_fails(*args, **kwargs):
        return ProcResult(exit_code=1, stdout="", stderr="fd broke")

    monkeypatch.setattr("lecode.tui.input.run_proc", _fd_fails)
    completer = PathCompleter(tmp_path)
    assert await _complete(completer, "./R") == ["README.md"]
    assert await _complete(completer, "./s") == ["src/"]


async def test_path_completer_cache_ttl(tmp_path, monkeypatch):
    calls = []

    async def _fd(*args, **kwargs):
        calls.append(1)
        return _fd_result("a.txt\n")

    now = [100.0]
    monkeypatch.setattr("lecode.tui.input.run_proc", _fd)
    completer = PathCompleter(tmp_path, clock=lambda: now[0])
    assert await _complete(completer, "./a") == ["a.txt"]
    assert await _complete(completer, "./a") == ["a.txt"]
    assert len(calls) == 1
    now[0] += 3.0
    assert await _complete(completer, "./a") == ["a.txt"]
    assert len(calls) == 2


async def _async(value):
    return value
