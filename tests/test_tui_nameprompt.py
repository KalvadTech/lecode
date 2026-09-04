"""Tests for the startup session-name prompt (pipe-input driven)."""

from __future__ import annotations

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lecode.session.storage import SessionStore
from lecode.tui.name_prompt import folder_sessions, pick_session, prompt_session_name


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


async def _prompt(store, keys: str) -> str | None:
    with create_pipe_input() as inp:
        inp.send_text(keys)
        session = PromptSession(input=inp, output=DummyOutput())
        return await prompt_session_name(store, session=session)


async def test_valid_name_returned(store):
    assert await _prompt(store, "my session\n") == "my session"


async def test_empty_rejected_then_valid(store, capsys):
    assert await _prompt(store, "\nvalid-name\n") == "valid-name"
    assert "session name must not be empty" in capsys.readouterr().out


async def test_invalid_rejected_then_valid(store, capsys):
    assert await _prompt(store, "bad/name\nok-name\n") == "ok-name"
    assert "must not contain path separators" in capsys.readouterr().out


async def test_duplicate_suffixed(store, capsys, tmp_path):
    store.create("foo", tmp_path)
    assert await _prompt(store, "foo\n") == "foo-2"
    assert "name taken, using 'foo-2'" in capsys.readouterr().out


async def test_ctrl_c_aborts(store):
    assert await _prompt(store, "\x03") is None


async def test_ctrl_d_aborts(store):
    assert await _prompt(store, "\x04") is None


async def test_abort_creates_no_session_file(store):
    assert await _prompt(store, "\x03") is None
    assert store.list_sessions() == []
    assert not store.sessions_dir.exists()


# -- bare -r session picker -------------------------------------------------------


async def _pick(store, cwd, keys: str):
    with create_pipe_input() as inp:
        inp.send_text(keys)
        session = PromptSession(input=inp, output=DummyOutput())
        return await pick_session(store, cwd, session=session)


def test_folder_sessions_filters_by_cwd(store, tmp_path):
    store.create("here", tmp_path)
    store.create("elsewhere", tmp_path / "other")
    assert [m.name for m in folder_sessions(store, tmp_path)] == ["here"]


async def test_pick_by_number(store, tmp_path):
    store.create("one", tmp_path)
    store.create("two", tmp_path)
    expected = folder_sessions(store, tmp_path)[1]  # list order, not insertion order
    picked = await _pick(store, tmp_path, "2\n")
    assert picked.id == expected.id


async def test_pick_empty_takes_most_recent(store, tmp_path):
    store.create("old", tmp_path)
    store.create("new", tmp_path)
    picked = await _pick(store, tmp_path, "\n")
    assert picked.name == "new"


async def test_pick_by_name(store, tmp_path):
    store.create("alpha", tmp_path)
    picked = await _pick(store, tmp_path, "alpha\n")
    assert picked.name == "alpha"


async def test_pick_by_id_prefix(store, tmp_path):
    meta = store.create("beta", tmp_path).meta
    picked = await _pick(store, tmp_path, meta.id[:8] + "\n")
    assert picked.id == meta.id


async def test_pick_invalid_then_valid(store, tmp_path, capsys):
    store.create("only", tmp_path)
    picked = await _pick(store, tmp_path, "99\n1\n")
    assert picked.name == "only"
    assert "error:" in capsys.readouterr().out


async def test_pick_lists_only_folder_sessions(store, tmp_path, capsys):
    store.create("here", tmp_path)
    store.create("elsewhere", tmp_path / "other")
    picked = await _pick(store, tmp_path, "\n")
    out = capsys.readouterr().out
    assert "here" in out and "elsewhere" not in out
    assert picked.name == "here"


async def test_pick_no_sessions_in_folder(store, tmp_path, capsys):
    store.create("elsewhere", tmp_path / "other")
    assert await _pick(store, tmp_path, "1\n") is None
    assert "no sessions" in capsys.readouterr().out


async def test_pick_ctrl_c_aborts(store, tmp_path):
    store.create("one", tmp_path)
    assert await _pick(store, tmp_path, "\x03") is None
