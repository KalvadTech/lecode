"""Tests for the startup session-name prompt (pipe-input driven)."""

from __future__ import annotations

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lecode.session.storage import SessionStore
from lecode.tui.name_prompt import prompt_session_name


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
