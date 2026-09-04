"""Tests for clipboard integration (OSC 52 / pbcopy / xclip) and OSC 8 links."""

from __future__ import annotations

import base64
from io import StringIO

from rich.console import Console
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.config.models import Config
from lecode.extras.proc import ProcResult
from lecode.session.storage import SessionStore
from lecode.tui.app import TuiApp
from lecode.tui.clipboard import copy_to_clipboard, osc8_link, osc52_sequence


def test_osc52_sequence_format():
    seq = osc52_sequence("hello")
    expected = base64.b64encode(b"hello").decode()
    assert seq == f"\x1b]52;c;{expected}\x1b\\"


def test_osc8_link_format():
    link = osc8_link("https://example.com", "example")
    assert link == "\x1b]8;;https://example.com\x1b\\example\x1b]8;;\x1b\\"


def test_osc8_link_no_color_is_plain():
    assert osc8_link("https://example.com", "example", no_color=True) == "example"


async def test_copy_tty_uses_osc52(monkeypatch, capsys):
    def _forbidden(*args, **kwargs):
        raise AssertionError("run_proc must not run when OSC 52 succeeds")

    monkeypatch.setattr("lecode.tui.clipboard.run_proc", _forbidden)
    assert await copy_to_clipboard("hello", tty=True) is True
    assert osc52_sequence("hello") in capsys.readouterr().out


async def test_copy_pbcopy_path(monkeypatch):
    calls = []

    async def _proc(argv, **kwargs):
        calls.append((argv, kwargs.get("input")))
        return ProcResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lecode.tui.clipboard.run_proc", _proc)
    assert await copy_to_clipboard("payload", tty=False) is True
    assert calls == [(["pbcopy"], "payload")]


async def test_copy_falls_back_to_xclip(monkeypatch):
    calls = []

    async def _proc(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "pbcopy":
            raise OSError("no pbcopy")
        return ProcResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lecode.tui.clipboard.run_proc", _proc)
    assert await copy_to_clipboard("payload", tty=False) is True
    assert calls == [["pbcopy"], ["xclip", "-selection", "clipboard"]]


async def test_copy_all_fail_returns_false(monkeypatch):
    async def _proc(argv, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr("lecode.tui.clipboard.run_proc", _proc)
    assert await copy_to_clipboard("payload", tty=False) is False


# -- /copy command ------------------------------------------------------------------


def make_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    out = StringIO()
    app = TuiApp(
        config,
        runtime,
        FakeProvider([]),
        session,
        store,
        console=Console(record=True, file=out, width=200),
    )
    return app, out


async def test_copy_command_copies_last_response(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch)
    copied = []

    async def _copy(text, **kwargs):
        copied.append(text)
        return True

    monkeypatch.setattr("lecode.tui.app.copy_to_clipboard", _copy)
    app._last_response = "the answer"
    await app.handle_command("/copy")
    assert copied == ["the answer"]
    assert "copied 10 chars" in out.getvalue()


async def test_copy_command_nothing_to_copy(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch)
    await app.handle_command("/copy")
    assert "nothing to copy" in out.getvalue()


async def test_copy_command_clipboard_unavailable(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch)

    async def _copy(text, **kwargs):
        return False

    monkeypatch.setattr("lecode.tui.app.copy_to_clipboard", _copy)
    app._last_response = "the answer"
    await app.handle_command("/copy")
    assert "clipboard unavailable" in out.getvalue()


async def test_last_response_tracked_after_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    app = TuiApp(
        config,
        runtime,
        FakeProvider([{"text": "final answer"}]),
        session,
        store,
        console=Console(record=True, file=StringIO(), width=200),
    )
    await app._submit("hi")
    await app._turn_task
    assert app._last_response == "final answer"
