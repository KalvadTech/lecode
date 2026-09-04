"""Tests for Unix-socket status signals."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

from tests.test_tui_app import make_app
from tests.test_worktree import commit_all, make_repo

from lecode.config.models import Config
from lecode.extras.status_signals import GIT_CONFLICT, START, STOP, StatusEmitter


def short_sock_path() -> Path:
    """A socket path short enough for the AF_UNIX sun_path limit (104 on macOS)."""
    return Path(f"/tmp/lecode-test-{os.getpid()}-{uuid.uuid4().hex[:6]}.sock")


def make_listener(path: Path) -> socket.socket:
    """A bound Unix datagram listener with a recv timeout."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(path))
    sock.settimeout(2)
    return sock


def recv_event(sock: socket.socket) -> dict:
    return json.loads(sock.recv(65536).decode())


def make_emitter(tmp_path, monkeypatch, *, enabled=True, socket_path=None):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.signals.enabled = enabled
    config.signals.socket_path = socket_path
    return StatusEmitter(config.signals, session="test-session"), config


# -- the emitter --------------------------------------------------------------------


def test_emit_sends_json_event(tmp_path, monkeypatch):
    path = short_sock_path()
    emitter, _ = make_emitter(tmp_path, monkeypatch, socket_path=str(path))
    listener = make_listener(path)
    emitter.emit(START)
    event = recv_event(listener)
    assert event["event"] == "start"
    assert event["session"] == "test-session"
    assert "ts" in event
    listener.close()


def test_disabled_emitter_is_inert(tmp_path, monkeypatch):
    path = tmp_path / "lecode.sock"
    emitter, _ = make_emitter(tmp_path, monkeypatch, enabled=False, socket_path=str(path))
    emitter.emit(START)  # must not raise, must not touch the path
    assert not path.exists()


def test_default_socket_path(tmp_path, monkeypatch):
    emitter, _ = make_emitter(tmp_path, monkeypatch)
    assert emitter.socket_path == tmp_path / "cfg" / "lecode.sock"


def test_failures_are_swallowed(tmp_path, monkeypatch):
    emitter, _ = make_emitter(
        tmp_path, monkeypatch, socket_path=str(tmp_path / "nonexistent" / "lecode.sock")
    )
    emitter.emit(START)  # no listener / no such directory — drops silently
    emitter.emit(GIT_CONFLICT, files=["a.py"])


def test_extra_fields(tmp_path, monkeypatch):
    path = short_sock_path()
    emitter, _ = make_emitter(tmp_path, monkeypatch, socket_path=str(path))
    listener = make_listener(path)
    emitter.emit(GIT_CONFLICT, files=["a.py", "b.py"])
    event = recv_event(listener)
    assert event["event"] == "git-conflict"
    assert event["files"] == ["a.py", "b.py"]
    listener.close()


# -- wiring: turns and worktree conflicts ----------------------------------------------


async def test_turn_start_stop_signals(tmp_path, monkeypatch):
    config = Config()
    config.signals.enabled = True
    config.signals.socket_path = str(short_sock_path())
    app, _, _ = make_app(tmp_path, monkeypatch, [{"text": "hi"}], config=config)
    listener = make_listener(Path(config.signals.socket_path))
    await app._submit("hello")
    await app._turn_task
    events = [recv_event(listener), recv_event(listener)]
    assert [e["event"] for e in events] == [START, STOP]
    assert all(e["session"] == "test-session" for e in events)
    listener.close()


async def test_signals_disabled_by_default(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [{"text": "hi"}])
    assert app._signals.enabled is False
    await app._submit("hello")
    await app._turn_task  # simply must not raise


async def test_git_conflict_signal_on_wt_merge(tmp_path, monkeypatch):
    config = Config()
    config.signals.enabled = True
    config.signals.socket_path = str(short_sock_path())
    await make_repo(tmp_path)
    app, _, _ = make_app(tmp_path, monkeypatch, [], config=config)
    listener = make_listener(Path(config.signals.socket_path))
    await app.handle_command("/worktree feat")
    wt_path = app._worktree.path
    (wt_path / "file.txt").write_text("worktree change\n", encoding="utf-8")
    await commit_all(wt_path, "worktree edit")
    (tmp_path / "file.txt").write_text("main change\n", encoding="utf-8")
    await commit_all(tmp_path, "main edit")
    await app.handle_command("/wt-merge")
    event = recv_event(listener)
    assert event["event"] == GIT_CONFLICT
    assert event["files"] == ["file.txt"]
    listener.close()
