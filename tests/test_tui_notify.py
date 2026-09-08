"""Tests for notifications (bell / afplay / paplay / aplay / osascript / notify-send)."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from rich.console import Console
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.agent.runner import Done
from lecode.config.models import Config, NotificationsConfig
from lecode.extras.proc import ProcResult
from lecode.session.storage import SessionStore
from lecode.tui.app import TuiApp
from lecode.tui.notify import Notifier


def make_notifier(captured: list, bells: list, **config_overrides):
    """A Notifier with fake player detection and run_proc capture."""
    config = NotificationsConfig(**config_overrides)
    return Notifier(
        config,
        which=lambda binary: f"/usr/bin/{binary}" if binary == "afplay" else None,
        file_exists=lambda path: True,
        bell=lambda: bells.append("\a"),
    )


@pytest.fixture
def proc_calls(monkeypatch):
    calls = []

    async def _proc(argv, **kwargs):
        calls.append(argv)
        return ProcResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("lecode.tui.notify.run_proc", _proc)
    return calls


async def test_finish_plays_afplay_with_volume(proc_calls):
    bells = []
    notifier = make_notifier(proc_calls, bells)
    await notifier.task_finish()
    assert proc_calls == [["afplay", "-v", "0.5", "/System/Library/Sounds/Glass.aiff"]]
    assert bells == []


async def test_custom_volume_passed_to_afplay(proc_calls):
    notifier = make_notifier(proc_calls, [], volume=0.9)
    await notifier.task_finish()
    assert proc_calls[0][2] == "0.9"


async def test_error_and_approval_sounds(proc_calls):
    notifier = make_notifier(proc_calls, [])
    await notifier.error()
    await notifier.approval_needed()
    assert proc_calls[0][-1] == "/System/Library/Sounds/Basso.aiff"
    assert proc_calls[1][-1] == "/System/Library/Sounds/Ping.aiff"


async def test_disabled_config_is_silent(proc_calls):
    bells = []
    notifier = make_notifier(proc_calls, bells, enabled=False)
    await notifier.task_finish()
    await notifier.error()
    await notifier.approval_needed()
    assert proc_calls == []
    assert bells == []


async def test_per_event_toggles(proc_calls):
    bells = []
    notifier = make_notifier(proc_calls, bells, on_error=False, on_approval=False)
    await notifier.error()
    await notifier.approval_needed()
    await notifier.task_finish()
    assert len(proc_calls) == 1  # only the finish sound


async def test_no_player_rings_bell(proc_calls):
    bells = []
    notifier = Notifier(
        NotificationsConfig(),
        which=lambda binary: None,
        file_exists=lambda path: True,
        bell=lambda: bells.append("\a"),
    )
    await notifier.task_finish()
    assert proc_calls == []
    assert bells == ["\a"]


async def test_no_sound_file_rings_bell(proc_calls):
    bells = []
    notifier = Notifier(
        NotificationsConfig(),
        which=lambda binary: "/usr/bin/afplay",
        file_exists=lambda path: False,
        bell=lambda: bells.append("\a"),
    )
    await notifier.task_finish()
    assert bells == ["\a"]


async def test_oserror_falls_back_to_bell(monkeypatch):
    async def _proc(argv, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr("lecode.tui.notify.run_proc", _proc)
    bells = []
    notifier = make_notifier([], bells)
    await notifier.task_finish()  # must not raise
    assert bells == ["\a"]


async def test_bell_itself_never_raises(monkeypatch):
    async def _proc(argv, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr("lecode.tui.notify.run_proc", _proc)

    def _bad_bell():
        raise OSError("closed stdout")

    notifier = Notifier(
        NotificationsConfig(),
        which=lambda b: "/usr/bin/afplay",
        file_exists=lambda p: True,
        bell=_bad_bell,
    )
    await notifier.task_finish()  # must not raise


# -- desktop channel ------------------------------------------------------------


def make_desktop_notifier(bells: list, binaries: tuple[str, ...], **kwargs):
    """A Notifier whose ``which`` finds exactly the given binaries.

    ``kwargs`` are split: ``session_name`` goes to the constructor, the rest
    to ``NotificationsConfig``.
    """
    session_name = kwargs.pop("session_name", None)
    return Notifier(
        NotificationsConfig(**kwargs),
        which=lambda binary: f"/usr/bin/{binary}" if binary in binaries else None,
        file_exists=lambda path: True,
        bell=lambda: bells.append("\a"),
        session_name=session_name,
    )


async def test_desktop_osascript_argv(proc_calls):
    notifier = make_desktop_notifier([], ("afplay", "osascript"), session_name="work")
    await notifier.task_finish()
    assert proc_calls == [
        ["afplay", "-v", "0.5", "/System/Library/Sounds/Glass.aiff"],
        ["osascript", "-e", 'display notification "task finished: work" with title "lecode"'],
    ]


async def test_desktop_notify_send_argv(proc_calls):
    notifier = make_desktop_notifier([], ("afplay", "notify-send"))
    await notifier.task_finish()
    assert proc_calls[1] == ["notify-send", "lecode", "task finished"]


async def test_no_desktop_binary_skips_desktop(proc_calls):
    notifier = make_desktop_notifier([], ("afplay",))
    await notifier.task_finish()
    assert proc_calls == [["afplay", "-v", "0.5", "/System/Library/Sounds/Glass.aiff"]]


async def test_desktop_channel_disabled(proc_calls):
    notifier = make_desktop_notifier([], ("afplay", "osascript"), desktop=False)
    await notifier.task_finish()
    assert proc_calls == [["afplay", "-v", "0.5", "/System/Library/Sounds/Glass.aiff"]]


async def test_sound_channel_disabled(proc_calls):
    bells = []
    notifier = make_desktop_notifier(bells, ("afplay", "osascript"), sound=False)
    await notifier.task_finish()
    assert proc_calls == [
        ["osascript", "-e", 'display notification "task finished" with title "lecode"']
    ]
    assert bells == []


async def test_error_desktop_escapes_and_truncates(proc_calls):
    notifier = make_desktop_notifier([], ("osascript",), sound=False)
    message = 'boom "quoted" \\ path\n' + "x" * 300
    await notifier.error(message)
    script = proc_calls[0][2]
    expected = 'boom \\"quoted\\" \\\\ path ' + "x" * 179
    assert script == f'display notification "{expected}" with title "lecode"'


async def test_approval_desktop_includes_tool_name(proc_calls):
    notifier = make_desktop_notifier([], ("notify-send",), sound=False)
    await notifier.approval_needed("bash")
    assert proc_calls == [["notify-send", "lecode", "approval needed: bash"]]


async def test_desktop_spawn_failure_never_raises(monkeypatch):
    async def _proc(argv, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr("lecode.tui.notify.run_proc", _proc)
    notifier = make_desktop_notifier([], ("osascript",), sound=False)
    await notifier.task_finish()  # must not raise


# -- app wiring ---------------------------------------------------------------------


class FakeNotifier:
    def __init__(self) -> None:
        self.finished = 0
        self.errors = 0
        self.approvals = 0

    async def task_finish(self):
        self.finished += 1

    async def error(self, message: str = ""):
        self.errors += 1

    async def approval_needed(self, tool_name: str = ""):
        self.approvals += 1


async def test_app_fires_finish_notification(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    app = TuiApp(
        config,
        runtime,
        FakeProvider([{"text": "hi"}]),
        session,
        store,
        console=Console(record=True, file=StringIO(), width=200),
    )
    fake = FakeNotifier()
    app._notifier = fake
    app._on_event(Done(stop_reason="done", turns=1))
    await asyncio.gather(*app._pending)
    assert fake.finished == 1
