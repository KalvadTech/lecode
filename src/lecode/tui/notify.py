"""Sound + desktop notifications, with a terminal-bell fallback for sound.

Sound: afplay/paplay/aplay; desktop: osascript (macOS) / notify-send
(freedesktop). Best-effort: every failure mode (no player, no sound file,
no desktop binary, spawn errors) falls back to the bell or silence —
nothing here ever raises. afplay gets ``-v <volume>`` from
``[notifications].volume``; paplay/aplay and the bell have no volume knob.
Channel toggles: ``sound`` / ``desktop``; per-event toggles:
``on_finish`` / ``on_error`` / ``on_approval`` (apply to both channels).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from collections.abc import Callable

from lecode.config.models import NotificationsConfig
from lecode.extras.proc import run_proc

#: Candidate sound files per event kind (macOS first, then freedesktop).
_SOUNDS = {
    "finish": (
        "/System/Library/Sounds/Glass.aiff",
        "/usr/share/sounds/freedesktop/stereo/complete.oga",
    ),
    "error": (
        "/System/Library/Sounds/Basso.aiff",
        "/usr/share/sounds/freedesktop/stereo/dialog-error.oga",
    ),
    "approval": (
        "/System/Library/Sounds/Ping.aiff",
        "/usr/share/sounds/freedesktop/stereo/bell.oga",
    ),
}

#: Player argv templates in preference order; only afplay honors {volume}.
_PLAYERS = (
    ("afplay", ("afplay", "-v", "{volume}", "{file}")),
    ("paplay", ("paplay", "{file}")),
    ("aplay", ("aplay", "{file}")),
)

_TITLE = "lecode"
_ERROR_MESSAGE_CAP = 200


def _applescript_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _osascript_argv(title: str, message: str) -> tuple[str, ...]:
    script = (
        f'display notification "{_applescript_escape(message)}" '
        f'with title "{_applescript_escape(title)}"'
    )
    return ("osascript", "-e", script)


def _notify_send_argv(title: str, message: str) -> tuple[str, ...]:
    return ("notify-send", title, message)


#: Desktop-notification argv builders in preference order.
_DESKTOP = (
    ("osascript", _osascript_argv),
    ("notify-send", _notify_send_argv),
)

_PLAY_TIMEOUT_S = 5.0


def _stdout_bell() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


class Notifier:
    """Notifies on turn finish / error / approval-needed events."""

    def __init__(
        self,
        config: NotificationsConfig,
        *,
        which: Callable[[str], str | None] = shutil.which,
        file_exists: Callable[[str], bool] = os.path.exists,
        bell: Callable[[], None] = _stdout_bell,
        session_name: str | None = None,
    ) -> None:
        self._config = config
        self._which = which
        self._file_exists = file_exists
        self._bell = bell
        self._session_name = session_name

    async def task_finish(self) -> None:
        if self._config.on_finish:
            message = "task finished"
            if self._session_name:
                message = f"task finished: {self._session_name}"
            await self._notify("finish", message)

    async def error(self, message: str = "") -> None:
        if self._config.on_error:
            text = " ".join(message.split())[:_ERROR_MESSAGE_CAP]
            await self._notify("error", text or "an error occurred")

    async def approval_needed(self, tool_name: str = "") -> None:
        if self._config.on_approval:
            message = f"approval needed: {tool_name}" if tool_name else "approval needed"
            await self._notify("approval", message)

    async def _notify(self, kind: str, message: str) -> None:
        if not self._config.enabled:
            return
        if self._config.sound:
            await self._play(kind)
        if self._config.desktop:
            # Flatten newlines — a raw one breaks the osascript string.
            await self._desktop(" ".join(message.split()))

    async def _play(self, kind: str) -> None:
        try:
            argv = self._player_argv(kind)
            if argv is None:
                self._bell()
                return
            result = await run_proc(list(argv), timeout=_PLAY_TIMEOUT_S)
            if result.exit_code != 0:
                self._bell()
        except Exception:
            with contextlib.suppress(Exception):  # notifications must never raise
                self._bell()

    async def _desktop(self, message: str) -> None:
        try:
            argv = self._desktop_argv(message)
            if argv is None:
                return  # no desktop binary — silently skip
            await run_proc(list(argv), timeout=_PLAY_TIMEOUT_S)
        except Exception:  # notifications must never raise
            pass

    def _desktop_argv(self, message: str) -> tuple[str, ...] | None:
        for binary, build in _DESKTOP:
            if self._which(binary):
                return build(_TITLE, message)
        return None

    def _player_argv(self, kind: str) -> tuple[str, ...] | None:
        sound = next((f for f in _SOUNDS[kind] if self._file_exists(f)), None)
        if sound is None:
            return None
        for binary, template in _PLAYERS:
            if self._which(binary):
                volume = str(self._config.volume)
                return tuple(part.format(volume=volume, file=sound) for part in template)
        return None
