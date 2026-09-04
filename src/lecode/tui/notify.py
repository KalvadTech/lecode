"""Audio notifications: afplay/paplay/aplay with a terminal-bell fallback.

Best-effort: every failure mode (no player, no sound file, player errors)
falls back to the bell or silence — nothing here ever raises. afplay gets
``-v <volume>`` from ``[notifications].volume``; paplay/aplay and the bell
have no volume knob. Per-event toggles: ``on_finish`` / ``on_error`` /
``on_approval``.
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

_PLAY_TIMEOUT_S = 5.0


def _stdout_bell() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


class Notifier:
    """Plays short sounds for turn finish / error / approval-needed events."""

    def __init__(
        self,
        config: NotificationsConfig,
        *,
        which: Callable[[str], str | None] = shutil.which,
        file_exists: Callable[[str], bool] = os.path.exists,
        bell: Callable[[], None] = _stdout_bell,
    ) -> None:
        self._config = config
        self._which = which
        self._file_exists = file_exists
        self._bell = bell

    async def task_finish(self) -> None:
        if self._config.on_finish:
            await self._play("finish")

    async def error(self) -> None:
        if self._config.on_error:
            await self._play("error")

    async def approval_needed(self) -> None:
        if self._config.on_approval:
            await self._play("approval")

    async def _play(self, kind: str) -> None:
        if not self._config.enabled:
            return
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

    def _player_argv(self, kind: str) -> tuple[str, ...] | None:
        sound = next((f for f in _SOUNDS[kind] if self._file_exists(f)), None)
        if sound is None:
            return None
        for binary, template in _PLAYERS:
            if self._which(binary):
                volume = str(self._config.volume)
                return tuple(part.format(volume=volume, file=sound) for part in template)
        return None
