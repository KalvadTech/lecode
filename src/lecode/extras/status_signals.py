"""Status signals: newline-delimited JSON events over a Unix datagram socket.

When ``[signals] enabled = true``, session lifecycle events are sent to
``socket_path`` (default ``<config_dir>/lecode.sock``) as datagrams — no
listener lifecycle to manage: with nobody bound, datagrams simply drop.
Signals are observational only; every failure is swallowed.

Event shape::

    {"event": "start"|"stop"|"git-conflict", "session": "<name>", "ts": "..."}
"""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Known signal events.
START = "start"
STOP = "stop"
GIT_CONFLICT = "git-conflict"

#: Socket file name under the config dir when no explicit path is set.
DEFAULT_SOCKET_NAME = "lecode.sock"


class StatusEmitter:
    """Fire-and-forget lifecycle events to a Unix datagram socket."""

    def __init__(self, config: Any, *, session: str, config_dir: Path | None = None) -> None:
        self._enabled = bool(getattr(config, "enabled", False))
        socket_path = getattr(config, "socket_path", None)
        if socket_path:
            self._path = Path(socket_path)
        else:
            if config_dir is None:
                from lecode.config.loader import config_dir as default_config_dir

                config_dir = default_config_dir()
            self._path = Path(config_dir) / DEFAULT_SOCKET_NAME
        #: Current session name; updated on session switch.
        self.session = session

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def socket_path(self) -> Path:
        return self._path

    def emit(self, event: str, **extra: Any) -> None:
        """Send one event as a JSON datagram; never raises."""
        if not self._enabled:
            return
        payload = {
            "event": event,
            "session": self.session,
            "ts": datetime.now(UTC).isoformat(),
            **extra,
        }
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                sock.sendto(json.dumps(payload).encode() + b"\n", str(self._path))
            finally:
                sock.close()
        except OSError:
            pass  # no listener, bad path, … — signals are best-effort
