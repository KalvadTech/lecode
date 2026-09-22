"""Best-effort lifecycle reporting for lecode running inside Herdr."""

from __future__ import annotations

import contextlib
import itertools
import os
import subprocess

SOURCE = "herdr:lecode"
AGENT = "lecode"
_TIMEOUT_S = 2.0
_sequence = itertools.count(1)


def report(state: str, *, message: str | None = None, session_id: str | None = None) -> None:
    """Report lifecycle state to Herdr, or do nothing outside a Herdr pane."""
    target = _target()
    if target is None:
        return
    binary, pane_id = target
    argv = [
        binary,
        "pane",
        "report-agent",
        pane_id,
        "--source",
        SOURCE,
        "--agent",
        AGENT,
        "--state",
        state,
        "--seq",
        str(next(_sequence)),
    ]
    if message:
        argv.extend(["--message", message])
    if session_id:
        argv.extend(["--agent-session-id", session_id])
    _run(argv)


def release() -> None:
    """Release Herdr lifecycle authority when the lecode session exits."""
    target = _target()
    if target is None:
        return
    binary, pane_id = target
    _run(
        [
            binary,
            "pane",
            "release-agent",
            pane_id,
            "--source",
            SOURCE,
            "--agent",
            AGENT,
        ]
    )


def _target() -> tuple[str, str] | None:
    if os.environ.get("HERDR_ENV") != "1":
        return None
    binary = os.environ.get("HERDR_BIN_PATH")
    pane_id = os.environ.get("HERDR_PANE_ID")
    return (binary, pane_id) if binary and pane_id else None


def _run(argv: list[str]) -> None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(argv, check=False, capture_output=True, timeout=_TIMEOUT_S)
