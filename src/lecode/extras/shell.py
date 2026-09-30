"""Detect the user's shell for command execution."""

from __future__ import annotations

import os
import shutil

#: Fallback when $SHELL is unset or does not resolve to an executable.
FALLBACK_SHELL = "/bin/sh"


def user_shell() -> str:
    """The user's shell: ``$SHELL`` resolved, else ``/bin/sh``.

    ``shutil.which`` accepts absolute paths and bare names alike; an unset,
    empty, or non-executable ``$SHELL`` falls back to ``/bin/sh`` — the
    previously hardcoded behavior.
    """
    shell = os.environ.get("SHELL", "")
    resolved = shutil.which(shell) if shell else None
    return resolved or FALLBACK_SHELL
