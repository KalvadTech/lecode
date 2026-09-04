"""Session naming rules.

Interactive startup requires a name (validated non-empty); duplicates are
suffixed ``-2``, ``-3``, … Headless mode uses :func:`auto_name`. The TUI
prompt loops on :func:`validate_name`; aborting is the TUI's job.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lecode.session.storage import SessionStore

MAX_NAME_LENGTH = 64


def validate_name(name: str | None) -> str | None:
    """Return an error message if ``name`` is unusable, else ``None``.

    Rules: non-empty/non-whitespace; no ``/`` or ``\\``; no control chars;
    no leading dot; at most 64 chars.
    """
    if name is None or not name.strip():
        return "session name must not be empty"
    stripped = name.strip()
    if len(stripped) > MAX_NAME_LENGTH:
        return f"session name must be at most {MAX_NAME_LENGTH} characters"
    if stripped.startswith("."):
        return "session name must not start with a dot"
    if "/" in stripped or "\\" in stripped:
        return "session name must not contain path separators"
    if any(ord(c) < 32 or ord(c) == 127 for c in stripped):
        return "session name must not contain control characters"
    return None


def unique_name(wanted: str, store: SessionStore) -> str:
    """A validated, deduplicated name: duplicates are suffixed ``-2``, ``-3``…"""
    name = wanted.strip()
    error = validate_name(name)
    if error is not None:
        raise ValueError(error)
    existing = {m.name for m in store.list_sessions()}
    if name not in existing:
        return name
    index = 2
    while True:
        suffix = f"-{index}"
        candidate = name[: MAX_NAME_LENGTH - len(suffix)] + suffix
        if candidate not in existing:
            return candidate
        index += 1


def auto_name(store: SessionStore | None = None) -> str:
    """Timestamp-based name for headless mode, deduplicated if a store is given."""
    base = "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if store is None:
        return base
    return unique_name(base, store)
