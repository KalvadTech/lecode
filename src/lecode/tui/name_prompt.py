"""The startup session-name prompt.

Interactive startup always begins here: a single inline prompt_toolkit line,
looped until :func:`~lecode.session.naming.validate_name` passes. Duplicates
are suffixed via :func:`~lecode.session.naming.unique_name` (``foo`` →
``foo-2``) with a notice. Ctrl-C/Ctrl-D abort with ``None`` — the caller
exits 0 before any session file is created.

Also home to :func:`pick_session`, the inline picker a bare ``-r/--resume``
opens: the folder's sessions are listed numbered and the user picks one by
number, name, or id prefix (empty input takes the most recent).
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession

from lecode.session.model import MetaRecord
from lecode.session.naming import unique_name, validate_name
from lecode.session.storage import SessionStore


async def prompt_session_name(
    store: SessionStore,
    *,
    prompt_text: str = "Session name: ",
    session: PromptSession | None = None,
) -> str | None:
    """Loop until a valid name is entered; ``None`` on Ctrl-C/Ctrl-D.

    ``session`` lets tests inject ``PromptSession(input=create_pipe_input())``.
    """
    prompt = session or PromptSession()
    while True:
        try:
            text = await prompt.prompt_async(prompt_text)
        except (KeyboardInterrupt, EOFError):
            return None
        error = validate_name(text)
        if error is not None:
            print(f"error: {error}")
            continue
        name = unique_name(text, store)
        if name != text.strip():
            print(f"name taken, using '{name}'")
        return name


def folder_sessions(store: SessionStore, cwd: Path | str) -> list[MetaRecord]:
    """Sessions created in ``cwd``, most recent first."""
    return store.list_sessions(cwd)


async def pick_session(
    store: SessionStore,
    cwd: Path | str,
    *,
    prompt_text: str = "Resume session [1]: ",
    session: PromptSession | None = None,
) -> MetaRecord | None:
    """List ``cwd``'s sessions and prompt for one; ``None`` on abort/none.

    Accepts a list number, an exact name, or a unique id prefix; empty input
    picks the most recent. Loops until the choice resolves.
    """
    sessions = folder_sessions(store, cwd)
    if not sessions:
        print(f"no sessions for {cwd} — start one without -r")
        return None
    print(f"sessions in {cwd}:")
    for i, meta in enumerate(sessions, 1):
        print(f"  {i}) {meta.name} · {meta.id[:8]} · {meta.created_at[:10]}")
    prompt = session or PromptSession()
    while True:
        try:
            text = await prompt.prompt_async(prompt_text)
        except (KeyboardInterrupt, EOFError):
            return None
        text = text.strip()
        if not text:
            return sessions[0]
        if text.isdigit() and 1 <= int(text) <= len(sessions):
            return sessions[int(text) - 1]
        matches = [m for m in sessions if m.name == text or m.id.startswith(text)]
        if len(matches) == 1:
            return matches[0]
        print("error: enter a list number, an exact name, or a unique id prefix")
