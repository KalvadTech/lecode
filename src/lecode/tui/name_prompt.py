"""The startup session-name prompt.

Interactive startup always begins here: a single inline prompt_toolkit line,
looped until :func:`~lecode.session.naming.validate_name` passes. Duplicates
are suffixed via :func:`~lecode.session.naming.unique_name` (``foo`` →
``foo-2``) with a notice. Ctrl-C/Ctrl-D abort with ``None`` — the caller
exits 0 before any session file is created.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession

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
