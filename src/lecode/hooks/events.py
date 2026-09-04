"""Lifecycle hook events and their JSON envelopes.

Every event delivered to a hook handler carries a JSON envelope on stdin:

.. code-block:: json

    {
      "event": "PreToolUse",
      "ts": "2026-09-01T12:00:00+00:00",
      "session": {"id": "...", "name": "..."},
      "tool": {"name": "bash", "args": {"command": "ls"}},
      "cwd": "/path/to/project"
    }

``tool`` is present for ``PreToolUse``/``PostToolUse``, ``result`` only for
``PostToolUse``, ``prompt`` only for ``UserPromptSubmit``, ``agent`` only for
``SubagentStart``/``SubagentEnd``. ``session`` is ``None`` outside a session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRE_TOOL_USE = "PreToolUse"
POST_TOOL_USE = "PostToolUse"
STOP = "Stop"
USER_PROMPT_SUBMIT = "UserPromptSubmit"
SESSION_START = "SessionStart"
SESSION_END = "SessionEnd"
SUBAGENT_START = "SubagentStart"
SUBAGENT_END = "SubagentEnd"

#: All known hook event names, in canonical order.
EVENTS = (
    PRE_TOOL_USE,
    POST_TOOL_USE,
    STOP,
    USER_PROMPT_SUBMIT,
    SESSION_START,
    SESSION_END,
    SUBAGENT_START,
    SUBAGENT_END,
)


def build_envelope(
    event: str,
    cwd: Path | str,
    *,
    session: Any | None = None,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    prompt: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    """Build the JSON envelope delivered to hook handlers on stdin."""
    envelope: dict[str, Any] = {
        "event": event,
        "ts": datetime.now(UTC).isoformat(),
        "session": (
            {"id": getattr(session, "id", None), "name": getattr(session, "name", None)}
            if session is not None
            else None
        ),
        "cwd": str(cwd),
    }
    if tool_name is not None:
        envelope["tool"] = {"name": tool_name, "args": tool_args or {}}
    if result is not None:
        envelope["result"] = result
    if prompt is not None:
        envelope["prompt"] = prompt
    if agent is not None:
        envelope["agent"] = agent
    return envelope
