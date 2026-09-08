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

``tool`` is present for tool and permission events, ``result`` for
``PostToolUse``/``PostToolUseFailure``, ``prompt`` only for
``UserPromptSubmit``, ``agent`` only for ``SubagentStart``/``SubagentEnd``.
``reason`` carries the ``Stop`` stop reason (and error text for the
``Notification`` error kind), ``decision`` the ``PermissionResult`` outcome,
``kind`` the ``Notification`` kind. ``session`` is ``None`` outside a session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRE_TOOL_USE = "PreToolUse"
POST_TOOL_USE = "PostToolUse"
POST_TOOL_USE_FAILURE = "PostToolUseFailure"
PERMISSION_REQUEST = "PermissionRequest"
PERMISSION_RESULT = "PermissionResult"
USER_PROMPT_SUBMIT = "UserPromptSubmit"
STOP = "Stop"
SESSION_START = "SessionStart"
SESSION_END = "SessionEnd"
SUBAGENT_START = "SubagentStart"
SUBAGENT_END = "SubagentEnd"
PRE_COMPACT = "PreCompact"
POST_COMPACT = "PostCompact"
INTERRUPT = "Interrupt"
NOTIFICATION = "Notification"

#: All known hook event names, in canonical order.
EVENTS = (
    PRE_TOOL_USE,
    POST_TOOL_USE,
    POST_TOOL_USE_FAILURE,
    PERMISSION_REQUEST,
    PERMISSION_RESULT,
    USER_PROMPT_SUBMIT,
    STOP,
    SESSION_START,
    SESSION_END,
    SUBAGENT_START,
    SUBAGENT_END,
    PRE_COMPACT,
    POST_COMPACT,
    INTERRUPT,
    NOTIFICATION,
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
    decision: str | None = None,
    reason: str | None = None,
    kind: str | None = None,
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
    if decision is not None:
        envelope["decision"] = decision
    if reason is not None:
        envelope["reason"] = reason
    if kind is not None:
        envelope["kind"] = kind
    return envelope
