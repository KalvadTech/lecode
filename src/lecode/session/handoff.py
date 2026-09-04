"""Session handoff: seed a new named session with a focused brief.

Phase 4 implements the plumbing — :func:`build_handoff_prompt` produces a
deterministic markdown brief. LLM-based synthesis lands with ``/handoff``.
"""

from __future__ import annotations

import json
from typing import Any

from lecode.session.model import MessageRecord
from lecode.session.storage import Session, SessionStore

#: Tool calls whose ``path`` argument counts as a touched file.
_FILE_TOOLS = ("read", "write", "edit")

_SNIPPET_LEN = 200
_DIGEST_MESSAGES = 6


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content or "").strip()


def files_touched(messages: list[MessageRecord]) -> list[str]:
    """Paths from read/write/edit tool calls, in first-seen order."""
    files: list[str] = []
    for record in messages:
        for call in record.message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") not in _FILE_TOOLS:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            path = args.get("path")
            if path and path not in files:
                files.append(path)
    return files


def build_handoff_prompt(messages: list[MessageRecord], goal_hint: str | None = None) -> str:
    """A markdown brief: goal hint, recent-exchange digest, files touched."""
    lines = ["# Session handoff", "", "## Goal", "", goal_hint or "(no goal hint provided)", ""]
    lines += ["## Recent context", ""]
    if messages:
        for record in messages[-_DIGEST_MESSAGES:]:
            text = _text_of(record.message) or "(tool calls)"
            if len(text) > _SNIPPET_LEN:
                text = text[:_SNIPPET_LEN] + "…"
            lines.append(f"- **{record.role}**: {text}")
    else:
        lines.append("- (empty session)")
    lines += ["", "## Files touched", ""]
    files = files_touched(messages)
    lines += [f"- `{f}`" for f in files] if files else ["- (none recorded)"]
    return "\n".join(lines) + "\n"


def handoff(source: Session, store: SessionStore, new_name: str) -> Session:
    """Create a new named session seeded with a brief of ``source``."""
    from lecode.session.naming import unique_name

    prompt = build_handoff_prompt(store.load_messages(source))
    session = store.create(
        name=unique_name(new_name, store),
        cwd=source.meta.cwd,
        model=source.meta.model,
        agent=source.meta.agent,
    )
    store.append_message(session, {"role": "user", "content": prompt})
    return session
