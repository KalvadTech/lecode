"""``/memory`` slash-command handlers (Phase 9 routes to these).

Plain functions returning renderable text; no I/O beyond the store.
"""

from __future__ import annotations

from itertools import groupby

from lecode.memory.store import MemoryStore

USAGE = "usage: /memory show | edit | search <pattern> | log [date] | notes"


def _render_hits(store: MemoryStore, pattern: str) -> str:
    try:
        hits = store.search(pattern)
    except ValueError as e:
        return f"error: {e}"
    if not hits:
        return "(no matches)"
    lines: list[str] = []
    for file, group in groupby(hits, key=lambda h: h.file):
        file_hits = list(group)
        lines.append(f"{file} ({len(file_hits)} hits):")
        for hit in file_hits:
            lines.append(f"  {hit.line_no}: {hit.snippet}")
    return "\n".join(lines)


def memory_command(args: list[str], store: MemoryStore) -> str:
    """Handle a ``/memory`` invocation; returns text for the feed."""
    sub = args[0] if args else "show"
    if sub == "show":
        content = store.read_long_term()
        return content if content.strip() else "(long-term memory is empty)"
    if sub == "edit":
        content = store.read_long_term(capped=False)
        # The TUI (Phase 8/9) opens $EDITOR on this; for now return it as-is.
        return "Edit MEMORY.md in your editor and save to apply. Current content:\n\n" + (
            content if content.strip() else "(empty)"
        )
    if sub == "search":
        if len(args) < 2:
            return "usage: /memory search <pattern>"
        return _render_hits(store, " ".join(args[1:]))
    if sub == "log":
        day = args[1] if len(args) > 1 else None
        content = store.read_daily(day)
        label = day or "today"
        return content if content.strip() else f"(no daily log for {label})"
    if sub == "notes":
        notes = store.list_notes()
        if not notes:
            return "(no notes)"
        return "\n".join(f"- {name}" for name in notes)
    return USAGE
