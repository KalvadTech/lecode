"""``/memory`` slash-command handlers (Phase 9 routes to these).

Plain functions returning renderable text; no I/O beyond the store.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import groupby

from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.memory.recall import RecallContext
from lecode.memory.store import MemoryStore, resolve_project_root

USAGE = (
    "usage: /memory show | edit | search <pattern> | log [date] | notes | facts [offset] | "
    "recall <fact-id> [offset] | read <session-id> <start-seq> <end-seq> [offset] | "
    "forget <fact-id> | correct <fact-id> <revision> <session-id> <start-seq> <end-seq> <text>"
)


async def memory_change_command(args: list[str], ctx: ToolContext, registry: ToolRegistry) -> str:
    """Slash mutations use exactly the tool permission/hook dispatch boundary."""
    if not args:
        return USAGE
    try:
        if args[0] == "forget" and len(args) == 2:
            payload = {"fact_id": args[1]}
        elif args[0] == "correct" and len(args) >= 7:
            if ctx.session_store is None:
                return "error: correction requires a parent session"
            snapshot = ctx.session_store.source_snapshot(
                args[3],
                int(args[4]),
                int(args[5]),
                project_root=ctx.project_root or resolve_project_root(ctx.cwd),
            )
            if snapshot.status != "valid" or snapshot.ref is None:
                return f"error: source is {snapshot.status}"
            payload = {
                "fact_id": args[1],
                "expected_revision": int(args[2]),
                "source_snapshot": asdict(snapshot.ref),
                "text": " ".join(args[6:]),
            }
        else:
            return USAGE
        _, result = await registry.dispatch_result(
            "memory-command", f"memory_{args[0]}", json.dumps(payload), ctx
        )
        return result.content
    except ValueError as e:
        return f"error: {e}"


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


def memory_command(
    args: list[str], store: MemoryStore, *, recall: RecallContext | None = None
) -> str:
    """Handle a ``/memory`` invocation; returns text for the feed."""
    sub = args[0] if args else "show"
    if sub in {"recall", "read", "facts"}:
        if recall is None:
            return "error: source recall is not available in this context"
        try:
            if sub == "facts" and len(args) in {1, 2}:
                return recall.list_facts({"offset": int(args[1]) if len(args) == 2 else 0})
            if sub == "recall" and len(args) in {2, 3}:
                return recall.recall(
                    {"fact_id": args[1], "offset": int(args[2]) if len(args) == 3 else 0}
                )
            if sub == "read" and len(args) in {4, 5}:
                return recall.recall(
                    {
                        "session_id": args[1],
                        "start_seq": int(args[2]),
                        "end_seq": int(args[3]),
                        "offset": int(args[4]) if len(args) == 5 else 0,
                    }
                )
        except ValueError as e:
            return f"error: {e}"
        return USAGE
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
