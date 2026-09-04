"""The four memory tools: memory_write / memory_edit / memory_read / memory_search.

Targets are ``long_term`` (MEMORY.md), ``daily``, ``scratchpad``, and
``note:<name>``. The store lives on ``ctx.extras["memory"]`` (put there by
``build_runtime`` when ``[memory] enabled = true``); all four tools return an
explanatory error when memory is disabled or no store is attached.
"""

from __future__ import annotations

from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.memory.store import MemoryStore

TARGETS = ("long_term", "daily", "scratchpad", "note:<name>")


def _store(ctx: ToolContext) -> MemoryStore | None:
    if not ctx.config.memory.enabled:
        return None
    store = ctx.extras.get("memory")
    return store if isinstance(store, MemoryStore) else None


def _unavailable(ctx: ToolContext) -> ToolResult:
    if not ctx.config.memory.enabled:
        return ToolResult("error: memory is disabled ([memory] enabled = false)", is_error=True)
    return ToolResult("error: memory store is not available in this context", is_error=True)


def _split_note(target: str) -> str | None:
    """The note name when ``target`` is ``note:<name>``, else ``None``."""
    return target[5:] if target.startswith("note:") else None


def _unknown_target(target: str) -> ToolResult:
    return ToolResult(f"error: unknown target {target!r} (want one of {TARGETS})", is_error=True)


def _page(content: str, offset: int, limit: int | None) -> str:
    """1-based line pagination."""
    lines = content.splitlines()
    if offset > 1:
        lines = lines[offset - 1 :]
    if limit is not None:
        lines = lines[:limit]
    return "\n".join(lines)


class MemoryWriteTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="memory_write",
            description="Write persistent memory: long_term | daily | scratchpad | note:<name>.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["append", "overwrite"]},
                },
                "required": ["content"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        if store is None:
            return _unavailable(ctx)
        content = str(args.get("content", ""))
        if not content.strip():
            return ToolResult("error: content must not be empty", is_error=True)
        mode = str(args.get("mode", "append"))
        if mode not in ("append", "overwrite"):
            return ToolResult(f"error: invalid mode {mode!r} (append|overwrite)", is_error=True)
        target = str(args.get("target", "long_term"))
        try:
            note = _split_note(target)
            if note is not None:
                existing = store.read_note(note)
                if mode == "append" and existing is not None:
                    store.write_note(note, existing.rstrip("\n") + "\n\n" + content)
                else:
                    store.write_note(note, content)
                return ToolResult(f"note {note!r} written ({mode})")
            if target == "long_term":
                if mode == "append":
                    store.append_long_term(content)
                else:
                    store.write_long_term(content)
            elif target == "daily":
                if mode == "append":
                    store.append_daily(content)
                else:
                    store.write_daily(content)
            elif target == "scratchpad":
                if mode == "append":
                    current = store.read_scratchpad().rstrip("\n")
                    store.write_scratchpad(f"{current}\n{content}" if current else content)
                else:
                    store.write_scratchpad(content)
            else:
                return _unknown_target(target)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(f"{target} updated ({mode})")


class MemoryEditTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="memory_edit",
            description="Exact-match replace in MEMORY.md (target long_term) or a note:<name>.",
            parameters={
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["old", "new"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        if store is None:
            return _unavailable(ctx)
        old, new = str(args["old"]), str(args["new"])
        target = str(args.get("target", "long_term"))
        try:
            note = _split_note(target)
            if note is not None:
                content = store.read_note(note)
                if content is None:
                    return ToolResult(f"error: no such note: {note!r}", is_error=True)
                store.write_note(note, MemoryStore._replace_unique(content, old, new))
                return ToolResult(f"note {note!r} edited")
            if target != "long_term":
                return _unknown_target(target)
            store.edit_long_term(old, new)
            return ToolResult("long-term memory edited")
        except KeyError:
            return ToolResult(f"error: text not found: {old[:80]!r}", is_error=True)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)


class MemoryReadTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="memory_read",
            description="Read memory: long_term | daily | scratchpad | note:<name>.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "date": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        if store is None:
            return _unavailable(ctx)
        target = str(args.get("target", "long_term"))
        day = args.get("date")
        try:
            note = _split_note(target)
            if note is not None:
                content = store.read_note(note)
                if content is None:
                    return ToolResult(f"error: no such note: {note!r}", is_error=True)
            elif target == "long_term":
                content = store.read_long_term(capped=False)
            elif target == "daily":
                content = store.read_daily(day)
            elif target == "scratchpad":
                content = store.read_scratchpad()
            else:
                return _unknown_target(target)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        offset = max(1, int(args.get("offset") or 1))
        limit = args.get("limit")
        paged = _page(content, offset, int(limit) if limit is not None else None)
        label = f"{target} ({day})" if day else target
        if not paged:
            return ToolResult(f"({label} is empty)")
        return ToolResult(paged, metadata={"target": target, "bytes": len(paged.encode())})


class MemorySearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="memory_search",
            description="Regex keyword search across all memory files (case-insensitive).",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        if store is None:
            return _unavailable(ctx)
        try:
            hits = store.search(str(args["pattern"]))
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        if not hits:
            return ToolResult("(no matches)")
        file_counts: dict[str, int] = {}
        for hit in hits:
            file_counts[hit.file] = file_counts.get(hit.file, 0) + 1
        lines: list[str] = []
        current_file = None
        for hit in hits:
            if hit.file != current_file:
                current_file = hit.file
                lines.append(f"{hit.file} ({file_counts[hit.file]} hits):")
            lines.append(f"  {hit.line_no}: {hit.snippet}")
        return ToolResult("\n".join(lines), metadata={"hits": len(hits)})


def memory_tools() -> list[Tool]:
    """The four memory tools, for registry assembly."""
    return [MemoryWriteTool(), MemoryEditTool(), MemoryReadTool(), MemorySearchTool()]
