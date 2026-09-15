"""Markdown memory tools and bounded, source-linked memory_recall.

Targets are ``long_term`` (MEMORY.md), ``daily``, ``scratchpad``, and
``note:<name>``. The store lives on ``ctx.extras["memory"]`` (put there by
``build_runtime`` when ``[memory] enabled = true``); tools return an
explanatory error when memory is disabled or no store is attached.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.memory.facts import FactStore
from lecode.memory.recall import RecallContext
from lecode.memory.store import MemoryStore, resolve_project_root
from lecode.session.storage import SourceRef

TARGETS = ("long_term", "daily", "scratchpad", "note:<name>")


@contextmanager
def _markdown_guard(ctx: ToolContext):
    if not ctx.config.memory.enabled:
        yield
        return
    facts = ctx.extras.get("facts")
    if ctx.recall_context is not None or (
        ctx.memory_generation is not None and not isinstance(facts, FactStore)
    ):
        raise ValueError("durable modification requires a parent context")
    if isinstance(facts, FactStore):
        with facts.guard_generation(ctx.memory_generation):
            yield
    else:
        yield  # Human-authored Markdown-only contexts have no managed fact store.


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
        try:
            with _markdown_guard(ctx):
                return self._mutate(args, ctx)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)

    def _mutate(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
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
        try:
            with _markdown_guard(ctx):
                return self._mutate(args, ctx)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)

    def _mutate(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
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


class MemoryRecallTool(Tool):
    def __init__(self, *, listing: bool = False) -> None:
        super().__init__(
            name="memory_list" if listing else "memory_recall",
            description=(
                "List project fact IDs, revisions, sources and validation status. "
                "Bounded; page by offset."
                if listing
                else "Recall exact source evidence by fact ID or explicit session ID and "
                "bounded sequence range. No global session scan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "start_seq": {"type": "integer"},
                    "end_seq": {"type": "integer"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
        )
        if listing:
            self.parameters = {
                "type": "object",
                "properties": {"offset": {"type": "integer", "minimum": 0}},
                "additionalProperties": False,
            }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.config.memory.enabled:
            return _unavailable(ctx)
        recall = ctx.recall_context
        if recall is None and ctx.session_store is not None:
            recall = RecallContext(
                ctx.session_store,
                ctx.extras.get("facts"),
                ctx.project_root or resolve_project_root(ctx.cwd),
                session_id=ctx.session.id if ctx.session is not None else None,
            )
        if recall is None:
            return _unavailable(ctx)
        try:
            return ToolResult(
                recall.list_facts(args) if self.name == "memory_list" else recall.recall(args)
            )
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)


class MemoryChangeTool(Tool):
    """Explicit selected-fact mutations, routed through normal permissions and hooks."""

    def __init__(self, action: str) -> None:
        properties: dict[str, Any] = {"fact_id": {"type": "string", "pattern": "^[a-f0-9]{64}$"}}
        if action == "correct":
            properties.update(
                {
                    "text": {"type": "string"},
                    "expected_revision": {"type": "integer", "minimum": 1},
                    "source_snapshot": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "seqs": {"type": "array", "items": {"type": "integer"}},
                            "digest": {"type": "string"},
                        },
                        "required": ["session_id", "seqs", "digest"],
                        "additionalProperties": False,
                    },
                }
            )
        super().__init__(
            name=f"memory_{action}",
            description=(
                "Correct one fact by ID and expected revision with an explicit persisted "
                "source_snapshot (session_id, seqs, digest from memory_recall)."
                if action == "correct"
                else "Forget one selected fact and its revisions. Excludes all contributing "
                "sources from managed memory; raw history and Markdown remain."
            ),
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.config.memory.enabled:
            return _unavailable(ctx)
        facts = ctx.extras.get("facts")
        if not isinstance(facts, FactStore) or ctx.session is None or ctx.session_store is None:
            return ToolResult(
                "error: durable modification requires a parent session", is_error=True
            )
        if set(args) != set(self.parameters["required"]):
            return ToolResult(
                "error: supply exactly the required selected-fact arguments", is_error=True
            )
        fact_id = args["fact_id"]
        if not isinstance(fact_id, str) or not re.fullmatch(r"[a-f0-9]{64}", fact_id):
            return ToolResult("error: invalid fact ID", is_error=True)
        try:
            if self.name == "memory_forget":
                changed = facts.forget(fact_id)
                for session_id in facts.pending_forget_sessions(fact_id):
                    ctx.session_store.flush_forgets(
                        ctx.session if session_id == ctx.session.id else session_id
                    )
                return ToolResult(
                    json.dumps({"fact_id": fact_id, "status": "forgotten", "changed": changed})
                )
            data = args["source_snapshot"]
            if not isinstance(data, dict) or set(data) != {"session_id", "seqs", "digest"}:
                raise ValueError("invalid source_snapshot")
            if not isinstance(data["seqs"], list) or not isinstance(data["digest"], str):
                raise ValueError("invalid source_snapshot")
            ref = SourceRef(data["session_id"], tuple(data["seqs"]), data["digest"])
            fact = facts.correct(
                fact_id,
                args["text"],
                expected_revision=args["expected_revision"],
                ref=ref,
                sessions=ctx.session_store,
                project_root=ctx.project_root or resolve_project_root(ctx.cwd),
                expected_generation=ctx.memory_generation,
            )
            return ToolResult(json.dumps({"fact_id": fact.id, "revision": fact.revision}))
        except (ValueError, KeyError) as e:
            return ToolResult(f"error: {e}", is_error=True)


def memory_tools() -> list[Tool]:
    """Memory readers and writers, for registry assembly."""
    return [
        MemoryWriteTool(),
        MemoryEditTool(),
        MemoryReadTool(),
        MemorySearchTool(),
        MemoryRecallTool(),
        MemoryRecallTool(listing=True),
        MemoryChangeTool("correct"),
        MemoryChangeTool("forget"),
    ]
