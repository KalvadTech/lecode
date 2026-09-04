"""Tool framework: Tool, ToolResult, ToolContext, ToolRegistry.

Permission gating lives in :meth:`ToolRegistry.dispatch`: every call is
checked against ``ctx.permission_checker`` first. Ask decisions need an
approver — headless/auto contexts set ``ctx.auto_approve`` to convert
Ask → Allow; the interactive UI sets ``ctx.approval_callback`` (AllowOnce /
AllowAlways(pattern), persisted via session grants / Deny); with neither,
Ask becomes a denial ("requires approval"). Deny is never converted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lecode.config.models import Config
from lecode.permission import AllowAlways, Decision, Deny, PermissionChecker
from lecode.telemetry import capture_exception, record_tool_call

#: Cap on tool-argument JSON size (guard against runaway payloads).
MAX_ARGS_BYTES = 1_000_000


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (object)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


@dataclass
class ToolContext:
    """Everything a tool needs at call time; later phases attach registries."""

    cwd: Path
    config: Config
    permission_checker: PermissionChecker
    session: Any | None = None  # lecode.session.storage.Session, when in a session
    session_store: Any | None = None  # SessionStore; grants persist when set
    session_perms: Any | None = None  # SessionPermissions backing the checker
    auto_approve: bool = False  # headless mode: Ask becomes Allow
    #: Interactive approver: async ``(tool_name, args, reason) -> ApprovalDecision``.
    #: Set by the TUI; ``None`` keeps Ask as a denial ("requires approval").
    approval_callback: Any | None = None
    read_paths: set[str] = field(default_factory=set)  # files read (edit guard)
    todos: list[dict[str, Any]] = field(default_factory=list)  # todo_write state
    extras: dict[str, Any] = field(default_factory=dict)  # advisor/memory/MCP later


def grant_always(ctx: ToolContext, tool: str, pattern: str) -> None:
    """Record a session "allow always" grant, persisting it when possible.

    Grants persist to the session JSONL only when the context carries both a
    session and its store.
    """
    if ctx.session_perms is not None:
        ctx.session_perms.grant(tool, pattern)
    if ctx.session is not None and ctx.session_store is not None:
        ctx.session_store.grant_permission(ctx.session, tool, pattern)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_tool_specs(self) -> list[dict[str, Any]]:
        """The OpenAI ``tools=[...]`` payload for all registered tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in (self._tools[name] for name in self.names())
        ]

    async def dispatch(
        self, call_id: str, name: str, args_json: str, ctx: ToolContext
    ) -> dict[str, Any]:
        """Permission-check, parse, and run one tool call; never raises.

        Returns a ``role: tool`` message dict for the conversation history.
        """
        message, _ = await self.dispatch_result(call_id, name, args_json, ctx)
        return message

    async def dispatch_result(
        self, call_id: str, name: str, args_json: str, ctx: ToolContext
    ) -> tuple[dict[str, Any], ToolResult]:
        """Like :meth:`dispatch`, but also returns the :class:`ToolResult`.

        The runner uses this to report error status through its events
        without leaking it into the wire message.
        """
        result = await self._execute(name, args_json, ctx)
        # Wire content: normally the text; tools may attach content parts
        # (the read tool's image rendering) via result metadata.
        content = result.metadata.get("content_parts") or result.content
        message = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }
        return message, result

    async def _execute(self, name: str, args_json: str, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(f"error: unknown tool '{name}'", is_error=True)
        if len(args_json) > MAX_ARGS_BYTES:
            return ToolResult("error: tool arguments too large", is_error=True)
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError as e:
            return ToolResult(f"error: invalid tool arguments JSON: {e}", is_error=True)
        if not isinstance(args, dict):
            return ToolResult("error: tool arguments must be a JSON object", is_error=True)

        check = ctx.permission_checker.check(name, args)
        if check.decision == Decision.DENY:
            return ToolResult(f"denied: {check.reason}", is_error=True)
        if check.decision == Decision.ASK and not ctx.auto_approve:
            if ctx.approval_callback is None:
                return ToolResult(
                    f"denied: requires approval ({check.reason})",
                    is_error=True,
                    metadata={"needs_approval": True},
                )
            approval = await ctx.approval_callback(name, args, check.reason)
            if isinstance(approval, Deny):
                return ToolResult(f"denied by user ({check.reason})", is_error=True)
            if isinstance(approval, AllowAlways):
                grant_always(ctx, name, approval.pattern)
            # AllowOnce / AllowAlways fall through to running the tool.

        try:
            started = time.monotonic()
            result = await tool.run(args, ctx)
            record_tool_call(name, is_error=result.is_error, duration_s=time.monotonic() - started)
            return result
        except Exception as e:  # tools never crash the loop
            record_tool_call(name, is_error=True, duration_s=time.monotonic() - started)
            capture_exception(e, context=f"tool:{name}")
            return ToolResult(f"error: {type(e).__name__}: {e}", is_error=True)
