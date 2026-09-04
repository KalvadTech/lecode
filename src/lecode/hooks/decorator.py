"""Decorating tools with lifecycle hooks at build time.

The permission checker runs first — ``ToolRegistry._execute`` returns early on
a checker Deny without ever calling ``tool.run``, so decorated hooks are never
consulted for checker-denied calls and can therefore only narrow. PreToolUse
fires before execution (Deny blocks, Ask requires approval unless the context
auto-approves, ``rewritten_input`` replaces the tool args); PostToolUse fires
after execution and is informational — its verdict is recorded in the result
metadata and cannot undo anything.
"""

from __future__ import annotations

from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from lecode.hooks.runner import HookDispatcher


def _wrap(tool: Tool, dispatcher: HookDispatcher) -> None:
    if getattr(tool, "_hooks_applied", False):
        return  # never double-wrap
    original_run = tool.run

    async def hooked_run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pre = await dispatcher.pre_tool_use(tool.name, args)
        if pre.verdict == "deny":
            return ToolResult(
                f"denied by hook: {pre.reason or 'PreToolUse hook'}",
                is_error=True,
                metadata={"hook_verdict": "deny"},
            )
        if pre.verdict == "ask" and not ctx.auto_approve:
            return ToolResult(
                f"denied: hook requires approval ({pre.reason or 'PreToolUse hook'})",
                is_error=True,
                metadata={"needs_approval": True, "hook_verdict": "ask"},
            )
        effective_args = pre.rewritten_input if pre.rewritten_input is not None else args
        result = await original_run(effective_args, ctx)
        post = await dispatcher.post_tool_use(
            tool.name, effective_args, result.content, result.is_error
        )
        if post.verdict != "allow" or post.reason:
            result.metadata["post_hook"] = {"verdict": post.verdict, "reason": post.reason}
        return result

    tool.run = hooked_run
    tool._hooks_applied = True  # type: ignore[attr-defined]


def apply_hooks(registry: ToolRegistry, dispatcher: HookDispatcher) -> ToolRegistry:
    """Wrap every tool in ``registry`` with the hook pipeline (in place)."""
    for name in registry.names():
        tool = registry.get(name)
        if tool is not None:
            _wrap(tool, dispatcher)
    return registry
