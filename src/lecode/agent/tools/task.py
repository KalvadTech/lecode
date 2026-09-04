"""The ``task`` tool: dispatch a subagent on a self-contained prompt.

The subagent runs a child agent loop with a lean tool registry (no ``task``
/ ``advisor``) under the agent's permission overlay; its final text is the
tool result. Calls are concurrency-safe — parallel ``task`` calls in one
turn run their children in parallel.

Seams (all via ``ctx.extras``): the provider under ``provider`` (installed
by the runner), the tool and agent registries under ``registry`` /
``agents`` (installed by the runtime builder), and the progress callback
under ``subagent_events`` (installed by the TUI).
"""

from __future__ import annotations

from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.extras.subagents import (
    AGENTS_EXTRA,
    REGISTRY_EXTRA,
    SUBAGENT_EVENTS_EXTRA,
    SubagentError,
    run_subagent,
)

#: Subagent used when the call does not name one.
DEFAULT_AGENT = "explore"


class TaskTool(Tool):
    """Run a subagent (default: explore) and return its final answer."""

    def __init__(self) -> None:
        super().__init__(
            name="task",
            description=(
                "Run a subagent on a self-contained task and get its final answer. "
                "Independent tasks can be dispatched in parallel in one turn. "
                f"Default agent: {DEFAULT_AGENT} (read-only codebase search)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The complete, self-contained task for the subagent.",
                    },
                    "agent": {
                        "type": "string",
                        "description": f"Subagent name (default: {DEFAULT_AGENT}).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short human-readable label for the task.",
                    },
                },
                "required": ["prompt"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult("error: task needs a prompt", is_error=True)
        registry = ctx.extras.get(REGISTRY_EXTRA)
        agents = ctx.extras.get(AGENTS_EXTRA)
        if registry is None or agents is None:
            return ToolResult("error: subagents are unavailable in this context", is_error=True)
        try:
            outcome = await run_subagent(
                ctx,
                registry,
                agents,
                name=str(args.get("agent") or DEFAULT_AGENT),
                prompt=prompt,
                on_event=ctx.extras.get(SUBAGENT_EVENTS_EXTRA),
            )
        except SubagentError as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(
            outcome.text or "(subagent returned no text)",
            metadata={
                "agent": outcome.agent,
                "turns": outcome.turns,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "cost_usd": outcome.cost_usd,
            },
        )


def make_tool() -> Tool:
    return TaskTool()
