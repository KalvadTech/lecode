"""The ``task`` tool: dispatch a subagent on a self-contained prompt.

The subagent runs a child agent loop with a lean tool registry (no ``task``)
under the agent's permission overlay; its final text is the
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
from lecode.extras.background import BACKGROUND_EXTRA, BackgroundError
from lecode.extras.subagents import (
    AGENTS_EXTRA,
    REGISTRY_EXTRA,
    SUBAGENT_EVENTS_EXTRA,
    SubagentError,
    run_subagent,
)
from lecode.extras.workers import WORKER_EXTRA
from lecode.extras.worktree import WorktreeError

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
                    "run_in_background": {
                        "type": "boolean",
                        "description": (
                            "Run detached and return a task id immediately; "
                            "track with the tasks_* tools"
                        ),
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
        # The TUI still owns its transient roster through run_subagent.
        manager = (
            None
            if ctx.extras.get(SUBAGENT_EVENTS_EXTRA) is not None
            else ctx.extras.get(WORKER_EXTRA)
        )
        if manager is not None:
            return await self._start_worker(args, ctx, manager, prompt)
        if args.get("run_in_background"):
            return self._start_background(args, ctx, registry, agents, prompt)
        try:
            outcome = await run_subagent(
                ctx,
                registry,
                agents,
                name=str(args.get("agent") or DEFAULT_AGENT),
                prompt=prompt,
                description=str(args.get("description") or ""),
                on_event=ctx.extras.get(SUBAGENT_EVENTS_EXTRA),
            )
        except SubagentError as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(
            outcome.text or "(subagent returned no text)",
            metadata={
                "agent": outcome.agent,
                "run_id": outcome.run_id,
                "turns": outcome.turns,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "cost_usd": outcome.cost_usd,
            },
        )

    async def _start_worker(
        self, args: dict[str, Any], ctx: ToolContext, manager: Any, prompt: str
    ) -> ToolResult:
        agent = str(args.get("agent") or DEFAULT_AGENT)
        description = str(args.get("description") or prompt[:60])
        background = bool(args.get("run_in_background"))
        try:
            worker = await manager.start(
                ctx,
                agent=agent,
                prompt=prompt,
                description=description,
                background=background,
            )
            if background:
                return ToolResult(
                    f"worker {worker.id} started ({agent}): {description}",
                    metadata={"worker_id": worker.id, "agent": agent},
                )
            outcome = await manager.wait(worker.id)
        except (SubagentError, WorktreeError, RuntimeError) as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(
            outcome.final_text or "(subagent returned no text)",
            metadata={
                "agent": agent,
                "worker_id": worker.id,
                "run_id": worker.id,
                "turns": outcome.turns,
                "input_tokens": outcome.usage_totals.input_tokens,
                "output_tokens": outcome.usage_totals.output_tokens,
                "cost_usd": outcome.usage_totals.cost_usd,
            },
        )

    def _start_background(
        self,
        args: dict[str, Any],
        ctx: ToolContext,
        registry: Any,
        agents: Any,
        prompt: str,
    ) -> ToolResult:
        manager = ctx.extras.get(BACKGROUND_EXTRA)
        if manager is None:
            return ToolResult(
                "error: background tasks are unavailable in this context", is_error=True
            )
        agent_name = str(args.get("agent") or DEFAULT_AGENT)
        description = str(args.get("description") or prompt[:60])
        on_event = ctx.extras.get(SUBAGENT_EVENTS_EXTRA)

        async def body(emit: Any) -> tuple[str, int | None]:
            try:
                # run_subagent builds fresh child extras itself — the parent's
                # "conversation" seam is never clobbered.
                outcome = await run_subagent(
                    ctx,
                    registry,
                    agents,
                    name=agent_name,
                    prompt=prompt,
                    description=description,
                    on_event=on_event,
                )
            except SubagentError as e:
                return f"error: {e}", 1
            text = outcome.text or "(subagent returned no text)"
            emit(text.encode("utf-8", errors="replace"))
            return text, 0

        try:
            record = manager.start("agent", description, body)
        except BackgroundError as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(f"background task {record.id} started ({agent_name}): {description}")


def make_tool() -> Tool:
    return TaskTool()
