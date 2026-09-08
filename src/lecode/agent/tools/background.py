"""The ``tasks_*`` tools: inspect and control background tasks.

Background tasks are started by ``bash``/``task`` with
``run_in_background: true`` and owned by the manager installed under
``ctx.extras["background"]``; without a manager these tools report a clear
error (e.g. inside subagent children, whose extras are fresh).
"""

from __future__ import annotations

from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.extras.background import (
    BACKGROUND_EXTRA,
    BackgroundError,
    BackgroundTask,
    BackgroundTaskManager,
)

_UNAVAILABLE = "error: background tasks are unavailable in this context"

#: Default/output caps for tasks_output and tasks_wait.
DEFAULT_OUTPUT_TAIL_BYTES = 4_000
DEFAULT_WAIT_TIMEOUT_S = 30.0
MAX_WAIT_TIMEOUT_S = 600.0


def _manager(ctx: ToolContext) -> BackgroundTaskManager | None:
    return ctx.extras.get(BACKGROUND_EXTRA)


def format_age(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m{total % 60:02d}s"


def format_row(record: BackgroundTask) -> str:
    """One ``/tasks``/``tasks_list`` row: id, kind, status, age, description."""
    exit_part = f", exit {record.exit_code}" if record.exit_code is not None else ""
    return (
        f"{record.id}  {record.kind}  {record.status}{exit_part}  "
        f"{format_age(record.age_s)}  {record.description}"
    )


def _final_report(record: BackgroundTask) -> str:
    output = record.output.strip()
    exit_part = f", exit {record.exit_code}" if record.exit_code is not None else ""
    head = f"{record.id} {record.status}{exit_part} after {format_age(record.age_s)}"
    return f"{head}\n{output}" if output else head


class TasksListTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="tasks_list",
            description="List background tasks (id, kind, status, age, description).",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        manager = _manager(ctx)
        if manager is None:
            return ToolResult(_UNAVAILABLE, is_error=True)
        records = manager.tasks()
        if not records:
            return ToolResult("(no background tasks)")
        return ToolResult("\n".join(format_row(record) for record in records))


class TasksOutputTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="tasks_output",
            description="Read the tail of a background task's output.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Task id (e.g. bg-1)"},
                    "tail": {
                        "type": "number",
                        "description": (
                            f"Bytes from the end (default {DEFAULT_OUTPUT_TAIL_BYTES})"
                        ),
                    },
                },
                "required": ["id"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        manager = _manager(ctx)
        if manager is None:
            return ToolResult(_UNAVAILABLE, is_error=True)
        tail = int(args.get("tail") or DEFAULT_OUTPUT_TAIL_BYTES)
        try:
            text = manager.output(str(args["id"]), tail_bytes=max(1, tail))
        except BackgroundError as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(text.rstrip("\n") or "(no output yet)")


class TasksStopTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="tasks_stop",
            description=("Stop a running background task (SIGTERM, then SIGKILL after a grace)."),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Task id (e.g. bg-1)"},
                },
                "required": ["id"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        manager = _manager(ctx)
        if manager is None:
            return ToolResult(_UNAVAILABLE, is_error=True)
        try:
            record = await manager.stop(str(args["id"]))
        except BackgroundError as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(_final_report(record))


class TasksWaitTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="tasks_wait",
            description="Wait for a background task to finish and return its output.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Task id (e.g. bg-1)"},
                    "timeout": {
                        "type": "number",
                        "description": (
                            f"Seconds (default {DEFAULT_WAIT_TIMEOUT_S}, max {MAX_WAIT_TIMEOUT_S})"
                        ),
                    },
                },
                "required": ["id"],
            },
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        manager = _manager(ctx)
        if manager is None:
            return ToolResult(_UNAVAILABLE, is_error=True)
        timeout = min(MAX_WAIT_TIMEOUT_S, float(args.get("timeout") or DEFAULT_WAIT_TIMEOUT_S))
        try:
            record = await manager.wait(str(args["id"]), timeout)
        except BackgroundError as e:
            return ToolResult(f"error: {e}", is_error=True)
        if record.status == "running":
            return ToolResult(f"{record.id} still running after {timeout}s")
        return ToolResult(_final_report(record))


def make_tools() -> list[Tool]:
    return [TasksListTool(), TasksOutputTool(), TasksStopTool(), TasksWaitTool()]
