"""Control persisted worker runs from their supervising conversation."""

from __future__ import annotations

from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.extras.subagents import SubagentError
from lecode.extras.workers import WORKER_CURRENT_EXTRA, WORKER_EXTRA

_ACTIONS = ("list", "send", "stop", "resume", "submit", "question", "integrate", "cleanup")


class WorkersTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="workers",
            description="List and control delegated workers. Workers can manage descendants only.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": list(_ACTIONS)},
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "interrupt": {"type": "boolean"},
                    "tree": {"type": "boolean"},
                },
                "required": ["action"],
            },
        )

    @staticmethod
    def _descendants(manager: Any, parent_id: str) -> set[str]:
        ids = set()
        pending = [parent_id]
        while pending:
            parent = pending.pop()
            for child in manager.children(parent):
                ids.add(child.id)
                pending.append(child.id)
        return ids

    def _validate(self, args: dict[str, Any]) -> str | None:
        action = args.get("action")
        allowed = {
            "list": {"action"},
            "send": {"action", "id", "text", "interrupt"},
            "stop": {"action", "id", "tree"},
            "resume": {"action", "id", "text"},
            "submit": {"action", "id"},
            "question": {"action", "text"},
            "integrate": {"action"},
            "cleanup": {"action"},
        }
        if action not in _ACTIONS or set(args) - allowed[action]:
            return "invalid workers action or arguments"
        if action in {"send", "stop", "resume", "submit"} and not isinstance(args.get("id"), str):
            return f"workers {action} needs an id"
        if action in {"send", "question"} and not isinstance(args.get("text"), str):
            return f"workers {action} needs text"
        if not isinstance(args.get("interrupt", False), bool) or not isinstance(
            args.get("tree", False), bool
        ):
            return "interrupt and tree must be booleans"
        return None

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        error = self._validate(args)
        if error is not None:
            return ToolResult(f"error: {error}", is_error=True)
        manager = ctx.extras.get(WORKER_EXTRA)
        if manager is None:
            return ToolResult("error: workers are unavailable in this context", is_error=True)
        action = args["action"]
        if action in {"integrate", "cleanup"}:
            return ToolResult(f"error: workers {action} is unavailable", is_error=True)
        current = ctx.extras.get(WORKER_CURRENT_EXTRA)
        if action == "list":
            workers = (
                manager.list()
                if current is None
                else [
                    worker
                    for worker in manager.list()
                    if worker.id in self._descendants(manager, current)
                ]
            )
            if not workers:
                return ToolResult("(no workers)")
            return ToolResult(
                "\n".join(
                    f"{worker.id}  {worker.state}  {worker.agent}  {worker.description}"
                    for worker in workers
                )
            )
        id = args.get("id")
        if action == "question":
            if current is None:
                return ToolResult("error: the main agent has no parent", is_error=True)
            try:
                manager.ask_parent(current, args["text"])
            except (KeyError, SubagentError) as e:
                return ToolResult(f"error: {e}", is_error=True)
            return ToolResult("question sent to parent")
        if current is not None and id not in self._descendants(manager, current):
            return ToolResult("error: workers can manage descendants only", is_error=True)
        try:
            if action == "send":
                message_id = await manager.send(id, args["text"], bool(args.get("interrupt")))
                return ToolResult(f"worker {id} message {message_id} queued")
            if action == "stop":
                await manager.stop(id, bool(args.get("tree")))
                return ToolResult(f"worker {id} stopped")
            if action == "resume":
                await manager.resume(id, args.get("text"))
                return ToolResult(f"worker {id} resumed")
            note = await manager.submit(id)
            return ToolResult(f"worker {id} submitted", metadata={"notification_id": note["id"]})
        except (KeyError, RuntimeError, SubagentError) as e:
            return ToolResult(f"error: {e}", is_error=True)


def make_tool() -> Tool:
    return WorkersTool()
