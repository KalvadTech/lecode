"""The ``todo_write`` tool: an in-session task list (stored on the context)."""

from __future__ import annotations

from lecode.agent.tools.base import Tool, ToolContext, ToolResult

VALID_STATUSES = ("pending", "in_progress", "done")
_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}


def render_todos(todos: list[dict]) -> str:
    if not todos:
        return "(no todos)"
    lines = []
    for i, todo in enumerate(todos, start=1):
        marker = _MARKERS.get(str(todo.get("status")), "[ ]")
        lines.append(f"{i}. {marker} {todo.get('title', '(untitled)')}")
    return "\n".join(lines)


class TodoWriteTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="todo_write",
            description="Replace the session todo list; renders the updated list back.",
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "status": {"type": "string", "enum": list(VALID_STATUSES)},
                            },
                            "required": ["title", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        todos = args.get("todos")
        if not isinstance(todos, list):
            return ToolResult("error: todos must be a list", is_error=True)
        cleaned: list[dict] = []
        for todo in todos:
            if not isinstance(todo, dict) or not todo.get("title"):
                return ToolResult("error: each todo needs a title", is_error=True)
            status = str(todo.get("status", "pending"))
            if status not in VALID_STATUSES:
                return ToolResult(
                    f"error: invalid status '{status}' (want one of {', '.join(VALID_STATUSES)})",
                    is_error=True,
                )
            cleaned.append({"title": str(todo["title"]), "status": status})
        ctx.todos.clear()
        ctx.todos.extend(cleaned)
        return ToolResult(render_todos(ctx.todos))


def make_tool() -> Tool:
    return TodoWriteTool()
