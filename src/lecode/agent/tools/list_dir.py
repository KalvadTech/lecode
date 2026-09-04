"""The ``list_dir`` tool: simple directory listing (pure Python)."""

from __future__ import annotations

from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult

MAX_ENTRIES = 500


class ListDirTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="list_dir",
            description="List a directory's entries, directories first.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Default cwd"}},
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = Path(str(args.get("path") or "."))
        if not path.is_absolute():
            path = ctx.cwd / path
        if not path.is_dir():
            return ToolResult(f"error: not a directory: {args.get('path')}", is_error=True)
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except OSError as e:
            return ToolResult(f"error: {e}", is_error=True)
        lines: list[str] = []
        for entry in entries[:MAX_ENTRIES]:
            lines.append(f"{entry.name}/" if entry.is_dir() else entry.name)
        if len(entries) > MAX_ENTRIES:
            lines.append(f"… {len(entries) - MAX_ENTRIES} more entries")
        return ToolResult("\n".join(lines) or "(empty directory)")


def make_tool() -> Tool:
    return ListDirTool()
