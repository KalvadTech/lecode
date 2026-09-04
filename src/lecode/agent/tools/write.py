"""The ``write`` tool: atomic file writes (tmp file + rename)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.lsp.tool import diagnostics_section


class WriteTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="write",
            description="Write content to a file atomically, creating parent directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = Path(str(args["path"]))
        if not path.is_absolute():
            path = ctx.cwd / path
        content = str(args["content"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            return ToolResult(f"error: {e}", is_error=True)
        ctx.read_paths.add(str(path))  # written files count as seen (edit guard)
        result = f"wrote {len(content.encode('utf-8'))} bytes to {path}"
        return ToolResult(result + await diagnostics_section(ctx, path))


def make_tool() -> Tool:
    return WriteTool()
