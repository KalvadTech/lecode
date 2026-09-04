"""The ``find_files`` tool: glob file search over ``fd``."""

from __future__ import annotations

import shutil
from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.extras.proc import run_proc

FD_PATH = shutil.which("fd") or "fd"

DEFAULT_MAX_RESULTS = 200


class FindFilesTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="find_files",
            description="Find files by glob pattern (fd). Faster than globbing by hand.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob, e.g. '*.py'"},
                    "path": {"type": "string", "description": "Root directory (default cwd)"},
                    "type": {"type": "string", "enum": ["file", "directory"]},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = str(args["pattern"])
        max_results = int(args.get("max_results") or DEFAULT_MAX_RESULTS)
        argv = [FD_PATH, "--color=never", "--glob", "--max-results", str(max_results + 1)]
        if args.get("type") == "file":
            argv += ["--type", "f"]
        elif args.get("type") == "directory":
            argv += ["--type", "d"]
        argv.append(pattern)
        root = str(args.get("path") or ".")
        argv.append(str(ctx.cwd / root if not root.startswith("/") else root))

        result = await run_proc(argv, cwd=ctx.cwd, timeout=30)
        if result.timed_out:
            return ToolResult("error: fd timed out", is_error=True)
        if result.exit_code != 0:
            return ToolResult(
                f"error: fd exited {result.exit_code}: {result.stderr.strip()[:500]}",
                is_error=True,
            )
        entries = [line for line in result.stdout.splitlines() if line]
        if not entries:
            return ToolResult("no files found")
        truncated = len(entries) > max_results
        entries = entries[:max_results]
        lines = [str(Path(entry)) for entry in entries]
        if truncated:
            lines.append("… more results (raise max_results)")
        return ToolResult("\n".join(lines), metadata={"count": len(entries)})


def make_tool() -> Tool:
    return FindFilesTool()
