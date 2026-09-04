"""The ``grep`` tool: regex search over ``rg --json``."""

from __future__ import annotations

import json
import shutil

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.extras.proc import run_proc

RG_PATH = shutil.which("rg") or "rg"

DEFAULT_MAX_RESULTS = 100
MAX_RESULTS = 500


class GrepTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="grep",
            description=(
                "Search file contents with a regex (ripgrep). Supports glob filters, "
                "case-insensitive search, and context lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory/file to search"},
                    "glob": {"type": "string", "description": "File filter, e.g. '*.py'"},
                    "ignore_case": {"type": "boolean"},
                    "context": {"type": "integer", "description": "Context lines (-C)"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = str(args["pattern"])
        max_results = min(MAX_RESULTS, int(args.get("max_results") or DEFAULT_MAX_RESULTS))
        argv = [RG_PATH, "--json", "--color=never", "--regexp", pattern]
        if args.get("ignore_case"):
            argv.append("--ignore-case")
        if args.get("glob"):
            argv += ["--glob", str(args["glob"])]
        context = int(args.get("context") or 0)
        if context > 0:
            argv += ["--context", str(context)]
        path = str(args.get("path") or ".")
        search_path = ctx.cwd / path if not path.startswith("/") else path
        argv.append(str(search_path))

        result = await run_proc(argv, cwd=ctx.cwd, timeout=60)
        if result.timed_out:
            return ToolResult("error: rg timed out", is_error=True)
        if result.exit_code == 1:
            return ToolResult("no matches")
        if result.exit_code >= 2 and not result.stdout.strip():
            return ToolResult(
                f"error: rg exited {result.exit_code}: {result.stderr.strip()[:500]}",
                is_error=True,
            )

        lines: list[str] = []
        match_count = 0
        for raw in result.stdout.splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            data = event.get("data", {})
            text = (data.get("lines", {}).get("text") or "").rstrip("\n")
            file_path = data.get("path", {}).get("text", "?")
            line_no = data.get("line_number")
            if kind == "match":
                match_count += 1
                if match_count > max_results:
                    continue
                lines.append(f"{file_path}:{line_no}: {text}")
            elif kind == "context":
                lines.append(f"{file_path}-{line_no}- {text}")
        if match_count > max_results:
            lines.append(f"… {match_count - max_results} more matches (raise max_results)")
        if not lines:
            return ToolResult("no matches")
        return ToolResult("\n".join(lines), metadata={"match_count": match_count})


def make_tool() -> Tool:
    return GrepTool()
