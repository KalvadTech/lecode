"""The ``lsp_diagnostics`` tool and the write/edit diagnostics appendix.

Both go through the :class:`~lecode.lsp.manager.LspManager` installed under
``ctx.extras["lsp"]`` by ``build_runtime`` (absent when ``[lsp] enabled =
false``). Everything is fail-open: a total ~3s budget caps the query and any
error yields no output, so LSP trouble never blocks the agent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.lsp.manager import Diagnostic
from lecode.lsp.registry import server_for_file

#: Extra key under which build_runtime installs the LspManager.
LSP_EXTRA = "lsp"

#: Cap on diagnostics shown in a write/edit result.
MAX_SHOWN = 20

#: Total budget for a post-write/edit diagnostics query.
SECTION_TIMEOUT = 3.0


def format_diagnostics(diagnostics: list[Diagnostic], cap: int = MAX_SHOWN) -> str:
    """Compact lines: ``line:col severity message`` (+ an elision line)."""
    lines = [f"{d.line}:{d.col} {d.severity} {d.message}" for d in diagnostics[:cap]]
    if len(diagnostics) > cap:
        lines.append(f"… and {len(diagnostics) - cap} more")
    return "\n".join(lines)


async def diagnostics_section(ctx: ToolContext, path: Path) -> str:
    """The ``## Diagnostics`` appendix for write/edit results ('' when none)."""
    manager = ctx.extras.get(LSP_EXTRA)
    if manager is None:
        return ""
    try:
        diagnostics = await asyncio.wait_for(manager.diagnostics_for(path), timeout=SECTION_TIMEOUT)
    except Exception:
        return ""  # fail-open: LSP problems never block the agent
    if not diagnostics:
        return ""
    return f"\n\n## Diagnostics\n{format_diagnostics(diagnostics)}"


class LspDiagnosticsTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="lsp_diagnostics",
            description=(
                "Get language-server diagnostics (errors, warnings) for a file. "
                "Use after writing or editing code to catch problems early."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file to check."},
                },
                "required": ["path"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = Path(str(args["path"]))
        if not path.is_absolute():
            path = ctx.cwd / path
        manager = ctx.extras.get(LSP_EXTRA)
        if manager is None:
            return ToolResult("LSP is disabled ([lsp] enabled = false)")
        if server_for_file(str(path), ctx.config) is None:
            return ToolResult(f"no LSP server for this filetype: {path.name}")
        diagnostics = await manager.diagnostics_for(path)  # fail-open: may be []
        if not diagnostics:
            return ToolResult(f"no diagnostics for {path}")
        return ToolResult(format_diagnostics(diagnostics))


def make_tool() -> Tool:
    return LspDiagnosticsTool()
