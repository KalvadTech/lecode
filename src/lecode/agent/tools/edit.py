"""The ``edit`` tool: fuzzy and CRC-anchored search/replace engines.

``fuzzy``: exact match first, then whitespace-normalized; exactly one
occurrence required (0 or 2+ is an error with diagnostics). ``crc``: replace
the span delimited by ``line_no:crc8`` anchors from anchored ``read`` output;
an anchor mismatch means the file changed — re-read before editing.

Editing a file never read (or written) in this session is refused unless
``force: true``.
"""

from __future__ import annotations

from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.agent.tools.read import line_anchor
from lecode.lsp.tool import diagnostics_section


def _normalize(text: str) -> str:
    """Whitespace-normalized form for fuzzy comparison."""
    return " ".join(text.split())


def _fuzzy_replace(
    content: str, old: str, new: str, replace_all: bool
) -> tuple[str | None, str | None]:
    """Return (new_content, error). Exactly one occurrence required."""
    count = content.count(old)
    if count == 1 or (count > 1 and replace_all):
        return content.replace(old, new), None
    if count > 1:
        return None, f"old_string occurs {count} times; pass replace_all or add more context"

    # Whitespace-normalized fallback, line-window based.
    lines = content.splitlines(keepends=True)
    old_lines = old.splitlines()
    if not old_lines:
        return None, "old_string not found"
    norm_old = [_normalize(line) for line in old_lines]
    matches: list[int] = []
    for start in range(len(lines) - len(old_lines) + 1):
        window = lines[start : start + len(old_lines)]
        if [_normalize(line) for line in window] == norm_old:
            matches.append(start)
    if len(matches) > 1 and not replace_all:
        at = ", ".join(str(m + 1) for m in matches)
        return None, f"old_string matches {len(matches)} locations (lines {at}); add more context"
    if not matches:
        return None, "old_string not found (exact or whitespace-normalized)"
    if replace_all and len(matches) > 1:
        for start in reversed(matches):
            lines[start : start + len(old_lines)] = [new + "\n"]
        return "".join(lines), None
    start = matches[0]
    lines[start : start + len(old_lines)] = [new + "\n"]
    return "".join(lines), None


def _parse_anchor(anchor: str) -> tuple[int, str] | None:
    if ":" not in anchor:
        return None
    line_no, _, crc = anchor.partition(":")
    if not line_no.isdigit() or not crc:
        return None
    return int(line_no), crc


def _crc_replace(
    lines: list[str], start_anchor: str, end_anchor: str | None, new: str
) -> tuple[str | None, str | None]:
    start = _parse_anchor(start_anchor)
    if start is None:
        return None, f"invalid start_anchor (want line_no:crc8): {start_anchor!r}"
    end = _parse_anchor(end_anchor) if end_anchor else start
    if end is None:
        return None, f"invalid end_anchor (want line_no:crc8): {end_anchor!r}"
    for label, (line_no, crc) in (("start", start), ("end", end)):
        if not (1 <= line_no <= len(lines)):
            return None, f"{label} anchor out of range: line {line_no} (file has {len(lines)})"
        if line_anchor(line_no, lines[line_no - 1]) != f"{line_no}:{crc}":
            return None, (
                f"file changed: {label} anchor {line_no}:{crc} does not match "
                f"line {line_no}; re-read the file"
            )
    if end[0] < start[0]:
        return None, "end anchor is before start anchor"
    lines[start[0] - 1 : end[0]] = [new + "\n"]
    return "".join(lines), None


class EditTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="edit",
            description=(
                "Search/replace in a file. Engine 'fuzzy' (default): exact then "
                "whitespace-normalized match of old_string. Engine 'crc': replace the "
                "span between line anchors from `read` with_anchors."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "engine": {"type": "string", "enum": ["fuzzy", "crc"]},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                    "start_anchor": {"type": "string", "description": "line_no:crc8"},
                    "end_anchor": {"type": "string", "description": "line_no:crc8"},
                    "force": {"type": "boolean", "description": "Skip the read-first guard"},
                },
                "required": ["path", "new_string"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = Path(str(args["path"]))
        if not path.is_absolute():
            path = ctx.cwd / path
        if not path.is_file():
            return ToolResult(f"error: no such file: {args['path']}", is_error=True)

        if str(path) not in ctx.read_paths and not args.get("force"):
            return ToolResult(
                f"error: {args['path']} has not been read in this session; "
                "read it first (or pass force=true)",
                is_error=True,
            )

        content = path.read_text(encoding="utf-8", errors="replace")
        engine = str(args.get("engine") or "fuzzy")
        new = str(args.get("new_string", ""))
        if engine == "crc":
            start_anchor = args.get("start_anchor")
            if not start_anchor:
                return ToolResult("error: crc engine requires start_anchor", is_error=True)
            new_content, error = _crc_replace(
                content.splitlines(keepends=True), str(start_anchor), args.get("end_anchor"), new
            )
        else:
            old = args.get("old_string")
            if old is None:
                return ToolResult("error: fuzzy engine requires old_string", is_error=True)
            new_content, error = _fuzzy_replace(
                content, str(old), new, bool(args.get("replace_all"))
            )
        if error is not None:
            return ToolResult(f"error: {error}", is_error=True)
        path.write_text(new_content or "", encoding="utf-8")
        result = f"edited {path} ({engine} engine)"
        return ToolResult(result + await diagnostics_section(ctx, path))


def make_tool() -> Tool:
    return EditTool()
