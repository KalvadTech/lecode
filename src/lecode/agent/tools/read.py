"""The ``read`` tool: paginated, line-numbered file reads.

Optional CRC anchors (``with_anchors``) prefix each line with
``line_no:crc8`` for the crc edit engine. Image files return a metadata note —
or, when the current model takes image input, the image itself as a content
part in the wire message (via the result's ``content_parts`` metadata).
"""

from __future__ import annotations

import zlib
from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.multimodal import check_modalities, load_attachment, to_content_parts
from lecode.providers.catalog import Catalog

DEFAULT_LIMIT = 2000
MAX_LIMIT = 10000

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def line_anchor(line_no: int, text: str) -> str:
    """``line_no:crc8`` anchor over the first non-space content of a line."""
    crc = zlib.crc32(text.strip().encode("utf-8", errors="replace")) & 0xFF
    return f"{line_no}:{crc:02x}"


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ctx.cwd / p


def _catalog(ctx: ToolContext) -> Catalog:
    """The session's live catalog; empty (fail-open) when none was fetched."""
    return ctx.catalog if ctx.catalog is not None else Catalog.default()


class ReadTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="read",
            description=(
                "Read a file with line numbers. Paginate with offset/limit; "
                "set with_anchors for CRC line anchors usable by the edit tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "offset": {"type": "integer", "description": "First line (1-based)"},
                    "limit": {
                        "type": "integer",
                        "description": f"Max lines (default {DEFAULT_LIMIT})",
                    },
                    "with_anchors": {
                        "type": "boolean",
                        "description": "Prefix lines with CRC anchors for the crc edit engine",
                    },
                },
                "required": ["path"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = _resolve(ctx, str(args.get("path", "")))
        if not path.exists():
            return ToolResult(f"error: no such file: {args.get('path')}", is_error=True)

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            size = path.stat().st_size
            ctx.read_paths.add(str(path))
            note = f"image file: {path.name} ({path.suffix[1:]}, {size} bytes)"
            try:
                attachment = load_attachment(path)
            except (OSError, ValueError):
                attachment = None  # e.g. over the 20 MB cap
            if (
                attachment is not None
                and check_modalities([attachment], ctx.config.llm.model, _catalog(ctx)) is None
            ):
                # The model takes image input: wire the image as a content part.
                parts: list = [{"type": "text", "text": note}]
                parts += to_content_parts([attachment])
                return ToolResult(note, metadata={"image": True, "content_parts": parts})
            return ToolResult(
                f"{note}; rendering as an image content part is not enabled for this model",
                metadata={"image": True, "path": str(path)},
            )

        if path.is_dir():
            return ToolResult(f"error: is a directory: {args.get('path')}", is_error=True)

        offset = max(1, int(args.get("offset") or 1))
        limit = min(MAX_LIMIT, int(args.get("limit") or DEFAULT_LIMIT))
        with_anchors = bool(args.get("with_anchors"))

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return ToolResult(f"error: {e}", is_error=True)

        total = len(lines)
        page = lines[offset - 1 : offset - 1 + limit]
        with_marks: list[str] = []
        for i, text in enumerate(page, start=offset):
            if with_anchors:
                with_marks.append(f"{line_anchor(i, text)}\t{text}")
            else:
                with_marks.append(f"{i}\t{text}")

        remaining = total - (offset - 1 + len(page))
        if remaining > 0:
            next_offset = offset + len(page)
            with_marks.append(f"… {remaining} more lines (continue with offset={next_offset})")

        # Repeat-read guard: same path + range read twice in a row.
        mtime = path.stat().st_mtime
        state = ctx.extras.setdefault("read.last", {})
        note = ""
        key = (str(path), offset, limit)
        if state.get("key") == key:
            if state.get("mtime") == mtime:
                note = "\n(note: already read, file unchanged)"
            else:
                note = "\n(note: re-read — file changed since last read)"
        state.update({"key": key, "mtime": mtime})

        ctx.read_paths.add(str(path))
        if not with_marks:
            with_marks = ["(empty file)"]
        return ToolResult("\n".join(with_marks) + note)


def make_tool() -> Tool:
    return ReadTool()
