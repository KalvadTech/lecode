"""Append-only output feed over a Rich ``Console``.

Normal scrollback only: no alternate screen, no ``Live`` regions that
repaint history. Streaming tokens print raw as they arrive (no markup, no
re-wrapping); reasoning tokens accumulate separately and render once at
stream end — as a muted one-liner when ``collapse_thinking`` is on, else as
a full muted panel. All output goes through the injected console so tests
can record with ``Console(record=True, file=StringIO())``.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from lecode.tui.themes import Theme

#: Max length of a rendered tool-call line.
TOOL_CALL_MAX_LEN = 120

#: Lines of a tool result shown before elision kicks in.
TOOL_RESULT_HEAD_LINES = 10


class Feed:
    """Renders user/assistant/tool/status output to a Rich console."""

    def __init__(self, console: Console, theme: Theme, collapse_thinking: bool = True) -> None:
        self._console = console
        self._theme = theme
        self._collapse_thinking = collapse_thinking
        self._streaming = False
        self._stream_printed = False
        self._thinking_parts: list[str] = []

    def set_theme(self, theme: Theme) -> None:
        """Swap the active theme (``/theme``); affects future output only."""
        self._theme = theme

    def user_message(self, text: str) -> None:
        """Echo the user's input as ``> text`` in the accent color."""
        self._console.print(Text(f"> {text}", style=self._theme.accent))

    def assistant_text(self, markdown: str) -> None:
        """Render a completed assistant message as Markdown."""
        self._console.print(Markdown(markdown))

    def stream_start(self) -> None:
        """Begin a streaming assistant turn."""
        self._streaming = True
        self._stream_printed = False
        self._thinking_parts = []

    def stream_token(self, text: str, *, thinking: bool = False) -> None:
        """Print a content token raw; accumulate reasoning tokens for stream end."""
        if thinking:
            self._thinking_parts.append(text)
            return
        self._console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
        self._stream_printed = True

    def stream_end(self) -> None:
        """Close the streamed line and flush any accumulated thinking."""
        self._streaming = False
        if self._stream_printed:
            self._console.print()
            self._stream_printed = False
        if not self._thinking_parts:
            return
        full = "".join(self._thinking_parts)
        count = len(self._thinking_parts)
        self._thinking_parts = []
        if self._collapse_thinking:
            self._console.print(Text(f"▸ thinking ({count} tokens)", style=self._theme.thinking))
        else:
            self._console.print(
                Panel(
                    Text(full, style=self._theme.thinking),
                    title="thinking",
                    border_style=self._theme.muted,
                )
            )

    def tool_call(self, name: str, args_preview: str) -> None:
        """Render ``⚙ name(args_preview)``, truncated to ~120 chars."""
        line = f"⚙ {name}({args_preview})"
        if len(line) > TOOL_CALL_MAX_LEN:
            line = line[: TOOL_CALL_MAX_LEN - 1] + "…"
        self._console.print(Text(line, style=self._theme.tool))

    def tool_result(self, name: str, content: str, is_error: bool = False) -> None:
        """Render a tool result head with ``… (N more lines)`` elision."""
        lines = content.splitlines()
        shown = lines[:TOOL_RESULT_HEAD_LINES]
        if len(lines) > TOOL_RESULT_HEAD_LINES:
            shown.append(f"… ({len(lines) - TOOL_RESULT_HEAD_LINES} more lines)")
        style = self._theme.error if is_error else self._theme.muted
        self._console.print(Text("\n".join(shown), style=style))

    def error(self, msg: str) -> None:
        """Render an error one-liner."""
        self._console.print(Text(f"✗ {msg}", style=self._theme.error))

    def info(self, msg: str) -> None:
        """Render an informational one-liner."""
        self._console.print(Text(msg, style=self._theme.muted))

    def permission(self, msg: str) -> None:
        """Render a permission-prompt one-liner."""
        self._console.print(Text(msg, style=self._theme.permission))

    def retrying(self, attempt: int, delay_s: float) -> None:
        """Render a retry notice one-liner."""
        self._console.print(
            Text(
                f"retrying (attempt {attempt}) in {delay_s:.1f}s…",
                style=self._theme.warning,
            )
        )
