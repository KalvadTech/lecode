"""Append-only output feed over a Rich ``Console``.

Normal scrollback only: no alternate screen, no ``Live`` regions that
repaint history. Everything printed here is newline-terminated complete
lines — prompt_toolkit's ``patch_stdout`` erase/redraw cycle overwrites any
partial (unterminated) line, so partial text must never reach the console.

Live streaming therefore goes to an in-layout region instead: when the TUI
binds ``stream_sink``/``stream_clear``, content tokens accumulate in
``_stream_parts`` and are mirrored to the sink (a window above the input);
the full text is flushed into the scrollback as one complete print at
stream end. Reasoning tokens accumulate separately and render once at
stream end — as a muted one-liner when ``collapse_thinking`` is on, else as
a full muted panel. All output goes through the injected console so tests
can record with ``Console(record=True, file=StringIO())``.

Logbook style: every discrete line carries a ``[HH:MM:SS]`` timestamp — except
the user-input echo, which prints verbatim — and action lines (tool calls,
tool results) end with the live ``ctx used/window · $cost-so-far`` segment
when a ``metrics`` callable is bound (the TUI binds it to the statusline
state).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from lecode.tui.statusline import context_meter, format_cost, human_tokens
from lecode.tui.themes import Theme

#: Max length of a rendered tool-call line.
TOOL_CALL_MAX_LEN = 120

#: Lines of a tool result shown before elision kicks in.
TOOL_RESULT_HEAD_LINES = 10

#: ``(context_used, context_window, cost_usd)`` at render time.
MetricsFn = Callable[[], tuple[int, int, float]]


class Feed:
    """Renders user/assistant/tool/status output to a Rich console."""

    def __init__(self, console: Console, theme: Theme, collapse_thinking: bool = True) -> None:
        self._console = console
        self._theme = theme
        self._collapse_thinking = collapse_thinking
        self._streaming = False
        self._stream_printed = False
        self._stream_parts: list[str] = []
        self._thinking_parts: list[str] = []
        #: Live context/cost source for the logbook suffix (None = no suffix).
        self.metrics: MetricsFn | None = None
        #: When bound (the TUI), content tokens are mirrored to an in-layout
        #: live region and flushed to the scrollback only once complete.
        self.stream_sink: Callable[[str], None] | None = None
        self.stream_clear: Callable[[], None] | None = None

    # -- logbook stamp ---------------------------------------------------------

    @staticmethod
    def _stamp() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _suffix(self) -> str:
        """`` · ctx 12.3k/200.0k · $0.0412`` when metrics are bound."""
        if self.metrics is None:
            return ""
        used, window, cost = self.metrics()
        return f" · ctx {human_tokens(used)}/{human_tokens(window)} · {format_cost(cost)}"

    # -- transient activity -----------------------------------------------------
    # No printed activity line: the statusline carries the spinner + activity
    # label. (A scrollback-level spinner used partial-line redraws, which
    # clobbered streamed output under patch_stdout.)

    def user_message(self, text: str) -> None:
        """Echo the user's input as ``> text`` — verbatim, no timestamp/suffix."""
        self._console.print(f"> {text}", markup=False, highlight=False)

    def assistant_text(self, markdown: str) -> None:
        """Render a completed assistant message as Markdown."""
        self._console.print(Markdown(markdown))

    def stream_start(self) -> None:
        """Begin a streaming assistant turn."""
        self._streaming = True
        self._stream_printed = False
        self._stream_parts = []
        self._thinking_parts = []

    def stream_token(self, text: str, *, thinking: bool = False) -> None:
        """Handle one content/reasoning token.

        Reasoning accumulates for stream end. Content accumulates too; with a
        bound sink it is mirrored live to the in-layout region, otherwise it
        prints raw immediately (non-TUI use, where no app redraws can eat it).
        """
        if thinking:
            self._thinking_parts.append(text)
            return
        if self.stream_sink is not None:
            self._stream_parts.append(text)
            self.stream_sink(text)
            self._stream_printed = True
        else:
            self._console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
            self._stream_printed = True

    def _flush_stream(self) -> None:
        """Close the open stream: print the accumulated text as one whole print.

        With a sink, the live region is cleared first and the full text lands
        in the scrollback newline-terminated (safe under patch_stdout).
        """
        if not self._stream_printed:
            return
        if self.stream_sink is not None:
            full = "".join(self._stream_parts)
            self._stream_parts = []
            if self.stream_clear is not None:
                self.stream_clear()
            self._console.print(full, markup=False, highlight=False, soft_wrap=True)
        else:
            self._console.print()
        self._stream_printed = False

    def stream_end(self) -> None:
        """Close the streamed line and flush any accumulated thinking."""
        self._streaming = False
        self._flush_stream()
        if not self._thinking_parts:
            return
        full = "".join(self._thinking_parts)
        count = len(self._thinking_parts)
        self._thinking_parts = []
        if self._collapse_thinking:
            self._console.print(
                Text(
                    f"[{self._stamp()}] ▸ thinking ({count} tokens)",
                    style=self._theme.thinking,
                )
            )
        else:
            self._console.print(
                Panel(
                    Text(full, style=self._theme.thinking),
                    title="thinking",
                    border_style=self._theme.muted,
                )
            )

    def llm_call(self, model: str, turn: int) -> None:
        """Log one LLM invocation: ``→ model (round N)``."""
        self._console.print(
            Text(f"[{self._stamp()}] → {model} (round {turn})", style=self._theme.muted)
        )

    def llm_response(
        self, model: str, turn: int, input_tokens: int, output_tokens: int, cost_usd: float
    ) -> None:
        """Log one finished LLM call: ``← model (round N) · ↑in · ↓out · $cost``."""
        # The streamed answer text has no trailing newline yet — close it first.
        self._flush_stream()
        line = (
            f"[{self._stamp()}] ← {model} (round {turn})"
            f" · ↑{human_tokens(input_tokens)} in · ↓{human_tokens(output_tokens)} out"
            f" · {format_cost(cost_usd)}"
        )
        self._console.print(Text(line, style=self._theme.muted))

    def tool_call(self, name: str, args_preview: str) -> None:
        """Render ``⚙ name(args_preview)``, truncated to ~120 chars."""
        line = f"[{self._stamp()}] ⚙ {name}({args_preview})"
        if len(line) > TOOL_CALL_MAX_LEN:
            line = line[: TOOL_CALL_MAX_LEN - 1] + "…"
        self._console.print(Text(line + self._suffix(), style=self._theme.tool))

    def tool_result(self, name: str, content: str, is_error: bool = False) -> None:
        """Render a tool result head with ``… (N more lines)`` elision."""
        lines = content.splitlines()
        shown = lines[:TOOL_RESULT_HEAD_LINES]
        if len(lines) > TOOL_RESULT_HEAD_LINES:
            shown.append(f"… ({len(lines) - TOOL_RESULT_HEAD_LINES} more lines)")
        if shown:
            shown[0] = f"[{self._stamp()}] {shown[0]}{self._suffix()}"
        style = self._theme.error if is_error else self._theme.muted
        self._console.print(Text("\n".join(shown), style=style))

    def turn_stats(
        self,
        *,
        context_used: int,
        context_window: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        session_cost_usd: float,
        tool_calls: int = 0,
        turns: int = 0,
        elapsed_s: float = 0.0,
    ) -> None:
        """Muted per-answer line: this answer first, then session totals."""
        _, pct = context_meter(context_used, context_window)
        parts = [
            f"answer: ↑{human_tokens(input_tokens)} in · ↓{human_tokens(output_tokens)} out"
            f" · {format_cost(cost_usd)}",
            f"total: {format_cost(session_cost_usd)}",
            f"ctx {human_tokens(context_used)}/{human_tokens(context_window)} ({pct}%)",
        ]
        activity = []
        if tool_calls:
            activity.append(f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}")
        if turns:
            activity.append(f"{turns} round{'s' if turns != 1 else ''}")
        if elapsed_s:
            activity.append(f"{elapsed_s:.1f}s")
        if activity:
            parts.append(" · ".join(activity))
        line = f"[{self._stamp()}] " + " · ".join(parts)
        self._console.print(Text(line, style=self._theme.muted))

    def review(self, model: str, feedback: str) -> None:
        """Pierre-mode feedback: a labelled block after the stats line."""
        self._console.print(Text(f"[{self._stamp()}] ◆ pierre ({model})", style=self._theme.accent))
        self._console.print(Text(feedback, style=self._theme.text))

    def error(self, msg: str) -> None:
        """Render an error one-liner."""
        self._console.print(Text(f"[{self._stamp()}] ✗ {msg}", style=self._theme.error))

    def info(self, msg: str) -> None:
        """Render an informational one-liner."""
        lines = msg.splitlines()
        if lines:
            lines[0] = f"[{self._stamp()}] {lines[0]}"
        self._console.print(Text("\n".join(lines), style=self._theme.muted))

    def permission(self, msg: str) -> None:
        """Render a permission-prompt one-liner."""
        self._console.print(Text(f"[{self._stamp()}] {msg}", style=self._theme.permission))

    def retrying(self, attempt: int, delay_s: float) -> None:
        """Render a retry notice one-liner."""
        self._console.print(
            Text(
                f"[{self._stamp()}] retrying (attempt {attempt}) in {delay_s:.1f}s…",
                style=self._theme.warning,
            )
        )
