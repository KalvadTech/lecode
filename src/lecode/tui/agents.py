"""Live agent-run roster: identity, status, activity, detail rendering.

Display state only — nothing here enters the model context. The roster
consumes :class:`SubagentProgress` events and answers three questions: what
is running now, what each run did, and what the open detail panel shows.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from rich.text import Text

from lecode.agent.runner import Done, Error, ToolCall, ToolResult
from lecode.extras.subagents import SubagentProgress
from lecode.permission.patterns import target_of
from lecode.tui.themes import Theme

#: Rows the compact roster shows before collapsing into ``+N more``.
ROSTER_VISIBLE_ROWS = 4

#: Activity entries retained per run (the tail wins).
RUN_ACTIVITY_MAX = 100

#: Result lines shown per tool in the detail panel.
RESULT_PREVIEW_LINES = 3

#: Height cap of the in-layout detail panel (rows).
DETAIL_MAX_ROWS = 12

_STATUS_GLYPHS = {
    "running": ("●", "accent"),
    "ok": ("✔", "success"),
    "error": ("✗", "error"),
    "cancelled": ("—", "muted"),
}


def _clip(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _target(name: str, args: str) -> str:
    """Best-effort target (path/command) from a tool call's raw arguments."""
    try:
        parsed = json.loads(args) if args.strip() else {}
    except json.JSONDecodeError:
        parsed = {}
    if isinstance(parsed, dict) and parsed:
        return target_of(name, parsed)
    return " ".join(args.split())


@dataclass
class ActivityEntry:
    """One tool call inside a child run, paired with its result when it lands."""

    call_id: str
    name: str
    args: str
    result: str = ""
    is_error: bool = False
    running: bool = True

    @property
    def target(self) -> str:
        return _clip(_target(self.name, self.args), 60)


@dataclass
class AgentRun:
    """The roster's per-run state."""

    run_id: str
    index: int
    agent: str
    description: str
    status: str = "running"
    activity: list[ActivityEntry] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)
    truncated: bool = False

    @property
    def current(self) -> str:
        """The one-line current action (what the roster row shows)."""
        if self.status != "running":
            return {"ok": "done", "error": "failed", "cancelled": "cancelled"}.get(
                self.status, self.status
            )
        active = next((entry for entry in reversed(self.activity) if entry.running), None)
        if active is None:
            return "thinking"
        return f"{active.name} {active.target}".strip()


class AgentRoster:
    """Append-only run registry keyed by run id; finished runs stay listed."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._order: list[str] = []

    def observe(self, progress: SubagentProgress) -> AgentRun:
        """Fold one child progress event into its run's state."""
        run = self._runs.get(progress.run_id)
        if run is None:
            run = AgentRun(
                run_id=progress.run_id,
                index=len(self._order) + 1,
                agent=progress.agent,
                description=progress.description,
            )
            self._runs[run.run_id] = run
            self._order.append(run.run_id)
        event = progress.event
        if isinstance(event, ToolCall):
            run.activity.append(
                ActivityEntry(call_id=event.id, name=event.name, args=event.arguments)
            )
            if len(run.activity) > RUN_ACTIVITY_MAX:
                run.activity = run.activity[-RUN_ACTIVITY_MAX:]
                run.truncated = True
        elif isinstance(event, ToolResult):
            entry = next((e for e in run.activity if e.call_id == event.id), None)
            if entry is not None:
                entry.result = event.content
                entry.is_error = event.is_error
                entry.running = False
        elif isinstance(event, Error):
            run.status = "error"
            run.error = event.message
        elif isinstance(event, Done) and run.status == "running":
            run.status = "ok"
        return run

    def finish(
        self, run_id: str, *, answer: str = "", error: str = "", is_error: bool = False
    ) -> AgentRun | None:
        """Close a run with its answer/error once the caller knows the outcome."""
        run = self._runs.get(run_id)
        if run is None:
            return None
        if answer:
            run.answer = answer
        if error:
            run.error = error
        if is_error or error:
            run.status = "error"
        elif run.status == "running":
            run.status = "ok"
        return run

    def cancel_running(self) -> list[AgentRun]:
        """Mark every still-running run cancelled (parent turn was cancelled)."""
        cancelled = [run for run in self._runs.values() if run.status == "running"]
        for run in cancelled:
            run.status = "cancelled"
        return cancelled

    def has_running(self) -> bool:
        return any(run.status == "running" for run in self._runs.values())

    def get(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def resolve(self, ref: str) -> AgentRun | None:
        """Resolve a run by roster number or run-id prefix (exact first)."""
        if ref.isdigit():
            wanted = int(ref)
            return next((run for run in self.runs() if run.index == wanted), None)
        exact = self._runs.get(ref)
        if exact is not None:
            return exact
        matches = [run for run in self.runs() if run.run_id.startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    def runs(self) -> list[AgentRun]:
        """Runs in start order (stable numbering)."""
        return [self._runs[run_id] for run_id in self._order]

    def visible(self, limit: int = ROSTER_VISIBLE_ROWS) -> tuple[list[AgentRun], int]:
        """``(shown, hidden_count)``: running first (newest first), then
        finished (newest first); stable indices come from the run itself."""
        runs = self.runs()
        running = [run for run in reversed(runs) if run.status == "running"]
        finished = [run for run in reversed(runs) if run.status != "running"]
        ordered = running + finished
        return ordered[:limit], max(0, len(ordered) - limit)


def _glyph(status: str, theme: Theme) -> tuple[str, str]:
    glyph, slot = _STATUS_GLYPHS.get(status, _STATUS_GLYPHS["running"])
    return glyph, getattr(theme, slot)


def roster_lines(roster: AgentRoster, theme: Theme, width: int) -> list[Text]:
    """The compact roster block (header + up to four rows + overflow)."""
    runs = roster.runs()
    if not runs:
        return []
    running = sum(1 for run in runs if run.status == "running")
    done = len(runs) - running
    header = f"agents · {running} running"
    if done:
        header += f" · {done} done"
    lines = [Text(header, style=theme.muted)]
    shown, hidden = roster.visible()
    for run in shown:
        glyph, style = _glyph(run.status, theme)
        line = Text()
        line.append(f"  {glyph} ", style=style)
        line.append(f"{run.index} {run.agent} ", style=theme.accent)
        line.append(_clip(run.description, max(12, width // 3)), style=theme.text)
        line.append(f" · {_clip(run.current, max(12, width // 3))}", style=theme.muted)
        lines.append(line)
    if hidden:
        lines.append(Text(f"  … +{hidden} more · /runs", style=theme.muted))
    return lines


def _preview(text: str, width: int, limit: int = RESULT_PREVIEW_LINES) -> list[str]:
    rows = text.splitlines()
    shown = [_clip(row, width) for row in rows[:limit]]
    if len(rows) > limit:
        shown.append(f"… ({len(rows) - limit} more lines)")
    return shown


def detail_lines(run: AgentRun | None, theme: Theme, width: int) -> list[Text]:
    """The per-run detail panel: identity, tool trail, answer/error."""
    if run is None:
        return []
    glyph, style = _glyph(run.status, theme)
    lines: list[Text] = []
    header = Text()
    header.append(f"  {glyph} ", style=style)
    header.append(f"{run.index} {run.agent} ", style=theme.accent)
    header.append(_clip(run.description, max(12, width // 2)), style=theme.text)
    header.append(f" · {run.current}", style=theme.muted)
    lines.append(header)
    if not run.activity:
        lines.append(Text("    (no tool calls yet)", style=theme.muted))
    for entry in run.activity:
        row = Text()
        row.append(f"    ⚙ {entry.name} ", style=theme.tool)
        row.append(_clip(entry.target, max(12, width - 20)), style=theme.text)
        if entry.running:
            row.append(" · running", style=theme.muted)
        elif entry.is_error:
            row.append(" · failed", style=theme.error)
        lines.append(row)
        if entry.result:
            preview_style = theme.error if entry.is_error else theme.muted
            for row_text in _preview(entry.result, max(12, width - 6)):
                lines.append(Text(f"      {row_text}", style=preview_style))
    if run.error and not run.answer:
        lines.append(Text(f"    error: {_clip(run.error, max(12, width - 14))}", style=theme.error))
    if run.answer:
        lines.append(Text("    answer:", style=theme.muted))
        for row_text in _preview(run.answer, max(12, width - 6)):
            lines.append(Text(f"      {row_text}", style=theme.text))
    return lines
