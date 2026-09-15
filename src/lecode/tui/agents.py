"""Live agent-run roster: identity, status, activity, detail rendering.

Display state only — nothing here enters the model context. The roster
consumes :class:`SubagentProgress` events and answers three questions: what
is running now, what each run did, and what the open detail panel shows.
"""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rich.text import Text

from lecode.agent.runner import Done, Error, ToolCall, ToolResult
from lecode.extras.subagents import SubagentProgress
from lecode.permission.patterns import target_of
from lecode.tui.statusline import format_cost, human_tokens
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
    "queued": ("○", "muted"),
    "waiting": ("◌", "warning"),
    "done": ("✔", "success"),
    "ok": ("✔", "success"),
    "error": ("✗", "error"),
    "failed": ("✗", "error"),
    "cancelled": ("—", "muted"),
    "stopped": ("■", "muted"),
    "interrupted": ("!", "warning"),
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
    parent_id: str | None = None
    depth: int = 0
    worker: bool = False
    origin: str = "delegated"
    cost_usd: float = 0.0
    usage_incomplete: bool = False
    context_used: int = 0
    context_window: int = 0
    elapsed_s: float = 0.0
    subtree_cost_usd: float = 0.0
    subtree_usage_incomplete: bool = False
    session_id: str | None = None

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

    def sync_worker(self, worker, *, context_window: int = 0) -> AgentRun:
        """Mirror the persistent worker state into the UI-only roster."""
        run = self._runs.get(worker.id)
        if run is None:
            run = AgentRun(
                run_id=worker.id,
                index=len(self._order) + 1,
                agent=worker.agent,
                description=worker.description,
                parent_id=worker.parent_id,
                depth=worker.depth,
                worker=True,
                origin=worker.origin,
            )
            self._runs[run.run_id] = run
            self._order.append(run.run_id)
        run.status = {"completed": "done", "failed": "error"}.get(worker.state, worker.state)
        run.parent_id = worker.parent_id
        run.depth = worker.depth
        run.worker = True
        run.origin = worker.origin
        run.error = worker.error or ""
        run.answer = worker.result.final_text if worker.result is not None else ""
        run.cost_usd = worker.usage_totals.cost_usd
        run.usage_incomplete = worker.usage_incomplete
        run.context_used = worker.usage_totals.context_tokens
        run.context_window = context_window
        run.session_id = getattr(worker, "session_id", None)
        started_at = getattr(worker, "started_at", "")
        now = datetime.now(UTC)
        with suppress(TypeError, ValueError):
            run.elapsed_s = max(0.0, (now - datetime.fromisoformat(started_at)).total_seconds())
        self._refresh_subtree_costs()
        return run

    def _refresh_subtree_costs(self) -> None:
        workers = [run for run in self._runs.values() if run.worker]
        children: dict[str | None, list[AgentRun]] = {}
        for run in workers:
            children.setdefault(run.parent_id, []).append(run)

        def total(run: AgentRun) -> tuple[float, bool]:
            cost = run.cost_usd
            incomplete = run.usage_incomplete
            for child in children.get(run.run_id, []):
                child_cost, child_incomplete = total(child)
                cost += child_cost
                incomplete |= child_incomplete
            run.subtree_cost_usd = cost
            run.subtree_usage_incomplete = incomplete
            return cost, incomplete

        for run in children.get(None, []):
            total(run)

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
        return any(run.status in {"queued", "running", "waiting"} for run in self._runs.values())

    def has_workers(self) -> bool:
        return any(run.worker for run in self._runs.values())

    def worker_total(self) -> tuple[float, bool]:
        workers = [run for run in self._runs.values() if run.worker]
        return sum(run.cost_usd for run in workers), any(run.usage_incomplete for run in workers)

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
        if self.has_workers():
            by_parent: dict[str | None, list[AgentRun]] = {}
            workers = [run for run in runs if run.worker]
            for run in workers:
                by_parent.setdefault(run.parent_id, []).append(run)
            ordered: list[AgentRun] = []

            def visit(parent_id: str | None) -> None:
                for child in by_parent.get(parent_id, []):
                    ordered.append(child)
                    visit(child.run_id)

            visit(None)
            ordered.extend(run for run in runs if not run.worker)
            return ordered[:limit], max(0, len(ordered) - limit)
        running = [run for run in reversed(runs) if run.status in {"queued", "running", "waiting"}]
        finished = [
            run for run in reversed(runs) if run.status not in {"queued", "running", "waiting"}
        ]
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
    running = sum(1 for run in runs if run.status in {"queued", "running", "waiting"})
    done = len(runs) - running
    header = (
        f"workers · {running} running" if roster.has_workers() else f"agents · {running} running"
    )
    if done:
        header += f" · {done} done"
    if roster.has_workers():
        cost, incomplete = roster.worker_total()
        header += f" · {format_cost(cost)}" + (" incomplete" if incomplete else "")
    lines = [Text(header, style=theme.muted)]
    shown, hidden = roster.visible()
    for run in shown:
        glyph, style = _glyph(run.status, theme)
        line = Text()
        indent = "  " * run.depth if run.worker else ""
        line.append(f"  {indent}{glyph} ", style=style)
        line.append(f"{run.index} {run.agent} ", style=theme.accent)
        line.append(_clip(run.description, max(12, width // 3)), style=theme.text)
        detail = _clip(run.current, max(12, width // 3))
        if run.worker:
            detail += f" · {format_cost(run.cost_usd)}" + ("?" if run.usage_incomplete else "")
            detail += f" · {run.elapsed_s:.0f}s"
        line.append(f" · {detail}", style=theme.muted)
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


def _message_lines(message: dict, theme: Theme) -> list[Text]:
    role = str(message.get("role", "unknown"))
    name = str(message.get("name") or "")
    label = f"    [{role}{f' {name}' if name else ''}]"
    lines = [Text(label, style=theme.muted)]
    content = message.get("content")
    if content not in (None, ""):
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=True)
        lines.extend(Text(f"      {row}", style=theme.text) for row in text.splitlines() or [""])
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        name = str(function.get("name", "tool"))
        args = str(function.get("arguments", ""))
        lines.append(Text(f"      ⚙ {name} {args}", style=theme.tool))
    return lines


def detail_lines(
    run: AgentRun | None,
    theme: Theme,
    width: int,
    transcript: list[dict] | None = None,
) -> list[Text]:
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
    if run.worker:
        cost = format_cost(run.cost_usd) + (" (incomplete)" if run.usage_incomplete else "")
        subtree = format_cost(run.subtree_cost_usd) + (
            " (incomplete)" if run.subtree_usage_incomplete else ""
        )
        context = "unknown"
        if run.context_window:
            context = f"{human_tokens(run.context_used)}/{human_tokens(run.context_window)}"
        lines.append(
            Text(
                f"    cost: {cost} · subtree: {subtree} · ctx: {context}"
                f" · elapsed: {run.elapsed_s:.0f}s",
                style=theme.muted,
            )
        )
    if transcript is not None:
        lines.append(Text("    transcript:", style=theme.muted))
        for message in transcript:
            lines.extend(_message_lines(message, theme))
        return lines
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
