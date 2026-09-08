"""The one fixed statusline.

Three-line layout, every element labelled::

    dir: <folder> · commit: <hash> · branch: <branch> · diff: <diff-stat>
    model: <model> · cost: <$0.00> · ctx: ▓▓▓░░ 84.0k/200k 42%
    session: <name> · agent: <agent> · in: 1.2k · out: 0.4k · <state>

Not user-configurable. Lines are truncated to the terminal width.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rich.text import Text

from lecode.extras.proc import run_proc
from lecode.tui.themes import Theme

#: Braille spinner frames cycled while the agent is running.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼ⴚ⦧⦇⦏⦉"

#: Width of the context meter in blocks.
METER_BLOCKS = 5

#: Git subprocess timeout in seconds.
_GIT_TIMEOUT_S = 2.0


class StatusLineState(StrEnum):
    """What the agent is doing right now."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    QUESTION = "question"


@dataclass
class GitInfo:
    """Snapshot of the repo state shown on the statusline's first line."""

    branch: str | None = None
    commit: str | None = None  # short hash
    diff: str | None = None  # compact shortstat, e.g. "±3 +10 -2"


@dataclass
class StatusState:
    """Everything the statusline needs to render one frame."""

    session_name: str
    agent: str
    model: str
    cwd: Path | str
    git: GitInfo | None = None
    #: Reasoning-level override label ("High", "None", …); None at baseline.
    reasoning: str | None = None
    context_used: int = 0
    context_window: int = 200_000
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    state: StatusLineState = StatusLineState.IDLE
    queued: int = 0
    steered: int = 0
    spinner_frame: int = 0
    #: What is running right now ("thinking", "running bash", …); shown with
    #: the spinner instead of a printed activity line (which flickered).
    activity: str | None = None


def human_tokens(n: int) -> str:
    """Format a token count compactly: ``400`` → ``0.4k``, ``1234`` → ``1.2k``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 100:
        return f"{n / 1000:.1f}k"
    return str(n)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for live context growth."""
    return max(1, len(text) // 4)


def format_cost(cost_usd: float) -> str:
    """Four decimals under one dollar, two decimals at or above."""
    return f"${cost_usd:.4f}" if cost_usd < 1 else f"${cost_usd:.2f}"


def context_meter(used: int, window: int) -> tuple[str, int]:
    """Return the ``(bar, percent)`` pair for the context meter."""
    ratio = min(max(used / window, 0.0), 1.0) if window > 0 else 0.0
    filled = 0 if ratio == 0 else math.ceil(ratio * METER_BLOCKS)
    return "▓" * filled + "░" * (METER_BLOCKS - filled), round(ratio * 100)


def _state_segment(state: StatusState) -> tuple[str, str]:
    """Plain text and theme-color name for the trailing state segment."""
    if state.state is StatusLineState.RUNNING:
        text = SPINNER_FRAMES[state.spinner_frame % len(SPINNER_FRAMES)]
        if state.activity:
            text += f" {state.activity}…"
        color = "accent"
    elif state.state is StatusLineState.AWAITING_APPROVAL:
        text = "awaiting approval"
        color = "permission"
    elif state.state is StatusLineState.QUESTION:
        text = "awaiting answer"
        color = "permission"
    else:
        text = "ready"
        color = "success"
    if state.queued > 0:
        text += f" +{state.queued}q"
    if state.steered > 0:
        text += f" +{state.steered}s"
    return text, color


def render_statusline(state: StatusState, theme: Theme, width: int = 100) -> Text:
    """Render the fixed three-line statusline, truncating each line to ``width``."""
    sep = Text(" · ", style=theme.muted)

    def labelled(line: Text, label: str, value: str, style: str, *, first: bool = False) -> None:
        if not first:
            line.append_text(sep.copy())
        line.append(f"{label}: ", style=theme.muted)
        line.append(value, style=style)

    # Line 1: dir · commit · branch · diff
    line1 = Text()
    labelled(line1, "dir", Path(state.cwd).name, theme.accent, first=True)
    if state.git is not None:
        for label, value in (
            ("commit", state.git.commit),
            ("branch", state.git.branch),
            ("diff", state.git.diff),
        ):
            if value:
                labelled(line1, label, value, theme.muted)

    # Line 2: model · cost · ctx meter x/y pct%
    bar, pct = context_meter(state.context_used, state.context_window)
    line2 = Text()
    labelled(line2, "model", state.model, theme.text, first=True)
    if state.reasoning is not None:
        line2.append_text(sep.copy())
        line2.append(state.reasoning, style=theme.muted)
    labelled(line2, "cost", format_cost(state.cost_usd), theme.muted)
    labelled(
        line2,
        "ctx",
        f"{bar} {human_tokens(state.context_used)}/{human_tokens(state.context_window)} {pct}%",
        theme.text,
    )

    # Line 3: session · agent · in/out tokens · state
    state_seg, state_color = _state_segment(state)
    line3 = Text()
    labelled(line3, "session", state.session_name, theme.accent, first=True)
    labelled(line3, "agent", state.agent, theme.accent)
    labelled(line3, "in", human_tokens(state.input_tokens), theme.muted)
    labelled(line3, "out", human_tokens(state.output_tokens), theme.muted)
    line3.append_text(sep.copy())
    line3.append(state_seg, style=getattr(theme, state_color))

    text = Text()
    for line in (line1, line2, line3):
        line.truncate(width, overflow="ellipsis")
        text.append_text(line)
        text.append("\n")
    text.truncate(max(0, len(text.plain) - 1))  # drop trailing newline
    return text


async def _git(args: list[str], cwd: Path | str) -> str | None:
    """Run one git command, returning stripped stdout or ``None`` on failure."""
    try:
        result = await run_proc(["git", *args], cwd=cwd, timeout=_GIT_TIMEOUT_S)
    except OSError:
        return None
    if result.exit_code != 0 or result.timed_out:
        return None
    return result.stdout.strip() or None


def _compact_shortstat(text: str | None) -> str | None:
    """Compact ``git diff --shortstat`` output: ``±3 +10 -2``; ``None`` when clean."""
    if not text:
        return None
    files = re.search(r"(\d+) files? changed", text)
    ins = re.search(r"(\d+) insertions?", text)
    dele = re.search(r"(\d+) deletions?", text)
    parts = [f"±{files.group(1)}" if files else ""]
    if ins:
        parts.append(f"+{ins.group(1)}")
    if dele:
        parts.append(f"-{dele.group(1)}")
    return " ".join(p for p in parts if p) or None


async def git_info(cwd: Path | str) -> GitInfo | None:
    """Return branch/commit/diff-stat for ``cwd``, or ``None`` outside a repo."""
    branch = await _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch is None:
        return None
    commit = await _git(["rev-parse", "--short", "HEAD"], cwd)
    stat = await _git(["diff", "--shortstat", "HEAD"], cwd)
    return GitInfo(branch=branch, commit=commit, diff=_compact_shortstat(stat))


class CachedGitInfo:
    """Small per-directory cache for :func:`git_info` with a TTL."""

    def __init__(
        self,
        ttl_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[GitInfo | None, float]] = {}

    async def get(self, cwd: Path | str) -> GitInfo | None:
        """Return the git info for ``cwd``, re-querying at most once per TTL."""
        key = str(cwd)
        now = self._clock()
        cached = self._cache.get(key)
        if cached is not None and now - cached[1] < self._ttl_s:
            return cached[0]
        info = await git_info(cwd)
        self._cache[key] = (info, now)
        return info
