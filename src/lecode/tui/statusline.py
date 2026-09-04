"""The one fixed statusline.

Single-line layout, segments joined with ``·``::

    <session-name> · <agent> · <model> · <cwd-basename>:<git-branch> ·
    ctx ▓▓▓░░ 42% · ↑1.2k ↓0.4k · $0.0123 · <state>

Not user-configurable beyond theme colors. On narrow terminals the
cwd/branch segment is truncated first, then the session name.
"""

from __future__ import annotations

import math
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

#: Minimum visible length before a segment stops absorbing truncation.
_MIN_CWD_LEN = 6
_MIN_SESSION_LEN = 4

#: Git subprocess timeout in seconds.
_GIT_TIMEOUT_S = 2.0


class StatusLineState(StrEnum):
    """What the agent is doing right now."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class StatusState:
    """Everything the statusline needs to render one frame."""

    session_name: str
    agent: str
    model: str
    cwd: Path | str
    git_branch: str | None = None
    context_used: int = 0
    context_window: int = 200_000
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    state: StatusLineState = StatusLineState.IDLE
    queued: int = 0
    steered: int = 0
    spinner_frame: int = 0


def human_tokens(n: int) -> str:
    """Format a token count compactly: ``400`` → ``0.4k``, ``1234`` → ``1.2k``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 100:
        return f"{n / 1000:.1f}k"
    return str(n)


def format_cost(cost_usd: float) -> str:
    """Four decimals under one dollar, two decimals at or above."""
    return f"${cost_usd:.4f}" if cost_usd < 1 else f"${cost_usd:.2f}"


def context_meter(used: int, window: int) -> tuple[str, int]:
    """Return the ``(bar, percent)`` pair for the context meter."""
    ratio = min(max(used / window, 0.0), 1.0) if window > 0 else 0.0
    filled = 0 if ratio == 0 else math.ceil(ratio * METER_BLOCKS)
    return "▓" * filled + "░" * (METER_BLOCKS - filled), round(ratio * 100)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return s[:max_len]
    return s[: max_len - 1] + "…"


def _state_segment(state: StatusState) -> tuple[str, str]:
    """Plain text and theme-color name for the trailing state segment."""
    if state.state is StatusLineState.RUNNING:
        text = SPINNER_FRAMES[state.spinner_frame % len(SPINNER_FRAMES)]
        color = "accent"
    elif state.state is StatusLineState.AWAITING_APPROVAL:
        text = "awaiting approval"
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
    """Render the fixed statusline, truncating to ``width`` if needed."""
    cwd_seg = Path(state.cwd).name
    if state.git_branch:
        cwd_seg += f":{state.git_branch}"
    bar, pct = context_meter(state.context_used, state.context_window)
    meter_seg = f"ctx {bar} {pct}%"
    tokens_seg = f"↑{human_tokens(state.input_tokens)} ↓{human_tokens(state.output_tokens)}"
    cost_seg = format_cost(state.cost_usd)
    state_seg, state_color = _state_segment(state)

    session_seg = state.session_name
    segments = [
        session_seg,
        state.agent,
        state.model,
        cwd_seg,
        meter_seg,
        tokens_seg,
        cost_seg,
        state_seg,
    ]
    total = sum(len(s) for s in segments) + len(" · ") * (len(segments) - 1)

    overflow = total - width
    if overflow > 0:
        cut = min(overflow, max(0, len(cwd_seg) - _MIN_CWD_LEN))
        cwd_seg = _truncate(cwd_seg, len(cwd_seg) - cut)
        overflow -= cut
    if overflow > 0:
        cut = min(overflow, max(0, len(session_seg) - _MIN_SESSION_LEN))
        session_seg = _truncate(session_seg, len(session_seg) - cut)

    text = Text()
    sep = Text(" · ", style=theme.muted)
    text.append(session_seg, style=theme.accent)
    text.append_text(sep)
    text.append(state.agent, style=theme.accent)
    text.append_text(sep.copy())
    text.append(state.model, style=theme.text)
    text.append_text(sep.copy())
    text.append(cwd_seg, style=theme.muted)
    text.append_text(sep.copy())
    text.append(f"ctx {bar} {pct}%", style=theme.text)
    text.append_text(sep.copy())
    text.append(tokens_seg, style=theme.muted)
    text.append_text(sep.copy())
    text.append(cost_seg, style=theme.muted)
    text.append_text(sep.copy())
    text.append(state_seg, style=getattr(theme, state_color))
    if len(text.plain) > width:
        text.truncate(width, overflow="ellipsis")
    return text


async def git_branch(cwd: Path | str) -> str | None:
    """Return the current git branch for ``cwd``, or ``None`` on any failure."""
    try:
        result = await run_proc(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            timeout=_GIT_TIMEOUT_S,
        )
    except OSError:
        return None
    if result.exit_code != 0 or result.timed_out:
        return None
    branch = result.stdout.strip()
    return branch or None


class CachedBranch:
    """Small per-directory cache for :func:`git_branch` with a TTL."""

    def __init__(
        self,
        ttl_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[str | None, float]] = {}

    async def get(self, cwd: Path | str) -> str | None:
        """Return the branch for ``cwd``, re-querying at most once per TTL."""
        key = str(cwd)
        now = self._clock()
        cached = self._cache.get(key)
        if cached is not None and now - cached[1] < self._ttl_s:
            return cached[0]
        branch = await git_branch(cwd)
        self._cache[key] = (branch, now)
        return branch
