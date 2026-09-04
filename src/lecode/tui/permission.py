"""Inline permission prompt state (y/a/n/ESC).

Rendering goes through the feed (normal scrollback) and the statusline's
``awaiting approval`` state; keypresses are intercepted by the main app's
keybindings, filtered on :attr:`ApprovalPrompt.is_pending`, so no nested
prompt_toolkit application ever fights over stdin.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from lecode.permission import ApprovalDecision

#: Target shown in the prompt line before truncation.
_TARGET_MAX_LEN = 60


@dataclass
class PendingApproval:
    tool_name: str
    target: str
    reason: str
    future: asyncio.Future[ApprovalDecision] = field(repr=False)


def approval_prompt_text(tool_name: str, target: str) -> str:
    """The one-line ask: ``allow bash 'ls'? (y)once (a)lways (n)deny — ESC denies``."""
    shown = target if len(target) <= _TARGET_MAX_LEN else target[: _TARGET_MAX_LEN - 1] + "…"
    return f"allow {tool_name} '{shown}'? (y)once (a)lways (n)deny — ESC denies"


class ApprovalPrompt:
    """At most one pending approval; resolved by keypress or cancelled."""

    def __init__(self) -> None:
        self._pending: PendingApproval | None = None

    @property
    def pending(self) -> PendingApproval | None:
        return self._pending

    @property
    def is_pending(self) -> bool:
        return self._pending is not None

    def request(self, tool_name: str, target: str, reason: str) -> asyncio.Future[ApprovalDecision]:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._pending = PendingApproval(tool_name, target, reason, future)
        return future

    def resolve(self, decision: ApprovalDecision) -> None:
        pending = self._pending
        if pending is not None and not pending.future.done():
            pending.future.set_result(decision)
        self._pending = None

    def cancel(self) -> None:
        pending = self._pending
        if pending is not None and not pending.future.done():
            pending.future.cancel()
        self._pending = None
