"""Inline permission prompt state (y/a/n/ESC).

Rendering goes through the feed (normal scrollback) and the statusline's
``awaiting approval`` state; keypresses are intercepted by the main app's
keybindings, filtered on :attr:`ApprovalPrompt.is_pending`, so no nested
prompt_toolkit application ever fights over stdin.

Concurrent askers (parent turn + workers) queue FIFO. Only the head is
resolved by keypresses; queued entries keep waiting until they reach the
front. Cancelling an awaiter removes exactly that entry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import partial

from lecode.permission import ApprovalDecision

#: Target shown in the prompt line before truncation.
_TARGET_MAX_LEN = 60


@dataclass
class PendingApproval:
    tool_name: str
    target: str
    reason: str
    future: asyncio.Future[ApprovalDecision] = field(repr=False)
    #: Attribution for "[w2 bash]" style prompts; None = the main turn.
    worker: str | None = None
    conversation: str = "main"
    allow_always: bool = True


def approval_prompt_text(tool_name: str, target: str, *, allow_always: bool = True) -> str:
    """The one-line ask: ``allow bash 'ls'? (y)once (a)lways (n)deny — ESC denies``."""
    if not allow_always:
        return f"{target}\n(y)es (n)o — ESC denies"
    shown = target if len(target) <= _TARGET_MAX_LEN else target[: _TARGET_MAX_LEN - 1] + "…"
    return f"allow {tool_name} '{shown}'? (y)once (a)lways (n)deny — ESC denies"


class ApprovalPrompt:
    """FIFO queue of pending approvals; keypresses resolve only the head."""

    def __init__(self) -> None:
        self._queue: list[PendingApproval] = []

    @property
    def pending(self) -> PendingApproval | None:
        return self._queue[0] if self._queue else None

    @property
    def is_pending(self) -> bool:
        return bool(self._queue)

    def request(
        self,
        tool_name: str,
        target: str,
        reason: str,
        *,
        worker: str | None = None,
        conversation: str = "main",
        allow_always: bool = True,
    ) -> asyncio.Future[ApprovalDecision]:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        entry = PendingApproval(
            tool_name, target, reason, future, worker, conversation, allow_always
        )
        self._queue.append(entry)
        future.add_done_callback(partial(self._on_future_done, entry))
        return future

    def resolve(self, decision: ApprovalDecision) -> None:
        entry = self.pending
        if entry is not None and not entry.future.done():
            entry.future.set_result(decision)
            self._queue.pop(0)

    def cancel(self, future: asyncio.Future[ApprovalDecision] | None = None) -> None:
        """Cancel the entry behind ``future`` only, or all outstanding when omitted."""
        if future is None:
            queue, self._queue = self._queue, []
            for item in queue:
                if not item.future.done():
                    item.future.cancel()
            return
        for i, item in enumerate(self._queue):
            if item.future is future:
                del self._queue[i]
                future.cancel()
                return

    def _on_future_done(
        self, entry: PendingApproval, future: asyncio.Future[ApprovalDecision]
    ) -> None:
        if future.cancelled() and entry in self._queue:
            self._queue.remove(entry)
