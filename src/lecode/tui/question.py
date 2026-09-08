"""Inline structured-question prompt state (1-4/enter/ESC).

Same shape as the permission prompt: rendering goes through the feed and
the statusline's ``awaiting answer`` state; keypresses are intercepted by
the main app's keybindings, filtered on :attr:`QuestionPrompt.is_pending`,
so no nested prompt_toolkit application ever fights over stdin. Questions
are handled sequentially — selecting an answer advances to the next one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

#: Cap on one rendered option line.
_OPTION_MAX_LEN = 100


@dataclass
class PendingQuestion:
    questions: list[dict[str, Any]]  # normalized by the ask_user tool
    future: asyncio.Future[list[dict[str, Any]]] = field(repr=False)
    index: int = 0
    answers: list[dict[str, Any]] = field(default_factory=list)
    #: Toggled option indices for the current multi-select question.
    selection: set[int] = field(default_factory=set)


def _clip(text: str) -> str:
    return text if len(text) <= _OPTION_MAX_LEN else text[: _OPTION_MAX_LEN - 1] + "…"


def question_prompt_text(question: dict[str, Any], selected: set[int] | None = None) -> str:
    """The rendered block: header tag, question, numbered options, key hint."""
    header = question.get("header")
    first = f"[{header}] {question['question']}" if header else str(question["question"])
    lines = [first]
    multi = bool(question.get("multi_select"))
    for i, option in enumerate(question["options"], start=1):
        label = _clip(str(option["label"]))
        if option.get("description"):
            label += f" — {_clip(str(option['description']))}"
        marker = ""
        if multi:
            marker = "[x] " if (i - 1) in (selected or set()) else "[ ] "
        lines.append(f"  {marker}{i}. {label}")
    hint = "1-4 toggle, enter confirms" if multi else "1-4 select"
    lines.append(f"{hint} — ESC dismisses")
    return "\n".join(lines)


class QuestionPrompt:
    """At most one pending question batch; resolved by keypress or cancelled."""

    def __init__(self) -> None:
        self._pending: PendingQuestion | None = None

    @property
    def pending(self) -> PendingQuestion | None:
        return self._pending

    @property
    def is_pending(self) -> bool:
        return self._pending is not None

    def request(self, questions: list[dict[str, Any]]) -> asyncio.Future[list[dict[str, Any]]]:
        future: asyncio.Future[list[dict[str, Any]]] = asyncio.get_running_loop().create_future()
        self._pending = PendingQuestion(questions, future)
        return future

    def current(self) -> dict[str, Any] | None:
        pending = self._pending
        if pending is None or pending.index >= len(pending.questions):
            return None
        return pending.questions[pending.index]

    def _advance(self, pending: PendingQuestion, answers: list[str]) -> None:
        pending.answers.append(
            {"question": pending.questions[pending.index]["question"], "answers": answers}
        )
        pending.index += 1
        pending.selection = set()
        if pending.index >= len(pending.questions):
            if not pending.future.done():
                pending.future.set_result(pending.answers)
            self._pending = None

    def select(self, option_index: int) -> str:
        """Digit keypress: single-select answers and advances; multi toggles.

        Returns ``"advanced"`` (render the next question), ``"toggled"``
        (re-render the current one), or ``"ignored"``.
        """
        pending = self._pending
        question = self.current()
        if pending is None or question is None:
            return "ignored"
        if not 0 <= option_index < len(question["options"]):
            return "ignored"
        if question.get("multi_select"):
            pending.selection ^= {option_index}
            return "toggled"
        self._advance(pending, [str(question["options"][option_index]["label"])])
        return "advanced"

    def confirm(self) -> str:
        """Enter on a multi-select question: record the toggled options."""
        pending = self._pending
        question = self.current()
        if pending is None or question is None or not question.get("multi_select"):
            return "ignored"
        labels = [str(question["options"][i]["label"]) for i in sorted(pending.selection)]
        self._advance(pending, labels)
        return "advanced"

    def dismiss(self) -> None:
        """ESC: the remaining questions are dismissed — the model decides."""
        pending = self._pending
        if pending is None:
            return
        for question in pending.questions[pending.index :]:
            pending.answers.append({"question": question["question"], "dismissed": True})
        if not pending.future.done():
            pending.future.set_result(pending.answers)
        self._pending = None

    def cancel(self) -> None:
        pending = self._pending
        if pending is not None and not pending.future.done():
            pending.future.cancel()
        self._pending = None
