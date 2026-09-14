"""Inline structured-question picker (arrows/space/enter/ESC).

Rendering reuses the themed picker panel: the options are rows and the
highlighted one is the cursor line, so the user picks with the arrow keys
instead of typing option numbers. Keypresses are intercepted by the main
app's keybindings, gated on :attr:`QuestionPrompt.is_pending`, so no nested
prompt_toolkit application ever fights over stdin. Questions are handled
sequentially — choosing an answer advances to the next one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

#: Cap on one rendered option line.
_OPTION_MAX_LEN = 100

#: Row styles, matching the picker panel's own classes.
_ROW_STYLE = "class:picker-menu.command"
_SELECTED_STYLE = "class:picker-menu.selected"

#: The always-present row that switches to a free-text answer.
_CUSTOM_ROW = "✎ Type your own answer…"


@dataclass
class PendingQuestion:
    questions: list[dict[str, Any]]  # normalized by the ask_user tool
    future: asyncio.Future[list[dict[str, Any]]] = field(repr=False)
    index: int = 0
    answers: list[dict[str, Any]] = field(default_factory=list)
    #: Toggled option indices for the current multi-select question.
    selection: set[int] = field(default_factory=set)
    #: Highlighted option for the cursor line.
    highlight: int = 0
    #: True while the user is typing a free-text answer (the custom row).
    custom: bool = False


def _clip(text: str) -> str:
    return text if len(text) <= _OPTION_MAX_LEN else text[: _OPTION_MAX_LEN - 1] + "…"


def question_heading(question: dict[str, Any]) -> str:
    """The picker panel's heading: header tag plus the question text."""
    header = question.get("header")
    first = f"[{header}] {question['question']}" if header else str(question["question"])
    return f" ask_user  {_clip(first)} "


def question_hint(question: dict[str, Any], custom: bool = False) -> str:
    """The picker panel's footer: the keys the current question accepts."""
    if custom:
        return " Type your answer in the box, Enter to submit, Esc back"
    if question.get("multi_select"):
        return " ↑↓ navigate  Space toggle  Enter confirm  Esc dismiss"
    return " ↑↓ navigate  Enter select  Esc dismiss"


def question_rows(
    question: dict[str, Any], highlight: int, selected: set[int]
) -> list[tuple[str, str]]:
    """Styled ``(style, text)`` rows for the picker panel's option list.

    The final row is the always-present free-text affordance; its index is
    ``len(options)`` and selecting it enters custom-answer mode.
    """
    multi = bool(question.get("multi_select"))
    rows: list[tuple[str, str]] = []
    for i, option in enumerate(question["options"]):
        if i:
            rows.append(("", "\n"))
        style = _SELECTED_STYLE if i == highlight else _ROW_STYLE
        marker = ("[x] " if i in selected else "[ ] ") if multi else ""
        text = marker + _clip(str(option["label"]))
        if option.get("description"):
            text += f" — {_clip(str(option['description']))}"
        rows.extend([(style, "> " if i == highlight else "  "), (style, text)])
    custom = len(question["options"])
    style = _SELECTED_STYLE if custom == highlight else _ROW_STYLE
    rows.append(("", "\n"))
    rows.extend([(style, "> " if custom == highlight else "  "), (style, _CUSTOM_ROW)])
    return rows


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
        pending.highlight = 0
        pending.custom = False
        if pending.index >= len(pending.questions):
            if not pending.future.done():
                pending.future.set_result(pending.answers)
            self._pending = None

    def move(self, delta: int) -> str:
        """Arrow key: move the highlight over the options and the custom row."""
        pending = self._pending
        question = self.current()
        if pending is None or question is None or pending.custom:
            return "ignored"
        count = len(question["options"]) + 1  # + the custom-answer row
        pending.highlight = (pending.highlight + delta) % count
        return "moved"

    def activate(self) -> str:
        """Enter/Tab/Space on the highlighted row: single-select answers and
        advances; multi-select toggles; the custom row enters typing mode.

        Returns ``"advanced"``, ``"toggled"``, ``"custom"`` or ``"ignored"``.
        """
        pending = self._pending
        question = self.current()
        if pending is None or question is None or pending.custom:
            return "ignored"
        if pending.highlight >= len(question["options"]):
            pending.custom = True
            return "custom"
        if question.get("multi_select"):
            pending.selection ^= {pending.highlight}
            return "toggled"
        self._advance(pending, [str(question["options"][pending.highlight]["label"])])
        return "advanced"

    def enter(self) -> str:
        """Enter key: confirm a multi-select question, else activate the row."""
        pending = self._pending
        question = self.current()
        if pending is None or question is None or pending.custom:
            return "ignored"  # custom mode waits for typed text
        if question.get("multi_select"):
            return self.confirm()
        return self.activate()

    def custom(self, text: str) -> str:
        """A typed custom answer (Enter with text in the input): use the text,
        keeping any toggled options on a multi-select question."""
        pending = self._pending
        question = self.current()
        text = text.strip()
        if pending is None or question is None or not text:
            return "ignored"
        answers = [text]
        if question.get("multi_select"):
            toggled = [str(question["options"][i]["label"]) for i in sorted(pending.selection)]
            answers = [*toggled, text]
        self._advance(pending, answers)
        return "advanced"

    def back(self) -> str:
        """Esc while typing a custom answer: return to the option list."""
        pending = self._pending
        if pending is None or not pending.custom:
            return "ignored"
        pending.custom = False
        return "back"

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
