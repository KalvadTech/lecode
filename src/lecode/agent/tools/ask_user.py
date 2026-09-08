"""The ``ask_user`` tool: structured multiple-choice questions mid-turn.

The tool itself only validates and delegates: the interactive UI installs
``ctx.question_callback`` (an inline keyboard-driven prompt, never a nested
prompt_toolkit app); headless/loop/chain/subagent contexts leave it ``None``
and get a graceful "decide yourself" result — not an error, so the model
proceeds instead of retrying.
"""

from __future__ import annotations

import json
from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult

#: Bounds on the question batch and each option list.
MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4

#: Returned when no interactive front-end installed the question callback.
UNAVAILABLE_CONTENT = (
    "user interaction is unavailable in this mode; use your best judgment and proceed"
)


def _validate(questions: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Normalize the ``questions`` argument; ``(cleaned, None)`` or ``(None, error)``."""
    if not isinstance(questions, list) or not questions:
        return None, "questions must be a non-empty array"
    if len(questions) > MAX_QUESTIONS:
        return None, f"too many questions: {len(questions)} (max {MAX_QUESTIONS})"
    cleaned: list[dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            return None, f"question {i} must be an object"
        text = q.get("question")
        if not isinstance(text, str) or not text.strip():
            return None, f"question {i} needs a non-empty 'question' string"
        options = q.get("options")
        if not isinstance(options, list) or not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
            return None, (
                f"question {i} needs {MIN_OPTIONS}-{MAX_OPTIONS} options "
                f"(got {len(options) if isinstance(options, list) else 'none'})"
            )
        cleaned_options: list[dict[str, Any]] = []
        for j, option in enumerate(options, start=1):
            label = option.get("label") if isinstance(option, dict) else None
            if not isinstance(label, str) or not label.strip():
                return None, f"question {i} option {j} needs a non-empty 'label' string"
            entry: dict[str, Any] = {"label": option["label"]}
            if isinstance(option.get("description"), str) and option["description"].strip():
                entry["description"] = option["description"]
            cleaned_options.append(entry)
        item: dict[str, Any] = {
            "question": text,
            "options": cleaned_options,
            "multi_select": bool(q.get("multi_select", False)),
        }
        if isinstance(q.get("header"), str) and q["header"].strip():
            item["header"] = q["header"]
        cleaned.append(item)
    return cleaned, None


class AskUserTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="ask_user",
            description=(
                "Ask the user 1-4 structured multiple-choice questions and wait for the "
                "answers. Each question offers 2-4 labelled options; set multi_select to "
                "let the user pick several. Use this to clarify ambiguous requirements "
                "instead of guessing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_QUESTIONS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "header": {
                                    "type": "string",
                                    "description": "short tag shown before the question",
                                },
                                "options": {
                                    "type": "array",
                                    "minItems": MIN_OPTIONS,
                                    "maxItems": MAX_OPTIONS,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "description": {"type": "string"},
                                        },
                                        "required": ["label"],
                                    },
                                },
                                "multi_select": {"type": "boolean"},
                            },
                            "required": ["question", "options"],
                        },
                    }
                },
                "required": ["questions"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        cleaned, error = _validate(args.get("questions"))
        if error is not None or cleaned is None:
            return ToolResult(f"error: {error}", is_error=True)
        if ctx.question_callback is None:
            return ToolResult(UNAVAILABLE_CONTENT)
        results = await ctx.question_callback(cleaned)
        content = json.dumps(results, ensure_ascii=False, separators=(",", ":"))
        if any(r.get("dismissed") for r in results if isinstance(r, dict)):
            content += "\ndismissed questions: the user declined to answer; use your best judgment."
        return ToolResult(content)


def make_tool() -> Tool:
    return AskUserTool()
