"""Tests for the ask_user tool: validation, callback dispatch, degradation."""

from __future__ import annotations

import json

import pytest

from lecode.agent.tools import core_tools
from lecode.agent.tools.ask_user import UNAVAILABLE_CONTENT, make_tool
from lecode.agent.tools.base import ToolRegistry

QUESTION = {
    "question": "Which approach?",
    "header": "design",
    "options": [
        {"label": "Alpha", "description": "the safe one"},
        {"label": "Beta"},
    ],
}


async def test_registered_in_core_tools():
    names = [tool.name for tool in core_tools()]
    assert "ask_user" in names


async def test_single_select_answers_shape(tool_ctx):
    seen = []

    async def answer(questions):
        seen.append(questions)
        return [{"question": questions[0]["question"], "answers": ["Beta"]}]

    tool_ctx.question_callback = answer
    result = await make_tool().run({"questions": [QUESTION]}, tool_ctx)
    assert not result.is_error
    assert json.loads(result.content) == [{"question": "Which approach?", "answers": ["Beta"]}]
    # the callback receives the normalized questions
    (question,) = seen[0]
    assert question["question"] == "Which approach?"
    assert question["header"] == "design"
    assert question["multi_select"] is False
    assert question["options"] == [
        {"label": "Alpha", "description": "the safe one"},
        {"label": "Beta"},
    ]


async def test_multi_select_answers(tool_ctx):
    async def answer(questions):
        return [{"question": questions[0]["question"], "answers": ["Alpha", "Beta"]}]

    tool_ctx.question_callback = answer
    args = {"questions": [{**QUESTION, "multi_select": True}]}
    result = await make_tool().run(args, tool_ctx)
    assert not result.is_error
    assert json.loads(result.content)[0]["answers"] == ["Alpha", "Beta"]


async def test_dismissal_passthrough(tool_ctx):
    async def dismiss(questions):
        return [{"question": questions[0]["question"], "dismissed": True}]

    tool_ctx.question_callback = dismiss
    result = await make_tool().run({"questions": [QUESTION]}, tool_ctx)
    assert not result.is_error
    assert json.loads(result.content.splitlines()[0]) == [
        {"question": "Which approach?", "dismissed": True}
    ]
    assert "best judgment" in result.content


@pytest.mark.parametrize(
    "questions",
    [
        None,
        "not-a-list",
        [],
        [QUESTION] * 5,  # too many
        [{"options": [{"label": "a"}, {"label": "b"}]}],  # missing question
        [{**QUESTION, "question": "  "}],  # blank question
        [{**QUESTION, "options": [{"label": "only"}]}],  # too few options
        [{**QUESTION, "options": [{"label": str(i)} for i in range(5)]}],  # too many
        [{**QUESTION, "options": [{"description": "no label"}, {"label": "b"}]}],
        ["not-an-object"],
    ],
)
async def test_malformed_args_error(tool_ctx, questions):
    async def answer(qs):  # pragma: no cover — must not be reached
        raise AssertionError("callback called on invalid args")

    tool_ctx.question_callback = answer
    result = await make_tool().run({"questions": questions}, tool_ctx)
    assert result.is_error
    assert result.content.startswith("error:")


async def test_no_callback_degrades_gracefully(tool_ctx):
    assert tool_ctx.question_callback is None
    result = await make_tool().run({"questions": [QUESTION]}, tool_ctx)
    assert not result.is_error  # not an error: the model should proceed, not retry
    assert result.content == UNAVAILABLE_CONTENT


async def test_dispatch_round_trip_through_registry(tool_ctx):
    """The full dispatch path: permission check (read-class) then the tool."""

    async def answer(questions):
        return [{"question": questions[0]["question"], "answers": ["Beta"]}]

    tool_ctx.question_callback = answer
    registry = ToolRegistry([make_tool()])
    args = json.dumps({"questions": [QUESTION]})
    message, result = await registry.dispatch_result("c1", "ask_user", args, tool_ctx)
    assert not result.is_error
    assert message["role"] == "tool"
    assert json.loads(message["content"])[0]["answers"] == ["Beta"]
