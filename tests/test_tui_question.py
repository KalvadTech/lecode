"""Tests for the inline question prompt (1-4/enter/ESC) and question callback."""

from __future__ import annotations

import asyncio
import json
from io import StringIO

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from tests.fakes import FakeProvider
from tests.test_tui_permission import wait_for

from lecode.agent.builder import build_runtime
from lecode.config.models import Config
from lecode.session.storage import SessionStore
from lecode.tui.app import TuiApp
from lecode.tui.question import QuestionPrompt, question_prompt_text
from lecode.tui.statusline import StatusLineState

QUESTIONS = [
    {
        "question": "Which approach?",
        "header": "design",
        "options": [{"label": "Alpha"}, {"label": "Beta"}, {"label": "Gamma"}],
        "multi_select": False,
    }
]


# -- prompt state + rendering ----------------------------------------------------


def test_question_prompt_text():
    text = question_prompt_text(QUESTIONS[0])
    assert "[design] Which approach?" in text
    assert "1. Alpha" in text and "2. Beta" in text and "3. Gamma" in text
    assert "1-4 select" in text and "ESC dismisses" in text


def test_question_prompt_text_multi_select_marks():
    question = {**QUESTIONS[0], "multi_select": True}
    text = question_prompt_text(question, {0, 2})
    assert "[x] 1. Alpha" in text
    assert "[ ] 2. Beta" in text
    assert "[x] 3. Gamma" in text
    assert "1-4 toggle, enter confirms" in text


async def test_question_prompt_single_select_resolves():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.is_pending
    assert prompt.select(1) == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["Beta"]}]
    assert not prompt.is_pending


async def test_question_prompt_out_of_range_ignored():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.select(3) == "ignored"  # only 3 options
    assert prompt.select(-1) == "ignored"
    assert prompt.is_pending
    prompt.dismiss()
    await future


async def test_question_prompt_multi_toggle_confirm():
    prompt = QuestionPrompt()
    questions = [{**QUESTIONS[0], "multi_select": True}]
    future = prompt.request(questions)
    assert prompt.select(0) == "toggled"
    assert prompt.select(2) == "toggled"
    assert prompt.select(2) == "toggled"  # toggles back off
    assert prompt.confirm() == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["Alpha"]}]


async def test_question_prompt_sequential_then_dismiss():
    prompt = QuestionPrompt()
    second = {
        "question": "Ship it?",
        "options": [{"label": "yes"}, {"label": "no"}],
        "multi_select": False,
    }
    future = prompt.request([QUESTIONS[0], second])
    assert prompt.select(0) == "advanced"
    assert prompt.current() == second  # advanced to the second question
    prompt.dismiss()
    assert await future == [
        {"question": "Which approach?", "answers": ["Alpha"]},
        {"question": "Ship it?", "dismissed": True},
    ]
    assert not prompt.is_pending


async def test_question_prompt_cancel():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    prompt.cancel()
    assert not prompt.is_pending
    with pytest.raises(asyncio.CancelledError):
        await future


# -- app integration ---------------------------------------------------------------


def make_app(tmp_path, monkeypatch, script):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False  # never play sounds in tests
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    out = StringIO()
    app = TuiApp(
        config,
        runtime,
        FakeProvider(script),
        session,
        store,
        console=Console(record=True, file=out, width=200),
    )
    return app, out


def ask_script(questions, reply="answered"):
    return [
        {"tool_calls": [{"name": "ask_user", "arguments": json.dumps({"questions": questions})}]},
        {"text": reply},
    ]


async def test_request_question_renders_and_restores_state(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch, [])
    task = asyncio.ensure_future(app._request_question(QUESTIONS))
    await wait_for(lambda: "Which approach?" in out.getvalue())
    assert app._status.state is StatusLineState.QUESTION
    assert app._question.is_pending
    assert app._question.select(1) == "advanced"
    assert await task == [{"question": "Which approach?", "answers": ["Beta"]}]
    assert app._status.state is StatusLineState.RUNNING
    assert not app._question.is_pending


async def test_pipe_question_digit_answers(tmp_path, monkeypatch):
    """Full flow: model calls ask_user, user presses '2', run completes."""
    app, out = make_app(tmp_path, monkeypatch, ask_script(QUESTIONS))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("2")
        await wait_for(lambda: "answered" in out.getvalue())  # turn fully done
        inp.send_text("/quit\r")
        assert await task == 0
    rendered = out.getvalue()
    assert '"answers":["Beta"]' in rendered  # tool result content


async def test_pipe_question_escape_dismisses(tmp_path, monkeypatch):
    app, out = make_app(tmp_path, monkeypatch, ask_script(QUESTIONS))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("\x1b")
        await wait_for(lambda: "answered" in out.getvalue())
        inp.send_text("/quit\r")
        assert await task == 0
    rendered = out.getvalue()
    assert '"dismissed":true' in rendered
    assert "best judgment" in rendered


async def test_pipe_question_multi_select_toggle_confirm(tmp_path, monkeypatch):
    questions = [{**QUESTIONS[0], "multi_select": True}]
    app, out = make_app(tmp_path, monkeypatch, ask_script(questions))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("1")
        await wait_for(lambda: "[x] 1. Alpha" in out.getvalue())
        inp.send_text("3")
        await wait_for(lambda: "[x] 3. Gamma" in out.getvalue())
        inp.send_text("\r")  # enter confirms
        await wait_for(lambda: "answered" in out.getvalue())
        inp.send_text("/quit\r")
        assert await task == 0
    assert '"answers":["Alpha","Gamma"]' in out.getvalue()
