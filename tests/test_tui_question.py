"""Tests for the inline question picker (arrows/space/enter/ESC) and question callback."""

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
from lecode.tui.question import QuestionPrompt, question_heading, question_hint, question_rows
from lecode.tui.statusline import StatusLineState

QUESTIONS = [
    {
        "question": "Which approach?",
        "header": "design",
        "options": [{"label": "Alpha"}, {"label": "Beta"}, {"label": "Gamma"}],
        "multi_select": False,
    }
]


def rows_text(rows) -> str:
    return "".join(text for _, text in rows)


# -- prompt state + rendering ----------------------------------------------------


def test_question_heading_and_hint():
    assert question_heading(QUESTIONS[0]) == " ask_user  [design] Which approach? "
    assert "↑↓ navigate" in question_hint(QUESTIONS[0])
    assert "Space toggle" not in question_hint(QUESTIONS[0])
    assert "Space toggle" in question_hint({**QUESTIONS[0], "multi_select": True})
    assert "Type your answer" in question_hint(QUESTIONS[0], custom=True)


def test_question_rows_highlight_and_custom_row():
    text = rows_text(question_rows(QUESTIONS[0], highlight=1, selected=set()))
    assert "> Beta" in text
    assert "  Alpha" in text and "  Gamma" in text
    assert "Type your own answer" in text


def test_question_rows_multi_select_marks():
    question = {**QUESTIONS[0], "multi_select": True}
    text = rows_text(question_rows(question, highlight=1, selected={0, 2}))
    assert "  [x] Alpha" in text
    assert "> [ ] Beta" in text
    assert "  [x] Gamma" in text


async def test_question_prompt_single_select_resolves():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.is_pending
    assert prompt.move(1) == "moved"
    assert prompt.enter() == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["Beta"]}]
    assert not prompt.is_pending


async def test_question_prompt_move_wraps():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.move(-1) == "moved"
    assert prompt.pending is not None and prompt.pending.highlight == 3  # the custom row
    assert prompt.move(-1) == "moved"
    assert prompt.pending.highlight == 2  # wraps to Gamma
    assert prompt.activate() == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["Gamma"]}]


async def test_question_prompt_multi_toggle_confirm():
    prompt = QuestionPrompt()
    questions = [{**QUESTIONS[0], "multi_select": True}]
    future = prompt.request(questions)
    assert prompt.activate() == "toggled"  # Alpha
    assert prompt.move(2) == "moved"
    assert prompt.activate() == "toggled"  # Gamma
    assert prompt.activate() == "toggled"  # toggles back off
    assert prompt.enter() == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["Alpha"]}]


async def test_question_prompt_custom_answer():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.custom("My own") == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["My own"]}]


async def test_question_prompt_custom_row_enters_mode():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.move(3) == "moved"  # past the 3 options, onto the custom row
    assert prompt.activate() == "custom"
    assert prompt.pending is not None and prompt.pending.custom
    assert prompt.custom("anything") == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["anything"]}]


async def test_question_prompt_back_from_custom_mode():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    prompt.move(3)
    prompt.activate()
    assert prompt.back() == "back"
    assert prompt.pending is not None and not prompt.pending.custom
    assert prompt.back() == "ignored"  # no longer in custom mode
    prompt.dismiss()
    await future


async def test_question_prompt_custom_answer_multi_keeps_toggles():
    prompt = QuestionPrompt()
    future = prompt.request([{**QUESTIONS[0], "multi_select": True}])
    assert prompt.activate() == "toggled"  # Alpha
    assert prompt.custom("something else") == "advanced"
    assert await future == [{"question": "Which approach?", "answers": ["Alpha", "something else"]}]


async def test_question_prompt_custom_blank_ignored():
    prompt = QuestionPrompt()
    future = prompt.request(QUESTIONS)
    assert prompt.custom("   ") == "ignored"
    assert prompt.is_pending
    prompt.dismiss()
    await future


async def test_question_prompt_sequential_then_dismiss():
    prompt = QuestionPrompt()
    second = {
        "question": "Ship it?",
        "options": [{"label": "yes"}, {"label": "no"}],
        "multi_select": False,
    }
    future = prompt.request([QUESTIONS[0], second])
    assert prompt.activate() == "advanced"  # first option, default highlight
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
    assert app._question.move(1) == "moved"
    assert app._question.enter() == "advanced"
    assert await task == [{"question": "Which approach?", "answers": ["Beta"]}]
    assert app._status.state is StatusLineState.RUNNING
    assert not app._question.is_pending


async def test_pipe_question_arrow_answers(tmp_path, monkeypatch):
    """Full flow: model calls ask_user, user picks with arrows, run completes."""
    app, out = make_app(tmp_path, monkeypatch, ask_script(QUESTIONS))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("\x1b[B\r")  # down to Beta, enter
        await wait_for(lambda: "answered" in out.getvalue())  # turn fully done
        inp.send_text("/quit\r")
        assert await task == 0
    rendered = out.getvalue()
    assert '"answers":["Beta"]' in rendered  # tool result content


async def test_pipe_question_custom_answer(tmp_path, monkeypatch):
    """Typing an answer during the picker submits it instead of an option."""
    app, out = make_app(tmp_path, monkeypatch, ask_script(QUESTIONS))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("use rust\r")
        await wait_for(lambda: "answered" in out.getvalue())
        inp.send_text("/quit\r")
        assert await task == 0
    assert '"answers":["use rust"]' in out.getvalue()


async def test_pipe_question_custom_row(tmp_path, monkeypatch):
    """Selecting the picker's custom row, then typing, submits the text."""
    app, out = make_app(tmp_path, monkeypatch, ask_script(QUESTIONS))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("\x1b[B\x1b[B\x1b[B\r")  # down onto the custom row, Enter
        inp.send_text("use rust\r")
        await wait_for(lambda: "answered" in out.getvalue())
        inp.send_text("/quit\r")
        assert await task == 0
    assert '"answers":["use rust"]' in out.getvalue()


async def test_pipe_question_custom_escape_returns_to_options(tmp_path, monkeypatch):
    """Esc while typing backs out to the options, dropping the draft."""
    app, out = make_app(tmp_path, monkeypatch, ask_script(QUESTIONS))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text("\x1b[B\x1b[B\x1b[B\r")  # onto the custom row, Enter
        inp.send_text("draft")
        inp.send_text("\x1b")  # back to options, draft dropped
        inp.send_text("\x1b[B\r")  # down wraps to Alpha, enter selects it
        await wait_for(lambda: "answered" in out.getvalue())
        inp.send_text("/quit\r")
        assert await task == 0
    assert '"answers":["Alpha"]' in out.getvalue()


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
    # The transcript shows the tool result summary; the full result (with the
    # "best judgment" instruction) is on the persisted tool message.
    tool_texts = [
        str(record.message.get("content"))
        for record in app.store.load_messages(app.session)
        if record.role == "tool"
    ]
    assert any("best judgment" in text for text in tool_texts)


async def test_pipe_question_multi_select_toggle_confirm(tmp_path, monkeypatch):
    questions = [{**QUESTIONS[0], "multi_select": True}]
    app, out = make_app(tmp_path, monkeypatch, ask_script(questions))
    with create_pipe_input() as inp:
        inp.send_text("help me choose\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "Which approach?" in out.getvalue())
        inp.send_text(" ")  # toggle Alpha
        inp.send_text("\x1b[B\x1b[B ")  # down to Gamma, toggle it
        inp.send_text("\r")  # enter confirms
        await wait_for(lambda: "answered" in out.getvalue())
        inp.send_text("/quit\r")
        assert await task == 0
    assert '"answers":["Alpha","Gamma"]' in out.getvalue()
