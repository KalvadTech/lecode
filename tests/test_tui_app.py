"""Tests for the interactive TUI application and the CLI interactive path."""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, ClassVar

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from tests.fakes import FakeProvider, sample_catalog
from typer.testing import CliRunner

from lecode.agent.builder import build_runtime
from lecode.cli import app as cli_app
from lecode.config.models import Config
from lecode.providers.types import Done, TokenDelta
from lecode.session.storage import SessionStore
from lecode.tui.app import QUEUE_LIMIT, TuiApp
from lecode.tui.statusline import StatusLineState


def make_app(tmp_path, monkeypatch, script, config=None):
    """A TuiApp over a FakeProvider with a recorded console."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = config or Config()
    config.notifications.enabled = False  # never play sounds in tests
    store = SessionStore()
    session = store.create("test-session", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    provider = FakeProvider(script)
    out = StringIO()
    console = Console(record=True, file=out, width=200)
    app = TuiApp(
        config, runtime, provider, session, store, console=console, catalog=sample_catalog()
    )
    return app, provider, out


class BlockingProvider:
    """Streams nothing until ``release`` is set; then a short answer."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.release = asyncio.Event()
        self.blocked = True

    def stream_chat(self, messages, model, tools=None, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], "model": model})
        return self._stream()

    async def _stream(self):
        if self.blocked:
            await self.release.wait()
        yield TokenDelta(text="done")
        yield Done(finish_reason="stop")


def make_blocking_app(tmp_path, monkeypatch, config=None):
    app, _, out = make_app(tmp_path, monkeypatch, [], config=config)
    provider = BlockingProvider()
    app._runner.provider = provider
    return app, provider, out


async def wait_for(cond, timeout=5.0):
    """Poll ``cond`` until true; fail the test on timeout."""
    for _ in range(int(timeout / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


# -- submissions / runner wiring --------------------------------------------


async def test_unknown_model_keeps_configured_window(tmp_path, monkeypatch):
    """A model outside the catalog falls back to the configured window."""
    config = Config()
    config.llm.model = "no/such-model"
    app, _, _ = make_app(tmp_path, monkeypatch, [], config=config)
    assert app.status.context_window == config.agent.context_window


def test_layout_is_chatbox_above_statusline(tmp_path, monkeypatch):
    """The input is a framed chatbox directly above the 3-line statusline;
    the frame's bottom border is the split between them. A conditional live
    region for streamed text sits above the chatbox, and the completion menu
    anchors below the input for slash commands."""
    from prompt_toolkit.layout.containers import (
        ConditionalContainer,
        FloatContainer,
        Window,
        to_container,
    )
    from prompt_toolkit.widgets import Frame

    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        pt_app = app._build_app(input=inp, output=DummyOutput())
    assert isinstance(app._chatbox, Frame) and app._chatbox.body is app._input_area
    children = pt_app.layout.container.children
    assert len(children) == 2
    floats_host = children[0]
    assert isinstance(floats_host, FloatContainer)
    assert len(floats_host.floats) == 1  # other trigger menus still follow the cursor
    inner = floats_host.content.children
    assert len(inner) == 4
    assert isinstance(inner[0], ConditionalContainer)  # live stream region
    assert inner[1] is to_container(app._chatbox)  # Frame unwraps to its HSplit
    assert isinstance(inner[2], ConditionalContainer)  # slash panel sizes to its rows
    assert isinstance(inner[3], Window)  # other dropdowns' space reservation
    assert isinstance(children[1], Window) and children[1].height == 3
    assert app._live_buffer is not None


def _buffer(app):
    return app._input_area.buffer


def _command_texts(state) -> list[str]:
    return [c.text for c in state.completions]


async def test_slash_menu_opens_with_all_commands(tmp_path, monkeypatch):
    """Typing '/' at input start opens the dropdown with every command."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/")
        await wait_for(lambda: _buffer(app).complete_state is not None)
        texts = _command_texts(_buffer(app).complete_state)
        assert "/quit " in texts and "/queue " in texts
        assert len([t for t in texts if t.startswith("/model")]) > 1  # several model-* rows
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_prefix_filters_while_typing(tmp_path, monkeypatch):
    """'/cop' narrows to commands starting with 'cop'."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/cop")
        await wait_for(
            lambda: (
                _buffer(app).complete_state is not None
                and "/copy " in _command_texts(_buffer(app).complete_state)
            )
        )
        for text in _command_texts(_buffer(app).complete_state):
            assert text.lower().startswith("/cop")
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_enter_fills_without_submitting(tmp_path, monkeypatch):
    """Enter fills the first match; only a second Enter submits it."""
    app, provider, out = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/cop")
        await wait_for(
            lambda: (
                _buffer(app).complete_state is not None
                and "/copy " in _command_texts(_buffer(app).complete_state)
            )
        )
        inp.send_text("\r")  # accept: fill, do not submit
        await wait_for(
            lambda: _buffer(app).text == "/copy " and _buffer(app).complete_state is None
        )
        assert provider.requests == []
        assert "nothing to copy" not in out.getvalue()
        inp.send_text("\r")  # now submit the filled command
        await wait_for(lambda: "nothing to copy" in out.getvalue())
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_arrows_navigate_and_wrap(tmp_path, monkeypatch):
    """Up/Down move through matches (Down = first, Up past first selects last)."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/mod")
        await wait_for(
            lambda: (
                _buffer(app).complete_state is not None
                and "/mode " in _command_texts(_buffer(app).complete_state)
            )
        )
        state = _buffer(app).complete_state
        first, last = state.completions[0].text, state.completions[-1].text
        inp.send_text("\x1b[B")  # Down -> first
        await wait_for(lambda: _buffer(app).text == first)
        inp.send_text("\x1b[A")  # Up from first -> deselect, restores what was typed
        await wait_for(lambda: _buffer(app).text == "/mod")
        inp.send_text("\x1b[A")  # Up again -> wrap to last
        await wait_for(lambda: _buffer(app).text == last)
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_tab_accepts_first_match(tmp_path, monkeypatch):
    """Tab fills the first match like Enter does."""
    app, provider, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/mod")
        await wait_for(
            lambda: (
                _buffer(app).complete_state is not None
                and "/mode " in _command_texts(_buffer(app).complete_state)
            )
        )
        inp.send_text("\t")
        await wait_for(lambda: _buffer(app).text == "/mode ")
        assert provider.requests == []
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_tab_accepts_navigated_match_without_submitting(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/mod")
        await wait_for(lambda: _buffer(app).complete_state is not None)
        inp.send_text("\x1b[B\x1b[B")
        await wait_for(lambda: _buffer(app).text == "/model ")
        inp.send_text("\t")
        await wait_for(lambda: _buffer(app).complete_state is None)
        assert _buffer(app).text == "/model "
        assert provider.requests == []
        assert out.getvalue() == ""
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_escape_dismisses_and_restores(tmp_path, monkeypatch):
    """Escape closes the menu and restores the typed text; later tabs reopen it."""
    app, provider, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/mod")
        await wait_for(lambda: _buffer(app).complete_state is not None)
        inp.send_text("\x1b[B")  # navigate: text becomes '/mode ' ... first
        await wait_for(lambda: _buffer(app).text != "/mod")
        inp.send_text("\x1b")  # Escape: restore '/mod', close menu
        await wait_for(lambda: _buffer(app).text == "/mod" and _buffer(app).complete_state is None)
        inp.send_text("\t")  # Tab reopens the menu
        await wait_for(lambda: _buffer(app).complete_state is not None)
        inp.send_text("\x1b")  # Escape again: dismiss without filling
        await wait_for(lambda: _buffer(app).complete_state is None and _buffer(app).text == "/mod")
        assert provider.requests == []
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_closes_after_command_name(tmp_path, monkeypatch):
    """A space after the command closes the menu and it stays closed."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/mod")
        await wait_for(lambda: _buffer(app).complete_state is not None)
        inp.send_text(" ")
        await wait_for(lambda: _buffer(app).text == "/mod ")
        await asyncio.sleep(0.3)  # give any spurious recompletion time to fire
        assert _buffer(app).complete_state is None
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_no_match_row_escape_dismisses_and_reopens(tmp_path, monkeypatch):
    """Escape dismisses the inert no-match row; editing or Tab brings it back."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("/zz")
        await wait_for(lambda: app._slash_menu_empty())
        inp.send_text("\x1b")  # Escape dismisses the row (no completion state)
        await wait_for(lambda: not app._slash_menu_empty())
        inp.send_text("z")  # editing the prefix reopens it
        await wait_for(lambda: app._slash_menu_empty())
        inp.send_text("\x1b")  # dismiss again
        await wait_for(lambda: not app._slash_menu_empty())
        inp.send_text("\t")  # Tab reopens the dismissed row
        await wait_for(lambda: app._slash_menu_empty())
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_slash_menu_mid_message_does_not_open(tmp_path, monkeypatch):
    """The menu only responds to '/' at the start of the input."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: app._input_area is not None)
        inp.send_text("hey /qu")
        await asyncio.sleep(0.3)
        assert _buffer(app).complete_state is None
        inp.send_text("\x15/quit\r")
        assert await task == 0


async def test_resume_restores_status_usage(tmp_path, monkeypatch):
    """A session with stored usage opens with the statusline pre-filled."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False
    store = SessionStore()
    session = store.create("resumed", tmp_path, model=config.llm.model)
    store.append_message(session, {"role": "user", "content": "hi"})
    store.append_message(
        session,
        {"role": "assistant", "content": "hello"},
        usage={"input_tokens": 12_000, "output_tokens": 800, "cost_usd": 0.02},
    )
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    out = StringIO()
    app = TuiApp(
        config,
        runtime,
        FakeProvider([]),
        session,
        store,
        console=Console(record=True, file=out, width=200),
    )
    assert app._status.context_used == 12_000
    assert app._status.input_tokens == 12_000
    assert app._status.output_tokens == 800
    assert app._status.cost_usd == 0.02

    # Undo drops the hidden turn from the live context line.
    await app.handle_command("/undo")
    assert app._status.context_used == 0


async def test_submit_streams_answer(tmp_path, monkeypatch):
    script = [{"text": ["Hello", " world"], "usage": {"input_tokens": 10, "output_tokens": 5}}]
    app, provider, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("hi")
    await app._turn_task
    rendered = out.getvalue()
    assert rendered.startswith("> hi\n")
    assert "Hello world" in rendered
    assert provider.requests[0]["messages"][-1] == {"role": "user", "content": "hi"}


def test_set_catalog_binds_late(tmp_path, monkeypatch):
    """The background catalog fetch lands after the chat opened: runner, ctx,
    and the statusline context window all rebind, and the feed announces it."""
    app, _, out = make_app(tmp_path, monkeypatch, [])
    fresh = sample_catalog()
    app.set_catalog(fresh, origin="live", count=42)
    assert app._catalog is fresh
    assert app._runner._catalog is fresh
    assert app.runtime.ctx.catalog is fresh
    assert "models: 42 fetched live" in out.getvalue()


def test_set_catalog_failed_fetch(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    app.set_catalog(sample_catalog(), origin="empty", count=0)
    assert "models: catalog unavailable" in out.getvalue()


async def test_submit_prints_per_answer_stats_line(tmp_path, monkeypatch):
    script = [{"text": "done", "usage": {"input_tokens": 1234, "output_tokens": 42}}]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("hi")
    await app._turn_task
    rendered = out.getvalue()
    assert "ctx 1.2k/" in rendered  # last call's prompt size over the model window
    assert "answer: ↑1.2k in · ↓42 out" in rendered
    assert "1 round" in rendered


async def test_reasoning_and_tool_events_render(tmp_path, monkeypatch):
    script = [
        {
            "reasoning": "let me think",
            "tool_calls": [{"name": "bash", "arguments": '{"command": "echo hi"}'}],
        },
        {"text": "all done"},
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("go")
    await app._turn_task
    rendered = out.getvalue()
    assert "▸ thinking (1 tokens)" in rendered
    assert "⚙ bash(" in rendered
    assert "all done" in rendered


async def test_queue_and_steer_while_running(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    await app._submit("first")
    await wait_for(lambda: len(provider.requests) == 1)
    await app._submit("queued-msg")
    await app._submit("steer-msg", steer=True)
    assert app._input_queue.qsize() == 1
    assert app._steer_queue.qsize() == 1
    assert app._status.queued == 1
    assert app._status.steered == 1
    # queued messages are NOT printed while waiting — they are listed in the
    # chatbox title instead
    rendered = out.getvalue()
    assert "> queued-msg" not in rendered
    assert "> steer-msg" not in rendered
    title = app._chatbox_title()
    assert "queue: queued-msg" in title
    assert "steer: steer-msg" in title
    provider.blocked = False
    provider.release.set()
    await wait_for(
        lambda: not app._turn_running() and app._input_queue.empty() and app._steer_queue.empty()
    )
    rendered = out.getvalue()
    # steer queue drains first: its message reaches the model before the queued one
    user_msgs = [
        m["content"] for req in provider.requests for m in req["messages"] if m["role"] == "user"
    ]
    assert user_msgs.index("steer-msg") < user_msgs.index("queued-msg")
    # echoed exactly once, when the model actually saw the message
    assert rendered.count("> queued-msg") == 1
    assert rendered.count("> steer-msg") == 1
    assert app._chatbox_title() == "message"  # drained: title back to rest
    assert app._status.queued == 0 and app._status.steered == 0


async def test_queue_limit_message(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    await app._submit("first")
    await wait_for(lambda: len(provider.requests) == 1)
    for i in range(QUEUE_LIMIT):
        await app._submit(f"q{i}")
    await app._submit("overflow")
    assert f"input queue is full ({QUEUE_LIMIT})" in out.getvalue()
    provider.blocked = False
    provider.release.set()
    await wait_for(lambda: not app._turn_running() and app._input_queue.empty())


async def test_ctrl_c_cancels_running_turn(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    await app._submit("go")
    await wait_for(lambda: len(provider.requests) == 1)
    assert app.cancel_turn() is True
    await app._turn_task
    assert "turn cancelled" in out.getvalue()
    assert app._status.state is StatusLineState.IDLE
    assert app.cancel_turn() is False  # nothing running now


async def test_provider_error_renders_and_recovers(tmp_path, monkeypatch):
    from lecode.providers.openai_compat import ProviderError

    error = ProviderError("boom", retryable=False)
    app, _, out = make_app(tmp_path, monkeypatch, [{"error": error}, {"text": "recovered"}])
    await app._submit("hi")
    await app._turn_task
    assert "✗" in out.getvalue() and "boom" in out.getvalue()
    await app._submit("again")
    await app._turn_task
    assert "recovered" in out.getvalue()


# -- shell-outs and slash commands -------------------------------------------


async def test_bang_runs_shell_without_llm(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [])
    await app._submit("!echo hello-shell")
    assert "hello-shell" in out.getvalue()
    assert provider.requests == []


async def test_double_bang_feeds_output_to_llm(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "noted"}])
    await app._submit("!!echo from-shell")
    await app._turn_task
    assert "from-shell" in out.getvalue()
    user_msg = provider.requests[0]["messages"][-1]
    assert user_msg["role"] == "user"
    assert "!echo from-shell" in user_msg["content"]
    assert "from-shell" in user_msg["content"]


async def test_bang_error_exit_code_is_error_result(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app._submit("!exit 3")
    assert "(no output)" in out.getvalue()


async def test_quit_and_exit_commands(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    await app._submit("/quit")
    assert app._quit is True
    app._quit = False
    await app._submit("/exit")
    assert app._quit is True


async def test_unknown_command_info(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app._submit("/bogus arg")
    assert "unknown command: /bogus" in out.getvalue()


# -- agent cycling / totals ----------------------------------------------------


def test_tab_cycles_primary_agents(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    base_checker = app._base_checker
    assert app._status.agent == "build"
    assert app.cycle_agent() == "plan"
    assert app._status.agent == "plan"
    assert app._runtime.ctx.permission_checker is not base_checker  # overlay applied
    assert app.cycle_agent() == "build"
    assert app._runtime.ctx.permission_checker is base_checker
    assert "agent: plan" in out.getvalue()


async def test_totals_line_on_exit(tmp_path, monkeypatch):
    script = [
        {
            "text": "hi",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001},
        }
    ]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("hello")
    await app._turn_task
    app.print_totals()
    assert "Session test-session: tokens 10 in / 5 out · cost $0.0010" in out.getvalue()


async def test_statusline_totals_update_after_turn(tmp_path, monkeypatch):
    script = [
        {
            "text": "hi",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001},
        }
    ]
    app, _, _ = make_app(tmp_path, monkeypatch, script)
    await app._submit("hello")
    await app._turn_task
    assert app._status.input_tokens == 10
    assert app._status.output_tokens == 5
    assert app._status.cost_usd == pytest.approx(0.001)
    assert app._status.context_used == 10


# -- end-to-end pipe-input smokes ---------------------------------------------


async def test_pipe_smoke_submit_answer_quit_totals(tmp_path, monkeypatch):
    script = [{"text": "streamed answer", "usage": {"input_tokens": 3, "output_tokens": 2}}]
    app, _, out = make_app(tmp_path, monkeypatch, script)
    with create_pipe_input() as inp:
        inp.send_text("hello\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "streamed answer" in out.getvalue())
        inp.send_text("/quit\r")
        code = await task
    assert code == 0
    assert "Session test-session: tokens 3 in / 2 out" in out.getvalue()


async def test_pipe_slash_command_echoed(tmp_path, monkeypatch):
    """A submitted slash command is echoed to the feed, like chat prompts."""
    app, _, out = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        inp.send_text("/model\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "model:" in out.getvalue())
        inp.send_text("/quit\r")
        code = await task
    assert code == 0
    assert "/model" in out.getvalue()  # the echo, not just the "model:" result


async def test_pipe_ctrl_c_exits_cleanly(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        inp.send_text("\x03")
        code = await app.run(input=inp, output=DummyOutput())
    assert code == 0


async def test_pipe_ctrl_d_exits_cleanly(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        inp.send_text("\x04")
        code = await app.run(input=inp, output=DummyOutput())
    assert code == 0


async def test_pipe_eof_exits_cleanly(tmp_path, monkeypatch):
    """stdin at EOF (non-tty pipe) is a clean exit, not a crash."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        pass  # closing the pipe = EOF
    code = await app.run(input=inp, output=DummyOutput())
    assert code == 0


async def test_pipe_submission_recorded_in_history(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [{"text": "ok"}])
    with create_pipe_input() as inp:
        inp.send_text("remember this\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: not app._turn_running() and app._turn_task is not None)
        inp.send_text("/quit\r")
        assert await task == 0
    from lecode.tui.input import SessionHistory

    assert "remember this" in SessionHistory(app._store, app.session).load_history_strings()


async def test_pipe_shift_enter_and_ctrl_j_insert_newline(tmp_path, monkeypatch):
    """Shift-Enter (kitty sequence) and Ctrl-J insert a newline; Enter submits."""
    script = [{"text": "first answer"}, {"text": "second answer"}]
    app, provider, _ = make_app(tmp_path, monkeypatch, script)
    with create_pipe_input() as inp:
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        inp.send_text("line one\x1b[13;2uline two\r")  # shift+enter, then enter
        await wait_for(lambda: len(provider.requests) == 1)
        inp.send_text("a\nb\r")  # ctrl-j, then enter
        await wait_for(lambda: len(provider.requests) == 2)
        inp.send_text("/quit\r")
        assert await task == 0
    contents = [r["messages"][-1]["content"] for r in provider.requests]
    assert contents[0] == "line one\nline two"
    assert contents[1] == "a\nb"


async def test_pipe_draft_persisted_on_eof_exit(tmp_path, monkeypatch):
    """Unsubmitted buffer text survives a restart as a draft."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        inp.send_text("unsubmitted draft")
        inp.close()  # EOF with text still in the buffer
        assert await app.run(input=inp, output=DummyOutput()) == 0
    from lecode.tui.input import SessionHistory

    assert SessionHistory(app._store, app.session).load_draft() == "unsubmitted draft"


# -- CLI wiring ------------------------------------------------------------------

runner = CliRunner()


class FakeTui:
    """Records construction args; ``run`` does nothing and exits 0."""

    instances: ClassVar[list[FakeTui]] = []

    def __init__(self, config, runtime, provider, session, store, **kwargs):
        self.config = config
        self.runtime = runtime
        self.session = session
        FakeTui.instances.append(self)

    async def run(self) -> int:
        return 0

    def attach_worktree(self, manager, info, original_cwd) -> None:
        self.worktree = info


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolated cwd + config dir; deps check, provider, prompt and TuiApp faked."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lecode.cli.check_dependencies", lambda: None)
    monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: object())
    FakeTui.instances = []
    monkeypatch.setattr("lecode.cli.TuiApp", FakeTui)
    return tmp_path


def _name_prompt(value):
    async def _prompt(store, **kwargs):
        return value

    return _prompt


def test_cli_abort_exits_zero_without_session(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt(None))
    result = runner.invoke(cli_app, [])
    assert result.exit_code == 0
    assert FakeTui.instances == []
    assert SessionStore().list_sessions() == []


def test_cli_interactive_creates_named_session(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("chatty"))
    result = runner.invoke(cli_app, [])
    assert result.exit_code == 0
    assert len(FakeTui.instances) == 1
    assert FakeTui.instances[0].session.name == "chatty"
    assert [m.name for m in SessionStore().list_sessions()] == ["chatty"]


def test_cli_default_mode_is_yolo(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("s"))
    result = runner.invoke(cli_app, [])
    assert result.exit_code == 0
    assert FakeTui.instances[0].runtime.ctx.permission_checker.mode == "yolo"


def test_cli_safe_flag_forces_readonly(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("s"))
    result = runner.invoke(cli_app, ["--safe"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].runtime.ctx.permission_checker.mode == "readonly"


def test_cli_read_only_alias_still_works(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("s"))
    result = runner.invoke(cli_app, ["--read-only"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].runtime.ctx.permission_checker.mode == "readonly"


def test_cli_resume_keeps_name_without_prompt(cli_env, monkeypatch):
    SessionStore().create("old-session", cli_env)

    async def _boom(store, **kwargs):
        raise AssertionError("name prompt must not run on --resume")

    monkeypatch.setattr("lecode.cli.prompt_session_name", _boom)
    result = runner.invoke(cli_app, ["-r", "old-session"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].session.name == "old-session"


def test_cli_continue_picks_latest(cli_env, monkeypatch):
    store = SessionStore()
    store.create("first", cli_env)
    store.create("second", cli_env)
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt(None))
    result = runner.invoke(cli_app, ["-c"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].session.name == "second"


def test_cli_resume_unknown_ref_fails(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt(None))
    result = runner.invoke(cli_app, ["-r", "nope"])
    assert result.exit_code == 2
    assert "nope" in result.output


def test_cli_bare_resume_picks_from_folder_sessions(cli_env, monkeypatch):
    SessionStore().create("old-session", cli_env)
    SessionStore().create("other-folder", cli_env / "other")

    async def _pick(store, cwd, **kwargs):
        assert [m.name for m in store.list_sessions() if m.cwd == str(cwd)] == ["old-session"]
        return store.resolve("old-session")

    monkeypatch.setattr("lecode.tui.name_prompt.pick_session", _pick)
    result = runner.invoke(cli_app, ["-r"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].session.name == "old-session"


def test_cli_bare_resume_abort_exits_zero(cli_env, monkeypatch):
    async def _pick(store, cwd, **kwargs):
        return None

    monkeypatch.setattr("lecode.tui.name_prompt.pick_session", _pick)
    result = runner.invoke(cli_app, ["-r"])
    assert result.exit_code == 0
    assert FakeTui.instances == []
    assert SessionStore().list_sessions() == []


def test_cli_no_color_lands_in_config(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("x"))
    result = runner.invoke(cli_app, ["--no-color"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].config.ui.no_color is True


async def test_ctrl_c_cancels_shell_out(tmp_path, monkeypatch):
    """Ctrl-C during a !cmd stops the command instead of quitting."""
    app, _, out = make_app(tmp_path, monkeypatch, [])
    task = asyncio.ensure_future(app._submit("!sleep 30"))
    await wait_for(lambda: app._shell_task is not None)
    assert app.cancel_action() is True
    await task
    assert "shell command cancelled" in out.getvalue()
    assert app.cancel_action() is False  # nothing running now


async def test_ctrl_c_cancels_plan_loop(tmp_path, monkeypatch):
    """Ctrl-C during /loop stops the loop instead of quitting."""
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    plan = tmp_path / "plan.md"
    plan.write_text("- [ ] task one\n- [ ] task two\n")
    app.start_loop(plan, max_iterations=5)
    await wait_for(lambda: len(provider.requests) == 1)
    assert app.cancel_turn() is False  # the loop is not the turn task
    assert app.cancel_action() is True
    await app._loop_task
    assert "loop stopped" in out.getvalue()
    assert app._status.state is StatusLineState.IDLE


async def test_context_meter_grows_during_turn(tmp_path, monkeypatch):
    """Streamed tokens and tool results grow ctx before real usage lands."""
    script = [
        {
            "text": ["word " * 200],  # ~1k chars ≈ 250 estimated tokens
            "tool_calls": [{"name": "list_dir", "arguments": '{"path": "."}'}],
        },
        {"text": "done", "usage": {"input_tokens": 5000, "output_tokens": 300}},
    ]
    app, _, _ = make_app(tmp_path, monkeypatch, script)
    start_ctx = app._status.context_used
    await app._submit("hi")
    await app._turn_task
    # during the turn the meter grew from streaming; at turn end the real
    # usage (context_tokens = 5000) replaced the estimate
    assert app._status.context_used == 5000
    assert app._status.context_used != start_ctx


async def test_estimate_tokens():
    from lecode.tui.statusline import estimate_tokens

    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 400) == 100


async def test_char_per_token_calibrates_from_usage(tmp_path, monkeypatch):
    """Real prompt usage moves the live estimate off the 4.0 default."""
    script = [{"text": "done", "usage": {"input_tokens": 5000, "output_tokens": 10}}]
    app, _, _ = make_app(tmp_path, monkeypatch, script)
    assert app._char_per_token == 4.0
    await app._submit("hi")
    await app._turn_task
    # chars/5000 tokens is a small ratio, clamped to 2.0; EMA: (4+2)/2 = 3.0
    assert app._char_per_token == 3.0
    assert app._estimate("x" * 300) == 100


async def test_switch_session_refused_when_locked(tmp_path, monkeypatch):
    """A session attached to another live process cannot be switched into."""
    app, _, out = make_app(tmp_path, monkeypatch, [])
    other = app._store.create("elsewhere", tmp_path, model="m")
    lock = app._store.acquire_lock(other)  # simulates the other lecode process
    assert lock is not None
    current = app.session
    assert app.switch_session(other) is False
    assert app.session is current  # unchanged
    assert "already open in another lecode process" in out.getvalue()
    lock.release()
    assert app.switch_session(other) is True  # free again after release


def test_cli_resume_locked_session_fails(cli_env, monkeypatch):
    from lecode.session.storage import SessionStore as _Store

    store = _Store()
    session = store.create("busy", cli_env)
    lock = store.acquire_lock(session)
    assert lock is not None
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt(None))
    result = runner.invoke(cli_app, ["-r", "busy"])
    assert result.exit_code == 2
    assert "already open in another lecode process" in result.output
    assert FakeTui.instances == []
