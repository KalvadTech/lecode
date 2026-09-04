"""Tests for the interactive TUI application and the CLI interactive path."""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, ClassVar

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from tests.fakes import FakeProvider
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
    app = TuiApp(config, runtime, provider, session, store, console=console)
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


def make_blocking_app(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
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


async def test_submit_streams_answer(tmp_path, monkeypatch):
    script = [{"text": ["Hello", " world"], "usage": {"input_tokens": 10, "output_tokens": 5}}]
    app, provider, out = make_app(tmp_path, monkeypatch, script)
    await app._submit("hi")
    await app._turn_task
    rendered = out.getvalue()
    assert "> hi" in rendered
    assert "Hello world" in rendered
    assert provider.requests[0]["messages"][-1] == {"role": "user", "content": "hi"}


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
    assert "> queued-msg" in rendered
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
        inp.send_text("hello\n")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: "streamed answer" in out.getvalue())
        inp.send_text("/quit\n")
        code = await task
    assert code == 0
    assert "Session test-session: tokens 3 in / 2 out" in out.getvalue()


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
        inp.send_text("remember this\n")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        await wait_for(lambda: not app._turn_running() and app._turn_task is not None)
        inp.send_text("/quit\n")
        assert await task == 0
    from lecode.tui.input import JsonlHistory, history_path

    assert "remember this" in JsonlHistory(history_path()).load_history_strings()


async def test_pipe_draft_persisted_on_eof_exit(tmp_path, monkeypatch):
    """Unsubmitted buffer text survives a restart as a draft."""
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    with create_pipe_input() as inp:
        inp.send_text("unsubmitted draft")
        inp.close()  # EOF with text still in the buffer
        assert await app.run(input=inp, output=DummyOutput()) == 0
    from lecode.tui.input import JsonlHistory, history_path

    assert JsonlHistory(history_path()).load_draft() == "unsubmitted draft"


# -- CLI wiring ------------------------------------------------------------------

runner = CliRunner()


class FakeTui:
    """Records construction args; ``run`` does nothing and exits 0."""

    instances: ClassVar[list[FakeTui]] = []

    def __init__(self, config, runtime, provider, session, store, **kwargs):
        self.config = config
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


def test_cli_no_color_lands_in_config(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("x"))
    result = runner.invoke(cli_app, ["--no-color"])
    assert result.exit_code == 0
    assert FakeTui.instances[0].config.ui.no_color is True
