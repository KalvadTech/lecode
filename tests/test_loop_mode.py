"""Tests for plan-file loop mode: parsing, the loop core, --loop, /loop."""

from __future__ import annotations

import re

import pytest
from tests.fakes import FakeProvider
from tests.test_tui_app import make_app, make_blocking_app, wait_for
from typer.testing import CliRunner

from lecode.cli import app as cli_app
from lecode.config.models import Config
from lecode.extras.loop_mode import (
    EXIT_DONE,
    EXIT_ERROR,
    EXIT_MAX_ITERATIONS,
    loop_session_name,
    run_plan_loop,
    unfinished_items,
)
from lecode.session.storage import SessionStore

runner = CliRunner()


# -- checklist parsing ---------------------------------------------------------


def test_unfinished_items_parsing():
    text = """# Plan

- [ ] first task
- [x] done task
- [X] also done
* [ ] bullet task
- not a checklist item
- [ ]last-in-line
"""
    assert unfinished_items(text) == ["first task", "bullet task"]


def test_loop_session_name_format():
    assert re.fullmatch(r"loop-\d{8}-\d{6}", loop_session_name())


# -- the loop core ---------------------------------------------------------------


def mark_first_done(plan) -> None:
    """What a successful agent does: flip the first open item."""
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("- [ ] ", "- [x] ", 1), encoding="utf-8"
    )


async def test_loop_completes_plan(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\n- [ ] first task\n- [ ] second task\n", encoding="utf-8")
    prompts: list[str] = []
    progress: list[str] = []
    texts: list[str] = []

    async def run_iteration(prompt: str) -> str:
        prompts.append(prompt)
        mark_first_done(plan)
        return "did one item"

    result = await run_plan_loop(
        run_iteration, plan, on_progress=progress.append, on_text=texts.append
    )
    assert result.exit_code == EXIT_DONE
    assert result.stop_reason == "done"
    assert result.iterations == 2
    assert "first task" in prompts[0] and "second task" in prompts[0]
    assert "second task" in prompts[1] and "first task" not in prompts[1]
    assert any("iteration 1" in line for line in progress)
    assert any("plan complete after 2" in line for line in progress)
    assert texts == ["did one item", "did one item"]


async def test_loop_max_iterations(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("- [ ] never done\n", encoding="utf-8")

    async def run_iteration(prompt: str) -> str:
        return "still thinking"

    result = await run_plan_loop(run_iteration, plan, max_iterations=3)
    assert result.exit_code == EXIT_MAX_ITERATIONS
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert result.remaining == ["never done"]


async def test_loop_already_complete_plan(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("- [x] all done\n", encoding="utf-8")
    calls = 0

    async def run_iteration(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return ""

    result = await run_plan_loop(run_iteration, plan)
    assert result.exit_code == EXIT_DONE
    assert result.iterations == 0
    assert calls == 0


async def test_loop_missing_plan_file(tmp_path):
    async def run_iteration(prompt: str) -> str:
        raise AssertionError("must not run")

    result = await run_plan_loop(run_iteration, tmp_path / "nope.md")
    assert result.exit_code == EXIT_ERROR
    assert "cannot read plan file" in result.error


async def test_loop_cmd_failure_feeds_next_prompt(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("- [ ] one\n- [ ] two\n", encoding="utf-8")
    prompts: list[str] = []

    async def run_iteration(prompt: str) -> str:
        prompts.append(prompt)
        mark_first_done(plan)
        return "ok"

    result = await run_plan_loop(run_iteration, plan, loop_cmd="echo boom && exit 3", cwd=tmp_path)
    assert result.exit_code == EXIT_DONE
    assert len(prompts) == 2
    assert "verification command" in prompts[1]
    assert "exit code 3" in prompts[1]
    assert "boom" in prompts[1]


async def test_loop_cmd_passing(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("- [ ] one\n- [ ] two\n", encoding="utf-8")
    prompts: list[str] = []

    async def run_iteration(prompt: str) -> str:
        prompts.append(prompt)
        mark_first_done(plan)
        return "ok"

    await run_plan_loop(run_iteration, plan, loop_cmd="true", cwd=tmp_path)
    assert "passed" in prompts[1]


# -- CLI --loop --------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolated cwd + config dir; deps check faked."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lecode.cli.check_dependencies", lambda: None)
    return tmp_path


def test_cli_loop_completes_plan(cli_env, monkeypatch):
    plan = cli_env / "plan.md"
    plan.write_text("- [ ] write the file\n", encoding="utf-8")
    script = [
        {"tool_calls": [{"name": "read", "arguments": '{"path": "plan.md"}'}]},
        {
            "tool_calls": [
                {
                    "name": "edit",
                    "arguments": (
                        '{"path": "plan.md", "old_string": "- [ ] write the file", '
                        '"new_string": "- [x] write the file"}'
                    ),
                }
            ]
        },
        {"text": "item done"},
    ]
    monkeypatch.setattr(
        "lecode.cli.build_provider", lambda config, api_key=None: FakeProvider(script)
    )
    result = runner.invoke(cli_app, ["--loop", "plan.md"])
    assert result.exit_code == 0, result.output
    assert "- [x] write the file" in plan.read_text(encoding="utf-8")
    assert "item done" in result.output  # agent text to stdout
    assert "iteration 1" in result.output  # progress to stderr (mixed by the runner)
    sessions = SessionStore().list_sessions()
    assert sessions and sessions[0].name.startswith("loop-")


def test_cli_loop_max_iterations_exit_3(cli_env, monkeypatch):
    (cli_env / "plan.md").write_text("- [ ] stuck\n", encoding="utf-8")
    script = [{"text": "no progress"}]
    monkeypatch.setattr(
        "lecode.cli.build_provider", lambda config, api_key=None: FakeProvider(script)
    )
    result = runner.invoke(cli_app, ["--loop", "plan.md", "--max-iterations", "2"])
    assert result.exit_code == 3
    assert "max iterations (2)" in result.output


def test_cli_loop_missing_plan_exit_1(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: FakeProvider([]))
    result = runner.invoke(cli_app, ["--loop", "nope.md"])
    assert result.exit_code == 1
    assert "cannot read plan file" in result.output


def test_cli_loop_and_prompt_are_exclusive(cli_env):
    result = runner.invoke(cli_app, ["-p", "hi", "--loop", "plan.md"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


# -- /loop command ------------------------------------------------------------------


def make_loop_app(tmp_path, monkeypatch, script):
    config = Config()
    config.permissions.mode = "yolo"
    return make_app(tmp_path, monkeypatch, script, config=config)


async def test_loop_command_runs_to_completion(tmp_path, monkeypatch):
    (tmp_path / "plan.md").write_text("- [ ] only task\n", encoding="utf-8")
    script = [
        {"tool_calls": [{"name": "read", "arguments": '{"path": "plan.md"}'}]},
        {
            "tool_calls": [
                {
                    "name": "edit",
                    "arguments": (
                        '{"path": "plan.md", "old_string": "- [ ] only task", '
                        '"new_string": "- [x] only task"}'
                    ),
                }
            ]
        },
        {"text": "did the task"},
    ]
    app, _, out = make_loop_app(tmp_path, monkeypatch, script)
    await app.handle_command("/loop plan.md")
    await wait_for(lambda: app._loop_task is not None and app._loop_task.done())
    rendered = out.getvalue()
    assert "loop started: plan.md" in rendered
    assert "loop done: plan complete after 1 iteration(s)" in rendered
    assert "- [x] only task" in (tmp_path / "plan.md").read_text(encoding="utf-8")
    # iterations are turns in the current session
    roles = [m.role for m in app.store.load_messages(app.session)]
    assert "user" in roles and "assistant" in roles


async def test_loop_command_stop(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("- [ ] never\n", encoding="utf-8")
    await app.handle_command("/loop plan.md")
    await wait_for(lambda: len(provider.requests) == 1)
    await app.handle_command("/loop stop")
    await wait_for(lambda: app._loop_task is not None and app._loop_task.done())
    assert "loop stopped" in out.getvalue()


async def test_loop_command_usage_errors(tmp_path, monkeypatch):
    app, _, out = make_loop_app(tmp_path, monkeypatch, [])
    await app.handle_command("/loop")
    await app.handle_command("/loop missing.md")
    await app.handle_command("/loop stop")
    rendered = out.getvalue()
    assert "usage: /loop <plan-file>" in rendered
    assert "no such plan file" in rendered
    assert "no loop running" in rendered


async def test_loop_command_refuses_second_loop(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("- [ ] never\n", encoding="utf-8")
    await app.handle_command("/loop plan.md")
    await wait_for(lambda: len(provider.requests) == 1)
    await app.handle_command("/loop plan.md")
    assert "already running" in out.getvalue()
    await app.handle_command("/loop stop")
    await wait_for(lambda: app._loop_task is not None and app._loop_task.done())


async def test_prompts_refused_while_loop_runs(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_text("- [ ] never\n", encoding="utf-8")
    await app.handle_command("/loop plan.md")
    await wait_for(lambda: len(provider.requests) == 1)
    await app._submit("hello")
    assert "a plan loop is running" in out.getvalue()
    assert app._input_queue.empty()
    await app.handle_command("/loop stop")
    await wait_for(lambda: app._loop_task is not None and app._loop_task.done())
