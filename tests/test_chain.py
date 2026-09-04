"""Tests for prompt chaining: run_chain, /chain, --chain."""

from __future__ import annotations

import pytest
from tests.fakes import FakeProvider
from tests.test_tui_app import make_app
from typer.testing import CliRunner

from lecode.agent.runner import AgentRunner
from lecode.agent.tools import ToolRegistry
from lecode.cli import app as cli_app
from lecode.extras.chain import PHASES, run_chain

runner = CliRunner()

TOPIC = "rewrite the parser"


def script_for_phases():
    return [{"text": f"out-{phase}"} for phase in PHASES]


# -- run_chain ---------------------------------------------------------------------


async def test_chain_phases_run_in_order(tool_ctx):
    provider = FakeProvider(script_for_phases())

    def factory():
        return AgentRunner(provider, ToolRegistry([]), tool_ctx)

    result = await run_chain(factory, TOPIC)
    assert [name for name, _ in result.phases] == list(PHASES)
    assert [output for _, output in result.phases] == [f"out-{p}" for p in PHASES]
    assert result.final == "out-review"
    assert len(provider.requests) == 4
    for request in provider.requests:
        assert TOPIC in request["messages"][-1]["content"]


async def test_chain_uses_phase_templates(tool_ctx):
    provider = FakeProvider(script_for_phases())

    def factory():
        return AgentRunner(provider, ToolRegistry([]), tool_ctx)

    await run_chain(factory, TOPIC)
    contents = [r["messages"][-1]["content"] for r in provider.requests]
    assert "Brainstorm aggressively" in contents[0]
    assert "ordered implementation plan" in contents[1]
    assert "Execute the plan below" in contents[2]
    assert "senior engineer" in contents[3]


async def test_chain_output_feeds_next_phase(tool_ctx):
    provider = FakeProvider(script_for_phases())

    def factory():
        return AgentRunner(provider, ToolRegistry([]), tool_ctx)

    await run_chain(factory, TOPIC)
    contents = [r["messages"][-1]["content"] for r in provider.requests]
    assert "out-brainstorm" not in contents[0]
    assert "out-brainstorm" in contents[1]
    assert "out-brainstorm" in contents[2] and "out-plan" in contents[2]
    assert all(f"out-{p}" in contents[3] for p in ("brainstorm", "plan", "code"))


async def test_chain_on_phase_callback(tool_ctx):
    provider = FakeProvider(script_for_phases())
    seen: list[tuple[str, str]] = []

    def factory():
        return AgentRunner(provider, ToolRegistry([]), tool_ctx)

    await run_chain(factory, TOPIC, on_phase=lambda phase, output: seen.append((phase, output)))
    assert seen == [(p, f"out-{p}") for p in PHASES]


async def test_chain_system_prompt(tool_ctx):
    provider = FakeProvider(script_for_phases())

    def factory():
        return AgentRunner(provider, ToolRegistry([]), tool_ctx)

    await run_chain(factory, TOPIC, system_prompt="You are lecode.")
    for request in provider.requests:
        assert request["messages"][0] == {"role": "system", "content": "You are lecode."}


# -- /chain command ------------------------------------------------------------------


async def test_chain_command(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, script_for_phases())
    await app.handle_command(f"/chain {TOPIC}")
    await app._turn_task
    rendered = out.getvalue()
    # the feed renders markdown through Rich, so literal "##" is consumed as a heading
    for phase in PHASES:
        assert f"out-{phase}" in rendered
    assert app._last_response == "out-review"
    # the chain is a side computation: nothing persisted to the session
    assert app.store.load_messages(app.session) == []
    assert len(provider.requests) == 4


async def test_chain_command_needs_topic(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/chain")
    assert "usage: /chain <topic>" in out.getvalue()


async def test_chain_command_busy_refused(tmp_path, monkeypatch):
    from tests.test_tui_app import make_blocking_app, wait_for

    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    await app._submit("first")
    await wait_for(lambda: len(provider.requests) == 1)
    await app.handle_command(f"/chain {TOPIC}")
    assert "finish or cancel the current turn first" in out.getvalue()
    provider.blocked = False
    provider.release.set()
    await wait_for(lambda: not app._turn_running())


# -- CLI --chain ----------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lecode.cli.check_dependencies", lambda: None)
    return tmp_path


def test_cli_chain_prints_phases(cli_env, monkeypatch):
    monkeypatch.setattr(
        "lecode.cli.build_provider",
        lambda config, api_key=None: FakeProvider(script_for_phases()),
    )
    result = runner.invoke(cli_app, ["--chain", TOPIC])
    assert result.exit_code == 0, result.output
    for phase in PHASES:
        assert f"## {phase}" in result.output
        assert f"out-{phase}" in result.output


def test_cli_chain_and_prompt_are_exclusive(cli_env):
    result = runner.invoke(cli_app, ["-p", "hi", "--chain", TOPIC])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
