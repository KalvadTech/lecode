"""Tests for the ``--hooks-test`` CLI dry-run."""

from __future__ import annotations

import json

import pytest
from tests.fakes import FakeProvider
from typer.testing import CliRunner

from lecode.cli import EXIT_ERROR, EXIT_OK, app
from lecode.hooks import EVENTS

runner = CliRunner()


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """An isolated config dir; returns a function writing the hooks table."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config_file = tmp_path / "cfg" / "config.toml"
    config_file.parent.mkdir(parents=True)

    def write(hooks_toml: str) -> None:
        config_file.write_text(f"schema_version = 1\n{hooks_toml}", encoding="utf-8")

    return write


def test_hooks_test_no_hooks_configured(cfg_dir):
    cfg_dir("")
    result = runner.invoke(app, ["--hooks-test"])
    assert result.exit_code == EXIT_OK
    assert "no hooks configured" in result.stdout


def test_hooks_test_prints_merged_verdicts(cfg_dir):
    command = json.dumps('echo \'{"verdict": "ask", "reason": "sure?"}\'')
    cfg_dir(f"[hooks]\nPreToolUse = [{command}]\n")
    result = runner.invoke(app, ["--hooks-test"])
    assert result.exit_code == EXIT_OK
    assert "PreToolUse: ask — sure?" in result.stdout


def test_hooks_test_failing_handler_exits_1(cfg_dir):
    allow = json.dumps('echo \'{"verdict": "allow"}\'')
    cfg_dir(f'[hooks]\nStop = ["exit 1"]\nPreToolUse = [{allow}]\n')
    result = runner.invoke(app, ["--hooks-test"])
    assert result.exit_code == EXIT_ERROR
    assert "handler failed" in result.stdout
    assert "PreToolUse: allow" in result.stdout


def test_hooks_test_unknown_event_warns(cfg_dir):
    cfg_dir('[hooks]\nBogus = ["echo hi"]\n')
    result = runner.invoke(app, ["--hooks-test"])
    assert result.exit_code == EXIT_OK
    assert "unknown hook event: Bogus" in result.stderr
    assert "no hooks configured" in result.stdout


def test_hooks_test_pre_tool_use_failure_is_deny(cfg_dir):
    cfg_dir('[hooks]\nPreToolUse = ["echo garbage"]\n')
    result = runner.invoke(app, ["--hooks-test"])
    assert result.exit_code == EXIT_ERROR
    assert "PreToolUse: deny" in result.stdout


def test_hooks_test_covers_all_15_events(cfg_dir):
    lines = "\n".join(f'{event} = ["true"]' for event in EVENTS)
    cfg_dir(f"[hooks]\n{lines}\n")
    result = runner.invoke(app, ["--hooks-test"])
    assert result.exit_code == EXIT_OK
    assert len(EVENTS) == 15
    for event in EVENTS:
        assert f"{event}: allow" in result.stdout


# -- headless fire sites (lecode -p) ----------------------------------------------


@pytest.fixture
def headless(tmp_path, monkeypatch):
    """Headless run with a fake provider; hooks append envelopes to a log."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.setattr("lecode.cli.find_missing_binaries", lambda: [])
    provider = FakeProvider([{"text": "done"}])
    monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: provider)
    log = tmp_path / "hooks.jsonl"
    config_file = tmp_path / "cfg" / "config.toml"
    config_file.parent.mkdir(parents=True)

    def write(hooks: dict[str, str]) -> None:
        lines = "\n".join(f"{event} = [{json.dumps(command)}]" for event, command in hooks.items())
        config_file.write_text(f"schema_version = 1\n[hooks]\n{lines}\n", encoding="utf-8")

    def read() -> list[dict]:
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    return provider, write, read, log


def test_headless_fires_session_prompt_and_stop_hooks(headless):
    provider, write, read, log = headless
    write(
        {
            "SessionStart": f"cat >> {log}; echo >> {log}",
            "SessionEnd": f"cat >> {log}; echo >> {log}",
            "UserPromptSubmit": f"cat >> {log}; echo >> {log}",
            "Stop": f"cat >> {log}; echo >> {log}",
        }
    )
    result = runner.invoke(app, ["-p", "hello hooks"])
    assert result.exit_code == EXIT_OK
    events = read()
    names = [e["event"] for e in events]
    assert names[0] == "SessionStart"
    assert names[-1] == "SessionEnd"
    prompt = next(e for e in events if e["event"] == "UserPromptSubmit")
    assert prompt["prompt"] == "hello hooks"
    stop = next(e for e in events if e["event"] == "Stop")
    assert stop["reason"] == "done"
    assert provider.requests  # the run actually happened


def test_headless_user_prompt_submit_deny_blocks(headless):
    provider, write, _, _log = headless
    deny = f"echo '{json.dumps({'verdict': 'deny', 'reason': 'no prompts today'})}'"
    write({"UserPromptSubmit": deny})
    result = runner.invoke(app, ["-p", "hello"])
    assert result.exit_code == EXIT_ERROR
    assert "prompt blocked by hook" in result.stderr
    assert "no prompts today" in result.stderr
    assert provider.requests == []  # the run never started


def test_headless_user_prompt_submit_handler_failure_blocks(headless):
    """Deny-safe: a crashing UserPromptSubmit handler blocks the prompt."""
    provider, write, _, _log = headless
    write({"UserPromptSubmit": "exit 1"})
    result = runner.invoke(app, ["-p", "hello"])
    assert result.exit_code == EXIT_ERROR
    assert "prompt blocked by hook" in result.stderr
    assert provider.requests == []
