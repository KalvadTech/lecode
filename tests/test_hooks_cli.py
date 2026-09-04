"""Tests for the ``--hooks-test`` CLI dry-run."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lecode.cli import EXIT_ERROR, EXIT_OK, app

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
