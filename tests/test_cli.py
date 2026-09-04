"""Tests for the CLI bootstrap (version flag, dependency gating)."""

from __future__ import annotations

from typer.testing import CliRunner

from lecode import __version__
from lecode.cli import EXIT_OK, EXIT_STARTUP, app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == EXIT_OK
    assert f"lecode {__version__}" in result.stdout


def test_version_flag_short():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == EXIT_OK
    assert __version__ in result.stdout


def test_startup_fails_when_binaries_missing(monkeypatch):
    monkeypatch.setattr("lecode.cli.find_missing_binaries", lambda: _fake_missing())
    result = runner.invoke(app, [])
    assert result.exit_code == EXIT_STARTUP
    assert "fd" in result.output


def test_startup_ok_when_binaries_present(monkeypatch):
    monkeypatch.setattr("lecode.cli.find_missing_binaries", lambda: [])
    monkeypatch.setattr("lecode.cli.run_interactive", lambda **kwargs: EXIT_OK)
    result = runner.invoke(app, [])
    assert result.exit_code == EXIT_OK


def _fake_missing():
    from lecode.deps import MissingBinary

    return [MissingBinary(name="fd", install_hint="brew install fd")]
