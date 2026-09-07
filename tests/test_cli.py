"""Tests for the CLI bootstrap (version flag, dependency gating)."""

from __future__ import annotations

import re

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


def test_help_flag_short():
    """``-h`` is an alias for ``--help``."""
    for flag in ("-h", "--help"):
        result = runner.invoke(app, [flag])
        assert result.exit_code == EXIT_OK
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "Usage: lecode" in plain


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


def test_fetch_catalog_uses_a_throwaway_client(monkeypatch):
    """The catalog fetch never shares the session client's connection pool.

    Pooling a connection on one ``asyncio.run`` loop and reusing it on the
    TUI's loop crashes TLS teardown with "Event loop is closed".
    """
    import lecode.cli as cli
    from lecode.config.models import Config
    from lecode.providers.catalog import Catalog
    from lecode.providers.live import LoadedCatalog

    closed: list[bool] = []

    class FakeClient:
        async def aclose(self):
            closed.append(True)

    sentinel = FakeClient()
    captured: dict = {}
    monkeypatch.setattr(cli, "build_provider", lambda config, api_key=None: sentinel)

    async def fake_load(client):
        captured["client"] = client
        return LoadedCatalog(Catalog.default(), "live")

    monkeypatch.setattr(cli, "load_catalog", fake_load)
    result = cli.fetch_catalog(Config())
    assert result.origin == "live"
    assert captured["client"] is sentinel
    assert closed  # built, used, and closed inside the fetch's own loop


def _fake_missing():
    from lecode.deps import MissingBinary

    return [MissingBinary(name="fd", install_hint="brew install fd")]
