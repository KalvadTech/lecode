"""Tests for the CLI bootstrap (version flag, dependency gating)."""

from __future__ import annotations

import re

import pytest
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


@pytest.mark.parametrize(
    "mode",
    [
        [],
        ["-p", "hi"],
        ["--loop", "plan.md"],
        ["--chain", "topic"],
        ["--resume", "existing"],
        ["--continue"],
    ],
)
def test_run_flags_override_config_in_every_mode(mode, tmp_path, monkeypatch):
    from lecode.auth import AuthError
    from lecode.session.storage import SessionStore

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lecode.cli.check_dependencies", lambda: None)
    config_path = tmp_path / "cfg" / "config.toml"
    config_path.parent.mkdir()
    original = 'schema_version = 1\n\n[llm]\nthinking = "low"\n'
    config_path.write_text(original)
    SessionStore().create("existing", tmp_path)

    async def name_prompt(store):
        return "new-session"

    monkeypatch.setattr("lecode.cli.prompt_session_name", name_prompt)
    captured = []

    def build_provider(config, api_key=None):
        captured.append(config)
        raise AuthError("stopped before network startup")

    monkeypatch.setattr("lecode.cli.build_provider", build_provider)
    result = runner.invoke(
        app,
        [
            *mode,
            "--thinking",
            "high",
            "--header",
            "X-Team: earlier",
            "--header",
            "x-team: runtime-value",
            "--header",
            "X-Route: https://example.test:8443",
            "--header",
            "X-Empty:",
        ],
    )
    assert result.exit_code == EXIT_STARTUP, result.output
    assert len(captured) == 1
    config = captured[0]
    assert config.llm.thinking == "high"
    assert config.llm._cli_headers == {
        "x-team": "runtime-value",
        "x-route": "https://example.test:8443",
        "x-empty": "",
    }
    assert "runtime-value" not in config.model_dump_json()
    assert "runtime-value" not in repr(config)
    assert config_path.read_text() == original
    assert all(
        "runtime-value" not in path.read_text() for path in config_path.parent.rglob("*.jsonl")
    )


@pytest.mark.parametrize(
    "header",
    [
        "sensitive-value",
        ": sensitive-value",
        "Bad Name: sensitive-value",
        "Bad/Name: sensitive-value",
        "X-Test: sensitive-value\r\nX-Injected: yes",
        "X-Test: sensitive-value\n",
        "X-Test: sensitive-value\x00",
        "X-Test: sensitive-value\x7f",
        "X-Test: sensitive-valueé",
    ],
)
def test_invalid_header_fails_before_startup_without_echoing_value(header, monkeypatch):
    def unexpected_startup():
        pytest.fail("invalid header reached startup")

    monkeypatch.setattr("lecode.cli.check_dependencies", unexpected_startup)
    result = runner.invoke(app, ["--header", header])
    assert result.exit_code == EXIT_STARTUP
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--header" in plain
    assert "sensitive-value" not in result.output


def test_invalid_thinking_fails_before_startup(monkeypatch):
    monkeypatch.setattr("lecode.cli.check_dependencies", lambda: pytest.fail("reached startup"))
    result = runner.invoke(app, ["--thinking", "ultra"])
    assert result.exit_code == EXIT_STARTUP
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--thinking" in plain


def _fake_missing():
    from lecode.deps import MissingBinary

    return [MissingBinary(name="fd", install_hint="brew install fd")]
