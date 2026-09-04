"""Tests for the --setup onboarding wizard and the first-run offer."""

from __future__ import annotations

import stat
import tomllib

import pytest
from typer.testing import CliRunner

from lecode import setup_wizard
from lecode.cli import app as cli_app
from lecode.setup_wizard import build_config, gather_answers, run_wizard

runner = CliRunner()


class FakeSession:
    """A PromptSession stand-in yielding scripted answers."""

    def __init__(self, answers):
        self._answers = iter(answers)
        self.messages: list[str] = []

    async def prompt_async(self, message=""):
        self.messages.append(message)
        try:
            return next(self._answers)
        except StopIteration:
            raise EOFError from None


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path / "cfg"


# -- the question flow -----------------------------------------------------------


async def test_wizard_happy_path_writes_config(cfg_dir):
    session = FakeSession(
        ["1", "sk-or-test-key", "1", "1", "", "n"]
    )  # provider, key, model, theme, notifications (default), advisor
    path = await run_wizard(session)
    assert path == cfg_dir / "config.toml"
    raw = tomllib.loads(path.read_text())
    assert raw["llm"]["provider"] == "openrouter"
    assert raw["llm"]["model"] == "openai/gpt-5-mini"
    assert raw["llm"]["api_key"] == "sk-or-test-key"
    assert raw["ui"]["theme"]
    assert raw["notifications"]["enabled"] is True  # empty answer → default yes
    assert "advisor" not in raw


async def test_wizard_config_file_is_owner_only(cfg_dir):
    await run_wizard(FakeSession(["1", "sk-or-key", "1", "1", "y", "n"]))
    mode = stat.S_IMODE((cfg_dir / "config.toml").stat().st_mode)
    assert mode == 0o600


async def test_wizard_custom_provider_asks_base_url(cfg_dir):
    session = FakeSession(["custom", "https://llm.local/v1", "local-key", "2", "1", "n", "n"])
    await run_wizard(session)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["llm"]["provider"] == "custom"
    assert raw["llm"]["base_url"] == "https://llm.local/v1"
    assert raw["llm"]["model"] == "openai/gpt-5"  # pick 2


async def test_wizard_base_url_validated(cfg_dir):
    session = FakeSession(["3", "ftp://nope", "https://ok.example/v1", "", "1", "1", "", "n"])
    await run_wizard(session)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["llm"]["base_url"] == "https://ok.example/v1"
    assert "api_key" not in raw["llm"]  # custom tolerates an empty key


async def test_wizard_key_required_loops_until_nonempty(cfg_dir):
    session = FakeSession(["openai", "", "", "sk-live", "1", "1", "", "n"])
    await run_wizard(session)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["llm"]["api_key"] == "sk-live"


async def test_wizard_advisor_opt_in(cfg_dir):
    await run_wizard(FakeSession(["1", "sk-or-key", "1", "1", "", "y"]))
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["advisor"] == {"enabled": True, "model": "openai/gpt-5-mini"}


async def test_wizard_notifications_off(cfg_dir):
    await run_wizard(FakeSession(["1", "sk-or-key", "1", "1", "n", "n"]))
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["notifications"]["enabled"] is False


def test_build_config_minimal_shape():
    raw = build_config(
        {
            "provider": "openai",
            "base_url": "",
            "api_key": "k",
            "model": "openai/gpt-5-mini",
            "theme": "default",
            "notifications": True,
            "advisor": False,
        }
    )
    assert raw == {
        "schema_version": 1,
        "llm": {"provider": "openai", "model": "openai/gpt-5-mini", "api_key": "k"},
        "ui": {"theme": "default"},
        "notifications": {"enabled": True},
    }


async def test_gather_answers_theme_menu_uses_real_themes(cfg_dir):
    session = FakeSession(["1", "sk-or-key", "1", "dracula", "", "n"])
    answers = await gather_answers(session)
    assert answers["theme"] == "dracula"  # exact theme name accepted


# -- --setup CLI flag --------------------------------------------------------------


class _FakeTty:
    def isatty(self):
        return True


def test_setup_non_tty_exits_2(cfg_dir):
    result = runner.invoke(cli_app, ["--setup"])  # CliRunner stdin is not a tty
    assert result.exit_code == 2
    assert "interactive terminal" in result.output


def test_setup_runs_wizard_and_exits(cfg_dir, monkeypatch):
    from lecode.cli import run_setup
    from lecode.config.loader import config_dir

    monkeypatch.setattr("sys.stdin", _FakeTty())
    called = []

    async def fake_wizard():
        called.append(True)
        return config_dir() / "config.toml"

    monkeypatch.setattr("lecode.cli.run_wizard", fake_wizard)
    assert run_setup() == 0
    assert called == [True]


def test_setup_cancelled_exits_1(cfg_dir, monkeypatch):
    from lecode.cli import run_setup

    monkeypatch.setattr("sys.stdin", _FakeTty())

    async def cancelled():
        raise KeyboardInterrupt

    monkeypatch.setattr("lecode.cli.run_wizard", cancelled)
    assert run_setup() == 1


# -- the first-run offer --------------------------------------------------------------


async def test_offer_no_runs_nothing(monkeypatch):
    monkeypatch.setattr(setup_wizard, "PromptSession", lambda: FakeSession(["n"]))
    ran = []

    async def fake_wizard(session=None):
        ran.append(True)

    monkeypatch.setattr(setup_wizard, "run_wizard", fake_wizard)
    assert await setup_wizard.offer_first_run_setup() is False
    assert ran == []


async def test_offer_yes_runs_wizard(monkeypatch):
    monkeypatch.setattr(setup_wizard, "PromptSession", lambda: FakeSession(["y"]))
    ran = []

    async def fake_wizard(session=None):
        ran.append(True)

    monkeypatch.setattr(setup_wizard, "run_wizard", fake_wizard)
    assert await setup_wizard.offer_first_run_setup() is True
    assert ran == [True]


async def test_offer_eof_means_no(monkeypatch):
    monkeypatch.setattr(setup_wizard, "PromptSession", lambda: FakeSession([]))
    assert await setup_wizard.offer_first_run_setup() is False


def _patch_past_offer(monkeypatch, offered):
    """Fake the setup offer and stop run_interactive at the name prompt."""

    async def fake_offer():
        offered.append(True)
        return False

    async def no_name(store, **kwargs):
        return None  # Ctrl-D at the name prompt → exit 0

    monkeypatch.setattr("lecode.cli.offer_first_run_setup", fake_offer)
    monkeypatch.setattr("lecode.cli.prompt_session_name", no_name)


def test_interactive_first_run_offers_setup(tmp_path, monkeypatch):
    """No config + tty → the offer runs before the session-name prompt."""
    from lecode.cli import run_interactive

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", _FakeTty())
    offered: list[bool] = []
    _patch_past_offer(monkeypatch, offered)
    assert run_interactive() == 0
    assert offered == [True]


def test_interactive_existing_config_skips_offer(tmp_path, monkeypatch):
    from lecode.cli import run_interactive

    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", _FakeTty())
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.toml").write_text("schema_version = 1\n")
    offered: list[bool] = []
    _patch_past_offer(monkeypatch, offered)
    assert run_interactive() == 0
    assert offered == []
