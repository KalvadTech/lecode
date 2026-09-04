"""Tests for the --setup onboarding wizard and the first-run offer."""

from __future__ import annotations

import json
import stat
import tomllib

import pytest
from typer.testing import CliRunner

from lecode import setup_wizard
from lecode.cli import app as cli_app
from lecode.setup_wizard import (
    build_config,
    detect_import_sources,
    gather_answers,
    import_from_opencode,
    import_from_pi,
    run_wizard,
)

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


@pytest.fixture
def clean_home(tmp_path):
    """An empty home dir so the wizard's import step finds nothing."""
    return tmp_path / "home"


# -- the question flow -----------------------------------------------------------


async def test_wizard_happy_path_writes_config(cfg_dir, clean_home):
    session = FakeSession(
        ["1", "sk-or-test-key", "1", "", "n"]
    )  # provider, key, model, notifications (default), advisor
    path = await run_wizard(session, home=clean_home)
    assert path == cfg_dir / "config.toml"
    raw = tomllib.loads(path.read_text())
    assert raw["llm"]["provider"] == "openrouter"
    assert raw["llm"]["model"] == "deepseek/deepseek-v4-flash"
    assert raw["llm"]["api_key"] == "sk-or-test-key"
    assert "ui" not in raw
    assert raw["notifications"]["enabled"] is True  # empty answer → default yes
    assert "advisor" not in raw


async def test_wizard_config_file_is_owner_only(cfg_dir, clean_home):
    await run_wizard(FakeSession(["1", "sk-or-key", "1", "y", "n"]), home=clean_home)
    mode = stat.S_IMODE((cfg_dir / "config.toml").stat().st_mode)
    assert mode == 0o600


async def test_wizard_custom_provider_asks_base_url(cfg_dir, clean_home):
    session = FakeSession(["custom", "https://llm.local/v1", "local-key", "2", "n", "n"])
    await run_wizard(session, home=clean_home)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["llm"]["provider"] == "custom"
    assert raw["llm"]["base_url"] == "https://llm.local/v1"
    assert raw["llm"]["model"] == "deepseek/deepseek-v4-pro"  # pick 2


async def test_wizard_base_url_validated(cfg_dir, clean_home):
    session = FakeSession(["2", "ftp://nope", "https://ok.example/v1", "", "1", "", "n"])
    await run_wizard(session, home=clean_home)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["llm"]["base_url"] == "https://ok.example/v1"
    assert "api_key" not in raw["llm"]  # custom tolerates an empty key


async def test_wizard_key_required_loops_until_nonempty(cfg_dir, clean_home):
    session = FakeSession(["openrouter", "", "", "sk-live", "1", "", "n"])
    await run_wizard(session, home=clean_home)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["llm"]["api_key"] == "sk-live"


async def test_wizard_advisor_opt_in(cfg_dir, clean_home):
    await run_wizard(FakeSession(["1", "sk-or-key", "1", "", "y"]), home=clean_home)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["advisor"] == {"enabled": True, "model": "deepseek/deepseek-v4-flash"}


async def test_wizard_notifications_off(cfg_dir, clean_home):
    await run_wizard(FakeSession(["1", "sk-or-key", "1", "n", "n"]), home=clean_home)
    raw = tomllib.loads((cfg_dir / "config.toml").read_text())
    assert raw["notifications"]["enabled"] is False


def test_build_config_minimal_shape():
    raw = build_config(
        {
            "provider": "openai",
            "base_url": "",
            "api_key": "k",
            "model": "openai/gpt-5-mini",
            "notifications": True,
            "advisor": False,
        }
    )
    assert raw == {
        "schema_version": 1,
        "llm": {"provider": "openai", "model": "openai/gpt-5-mini", "api_key": "k"},
        "notifications": {"enabled": True},
    }


async def test_model_menu_shows_context_and_price(cfg_dir, clean_home, capsys):
    session = FakeSession(["1", "sk-or-key", "2", "", "n"])
    answers = await gather_answers(session, home=clean_home)
    menu = capsys.readouterr().out
    assert "2) deepseek/deepseek-v4-pro — ctx 1.0M · $1.04/M in · $2.08/M out" in menu
    assert answers["model"] == "deepseek/deepseek-v4-pro"  # bare id returned


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


# -- import from pi / opencode -----------------------------------------------------


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_detect_import_sources_empty(tmp_path):
    assert detect_import_sources(tmp_path) == []


def test_detect_import_sources_finds_both(tmp_path):
    _write(tmp_path / ".pi" / "agent" / "settings.json", {})
    _write(tmp_path / ".config" / "opencode" / "opencode.json", {})
    assert detect_import_sources(tmp_path) == ["pi", "opencode"]


def test_import_from_pi_openrouter(tmp_path):
    _write(
        tmp_path / ".pi" / "agent" / "settings.json",
        {"defaultProvider": "openrouter", "defaultModel": "deepseek/deepseek-v4-pro"},
    )
    _write(
        tmp_path / ".pi" / "agent" / "auth.json",
        {"openrouter": {"type": "api_key", "key": "sk-or-pi"}},
    )
    assert import_from_pi(tmp_path) == {
        "provider": "openrouter",
        "api_key": "sk-or-pi",
        "model": "deepseek/deepseek-v4-pro",
    }


def test_import_from_pi_custom_provider(tmp_path):
    _write(
        tmp_path / ".pi" / "agent" / "settings.json",
        {"defaultProvider": "local", "defaultModel": "qwen3"},
    )
    _write(tmp_path / ".pi" / "agent" / "auth.json", {"local": {"key": "local-key"}})
    _write(
        tmp_path / ".pi" / "agent" / "models.json",
        {"providers": {"local": {"baseUrl": "http://localhost:1234/v1"}}},
    )
    assert import_from_pi(tmp_path) == {
        "provider": "custom",
        "base_url": "http://localhost:1234/v1",
        "api_key": "local-key",
        "model": "qwen3",
    }


def test_import_from_pi_unsupported_provider(tmp_path):
    _write(
        tmp_path / ".pi" / "agent" / "settings.json",
        {"defaultProvider": "anthropic", "defaultModel": "claude-opus-4-8"},
    )
    assert import_from_pi(tmp_path) == {}


def test_import_from_pi_missing_files(tmp_path):
    assert import_from_pi(tmp_path) == {}


def test_import_from_opencode_openrouter(tmp_path):
    _write(
        tmp_path / ".config" / "opencode" / "opencode.json",
        {"model": "openrouter/moonshotai/kimi-k2.6"},
    )
    _write(
        tmp_path / ".local" / "share" / "opencode" / "auth.json",
        {"openrouter": {"type": "api", "key": "sk-or-oc"}},
    )
    assert import_from_opencode(tmp_path) == {
        "provider": "openrouter",
        "model": "moonshotai/kimi-k2.6",
        "api_key": "sk-or-oc",
    }


def test_import_from_opencode_custom_base_url(tmp_path):
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    (tmp_path / ".config" / "opencode" / "opencode.jsonc").write_text(
        '{\n  // comment\n  "model": "corp/qwen3",\n'
        '  "provider": {"corp": {"options": {"baseURL": "https://corp.example/v1"}}}\n}'
    )
    assert import_from_opencode(tmp_path) == {
        "provider": "custom",
        "base_url": "https://corp.example/v1",
        "model": "qwen3",
    }


def test_import_from_opencode_no_model(tmp_path):
    _write(tmp_path / ".config" / "opencode" / "opencode.json", {"$schema": "x"})
    assert import_from_opencode(tmp_path) == {}


async def test_wizard_import_step_prefills_answers(cfg_dir, tmp_path, capsys):
    home = tmp_path / "home"
    _write(
        home / ".pi" / "agent" / "settings.json",
        {"defaultProvider": "openrouter", "defaultModel": "z-ai/glm-4.7"},
    )
    _write(home / ".pi" / "agent" / "auth.json", {"openrouter": {"key": "sk-or-imported"}})
    # import pick, provider (default), key (default), model (default), notif, advisor
    session = FakeSession(["1", "", "", "", "", "n"])
    answers = await gather_answers(session, home=home)
    out = capsys.readouterr().out
    assert "imported from pi" in out
    assert "api key found" in out  # key redacted from the summary
    assert "sk-or-imported" not in out
    assert answers["provider"] == "openrouter"
    assert answers["api_key"] == "sk-or-imported"
    assert answers["model"] == "z-ai/glm-4.7"


async def test_wizard_import_skip_asks_normally(cfg_dir, tmp_path, capsys):
    home = tmp_path / "home"
    _write(home / ".pi" / "agent" / "settings.json", {"defaultProvider": "openrouter"})
    session = FakeSession(["skip", "1", "sk-manual", "1", "", "n"])
    answers = await gather_answers(session, home=home)
    assert "imported" not in capsys.readouterr().out
    assert answers["api_key"] == "sk-manual"


async def test_wizard_import_unimportable_source_falls_back(cfg_dir, tmp_path, capsys):
    home = tmp_path / "home"
    _write(
        home / ".pi" / "agent" / "settings.json",
        {"defaultProvider": "anthropic", "defaultModel": "claude-opus-4-8"},
    )
    session = FakeSession(["pi", "1", "sk-manual", "1", "", "n"])
    answers = await gather_answers(session, home=home)
    assert "nothing importable" in capsys.readouterr().out
    assert answers["provider"] == "openrouter"
