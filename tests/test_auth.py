"""Tests for the API-key priority chain and auth policies."""

from __future__ import annotations

import pytest

from lecode.auth import AuthError, resolve_api_key
from lecode.config.models import Config

ALL_KEY_VARS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "MY_LLM_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ALL_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


def _config(**kwargs) -> Config:
    return Config.model_validate(kwargs)


def test_cli_flag_wins_over_env_and_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    config = _config(llm={"api_key": "config-key"})
    resolved = resolve_api_key("openrouter", config, cli_key="cli-key")
    assert resolved.key == "cli-key"
    assert resolved.source == "cli"


def test_openrouter_env_order(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    resolved = resolve_api_key("openrouter", _config())
    assert (resolved.key, resolved.source) == ("or-key", "env")


def test_openrouter_falls_back_to_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    resolved = resolve_api_key("openrouter", _config())
    assert (resolved.key, resolved.source) == ("oa-key", "env")


def test_generic_provider_uses_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    resolved = resolve_api_key("openai", _config())
    assert (resolved.key, resolved.source) == ("oa-key", "env")


def test_config_file_key_is_last_resort():
    config = _config(llm={"api_key": "config-key"})
    resolved = resolve_api_key("openrouter", config)
    assert (resolved.key, resolved.source) == ("config-key", "config")


def test_custom_provider_uses_its_env_var(monkeypatch):
    monkeypatch.setenv("MY_LLM_KEY", "custom-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    config = _config(
        custom_providers={"myllm": {"base_url": "http://x", "api_key_env": "MY_LLM_KEY"}}
    )
    resolved = resolve_api_key("myllm", config)
    assert (resolved.key, resolved.source) == ("custom-env-key", "env")


def test_custom_provider_falls_back_to_config_key():
    config = _config(
        llm={"api_key": "config-key"},
        custom_providers={"myllm": {"base_url": "http://x", "api_key_env": "MY_LLM_KEY"}},
    )
    resolved = resolve_api_key("myllm", config)
    assert (resolved.key, resolved.source) == ("config-key", "config")


def test_policy_required_raises_without_key():
    config = _config(llm={"auth_policy": "required"})
    with pytest.raises(AuthError, match="required"):
        resolve_api_key("openrouter", config)


def test_policy_required_passes_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    config = _config(llm={"auth_policy": "required"})
    assert resolve_api_key("openrouter", config).key == "k"


def test_policy_none_never_sends_a_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    config = _config(llm={"auth_policy": "none", "api_key": "k2"})
    resolved = resolve_api_key("openrouter", config, cli_key="k3")
    assert (resolved.key, resolved.source) == (None, "none")


def test_policy_auto_allows_keyless():
    resolved = resolve_api_key("openrouter", _config())
    assert (resolved.key, resolved.source) == (None, "none")


def test_custom_provider_policy_overrides_global(monkeypatch):
    monkeypatch.setenv("MY_LLM_KEY", "k")
    config = _config(
        llm={"auth_policy": "none"},
        custom_providers={
            "myllm": {
                "base_url": "http://x",
                "api_key_env": "MY_LLM_KEY",
                "auth_policy": "auto",
            }
        },
    )
    assert resolve_api_key("myllm", config).key == "k"
    assert resolve_api_key("openrouter", config).key is None
