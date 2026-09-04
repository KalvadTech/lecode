"""Tests for provider resolution (openrouter, custom, --base-url) and build_client."""

from __future__ import annotations

import pytest

from lecode.auth import ResolvedKey, resolve_api_key
from lecode.config.models import Config
from lecode.providers import build_client, resolve_provider
from lecode.providers.openrouter import OPENROUTER_BASE_URL


def _config(**kwargs) -> Config:
    return Config.model_validate(kwargs)


def test_openrouter_default():
    spec = resolve_provider(_config())
    assert spec.name == "openrouter"
    assert spec.base_url == OPENROUTER_BASE_URL
    assert spec.headers["X-Title"] == "lecode"
    assert spec.model == "deepseek/deepseek-v4-flash"


def test_openai_is_not_a_builtin():
    with pytest.raises(ValueError, match="unknown provider 'openai'"):
        resolve_provider(_config(llm={"provider": "openai"}))


def test_cli_provider_flag_overrides_config():
    spec = resolve_provider(_config(llm={"provider": "nope"}), cli_provider="openrouter")
    assert spec.name == "openrouter"
    assert spec.base_url == OPENROUTER_BASE_URL


def test_custom_provider_from_config():
    config = _config(
        custom_providers={
            "local": {
                "base_url": "http://localhost:11434/v1",
                "api_key_env": "OLLAMA_KEY",
                "headers": {"X-Gateway": "corp"},
                "auth_policy": "none",
            }
        },
        llm={"provider": "local"},
    )
    spec = resolve_provider(config)
    assert spec.name == "local"
    assert spec.base_url == "http://localhost:11434/v1"
    assert spec.headers == {"X-Gateway": "corp"}
    assert spec.auth_policy == "none"


def test_cli_base_url_makes_adhoc_custom_provider():
    config = _config(llm={"auth_policy": "required", "tls_verify": False})
    spec = resolve_provider(config, cli_base_url="http://localhost:8080/v1")
    assert spec.name == "custom"
    assert spec.base_url == "http://localhost:8080/v1"
    assert spec.auth_policy == "required"
    assert spec.tls_verify is False
    assert spec.headers == {}


def test_llm_base_url_overrides_builtin_default():
    config = _config(llm={"provider": "openrouter", "base_url": "https://proxy.corp/v1"})
    spec = resolve_provider(config)
    assert spec.base_url == "https://proxy.corp/v1"
    assert spec.headers["X-Title"] == "lecode"  # preset headers kept


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_provider(_config(llm={"provider": "nope"}))


def test_build_client_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    config = _config()
    spec = resolve_provider(config)
    resolved = resolve_api_key(spec.name, config)
    assert resolved.source == "env"
    client = build_client(spec, resolved)
    assert client._client.headers["Authorization"] == "Bearer sk-or-env"
    assert client._client.headers["X-Title"] == "lecode"
    assert str(client._client.base_url).rstrip("/") == OPENROUTER_BASE_URL


def test_build_client_keyless():
    client = build_client(
        resolve_provider(_config(), cli_base_url="http://localhost:8080/v1"),
        ResolvedKey(key=None, source="none"),
    )
    assert "authorization" not in client._client.headers
