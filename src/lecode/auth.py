"""API-key resolution and auth-policy enforcement.

Priority chain: CLI flag > environment variable > config file. Providers may
run keyless against local endpoints, governed by an auth policy:

- ``required`` — a key must resolve, else :class:`AuthError` is raised
- ``none``     — no key is ever sent (for endpoints that reject auth headers)
- ``auto``     — use the key if one resolves, run keyless otherwise
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from lecode.config.models import Config

KeySource = Literal["cli", "env", "config", "none"]

#: Environment variables checked per built-in provider, in order.
_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
}

#: Fallback env chain for any custom OpenRouter-compatible provider.
_DEFAULT_ENV_VARS = ("OPENAI_API_KEY",)


class AuthError(RuntimeError):
    """Raised when a provider's auth policy requires a key but none resolves."""


@dataclass(frozen=True)
class ResolvedKey:
    key: str | None
    source: KeySource


def resolve_api_key(
    provider_name: str,
    config: Config,
    cli_key: str | None = None,
) -> ResolvedKey:
    """Resolve the API key for ``provider_name`` and enforce its auth policy."""
    custom = config.custom_providers.get(provider_name)
    policy = custom.auth_policy if custom is not None else config.llm.auth_policy

    if policy == "none":
        return ResolvedKey(key=None, source="none")

    key: str | None = None
    source: KeySource = "none"

    if cli_key:
        key, source = cli_key, "cli"
    else:
        if custom is not None:
            env_names = (custom.api_key_env,) if custom.api_key_env else ()
        else:
            env_names = _ENV_VARS.get(provider_name, _DEFAULT_ENV_VARS)
        for name in env_names:
            value = os.environ.get(name)
            if value:
                key, source = value, "env"
                break
        if key is None and config.llm.api_key:
            key, source = config.llm.api_key, "config"

    if key is None and policy == "required":
        raise AuthError(
            f"provider '{provider_name}' has auth_policy = 'required' but no API key "
            "was found (checked CLI flag, environment, and config file)"
        )
    return ResolvedKey(key=key, source=source if key is not None else "none")
