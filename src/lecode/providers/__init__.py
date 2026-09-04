"""Provider layer: one streaming client, the OpenRouter preset, and the catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lecode.auth import ResolvedKey
from lecode.config.models import Config
from lecode.providers.catalog import Catalog, ModelInfo
from lecode.providers.openai_compat import ChatClient, ProviderError
from lecode.providers.openrouter import APP_HEADERS, OPENROUTER_BASE_URL

__all__ = [
    "Catalog",
    "ChatClient",
    "ModelInfo",
    "ProviderError",
    "ProviderSpec",
    "build_client",
    "resolve_provider",
]

#: Provider names with built-in base URLs. Anything else is a custom
#: OpenRouter-compatible endpoint via ``[custom_providers]`` or ``--base-url``.
BUILTIN_PROVIDERS = ("openrouter",)


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to build a client for one provider."""

    name: str
    base_url: str
    model: str
    headers: dict[str, str] = field(default_factory=dict)
    auth_policy: Literal["auto", "required", "none"] = "auto"
    tls_verify: bool = True


def resolve_provider(
    config: Config,
    cli_base_url: str | None = None,
    cli_provider: str | None = None,
) -> ProviderSpec:
    """Resolve which endpoint to talk to.

    Precedence: ``--base-url`` (ad-hoc provider named ``custom``) >
    ``--provider`` > ``[llm].provider``. The only built-in is ``openrouter``;
    anything else must exist in ``[custom_providers]`` (any OpenRouter-
    compatible endpoint). ``[llm].base_url`` overrides a built-in's default
    base URL.
    """
    if cli_base_url:
        return ProviderSpec(
            name="custom",
            base_url=cli_base_url,
            model=config.llm.model,
            auth_policy=config.llm.auth_policy,
            tls_verify=config.llm.tls_verify,
        )

    name = cli_provider or config.llm.provider
    base_url = config.llm.base_url
    headers: dict[str, str] = {}
    auth_policy = config.llm.auth_policy

    if name == "openrouter":
        base_url = base_url or OPENROUTER_BASE_URL
        headers = dict(APP_HEADERS)
    elif name in config.custom_providers:
        custom = config.custom_providers[name]
        base_url = base_url or custom.base_url
        headers = dict(custom.headers)
        auth_policy = custom.auth_policy
    else:
        raise ValueError(
            f"unknown provider '{name}'; expected 'openrouter' "
            "or a [custom_providers] entry (any OpenRouter-compatible endpoint)"
        )

    return ProviderSpec(
        name=name,
        base_url=base_url,
        model=config.llm.model,
        headers=headers,
        auth_policy=auth_policy,
        tls_verify=config.llm.tls_verify,
    )


def build_client(spec: ProviderSpec, resolved_key: ResolvedKey) -> ChatClient:
    """Instantiate the streaming client for a resolved provider + key."""
    return ChatClient(
        spec.base_url,
        api_key=resolved_key.key,
        default_headers=spec.headers,
        tls_verify=spec.tls_verify,
    )
