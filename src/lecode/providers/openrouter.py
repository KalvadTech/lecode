"""OpenRouter preset over the generic OpenAI-compatible client.

Adds the app-identity headers, maps the live ``/models`` catalog into
:class:`~lecode.providers.catalog.ModelInfo` entries, and builds ``provider``
routing hints for ``extra_body`` pass-through. Prompt caching needs no code
here: message ``cache_control`` fields are serialized untouched by the client.
"""

from __future__ import annotations

from typing import Any

from lecode.providers.catalog import Modalities, ModelInfo, Pricing
from lecode.providers.openai_compat import ChatClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: App-identity headers sent on every OpenRouter request.
APP_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/wowi42/lecode",
    "X-Title": "lecode",
}


def openrouter_client(
    api_key: str | None = None,
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: Any = None,
    tls_verify: bool = True,
) -> ChatClient:
    """A :class:`ChatClient` preconfigured for OpenRouter."""
    headers = dict(APP_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return ChatClient(
        OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers=headers,
        timeout=timeout,
        tls_verify=tls_verify,
    )


def _per_million(pricing: dict[str, Any], key: str) -> float:
    """Convert OpenRouter per-token pricing (string) to per-million-token float."""
    try:
        return float(pricing.get(key) or 0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def map_remote_model(item: dict[str, Any]) -> ModelInfo | None:
    """Map one ``/models`` entry to :class:`ModelInfo`.

    Tolerates missing fields; returns ``None`` only for entries without an
    id. Entries from a plain OpenAI-shaped endpoint carry just an id — those
    get a default 128k context window and zeroed pricing.
    """
    model_id = item.get("id")
    if not model_id:
        return None
    context_length = item.get("context_length") or 128_000
    architecture = item.get("architecture") or {}
    supported = item.get("supported_parameters") or []
    top_provider = item.get("top_provider") or {}
    return ModelInfo(
        id=model_id,
        name=item.get("name") or model_id,
        context_window=int(context_length),
        max_output=top_provider.get("max_completion_tokens") or None,
        pricing=Pricing(
            prompt=_per_million(item.get("pricing") or {}, "prompt"),
            completion=_per_million(item.get("pricing") or {}, "completion"),
        ),
        modalities=Modalities(
            input=list(architecture.get("input_modalities") or ["text"]),
            output=list(architecture.get("output_modalities") or ["text"]),
        ),
        supports_tools="tools" in supported if supported else True,
        supports_reasoning="reasoning" in supported,
    )


async def fetch_remote_catalog(client: ChatClient) -> list[ModelInfo]:
    """Fetch the live OpenRouter catalog as :class:`ModelInfo` entries."""
    entries = []
    for item in await client.list_models():
        mapped = map_remote_model(item)
        if mapped is not None:
            entries.append(mapped)
    return entries


def routing_extra_body(
    order: list[str] | None = None,
    *,
    allow_fallbacks: bool | None = None,
    sort: str | None = None,
) -> dict[str, Any]:
    """OpenRouter ``provider`` routing hints, for ``extra_body`` pass-through."""
    provider: dict[str, Any] = {}
    if order:
        provider["order"] = list(order)
    if allow_fallbacks is not None:
        provider["allow_fallbacks"] = allow_fallbacks
    if sort is not None:
        provider["sort"] = sort
    return {"provider": provider}
