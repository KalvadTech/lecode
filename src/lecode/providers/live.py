"""Live model catalog: fetch the provider's ``/models``, cache, fall back.

Startup resolution order:

1. live fetch from the resolved endpoint (bounded by a short timeout)
2. the on-disk cache, but only when it was fetched from the same base URL
3. the bundled static snapshot

Live and cache entries merge **over** the bundled catalog, so ids the
endpoint does not list still resolve with bundled metadata. Every failure
mode is swallowed — a missing catalog must never block startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lecode.providers.catalog import Catalog, ModelInfo
from lecode.providers.openai_compat import ChatClient
from lecode.providers.openrouter import fetch_remote_catalog

log = logging.getLogger(__name__)

CACHE_FILENAME = "models-cache.json"

#: Bound on the live ``/models`` fetch at startup.
FETCH_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class LoadedCatalog:
    """The catalog plus where it came from (for the loading screen)."""

    catalog: Catalog
    origin: str  # "live" | "cache" | "bundled"
    remote_count: int = 0


def _cache_path(config_dir: Path) -> Path:
    return config_dir / CACHE_FILENAME


def _read_cache(config_dir: Path, base_url: str) -> list[ModelInfo] | None:
    """Cached entries for ``base_url``; ``None`` when absent/stale/foreign."""
    path = _cache_path(config_dir)
    try:
        raw = json.loads(path.read_text("utf-8"))
        if raw.get("base_url") != base_url:
            return None
        return [ModelInfo.model_validate(e) for e in raw.get("models") or []] or None
    except (OSError, ValueError, TypeError) as e:
        log.debug("model catalog cache unreadable: %s", e)
        return None


def _write_cache(config_dir: Path, base_url: str, entries: list[ModelInfo]) -> None:
    """Persist the fetched catalog; best effort, never raises."""
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_url": base_url,
            "fetched_at": datetime.now(UTC).isoformat(),
            "models": [e.model_dump() for e in entries],
        }
        _cache_path(config_dir).write_text(json.dumps(payload), encoding="utf-8")
    except OSError as e:
        log.debug("model catalog cache write failed: %s", e)


async def load_catalog(
    client: ChatClient,
    config_dir: Path,
    *,
    timeout_s: float = FETCH_TIMEOUT_S,
) -> LoadedCatalog:
    """Resolve the startup catalog for ``client`` (live → cache → bundled)."""
    bundled = Catalog.default()
    base_url = getattr(client, "base_url", "")
    try:
        entries = await asyncio.wait_for(fetch_remote_catalog(client), timeout_s)
    except Exception as e:  # any fetch/parse failure falls through to cache
        log.debug("model catalog fetch failed: %s", e)
        entries = []
    if entries:
        _write_cache(config_dir, base_url, entries)
        return LoadedCatalog(bundled.merge(entries), "live", len(entries))
    cached = _read_cache(config_dir, base_url)
    if cached:
        return LoadedCatalog(bundled.merge(cached), "cache", len(cached))
    return LoadedCatalog(bundled, "bundled")
