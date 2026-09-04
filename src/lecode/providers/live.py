"""Live model catalog: fetch the provider's ``/models`` at startup.

There is no bundled snapshot and nothing is cached on disk: when the fetch
fails the catalog is simply empty, and every consumer fails open (unknown
models get no pricing/modality metadata but still work).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from lecode.providers.catalog import Catalog
from lecode.providers.openai_compat import ChatClient
from lecode.providers.openrouter import fetch_remote_catalog

log = logging.getLogger(__name__)

#: Bound on the live ``/models`` fetch at startup.
FETCH_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class LoadedCatalog:
    """The catalog plus where it came from (for the loading screen)."""

    catalog: Catalog
    origin: str  # "live" | "empty"
    remote_count: int = 0


async def load_catalog(
    client: ChatClient,
    *,
    timeout_s: float = FETCH_TIMEOUT_S,
) -> LoadedCatalog:
    """Resolve the startup catalog for ``client`` (live, else empty)."""
    try:
        entries = await asyncio.wait_for(fetch_remote_catalog(client), timeout_s)
    except Exception as e:  # any fetch/parse failure → empty catalog
        log.debug("model catalog fetch failed: %s", e)
        entries = []
    if entries:
        return LoadedCatalog(Catalog(entries), "live", len(entries))
    return LoadedCatalog(Catalog.default(), "empty")
