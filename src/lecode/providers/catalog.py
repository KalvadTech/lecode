"""Static model catalog.

A bundled JSON snapshot of well-known models (OpenRouter ids) with context
window, pricing, modality, and capability flags. Loaded via
``importlib.resources``; :meth:`Catalog.merge` is the hook Phase 3 uses to
fold in the live OpenRouter catalog refresh.
"""

from __future__ import annotations

import json
from importlib import resources

from pydantic import BaseModel


class Pricing(BaseModel):
    """USD per million tokens."""

    prompt: float
    completion: float


class Modalities(BaseModel):
    input: list[str]
    output: list[str]


class ModelInfo(BaseModel):
    id: str
    name: str
    context_window: int
    max_output: int | None = None
    pricing: Pricing
    modalities: Modalities
    supports_tools: bool = True
    supports_reasoning: bool = False


class ModelNotFoundError(KeyError):
    """No catalog entry matched the query."""


class AmbiguousModelError(KeyError):
    """A prefix query matched more than one catalog entry."""

    def __init__(self, query: str, matches: list[str]) -> None:
        self.query = query
        self.matches = matches
        super().__init__(f"ambiguous model '{query}', matches: {', '.join(matches)}")


class Catalog:
    """Lookup over a set of :class:`ModelInfo` entries, keyed by id."""

    def __init__(self, entries: list[ModelInfo]) -> None:
        self._entries: dict[str, ModelInfo] = {e.id: e for e in entries}

    @classmethod
    def default(cls) -> Catalog:
        """Load the bundled catalog from ``lecode.data/models.json``."""
        text = resources.files("lecode.data").joinpath("models.json").read_text("utf-8")
        return cls([ModelInfo.model_validate(e) for e in json.loads(text)])

    def get(self, query: str) -> ModelInfo:
        """Resolve ``query`` by exact id, unique id prefix, or name (any case)."""
        if query in self._entries:
            return self._entries[query]

        prefix_matches = [e for e in self._entries.values() if e.id.startswith(query)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if len(prefix_matches) > 1:
            raise AmbiguousModelError(query, sorted(e.id for e in prefix_matches))

        lowered = query.lower()
        name_matches = [e for e in self._entries.values() if e.name.lower() == lowered]
        if len(name_matches) == 1:
            return name_matches[0]
        if len(name_matches) > 1:
            raise AmbiguousModelError(query, sorted(e.id for e in name_matches))

        raise ModelNotFoundError(f"unknown model: {query}")

    def all(self) -> list[ModelInfo]:
        """All entries, sorted by id."""
        return [self._entries[k] for k in sorted(self._entries)]

    def modalities_for(self, model_id: str) -> Modalities:
        """Input/output modalities for a model (accepts any lookup form)."""
        return self.get(model_id).modalities

    def merge(self, entries: list[ModelInfo]) -> Catalog:
        """Return a new catalog with ``entries`` merged in by id.

        Incoming entries replace same-id entries and are appended otherwise;
        this is the hook for the Phase 3 remote catalog refresh.
        """
        merged = dict(self._entries)
        for entry in entries:
            merged[entry.id] = entry
        return Catalog(list(merged.values()))
