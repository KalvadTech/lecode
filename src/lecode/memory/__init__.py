"""Persistent project memory: markdown store, tools, injection, commands."""

from __future__ import annotations

from lecode.memory.commands import memory_command
from lecode.memory.store import (
    DEFAULT_MAX_BYTES,
    MAX_SEARCH_HITS,
    MemoryStore,
    SearchHit,
    memory_injection,
    memory_root,
    project_slug,
)
from lecode.memory.tools import memory_tools

__all__ = [
    "DEFAULT_MAX_BYTES",
    "MAX_SEARCH_HITS",
    "MemoryStore",
    "SearchHit",
    "memory_command",
    "memory_injection",
    "memory_root",
    "memory_tools",
    "project_slug",
]
