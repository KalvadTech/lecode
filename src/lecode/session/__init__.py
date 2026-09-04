"""JSONL sessions: storage, naming, handoff, and stats."""

from __future__ import annotations

from lecode.session.naming import auto_name, unique_name, validate_name
from lecode.session.storage import (
    AmbiguousSessionError,
    Session,
    SessionNotFoundError,
    SessionStore,
)

__all__ = [
    "AmbiguousSessionError",
    "Session",
    "SessionNotFoundError",
    "SessionStore",
    "auto_name",
    "unique_name",
    "validate_name",
]
