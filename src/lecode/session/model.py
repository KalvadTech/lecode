"""JSONL session record models.

A session file is append-only: one JSON record per line, discriminated by
``type``. History is never rewritten — undo/redo/compaction only append
records; :func:`parse_record` tolerates corrupt lines (returns ``None``).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

#: On-disk session schema version.
SESSION_SCHEMA_VERSION = 1


class MetaRecord(BaseModel):
    """First line of every session file."""

    type: Literal["meta"] = "meta"
    id: str
    name: str
    cwd: str
    created_at: str
    agent: str = "build"
    model: str | None = None
    schema_version: int = SESSION_SCHEMA_VERSION


class MessageRecord(BaseModel):
    """One chat message. ``message`` carries the Phase 3 ChatMessage shape."""

    type: Literal["message"] = "message"
    seq: int
    ts: str
    role: str
    message: dict[str, Any]
    #: Token/cost usage for assistant messages: input_tokens/output_tokens/cost_usd.
    usage: dict[str, Any] | None = None


class EventRecord(BaseModel):
    """A session event: compact, restore_point, rename, permission_grant, redo."""

    type: Literal["event"] = "event"
    seq: int
    ts: str
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class TombstoneRecord(BaseModel):
    """Replay cutoff: records with ``seq > up_to_seq`` are hidden from replay.

    (``up_to_seq`` is the highest still-visible seq; undo hides the last user
    turn by tombstoning at ``user_seq - 1``.)
    """

    type: Literal["tombstone"] = "tombstone"
    seq: int
    ts: str
    up_to_seq: int


Record = Annotated[
    MetaRecord | MessageRecord | EventRecord | TombstoneRecord,
    Field(discriminator="type"),
]

_record_adapter = TypeAdapter(Record)


def parse_record(line: str) -> Record | None:
    """Parse one JSONL line; return ``None`` on any corruption."""
    try:
        return _record_adapter.validate_json(line)
    except ValueError:
        return None
