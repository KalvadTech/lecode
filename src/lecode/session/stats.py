"""Session usage statistics and cost reporting.

Token totals come from stored per-message usage; cost uses recorded
``cost_usd`` when present, else falls back to catalog pricing for the
session's model (per-million-token rates).
"""

from __future__ import annotations

from dataclasses import dataclass

from lecode.providers.catalog import AmbiguousModelError, Catalog, ModelNotFoundError
from lecode.session.model import EventRecord, MessageRecord, TombstoneRecord
from lecode.session.storage import Session, SessionStore


@dataclass(frozen=True)
class Stats:
    message_count: int
    role_counts: dict[str, int]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    #: Context fill after the last visible assistant turn (its input tokens).
    context_tokens: int
    created_at: str
    last_active: str | None
    tombstone_count: int
    completed_turns: int = 0
    unknown_usage_calls: int = 0
    #: True when any counted model call has missing usage or unknown cost.
    usage_incomplete: bool = False
    input_tokens_known: bool = False
    output_tokens_known: bool = False
    cost_usd_known: bool = False


def _usage_tokens(usage: dict) -> tuple[int, int]:
    """Accept both our (input/output) and OpenAI-style (prompt/completion) keys."""
    in_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return in_tokens, out_tokens


def session_stats(
    store: SessionStore,
    session: Session,
    catalog: Catalog | None = None,
    *,
    include_message_usage: bool = True,
) -> Stats:
    """Aggregate history; event-only usage lets workers reconcile auxiliary calls."""
    message_count = completed_turns = tombstone_count = 0
    last_active = None
    context_candidates: list[tuple[int, int]] = []
    replay_events: list[EventRecord | TombstoneRecord] = []

    role_counts: dict[str, int] = {}
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    unknown_usage_calls = 0
    usage_incomplete = False
    known = {"input_tokens": False, "output_tokens": False, "cost_usd": False}

    def count_known(usage: dict, *, per_call: bool = False) -> None:
        nonlocal usage_incomplete
        aliases = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}
        for key in known:
            present = usage.get(key) is not None or usage.get(aliases.get(key, key)) is not None
            available = bool(usage.get(f"{key}_known", present))
            known[key] |= available
            if per_call and key in aliases and not available:
                usage_incomplete = True

    for record in store.iter_records(session):
        if isinstance(record, MessageRecord | EventRecord | TombstoneRecord):
            last_active = max(last_active or record.ts, record.ts)
        if isinstance(record, TombstoneRecord):
            tombstone_count += 1
            replay_events.append(record)
        elif isinstance(record, EventRecord) and record.kind == "redo":
            replay_events.append(record)
        if not isinstance(record, MessageRecord):
            continue
        message_count += 1
        if record.role == "assistant":
            completed_turns += not record.message.get("incomplete", False)
            in_tok, _ = _usage_tokens(record.usage or {})
            if in_tok:
                context_candidates.append((record.seq, in_tok))
        role_counts[record.role] = role_counts.get(record.role, 0) + 1
        if not include_message_usage:
            continue
        usage = record.usage or {}
        count_known(usage, per_call=record.role == "assistant")
        usage_incomplete |= bool(usage.get("incomplete")) or (
            record.role == "assistant" and not usage
        )
        in_tok, out_tok = _usage_tokens(usage)
        input_tokens += in_tok
        output_tokens += out_tok
        if usage.get("cost_usd") is not None:
            cost_usd += float(usage["cost_usd"])
        elif usage and (record.role == "assistant" or in_tok or out_tok):
            if not session.meta.model:
                usage_incomplete = True
                continue
            if catalog is None:
                catalog = Catalog.default()
            try:
                pricing = catalog.get(session.meta.model).pricing
            except (ModelNotFoundError, AmbiguousModelError):
                usage_incomplete = True
                continue
            cost_usd += (in_tok * pricing.prompt + out_tok * pricing.completion) / 1_000_000
            usage_incomplete |= not pricing.known
            known["cost_usd"] |= pricing.known

    # Auxiliary/partial model calls and worker dispatches carry usage on events.
    for record in store.iter_records(session):
        if not isinstance(record, EventRecord):
            continue
        if record.kind == "usage_incomplete":
            usage_incomplete = True
            continue
        if record.kind in {"pierre", "compact", "memory_usage", "provider_usage"}:
            usage = record.data.get("usage") or {}
            if record.data.get("usage") is None:
                unknown_usage_calls += 1
                usage_incomplete = True
        elif record.kind == "worker_usage":
            usage = record.data.get("usage") or record.data
        elif record.kind == "agent_run":
            usage = record.data
            usage_incomplete |= bool(usage.get("usage_incomplete"))
        else:
            continue
        usage_incomplete |= bool(record.data.get("incomplete") or usage.get("incomplete"))
        count_known(
            usage, per_call=record.kind in {"pierre", "compact", "memory_usage", "provider_usage"}
        )
        in_tok, out_tok = _usage_tokens(usage)
        input_tokens += in_tok
        output_tokens += out_tok
        if usage.get("cost_usd") is not None:
            cost_usd += float(usage["cost_usd"])
        elif (in_tok or out_tok) and record.data.get("model"):
            if catalog is None:
                catalog = Catalog.default()
            try:
                pricing = catalog.get(record.data["model"]).pricing
            except (ModelNotFoundError, AmbiguousModelError):
                usage_incomplete = True
                continue
            cost_usd += (in_tok * pricing.prompt + out_tok * pricing.completion) / 1e6
            usage_incomplete |= not pricing.known
            known["cost_usd"] |= pricing.known
        else:
            usage_incomplete = True

    # Retain only usage/undo metadata, never the transcript or tool payloads.
    tombstones = store._active_tombstones(replay_events)
    context_tokens = next(
        (
            tokens
            for seq, tokens in reversed(context_candidates)
            if not store._is_hidden(seq, tombstones)
        ),
        0,
    )
    return Stats(
        message_count=message_count,
        role_counts=role_counts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        context_tokens=context_tokens,
        created_at=session.meta.created_at,
        last_active=last_active,
        tombstone_count=tombstone_count,
        completed_turns=completed_turns,
        unknown_usage_calls=unknown_usage_calls,
        usage_incomplete=usage_incomplete,
        input_tokens_known=known["input_tokens"],
        output_tokens_known=known["output_tokens"],
        cost_usd_known=known["cost_usd"],
    )
