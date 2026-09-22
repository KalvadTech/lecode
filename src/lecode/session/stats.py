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
    unknown_usage_calls: int = 0
    #: True when any counted model call has missing usage or unknown cost.
    usage_incomplete: bool = False


def _usage_tokens(usage: dict) -> tuple[int, int]:
    """Accept both our (input/output) and OpenAI-style (prompt/completion) keys."""
    in_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return in_tokens, out_tokens


def session_stats(store: SessionStore, session: Session, catalog: Catalog | None = None) -> Stats:
    """Aggregate stats over the full record history (tombstones included)."""
    records = store.read_records(session)
    messages = [r for r in records if isinstance(r, MessageRecord)]

    role_counts: dict[str, int] = {}
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    unknown_usage_calls = 0
    usage_incomplete = False

    for record in messages:
        role_counts[record.role] = role_counts.get(record.role, 0) + 1
        usage = record.usage or {}
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

    # Auxiliary model calls and worker dispatches carry their usage on events.
    for record in records:
        if not isinstance(record, EventRecord):
            continue
        if record.kind in {"pierre", "compact", "memory_usage"}:
            usage = record.data.get("usage") or {}
            if record.data.get("usage") is None:
                unknown_usage_calls += 1
                usage_incomplete = True
        elif record.kind == "worker_usage":
            usage = record.data.get("usage") or record.data
        else:
            continue
        usage_incomplete |= bool(record.data.get("incomplete") or usage.get("incomplete"))
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
        else:
            usage_incomplete = True

    timestamps = [
        r.ts for r in records if isinstance(r, MessageRecord | EventRecord | TombstoneRecord)
    ]
    # Context fill = input tokens of the last *visible* assistant message
    # (tombstoned turns no longer sit in the context window).
    context_tokens = 0
    for record in reversed(store.load_messages(session)):
        if record.role == "assistant" and record.usage:
            in_tok, _ = _usage_tokens(record.usage)
            if in_tok:
                context_tokens = in_tok
                break
    return Stats(
        message_count=len(messages),
        role_counts=role_counts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        context_tokens=context_tokens,
        created_at=session.meta.created_at,
        last_active=max(timestamps) if timestamps else None,
        tombstone_count=sum(isinstance(r, TombstoneRecord) for r in records),
        unknown_usage_calls=unknown_usage_calls,
        usage_incomplete=usage_incomplete,
    )
