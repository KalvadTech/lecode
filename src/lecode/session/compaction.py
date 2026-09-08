"""Context compaction: summarize old messages, keep the recent tail.

The summarize-and-record core shared by ``/compact`` and the runner's
automatic trigger. The provider condenses everything older than the last
few messages; the summary is recorded as an append-only compact event, so
the full history stays on disk and :meth:`SessionStore.load_for_model`
replays summary + tail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lecode.hooks import POST_COMPACT, PRE_COMPACT
from lecode.session.storage import Session, SessionStore

if TYPE_CHECKING:
    from lecode.hooks import HookDispatcher

#: Recent messages kept raw by compaction; everything older is summarized.
COMPACT_KEEP_TAIL = 4

#: Chars of transcript sent to the summarizer.
COMPACT_TRANSCRIPT_CAP = 100_000

COMPACT_PROMPT = (
    "Summarize this conversation for continuation by an AI coding agent. "
    "Capture the goal, decisions made, files touched, and the current state "
    "of the work. Be compact (a few hundred words at most); plain text."
)


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return " ".join(parts).strip()
    return str(content or "").strip()


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def compact_session(
    provider: Any,
    store: SessionStore,
    session: Session,
    model: str,
    *,
    hooks: HookDispatcher | None = None,
) -> str | None:
    """Summarize all but the last few messages and record the compaction.

    Returns the summary, or ``None`` when there is too little history or the
    provider call failed — callers continue uncompacted (fail-open).
    ``hooks`` (when given) fires the observational PreCompact/PostCompact
    events around the summarize-and-record step.
    """
    messages = store.load_messages(session)
    if len(messages) <= COMPACT_KEEP_TAIL:
        return None
    older, tail = messages[:-COMPACT_KEEP_TAIL], messages[-COMPACT_KEEP_TAIL:]
    transcript = "\n".join(f"{m.role}: {_text_of(m.message)}" for m in older)
    transcript = _clip(transcript, COMPACT_TRANSCRIPT_CAP)
    if hooks is not None:
        await hooks.fire(PRE_COMPACT)
    try:
        completed = await provider.complete(
            [
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": transcript},
            ],
            model=model,
        )
    except Exception:
        return None
    summary = (completed.content or "").strip()
    if not summary:
        return None
    store.compact(session, summary, keep_from_seq=tail[0].seq)
    if hooks is not None:
        await hooks.fire(POST_COMPACT)
    return summary
