"""Context compaction: summarize old messages, keep the recent tail.

The summarize-and-record core shared by ``/compact`` and the runner's
automatic trigger. The provider condenses a bounded prefix of the visible
messages; the summary and its exact covered range are recorded as an
append-only compact event, so the full history stays on disk and
:meth:`SessionStore.load_for_model` replays summary + tail.
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


def _coverage(messages: list[Any]) -> int:
    """How many leading visible messages the summarizer can wholly cover.

    Taking a prefix (never truncating from the end) keeps the summarized range
    equal to the range replay removes. The kept tail never starts on a ``tool``
    message: the boundary walks back until the assistant that issued the
    matching tool calls is kept too. Zero when nothing is safely coverable.
    """
    limit = len(messages) - COMPACT_KEEP_TAIL
    if limit <= 0:
        return 0
    covered = 0
    size = 0
    while covered < limit:
        line = f"{messages[covered].role}: {_text_of(messages[covered].message)}"
        size += len(line) + (1 if covered else 0)
        if size > COMPACT_TRANSCRIPT_CAP:
            break
        covered += 1
    while covered and messages[covered].role == "tool":
        covered -= 1
    return covered


async def compact_session(
    provider: Any,
    store: SessionStore,
    session: Session,
    model: str,
    *,
    hooks: HookDispatcher | None = None,
) -> str | None:
    """Summarize a prefix of the visible messages and record the compaction.

    The covered prefix is bounded by :data:`COMPACT_TRANSCRIPT_CAP`; the kept
    tail is everything after it, so ``keep_from_seq`` and the recorded source
    range exactly partition the visible history. Returns the summary, or
    ``None`` when nothing is safely coverable or the provider call failed —
    callers continue uncompacted (fail-open). ``hooks`` (when given) fires the
    observational PreCompact/PostCompact events around the summarize step.
    """
    messages = store.visible_messages(session)
    covered = _coverage(messages)
    if covered == 0:
        return None
    prefix, tail = messages[:covered], messages[covered:]
    transcript = "\n".join(f"{m.role}: {_text_of(m.message)}" for m in prefix)
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
    store.compact(
        session,
        summary,
        keep_from_seq=tail[0].seq,
        source_start_seq=prefix[0].seq,
        source_end_seq=prefix[-1].seq,
    )
    if hooks is not None:
        await hooks.fire(POST_COMPACT)
    return summary
