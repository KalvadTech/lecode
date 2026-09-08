"""AI-generated session titles.

One non-blocking provider request turns the first user message of an
auto-named session into a short title; failures and unusable output yield
``None`` so callers keep the fallback name. No retries, no extra config.
"""

from __future__ import annotations

from typing import Any

from lecode.session.naming import sanitize_title

TITLE_PROMPT = (
    "Write a short title for a chat session with an AI coding agent, based on "
    "the user's first message. Reply with only the title: 3-6 words, "
    "descriptive, in the same language as the request. No quotes, no trailing "
    "period, no explanation."
)


async def generate_title(provider: Any, model: str, text: str) -> str | None:
    """Ask ``provider`` for a session title for ``text``; ``None`` on failure."""
    try:
        completed = await provider.complete(
            [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": text},
            ],
            model=model,
        )
    except Exception:
        return None
    return sanitize_title(completed.content)
