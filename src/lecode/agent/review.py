"""Pierre mode: a post-task review by a second model.

When ``[pierre] enabled = true``, every completed run ends with one extra
non-streaming call: the reviewer model gets the user's request and the
agent's final answer and returns short feedback ("did it deliver?").

Fail-open by design — any error (offline, unknown model, empty answer)
yields ``None`` and the run completes without a review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lecode.context.resources import load_text
from lecode.providers.types import CompletedMessage, collect


@dataclass(frozen=True)
class ReviewOutcome:
    """One pierre review: the feedback text and the call's token usage."""

    feedback: str
    model: str
    usage: dict[str, Any] | None = None


def message_text(message: dict[str, Any]) -> str:
    """The text of a message, whether plain or content parts."""
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content or "").strip()


def user_request(messages: list[dict[str, Any]]) -> str:
    """The request under review: the last user message's text."""
    for message in reversed(messages):
        if message.get("role") == "user":
            text = message_text(message)
            if text:
                return text
    return ""


async def review(
    provider: Any,
    model: str,
    *,
    request: str,
    response: str,
    cwd: Path | None = None,
) -> ReviewOutcome | None:
    """One reviewer call; ``None`` on any failure (fail-open)."""
    if not request or not response:
        return None
    prompt = load_text("prompts", "pierre.md", cwd=cwd)
    user = f"## User request\n\n{request}\n\n## Agent's final answer\n\n{response}"
    try:
        complete = getattr(provider, "complete", None)
        if complete is not None:
            completed: CompletedMessage = await complete(
                [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                model=model,
            )
        else:
            completed = await collect(
                provider.stream_chat(
                    [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                    model=model,
                )
            )
    except Exception:
        return None
    feedback = (completed.content or "").strip()
    if not feedback:
        return None
    return ReviewOutcome(feedback=feedback, model=model, usage=completed.usage)
