"""Tests for AI-generated session titles (first user message → short title)."""

from __future__ import annotations

from typing import Any

from lecode.providers.types import CompletedMessage
from lecode.session.title import generate_title


class TitleFake:
    """Provider stub answering only ``complete`` calls."""

    def __init__(self, content: str | None = "Fix login bug", fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.requests: list[tuple[list[dict], str]] = []

    async def complete(self, messages: list[dict], model: str, **kwargs: Any) -> CompletedMessage:
        self.requests.append((messages, model))
        if self.fail:
            raise RuntimeError("provider down")
        return CompletedMessage(content=self.content or "")


async def test_generate_title_returns_sanitized_title():
    fake = TitleFake(content="Fix login bug.\n\nmore context ignored.")
    assert await generate_title(fake, "openai/gpt-5-mini", "help me fix the login bug") == (
        "Fix login bug"
    )
    messages, model = fake.requests[0]
    assert model == "openai/gpt-5-mini"
    assert messages[0]["role"] == "system"
    assert "3-6 words" in str(messages[0]["content"])
    assert messages[1] == {"role": "user", "content": "help me fix the login bug"}


async def test_generate_title_provider_failure_returns_none():
    fake = TitleFake(fail=True)
    assert await generate_title(fake, "m", "prompt") is None


async def test_generate_title_unusable_output_returns_none():
    fake = TitleFake(content=".dotfile")
    assert await generate_title(fake, "m", "prompt") is None
    assert await generate_title(TitleFake(content=""), "m", "prompt") is None
