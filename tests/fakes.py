"""In-memory fake provider implementing the Phase 3 stream protocol.

Drives agent-loop tests without HTTP. Each ``stream_chat`` call pops one
script entry; every request (messages/tools/model snapshot) is recorded in
``.requests`` for assertions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from lecode.providers.types import (
    CompletedMessage,
    Done,
    ReasoningDelta,
    StreamEvent,
    TokenDelta,
    ToolCallDelta,
    Usage,
    collect,
)

#: One scripted turn. Keys (all optional):
#:   text: str | list[str]           — token chunks
#:   reasoning: str | list[str]      — reasoning chunks
#:   tool_calls: [{id, name, arguments}]  — arguments is a JSON string
#:   error: ProviderError            — raised instead of streaming
#:   usage: dict                     — usage event before Done
#:   finish_reason: str              — default: "tool_calls" / "stop"
ScriptEntry = dict[str, Any]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


class FakeProvider:
    """Scripted streaming responder with request capture."""

    def __init__(self, script: list[ScriptEntry]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def stream_chat(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(
            {
                "messages": [dict(m) for m in messages],
                "model": model,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        entry = self.script.pop(0) if self.script else {"text": "(no scripted response left)"}
        return self._stream(entry)

    async def complete(self, messages: list[dict], model: str, **kwargs: Any) -> CompletedMessage:
        return await collect(self.stream_chat(messages, model, **kwargs))

    async def _stream(self, entry: ScriptEntry) -> AsyncIterator[StreamEvent]:
        error = entry.get("error")
        if error is not None:
            raise error
        for text in _as_list(entry.get("reasoning")):
            yield ReasoningDelta(text=text)
        for text in _as_list(entry.get("text")):
            yield TokenDelta(text=text)
        for index, call in enumerate(entry.get("tool_calls") or []):
            arguments = str(call.get("arguments", "{}"))
            midpoint = max(1, len(arguments) // 2)
            yield ToolCallDelta(
                index=index,
                id=str(call.get("id", f"call_{index}")),
                name=str(call.get("name", "")),
            )
            # arguments in two chunks, split mid-token, to exercise accumulation
            yield ToolCallDelta(index=index, arguments_chunk=arguments[:midpoint])
            yield ToolCallDelta(index=index, arguments_chunk=arguments[midpoint:])
        if usage := entry.get("usage"):
            yield Usage(usage=usage)
        default_finish = "tool_calls" if entry.get("tool_calls") else "stop"
        yield Done(finish_reason=entry.get("finish_reason") or default_finish)
