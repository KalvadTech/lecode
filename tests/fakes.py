"""In-memory fake provider implementing the Phase 3 stream protocol.

Drives agent-loop tests without HTTP. Each ``stream_chat`` call pops one
script entry; every request (messages/tools/model snapshot) is recorded in
``.requests`` for assertions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from lecode.providers.catalog import Catalog, ModelInfo
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


def sample_catalog() -> Catalog:
    """A small catalog standing in for the live ``/models`` fetch in tests.

    There is no bundled catalog anymore, so tests that need model metadata
    (pricing, context windows, modalities) build this explicitly.
    """

    def entry(
        id: str,
        name: str,
        context_window: int,
        prompt: float,
        completion: float,
        inputs: list[str] | None = None,
    ) -> ModelInfo:
        return ModelInfo.model_validate(
            {
                "id": id,
                "name": name,
                "context_window": context_window,
                "pricing": {"prompt": prompt, "completion": completion},
                "modalities": {"input": inputs or ["text"], "output": ["text"]},
            }
        )

    return Catalog(
        [
            entry("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", 1048576, 0.09, 0.18),
            entry("openai/gpt-5", "GPT-5", 400000, 1.25, 10.0, ["text", "image"]),
            entry("openai/gpt-5-mini", "GPT-5 Mini", 400000, 0.25, 2.0, ["text", "image"]),
            entry("openai/gpt-5-nano", "GPT-5 Nano", 400000, 0.05, 0.4),
            entry(
                "anthropic/claude-sonnet-4",
                "Claude Sonnet 4",
                200000,
                3.0,
                15.0,
                ["text", "image", "pdf"],
            ),
            entry("moonshotai/kimi-k2.6", "Kimi K2.6", 262144, 0.6, 2.5),
            entry("openai/gpt-4o", "GPT-4o", 128000, 2.5, 10.0, ["text", "image"]),
            entry("deepseek/deepseek-r1", "DeepSeek R1", 163840, 0.55, 2.19),
            entry(
                "google/gemini-2.5-pro",
                "Gemini 2.5 Pro",
                1048576,
                1.25,
                10.0,
                ["text", "image", "audio"],
            ),
            # models the setup wizard's tests pin as the provider list
            entry("tencent/hy4-preview", "Tencent Hy4 Preview", 1048576, 0.834, 2.501),
            entry(
                "deepseek/deepseek-v4-flash-0731", "DeepSeek V4 Flash 0731", 1310720, 0.065, 0.18
            ),
            entry("z-ai/glm-5.3-flash", "GLM 5.3 Flash", 1310720, 0.075, 0.25),
            entry("z-ai/glm-5.2", "GLM 5.2", 1048576, 0.966, 3.036),
        ]
    )


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
