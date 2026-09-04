"""Message and streaming types for the OpenAI-compatible provider protocol.

Outgoing messages are plain dicts (``TypedDict`` for structure only) so that
provider-specific extensions — e.g. prompt-caching ``cache_control`` markers —
pass through to the wire untouched.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Required, TypedDict


class TextPart(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]


class ImageUrl(TypedDict, total=False):
    url: Required[str]  # https URL or data URI (base64)
    detail: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


class FileData(TypedDict, total=False):
    file_data: Required[str]  # data URI (base64)
    filename: str


class FilePart(TypedDict):
    type: Literal["file"]
    file: FileData


#: A multimodal content part inside a message's ``content`` list.
ContentPart = TextPart | ImagePart | FilePart


class ToolCallFunction(TypedDict):
    name: str
    arguments: str  # JSON-encoded arguments


class ToolCallDict(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessage(TypedDict, total=False):
    """One chat message. Extra keys (``cache_control`` …) pass through verbatim."""

    role: Required[str]
    content: str | list[ContentPart] | None
    tool_calls: list[ToolCallDict]
    tool_call_id: str  # for role="tool" results
    name: str


@dataclass(frozen=True)
class TokenDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """One streamed fragment of a tool call; deltas pair by ``index``."""

    index: int
    id: str = ""
    name: str = ""
    arguments_chunk: str = ""


@dataclass(frozen=True)
class Usage:
    usage: dict[str, Any]


@dataclass(frozen=True)
class Done:
    finish_reason: str | None = None


#: Everything a chat stream can yield.
StreamEvent = TokenDelta | ReasoningDelta | ToolCallDelta | Usage | Done


@dataclass
class CompletedMessage:
    """A fully assembled assistant turn."""

    role: str = "assistant"
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCallDict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    def as_message(self) -> ChatMessage:
        """Render as a message dict suitable for appending to the history."""
        message: ChatMessage = {"role": self.role, "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


async def collect(stream: AsyncIterator[StreamEvent]) -> CompletedMessage:
    """Assemble a stream of events into a :class:`CompletedMessage`.

    Tool-call deltas are paired by ``index``; argument chunks concatenate in
    arrival order.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    async for event in stream:
        if isinstance(event, TokenDelta):
            text_parts.append(event.text)
        elif isinstance(event, ReasoningDelta):
            reasoning_parts.append(event.text)
        elif isinstance(event, ToolCallDelta):
            call = calls.setdefault(event.index, {"id": "", "name": "", "arguments": ""})
            call["id"] += event.id
            call["name"] += event.name
            call["arguments"] += event.arguments_chunk
        elif isinstance(event, Usage):
            usage = event.usage
        elif isinstance(event, Done):
            finish_reason = event.finish_reason

    tool_calls: list[ToolCallDict] = [
        {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        }
        for _, call in sorted(calls.items())
    ]
    return CompletedMessage(
        content="".join(text_parts),
        reasoning="".join(reasoning_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )
