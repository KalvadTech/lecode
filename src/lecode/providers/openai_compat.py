"""The one streaming client: OpenAI Chat Completions over httpx with SSE.

Used directly for generic OpenAI-compatible endpoints and by the OpenRouter
preset. SSE is parsed directly: lines from ``client.stream()`` are split into
``data:`` events, ``[DONE]`` terminates, comment/keep-alive lines are ignored.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from lecode.providers.types import (
    ChatMessage,
    CompletedMessage,
    Done,
    ReasoningDelta,
    StreamEvent,
    TokenDelta,
    ToolCallDelta,
    Usage,
    collect,
)

#: HTTP statuses that justify an automatic retry.
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    """A provider failure with retry classification."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable


def _error_from_response(status: int, body: str) -> ProviderError:
    """Map a non-2xx response to a :class:`ProviderError`.

    Understands OpenRouter-style ``{"error": {"message", "code"}}`` bodies.
    """
    message = body or f"HTTP {status}"
    try:
        data = json.loads(body)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"])
    except (json.JSONDecodeError, AttributeError):
        pass
    return ProviderError(
        message,
        status=status,
        body=body,
        retryable=status in RETRYABLE_STATUSES,
    )


def _error_from_stream_chunk(chunk: dict[str, Any]) -> ProviderError:
    """Map an in-stream ``{"error": ...}`` event to a :class:`ProviderError`."""
    error = chunk.get("error")
    error = error if isinstance(error, dict) else {}
    message = str(error.get("message") or "provider stream error")
    code = error.get("code")
    status = code if isinstance(code, int) else None
    return ProviderError(
        message,
        status=status,
        body=json.dumps(chunk),
        retryable=status in RETRYABLE_STATUSES if status else False,
    )


async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the payload of each SSE ``data:`` event (multi-line aware)."""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        line = line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
    if data_lines:
        yield "\n".join(data_lines)


class ChatClient:
    """Async streaming client for one OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
        tls_verify: bool = True,
        default_extra_body: dict[str, Any] | None = None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if default_headers:
            headers.update(default_headers)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if timeout is None:
            timeout = httpx.Timeout(30.0, connect=10.0, read=300.0)
        self.base_url = base_url.rstrip("/")
        #: Merged into every request payload (per-call extra_body wins).
        self._default_extra_body = dict(default_extra_body or {})
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            verify=tls_verify,
        )

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _chat_payload(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,  # passed through untouched (cache_control survives)
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        payload.update(self._default_extra_body)
        if extra_body:
            payload.update(extra_body)
        return payload

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one chat completion turn as :class:`StreamEvent` items."""
        payload = self._chat_payload(
            messages,
            model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )
        finish_reason: str | None = None
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise _error_from_response(response.status_code, body)
                async for data in _iter_sse_data(response):
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if isinstance(chunk, dict) and "error" in chunk:
                        raise _error_from_stream_chunk(chunk)
                    usage = chunk.get("usage")
                    if usage:
                        # OpenRouter's usage extension reports the real billed
                        # amount as ``cost``; normalize to our ``cost_usd``.
                        if "cost" in usage and "cost_usd" not in usage:
                            usage["cost_usd"] = usage["cost"]
                        yield Usage(usage=usage)
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield TokenDelta(text=content)
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            yield ReasoningDelta(text=reasoning)
                        for tool_call in delta.get("tool_calls") or []:
                            function = tool_call.get("function") or {}
                            yield ToolCallDelta(
                                index=tool_call.get("index", 0),
                                id=tool_call.get("id") or "",
                                name=function.get("name") or "",
                                arguments_chunk=function.get("arguments") or "",
                            )
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        except httpx.TransportError as e:
            raise ProviderError(str(e), retryable=True) from e
        yield Done(finish_reason=finish_reason)

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> CompletedMessage:
        """Non-streaming helper (pierre/summarization): collects a stream."""
        return await collect(self.stream_chat(messages, model, **kwargs))

    async def list_models(self) -> list[dict[str, Any]]:
        """GET ``/models``; returns the raw ``data`` entries."""
        try:
            response = await self._client.get("/models")
        except httpx.TransportError as e:
            raise ProviderError(str(e), retryable=True) from e
        if response.status_code != 200:
            raise _error_from_response(response.status_code, response.text)
        data = response.json()
        entries = data.get("data") if isinstance(data, dict) else None
        return entries if isinstance(entries, list) else []
