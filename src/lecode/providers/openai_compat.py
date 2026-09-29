"""The one streaming client: OpenAI Chat Completions over httpx with SSE.

Used directly for generic OpenAI-compatible endpoints and by the OpenRouter
preset. SSE is parsed directly: lines from ``client.stream()`` are split into
``data:`` events, ``[DONE]`` terminates, comment/keep-alive lines are ignored.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

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

# Known structured codes. Never infer a failure category from message text.
_SYMBOLIC_STATUSES = {
    "authentication": 401,
    "invalid_api_key": 401,
    "payment_required": 402,
    "insufficient_quota": 402,
    "credit_balance_exhausted": 402,
    "model_not_found": 404,
    "not_found": 404,
    "rate_limit_exceeded": 429,
    "provider_overloaded": 503,
    "provider_unavailable": 502,
    "server": 500,
    "server_error": 500,
    "timeout": 408,
}

type ErrorCategory = Literal[
    "authentication", "budget", "model_not_found", "rate_limit", "upstream", "stream"
]


def _category_from_status(status: int | None) -> ErrorCategory | None:
    if status is None:
        return None
    categories: dict[int, ErrorCategory] = {
        401: "authentication",
        402: "budget",
        404: "model_not_found",
        429: "rate_limit",
    }
    if status in categories:
        return categories[status]
    if status in {408, 409} or 500 <= status < 600:
        return "upstream"
    return None


class ProviderError(Exception):
    """A provider failure with a semantic category and retry classification."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retryable: bool = False,
        category: ErrorCategory | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.category = category or _category_from_status(status)
        if self.category is None and status is None and retryable:
            self.category = "upstream"
        self.retryable = retryable and self.category != "budget"


def _provider_error(
    error: Any, *, status: int | None, body: str, stream: bool = False
) -> ProviderError:
    """Classify HTTP and SSE failures using the same structured fields."""
    error = error if isinstance(error, dict) else {}
    metadata = error.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    semantic_status = None
    for value in (metadata.get("error_type"), error.get("code"), error.get("type")):
        if isinstance(value, str) and value in _SYMBOLIC_STATUSES:
            semantic_status = _SYMBOLIC_STATUSES[value]
            break

    code = error.get("code")
    if isinstance(code, str) and len(code) == 3 and code.isascii() and code.isdecimal():
        code = int(code)
    if status is None:
        status = code if type(code) is int and 400 <= code < 600 else semantic_status
    effective_status = semantic_status or status
    category = _category_from_status(effective_status)
    message = error.get("message") or (
        "provider stream error" if stream else body or f"HTTP {status}"
    )
    return ProviderError(
        str(message),
        status=status,
        body=body,
        retryable=status in RETRYABLE_STATUSES,
        category=category or ("stream" if stream else None),
    )


def _error_from_response(status: int, body: str) -> ProviderError:
    """Map a non-2xx response to a :class:`ProviderError`."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    return _provider_error(error, status=status, body=body)


def _error_from_stream_chunk(chunk: dict[str, Any]) -> ProviderError:
    """Map an in-stream ``{"error": ...}`` event to a :class:`ProviderError`."""
    return _provider_error(chunk.get("error"), status=None, body=json.dumps(chunk), stream=True)


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
                    try:
                        chunk = json.loads(data)
                        if not isinstance(chunk, dict):
                            raise TypeError("stream event must be an object")
                        if "error" in chunk:
                            raise _error_from_stream_chunk(chunk)
                        usage = chunk.get("usage")
                        if usage is not None and not isinstance(usage, dict):
                            raise TypeError("stream usage must be an object")
                        if usage:
                            for field in (
                                "input_tokens",
                                "output_tokens",
                                "prompt_tokens",
                                "completion_tokens",
                            ):
                                if usage.get(field) is not None:
                                    int(usage[field])
                            for field in ("cost", "cost_usd"):
                                if usage.get(field) is not None:
                                    float(usage[field])
                            # OpenRouter reports the real billed amount as ``cost``.
                            if "cost" in usage and "cost_usd" not in usage:
                                usage["cost_usd"] = usage["cost"]
                            yield Usage(usage=usage)
                        choices = chunk.get("choices")
                        if choices is not None and not isinstance(choices, list):
                            raise TypeError("stream choices must be a list")
                        for choice in choices or []:
                            delta = choice.get("delta")
                            if delta is not None and not isinstance(delta, dict):
                                raise TypeError("stream delta must be an object")
                            delta = delta or {}
                            content = delta.get("content")
                            reasoning_content = delta.get("reasoning_content")
                            reasoning = delta.get("reasoning")
                            finish = choice.get("finish_reason")
                            if any(
                                value is not None and not isinstance(value, str)
                                for value in (content, reasoning_content, reasoning, finish)
                            ):
                                raise TypeError("stream text must be a string")
                            if finish == "error":
                                raise ProviderError("provider stream error", category="stream")
                            reasoning = reasoning_content or reasoning
                            if content:
                                yield TokenDelta(text=content)
                            if reasoning:
                                yield ReasoningDelta(text=reasoning)
                            tool_calls = delta.get("tool_calls")
                            if tool_calls is not None and not isinstance(tool_calls, list):
                                raise TypeError("stream tool calls must be a list")
                            for tool_call in tool_calls or []:
                                function = tool_call.get("function")
                                if function is not None and not isinstance(function, dict):
                                    raise TypeError("stream function must be an object")
                                function = function or {}
                                index = tool_call.get("index", 0)
                                if (
                                    type(index) is not int
                                    or index < 0
                                    or any(
                                        value is not None and not isinstance(value, str)
                                        for value in (
                                            tool_call.get("id"),
                                            function.get("name"),
                                            function.get("arguments"),
                                        )
                                    )
                                ):
                                    raise TypeError("invalid stream tool call")
                                yield ToolCallDelta(
                                    index=index,
                                    id=tool_call.get("id") or "",
                                    name=function.get("name") or "",
                                    arguments_chunk=function.get("arguments") or "",
                                )
                            if finish:
                                finish_reason = finish
                    except (ValueError, AttributeError, TypeError, OverflowError) as e:
                        raise ProviderError(
                            "malformed provider stream event", category="stream"
                        ) from e
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
