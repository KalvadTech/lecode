"""The multi-turn streaming agent loop.

One turn: stream from the provider (retrying retryable errors), collect the
:class:`~lecode.providers.types.CompletedMessage`, persist it, then dispatch
tool calls in parallel and loop. The runner never renders — it reports
progress through the ``on_event`` callback using the event taxonomy from the
build plan (``Token``/``Reasoning``/``ToolCall``/``ToolResult``/``Error``/
``Retrying``/``Done``).

Stop reasons: ``"done"`` (final text answer), ``"empty"`` (the provider
returned no text and no tool calls three nudges in a row), ``"max_turns"``.
Provider errors propagate after an ``Error`` event; cancellation propagates
after partial state is persisted.

The provider seam is duck-typed: anything with ``stream_chat(messages, model,
tools=...)`` works (:class:`~lecode.providers.openai_compat.ChatClient` and
the test ``FakeProvider`` both qualify).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from lecode.agent.review import review, user_request
from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.config.models import Config
from lecode.providers.catalog import Catalog, ModelNotFoundError
from lecode.providers.openai_compat import ProviderError
from lecode.providers.retry import retry_async
from lecode.providers.types import (
    ChatMessage,
    CompletedMessage,
    ReasoningDelta,
    StreamEvent,
    TokenDelta,
    ToolCallDelta,
    Usage,
)
from lecode.providers.types import (
    Done as StreamDone,
)
from lecode.telemetry import capture_exception, record_turn

# -- runner events (the plan's taxonomy) --------------------------------------


@dataclass(frozen=True)
class Token:
    text: str


@dataclass(frozen=True)
class Reasoning:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool


@dataclass(frozen=True)
class Error:
    message: str


@dataclass(frozen=True)
class Retrying:
    attempt: int
    error: str
    delay: float


@dataclass(frozen=True)
class Done:
    stop_reason: str
    turns: int


@dataclass(frozen=True)
class Review:
    """Pierre-mode feedback on the finished run (request vs result)."""

    feedback: str
    model: str


#: Everything the runner reports through ``on_event``.
AgentEvent = Token | Reasoning | ToolCall | ToolResult | Error | Retrying | Done | Review

#: on_event(event) — sync or async.
OnEvent = Callable[[AgentEvent], Any]

#: System nudge injected when a turn comes back with no text and no tool calls.
EMPTY_NUDGE = (
    "Your last response was empty. Reply to the user now — either with a text "
    "answer or with tool calls."
)

#: User message appended when a turn was cut off by the output limit.
CONTINUE_PROMPT = "Please continue."

#: How many empty-turn nudges before giving up with stop_reason "empty".
EMPTY_NUDGE_LIMIT = 3

#: Finish reasons that mean the answer was truncated and should continue.
LENGTH_FINISH_REASONS = frozenset({"length", "max_tokens"})


@dataclass(frozen=True)
class UsageTotals:
    """Accumulated usage over a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    #: Prompt size of the last API call — the real context fill (the
    #: accumulated ``input_tokens`` double-counts across tool-call rounds).
    context_tokens: int = 0


@dataclass(frozen=True)
class RunResult:
    final_text: str
    turns: int
    stop_reason: str
    usage_totals: UsageTotals
    #: Tool calls executed across all rounds of the run.
    tool_calls: int = 0
    #: Wall-clock seconds for the whole run.
    elapsed_s: float = 0.0
    #: Pierre-mode feedback (``[pierre] enabled``); ``None`` when not reviewed.
    review: str | None = None


def _usage_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    """Accept both our (input/output) and OpenAI-style (prompt/completion) keys."""
    in_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return in_tokens, out_tokens


class AgentRunner:
    """Multi-turn streaming loop over a provider + tool registry."""

    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        ctx: ToolContext,
        *,
        session: Any | None = None,
        store: Any | None = None,
        config: Config | None = None,
        steer_queue: asyncio.Queue[Any] | None = None,
        input_queue: asyncio.Queue[Any] | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.ctx = ctx
        self.session = session
        self.store = store
        self.config = config or ctx.config
        self.model = self.config.llm.model
        #: Priority queue drained first between turns (TUI Alt+Enter), then input.
        self.steer_queue = steer_queue
        self.input_queue = input_queue
        self._catalog: Catalog | None = catalog
        #: Partially collected turn, for cancellation-safe persistence.
        self._partial: CompletedMessage | None = None
        # Advisor seam: the advisor tool reaches the provider through ctx.
        self.ctx.extras["provider"] = provider

    async def run(
        self,
        messages: list[ChatMessage],
        on_event: OnEvent | None = None,
    ) -> RunResult:
        """Run the loop from ``messages`` until done, empty, or max turns."""
        history: list[ChatMessage] = list(messages)
        # Advisor seam: the live conversation, visible to the advisor tool.
        self.ctx.extras["conversation"] = history
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        context_tokens = 0
        turns = 0
        tool_calls = 0
        started_at = time.monotonic()
        empty_retries = 0
        final_text = ""
        continuing = False
        stop_reason = "done"
        max_turns = self.config.agent.max_turns

        try:
            while True:
                if turns >= max_turns:
                    stop_reason = "max_turns"
                    break
                if turns > 0:
                    await self._drain_queues(history)
                    if self.config.agent.turn_cooldown_ms > 0:
                        await asyncio.sleep(self.config.agent.turn_cooldown_ms / 1000)

                self._partial = None
                completed = await self._stream_turn(history, on_event)
                turns += 1

                in_tok, out_tok, cost = self._turn_cost(completed)
                input_tokens += in_tok
                output_tokens += out_tok
                cost_usd += cost
                context_tokens = in_tok or context_tokens
                history.append(completed.as_message())
                self._persist_assistant(completed, in_tok, out_tok, cost)

                if completed.tool_calls:
                    final_text = ""
                    continuing = False
                    results = await self._run_tools(completed, on_event)
                    tool_calls += len(results)
                    history.extend(results)
                    continue

                if not completed.content:
                    empty_retries += 1
                    if empty_retries > EMPTY_NUDGE_LIMIT:
                        stop_reason = "empty"
                        break
                    history.append({"role": "system", "content": EMPTY_NUDGE})
                    continue

                empty_retries = 0
                if completed.finish_reason in LENGTH_FINISH_REASONS:
                    final_text += completed.content
                    history.append({"role": "user", "content": CONTINUE_PROMPT})
                    continuing = True
                    continue
                final_text = (final_text if continuing else "") + completed.content
                stop_reason = "done"
                break
        except asyncio.CancelledError:
            self._persist_partial(history)
            raise

        await self._emit(on_event, Done(stop_reason=stop_reason, turns=turns))

        # Pierre mode: a second model reviews the request vs the result.
        review_text: str | None = None
        pierre = self.config.pierre
        if pierre.enabled and stop_reason == "done" and final_text:
            outcome = await review(
                self.provider,
                pierre.model or self.model,
                request=user_request(messages),
                response=final_text,
                cwd=self.ctx.cwd,
            )
            if outcome is not None:
                review_text = outcome.feedback
                in_tok, out_tok, cost = self._usage_cost(outcome.model, outcome.usage or {})
                input_tokens += in_tok
                output_tokens += out_tok
                cost_usd += cost
                if self.session is not None and self.store is not None:
                    self.store.append_event(
                        self.session,
                        "pierre",
                        {
                            "model": outcome.model,
                            "feedback": outcome.feedback,
                            "usage": {
                                "input_tokens": in_tok,
                                "output_tokens": out_tok,
                                "cost_usd": cost,
                            },
                        },
                    )
                await self._emit(on_event, Review(feedback=outcome.feedback, model=outcome.model))

        elapsed_s = time.monotonic() - started_at
        record_turn(
            model=self.model,
            stop_reason=stop_reason,
            turns=turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            elapsed_s=elapsed_s,
        )
        return RunResult(
            final_text=final_text,
            turns=turns,
            stop_reason=stop_reason,
            usage_totals=UsageTotals(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                context_tokens=context_tokens,
            ),
            tool_calls=tool_calls,
            elapsed_s=elapsed_s,
            review=review_text,
        )

    # -- one turn --------------------------------------------------------------

    async def _stream_turn(
        self, history: list[ChatMessage], on_event: OnEvent | None
    ) -> CompletedMessage:
        tools = self.registry.openai_tool_specs() or None
        thinking = self.config.llm.thinking
        reasoning_effort = None if thinking == "none" else thinking

        async def invoke() -> CompletedMessage:
            stream = self.provider.stream_chat(
                history, model=self.model, tools=tools, reasoning_effort=reasoning_effort
            )
            return await self._collect(stream, on_event)

        def on_retry(attempt: int, error: ProviderError, delay: float) -> Any:
            return self._emit(on_event, Retrying(attempt=attempt, error=str(error), delay=delay))

        try:
            return await retry_async(invoke, on_retry=on_retry)
        except ProviderError as e:
            await self._emit(on_event, Error(message=str(e)))
            capture_exception(e, context="provider")
            raise

    async def _collect(
        self, stream: AsyncIterator[StreamEvent], on_event: OnEvent | None
    ) -> CompletedMessage:
        """Assemble one turn, forwarding deltas to ``on_event`` as they stream."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        try:
            async for event in stream:
                if isinstance(event, TokenDelta):
                    text_parts.append(event.text)
                    await self._emit(on_event, Token(text=event.text))
                elif isinstance(event, ReasoningDelta):
                    reasoning_parts.append(event.text)
                    await self._emit(on_event, Reasoning(text=event.text))
                elif isinstance(event, ToolCallDelta):
                    call = calls.setdefault(event.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] += event.id
                    call["name"] += event.name
                    call["arguments"] += event.arguments_chunk
                elif isinstance(event, Usage):
                    usage = event.usage
                elif isinstance(event, StreamDone):
                    finish_reason = event.finish_reason
        except asyncio.CancelledError:
            self._partial = self._assemble(text_parts, reasoning_parts, calls, usage, None)
            raise
        return self._assemble(text_parts, reasoning_parts, calls, usage, finish_reason)

    @staticmethod
    def _assemble(
        text_parts: list[str],
        reasoning_parts: list[str],
        calls: dict[int, dict[str, str]],
        usage: dict[str, Any] | None,
        finish_reason: str | None,
    ) -> CompletedMessage:
        tool_calls = [
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

    # -- tool dispatch -----------------------------------------------------------

    async def _run_tools(
        self, completed: CompletedMessage, on_event: OnEvent | None
    ) -> list[ChatMessage]:
        """Dispatch all tool calls in parallel; results pair by call id."""
        for call in completed.tool_calls:
            function = call["function"]
            await self._emit(
                on_event,
                ToolCall(id=call["id"], name=function["name"], arguments=function["arguments"]),
            )
        tasks = [
            asyncio.ensure_future(
                self.registry.dispatch_result(
                    call["id"],
                    call["function"]["name"],
                    call["function"]["arguments"],
                    self.ctx,
                )
            )
            for call in completed.tool_calls
        ]
        try:
            pairs = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # Cancel in-flight tools; persist the results that did complete.
            for task in tasks:
                task.cancel()
            for task in tasks:
                if task.done() and not task.cancelled():
                    self._persist_message(task.result()[0])
            raise
        messages: list[ChatMessage] = []
        for call, (message, result) in zip(completed.tool_calls, pairs, strict=True):
            messages.append(message)
            self._persist_message(message)
            await self._emit(
                on_event,
                ToolResult(
                    id=call["id"],
                    name=call["function"]["name"],
                    content=result.content,
                    is_error=result.is_error,
                ),
            )
        return messages

    # -- queues -------------------------------------------------------------------

    async def _drain_queues(self, history: list[ChatMessage]) -> None:
        """Drain the steer queue first (priority), then the input queue.

        Drained items are appended to the history as user messages.
        """
        for queue in (self.steer_queue, self.input_queue):
            if queue is None:
                continue
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                message: ChatMessage = {"role": "user", "content": item}
                history.append(message)
                self._persist_message(message)

    # -- usage / cost ---------------------------------------------------------------

    def _usage_cost(self, model: str, usage: dict[str, Any]) -> tuple[int, int, float]:
        """(input tokens, output tokens, cost in USD) for one usage dict."""
        in_tok, out_tok = _usage_tokens(usage)
        if usage.get("cost_usd") is not None:
            return in_tok, out_tok, float(usage["cost_usd"])
        if not (in_tok or out_tok):
            return in_tok, out_tok, 0.0
        if self._catalog is None:
            self._catalog = Catalog.default()
        try:
            pricing = self._catalog.get(model).pricing
        except ModelNotFoundError:
            return in_tok, out_tok, 0.0
        return in_tok, out_tok, (in_tok * pricing.prompt + out_tok * pricing.completion) / 1e6

    def _turn_cost(self, completed: CompletedMessage) -> tuple[int, int, float]:
        """(input tokens, output tokens, cost in USD) for one turn."""
        return self._usage_cost(self.model, completed.usage or {})

    # -- persistence ------------------------------------------------------------------

    def _persist_assistant(
        self, completed: CompletedMessage, in_tok: int, out_tok: int, cost: float
    ) -> None:
        usage = None
        if completed.usage is not None:
            usage = {"input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost}
        self._persist_message(completed.as_message(), usage)

    def _persist_message(self, message: ChatMessage, usage: dict[str, Any] | None = None) -> None:
        if self.session is not None and self.store is not None:
            self.store.append_message(self.session, dict(message), usage=usage)

    def _persist_partial(self, history: list[ChatMessage]) -> None:
        """On cancellation, keep whatever partial assistant turn exists."""
        partial = self._partial
        self._partial = None
        if partial is None or not (partial.content or partial.tool_calls):
            return
        history.append(partial.as_message())
        self._persist_message(partial.as_message())

    # -- events -------------------------------------------------------------------------

    @staticmethod
    async def _emit(on_event: OnEvent | None, event: AgentEvent) -> None:
        if on_event is None:
            return
        result = on_event(event)
        if inspect.isawaitable(result):
            await result
