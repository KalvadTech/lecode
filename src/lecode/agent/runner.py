"""The multi-turn streaming agent loop.

One turn: stream from the provider (retrying retryable errors), collect the
:class:`~lecode.providers.types.CompletedMessage`, persist it, then dispatch
tool calls in parallel and loop. The runner never renders — it reports
progress through the ``on_event`` callback using the event taxonomy from the
build plan (``Token``/``Reasoning``/``ToolCall``/``ToolResult``/``Error``/
``Retrying``/``Done``).

Stop reasons: ``"done"`` (final text answer), ``"empty"`` (the provider
returned no text and no tool calls three nudges in a row), ``"max_turns"``,
``"context_overflow"`` (the request cannot safely fit the current model budget).
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
from dataclasses import dataclass, field, replace
from typing import Any

from lecode.agent.review import review, user_request
from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.config.models import Config
from lecode.extras.background import BACKGROUND_EXTRA
from lecode.hooks import STOP
from lecode.providers.catalog import AmbiguousModelError, Catalog, ModelNotFoundError
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
from lecode.session.compaction import (
    compact_session,
    context_limits,
    estimate_request,
    request_size,
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
    #: Tool-attached metadata (e.g. the task tool's run id for roster lookup).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Error:
    message: str


@dataclass(frozen=True)
class Retrying:
    attempt: int
    error: str
    delay: float


@dataclass(frozen=True)
class LlmCall:
    """The runner is invoking the model — one event per round, before streaming."""

    model: str
    turn: int


@dataclass(frozen=True)
class LlmResponse:
    """One model invocation finished — per-call usage for the logbook."""

    model: str
    turn: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    #: Characters sent in this call's prompt — calibrates live token estimates.
    prompt_chars: int = 0
    usage_incomplete: bool = False
    #: Model-call seconds, including latency/retries but excluding tool execution.
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class QueuedMessage:
    """A queued user message was drained into the conversation mid-run.

    The TUI echoes it into the logbook at this point — not at submit time —
    so a queued message is printed exactly once, when the model sees it.
    """

    content: Any


@dataclass(frozen=True)
class CompactionStarted:
    """Automatic compaction is summarizing the older context."""

    context_tokens: int
    threshold: int


@dataclass(frozen=True)
class CompactionFinished:
    """Automatic compaction recorded a summary and rebuilt the history."""

    summary_chars: int


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
AgentEvent = (
    Token
    | Reasoning
    | ToolCall
    | ToolResult
    | Error
    | Retrying
    | LlmCall
    | LlmResponse
    | QueuedMessage
    | CompactionStarted
    | CompactionFinished
    | Done
    | Review
)

#: on_event(event) — sync or async.
OnEvent = Callable[[AgentEvent], Any]


def _prompt_chars(history: list[dict[str, Any]]) -> int:
    """Text characters in the prompt about to be sent (system + history)."""
    total = 0
    for message in history:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
    return total


#: System nudge injected when a turn comes back with no text and no tool calls.
EMPTY_NUDGE = (
    "Your last response was empty. Reply to the user now — either with a text "
    "answer or with tool calls."
)

#: User message appended when a turn was cut off by the output limit.
CONTINUE_PROMPT = "Please continue."

#: How many empty-turn nudges before giving up with stop_reason "empty".
EMPTY_NUDGE_LIMIT = 3


class ContextPaused(Exception):
    """A request cannot safely use the current context."""


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
    unknown_usage_calls: int = 0
    usage_incomplete: bool = False


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
        refresh_prompt: Callable[[], str] | None = None,
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
        #: Recomputes the live system prompt at each request boundary.
        self._refresh_prompt = refresh_prompt
        #: Partially collected turn, for cancellation-safe persistence.
        self._partial: CompletedMessage | None = None
        self._bytes_per_token = 3.0
        self._calibration_model = self.model
        # Subagent seam: child runners reach the provider through ctx.
        self.ctx.extras["provider"] = provider
        self._seen_generation = self.memory_generation()
        self._request_generation = self._seen_generation

    async def run(
        self,
        messages: list[ChatMessage],
        on_event: OnEvent | None = None,
        *,
        expected_generation: int | None = None,
    ) -> RunResult:
        """Run the loop from ``messages`` until done, empty, or max turns."""
        history: list[ChatMessage] = list(messages)
        if self._refresh_prompt is not None and self.store is not None and self.session is not None:
            replay = self.store.load_for_model(self.session)
            if replay and replay[0].get("role") == "system" and history[:1] == replay[:1]:
                history.insert(0, {"role": "system", "content": ""})
        # The live conversation, visible through ctx (subagents, hooks).
        self.ctx.extras["conversation"] = history
        manager = self.ctx.extras.get("workers")
        if manager is not None and self.session is not None:
            manager.repair_interrupted_tools(self.session, history)
        # Background tasks finished between runs surface at the start.
        await self._drain_background(history, on_event)
        await self._consume_workers(history, on_event)
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        usage_incomplete = False
        unknown_usage_calls = 0

        def memory_usage(usage: dict | None) -> None:
            nonlocal input_tokens, output_tokens, cost_usd, usage_incomplete, unknown_usage_calls
            if usage is None:
                unknown_usage_calls += 1
                usage_incomplete = True
            else:
                in_tok, out_tok, cost, incomplete = self._usage_cost(self.model, usage)
                input_tokens += in_tok
                output_tokens += out_tok
                cost_usd += cost
                usage_incomplete |= incomplete

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
            self._sync_memory(
                history,
                force=self.memory_generation() > 0,
                expected_generation=expected_generation,
                fresh_request=messages[-1]
                if messages and messages[-1].get("role") == "user"
                else None,
            )
            self._request_generation = self._seen_generation
            while True:
                if turns >= max_turns:
                    stop_reason = "max_turns"
                    break
                if turns > 0:
                    await self._drain_queues(history, on_event)
                    await self._consume_workers(history, on_event)
                    if self.config.agent.turn_cooldown_ms > 0:
                        await asyncio.sleep(self.config.agent.turn_cooldown_ms / 1000)

                self._sync_memory(history, expected_generation=expected_generation)
                self._refresh_system(history)
                specs = self.registry.openai_tool_specs() or None
                window, headroom = context_limits(self.model, self.config, self._catalog)
                estimated = estimate_request(history, specs, bytes_per_token=self._bytes_per_token)
                while await self._maybe_compact(history, on_event, estimated, turns, memory_usage):
                    updated = estimate_request(
                        history, specs, bytes_per_token=self._bytes_per_token
                    )
                    if updated >= estimated:
                        estimated = updated
                        break
                    estimated = updated
                if estimated + headroom > window:
                    await self._emit(
                        on_event,
                        Error(
                            message=(
                                "Context cannot safely fit the current model's input "
                                "and output budget; paused."
                            )
                        ),
                    )
                    stop_reason = "context_overflow"
                    break

                self._partial = None
                self._request_generation = self._seen_generation
                prompt_chars = _prompt_chars(history)
                source_version = (
                    self.store.source_version(self.session, include_worker_events=False)
                    if self.store is not None and self.session is not None
                    else None
                )
                await self._emit(on_event, LlmCall(model=self.model, turn=turns + 1))
                call_started_at = time.monotonic()
                completed = await self._stream_turn(history, on_event, source_version)
                call_elapsed_s = time.monotonic() - call_started_at
                turns += 1

                in_tok, out_tok, cost, incomplete = self._turn_cost(completed)
                usage_incomplete |= incomplete
                if in_tok:
                    self._bytes_per_token = min(
                        self._bytes_per_token, request_size(history, specs) / in_tok
                    )
                self._check_generation()
                if (
                    self.store is not None
                    and self.session is not None
                    and self.store.source_version(self.session, include_worker_events=False)
                    != source_version
                ):
                    raise ContextPaused(
                        "Session sources changed during the request; response discarded."
                    )
                await self._emit(
                    on_event,
                    LlmResponse(
                        model=self.model,
                        turn=turns,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=cost,
                        prompt_chars=prompt_chars,
                        usage_incomplete=incomplete,
                        elapsed_s=call_elapsed_s,
                    ),
                )
                input_tokens += in_tok
                output_tokens += out_tok
                cost_usd += cost
                context_tokens = in_tok or context_tokens
                history.append(completed.as_message())
                self._persist_assistant(completed, in_tok, out_tok, cost, incomplete)

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
                manager = self.ctx.extras.get("workers")
                worker_id = self.ctx.extras.get("worker_id")
                if (
                    turns >= max_turns
                    and manager is not None
                    and (
                        any(w.is_active for w in manager.descendants(worker_id))
                        or manager.pending_notifications(worker_id)
                        or (
                            worker_id is not None
                            and (manager.pending(worker_id) or manager.questions(worker_id))
                        )
                        or any(
                            q is not None and not q.empty()
                            for q in (self.steer_queue, self.input_queue)
                        )
                    )
                ):
                    stop_reason = "max_turns"
                    break
                if turns < max_turns and await self._consume_workers(
                    history, on_event, completing=True
                ):
                    continue
                stop_reason = "done"
                break
        except ContextPaused as exc:
            final_text = ""
            stop_reason = "context_overflow"
            await self._emit(on_event, Error(message=str(exc)))
        except asyncio.CancelledError:
            self._persist_partial(history)
            raise

        if stop_reason != "done":
            manager = self.ctx.extras.get("workers")
            message = (
                manager.stop_message(self.ctx.extras.get("worker_id"), stop_reason)
                if manager is not None
                else f"Run stopped: {stop_reason}."
            )
            if self.session is not None and self.store is not None:
                self.store.append_event(
                    self.session,
                    "run_stopped",
                    {
                        "reason": stop_reason,
                        "message": message,
                    },
                )
            await self._emit(on_event, Error(message))
        await self._emit(on_event, Done(stop_reason=stop_reason, turns=turns))
        hooks = self.ctx.extras.get("hooks")
        if hooks is not None and hooks.handlers.get(STOP):
            await hooks.fire(STOP, reason=stop_reason)
        if self.memory_generation() != self._request_generation:
            final_text = ""
            stop_reason = "context_overflow"

        # Pierre mode: a second model reviews the request vs the result.
        review_text: str | None = None
        pierre = self.config.pierre
        if pierre.enabled and stop_reason == "done" and final_text:
            review_generation = self.memory_generation()
            outcome = await review(
                self.provider,
                pierre.model or self.model,
                request=user_request(messages),
                response=final_text,
                cwd=self.ctx.cwd,
            )
            if self.memory_generation() != review_generation:
                outcome = None
                final_text = ""
                stop_reason = "context_overflow"
            if outcome is not None:
                review_text = outcome.feedback
                in_tok, out_tok, cost, incomplete = self._usage_cost(outcome.model, outcome.usage)
                usage_incomplete |= incomplete
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
                                "incomplete": incomplete,
                            },
                        },
                    )
                await self._emit(on_event, Review(feedback=outcome.feedback, model=outcome.model))

        if self.memory_generation() != self._request_generation:
            final_text = ""
            review_text = None
            stop_reason = "context_overflow"
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
                unknown_usage_calls=unknown_usage_calls,
                usage_incomplete=usage_incomplete,
            ),
            tool_calls=tool_calls,
            elapsed_s=elapsed_s,
            review=review_text,
        )

    # -- one turn --------------------------------------------------------------

    def memory_generation(self) -> int:
        if self.ctx.recall_context is not None:
            return self.ctx.recall_context.generation()
        facts = self.ctx.extras.get("facts")
        if facts is not None:
            return facts.generation()
        if self.store is not None and self.session is not None:
            return self.store.memory_generation(self.session.id)
        return 0

    def _sync_memory(
        self,
        history: list[ChatMessage],
        *,
        force: bool = False,
        expected_generation: int | None = None,
        fresh_request: ChatMessage | None = None,
    ) -> None:
        generation = self.memory_generation()
        if expected_generation is not None and generation != expected_generation:
            raise ContextPaused("Memory exclusions changed; derived input discarded.")
        if self.store is None or self.session is None:
            if generation != self._seen_generation:
                raise ContextPaused("Memory exclusions changed; discard derived context and retry.")
            return
        reset = force or generation != self._seen_generation
        history[:] = self.store.refresh_model_history(
            self.session, history, reset=reset, fresh_request=fresh_request
        )
        if reset:
            self._seen_generation = generation
            self._refresh_system(history)

    def _check_generation(self) -> None:
        if self.memory_generation() != self._request_generation:
            raise ContextPaused("Memory exclusions changed during the request; response discarded.")

    def _refresh_system(self, history: list[ChatMessage]) -> None:
        if self._calibration_model != self.model:
            self._bytes_per_token = 3.0
            self._calibration_model = self.model
        if self._refresh_prompt is not None:
            prompt: ChatMessage = {"role": "system", "content": self._refresh_prompt()}
            if history and history[0].get("role") == "system":
                history[0] = prompt
            else:
                history.insert(0, prompt)

    async def _stream_turn(
        self, history: list[ChatMessage], on_event: OnEvent | None, source_version: tuple | None
    ) -> CompletedMessage:
        thinking = self.config.llm.thinking
        reasoning_effort = None if thinking == "none" else thinking

        async def invoke() -> CompletedMessage:
            self._check_generation()
            self._refresh_system(history)
            tools = self.registry.openai_tool_specs() or None
            window, headroom = context_limits(self.model, self.config, self._catalog)
            if (
                estimate_request(history, tools, bytes_per_token=self._bytes_per_token) + headroom
                > window
            ):
                raise ContextPaused("Refreshed context exceeds the current model budget; paused.")
            if (
                self.store is not None
                and self.session is not None
                and (
                    source_version is None
                    or self.store.source_version(self.session, include_worker_events=False)
                    != source_version
                )
            ):
                raise ContextPaused(
                    "Session changed before provider request; paused. Reload the session."
                )
            stream = self.provider.stream_chat(
                history,
                model=self.model,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_tokens=headroom,
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
                    replace(self.ctx, memory_generation=self._request_generation),
                )
            )
            for call in completed.tool_calls
        ]
        manager = self.ctx.extras.get("workers")
        worker_id = self.ctx.extras.get("worker_id")
        try:
            if manager is not None:
                pairs = await manager.await_tools(worker_id, tasks)
            else:
                pairs = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # Cancel in-flight tools; persist the results that did complete.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                usage = manager.result_usage(task) if manager else None
                if task.done() and not task.cancelled():
                    self._persist_message(task.result()[0], usage)
            raise
        messages: list[ChatMessage] = []
        for call, task, (message, result) in zip(completed.tool_calls, tasks, pairs, strict=True):
            messages.append(message)
            self._persist_message(message, manager.result_usage(task) if manager else None)
            await self._emit(
                on_event,
                ToolResult(
                    id=call["id"],
                    name=call["function"]["name"],
                    content=result.content,
                    is_error=result.is_error,
                    metadata=result.metadata,
                ),
            )
        return messages

    # -- queues -------------------------------------------------------------------

    async def _drain_queues(
        self, history: list[ChatMessage], on_event: OnEvent | None, *, prefetched=None
    ) -> bool:
        """Drain the steer queue first (priority), then the input queue.

        Drained items are appended to the history as user messages, with a
        ``QueuedMessage`` event each so the TUI echoes them when the model
        actually sees them.
        """
        messages = []
        prefetched = dict(prefetched or {})
        for queue in (self.steer_queue, self.input_queue):
            if queue is None:
                continue
            while True:
                try:
                    item = prefetched.pop(queue) if queue in prefetched else queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                message: ChatMessage = {"role": "user", "content": item}
                history.append(message)
                self._persist_message(message)
                messages.append(item)
        # Persist every fetched item before yielding, including cancellation races.
        for item in messages:
            await self._emit(on_event, QueuedMessage(content=item))
        await self._drain_background(history, on_event)
        return bool(messages)

    async def _drain_background(self, history: list[ChatMessage], on_event: OnEvent | None) -> None:
        """Feed background-task completions in as synthetic user messages.

        Drained at run start and between turns so the model hears about a
        finished background task at the next opportunity.
        """
        manager = self.ctx.extras.get(BACKGROUND_EXTRA)
        if manager is None:
            return
        for note in manager.drain_notifications():
            message: ChatMessage = {"role": "user", "content": note}
            history.append(message)
            self._persist_message(message, derived=True)
            await self._emit(on_event, QueuedMessage(content=note))

    async def _consume_workers(
        self, history: list[ChatMessage], on_event: OnEvent | None, *, completing=False
    ) -> bool:
        """Deliver worker inboxes only between model turns, never mid tool batch."""
        manager = self.ctx.extras.get("workers")
        if manager is None:
            return False
        worker_id = self.ctx.extras.get("worker_id")
        getters = {}
        if completing and worker_id is None:
            for queue in (self.steer_queue, self.input_queue):
                if queue is not None and queue not in getters:
                    getters[queue] = asyncio.create_task(queue.get())
                    getters[queue].add_done_callback(manager._signal)
        queued = False
        try:
            items = await manager.boundary(
                worker_id,
                history,
                completing=completing,
                input_ready=lambda: any(task.done() for task in getters.values()),
            )
        finally:
            for task in getters.values():
                if not task.done():
                    task.cancel()
            if getters:
                await asyncio.gather(*getters.values(), return_exceptions=True)
                prefetched = {
                    q: task.result() for q, task in getters.items() if not task.cancelled()
                }
                queued = await self._drain_queues(history, on_event, prefetched=prefetched)
        for item in items:
            await self._emit(on_event, QueuedMessage(content=item["text"]))
        return bool(items or queued)

    # -- automatic compaction -----------------------------------------------------

    async def _maybe_compact(
        self,
        history: list[ChatMessage],
        on_event: OnEvent | None,
        context_tokens: int,
        turns: int,
        on_usage: Callable[[dict | None], None],
    ) -> bool:
        """Compact against the whole outgoing request, including new tool results."""
        compaction = self.config.compaction
        if not compaction.enabled or self.session is None or self.store is None:
            return False
        window, headroom = context_limits(self.model, self.config, self._catalog)
        threshold = window - headroom
        if compaction.mid_turn_threshold is not None and turns >= 1:
            # Tool-loop rounds after the first trigger at this absolute count.
            threshold = min(threshold, int(compaction.mid_turn_threshold))
        if context_tokens < threshold:
            return False
        await self._emit(
            on_event, CompactionStarted(context_tokens=context_tokens, threshold=threshold)
        )
        source_version = self.store.source_version(self.session, include_derivations=False)
        replay = self.store.load_for_model(self.session)
        summary = await compact_session(
            self.provider,
            self.store,
            self.session,
            self.model,
            hooks=self.ctx.extras.get("hooks"),
            config=self.config,
            catalog=self._catalog,
            on_usage=on_usage,
            ctx=self.ctx,
        )
        if self.store.source_version(self.session, include_derivations=False) != source_version:
            raise ContextPaused(
                "Session sources changed during compaction; paused. Reload the session."
            )
        if summary is None:
            return False  # fail-open: keep going uncompacted
        # Rebuild in place so ctx.extras["conversation"] stays valid.
        rebuilt = self.store.load_for_model(self.session)
        remaining = list(history)
        # Remove only the covered prefix, preserving raw tail and transient ordering.
        # ponytail: linear searches; index identities if very large tails become costly.
        for message in replay[: len(replay) - len(rebuilt) + 1]:
            if message in remaining:
                remaining.remove(message)
        split = 0
        while split < len(remaining) and remaining[split].get("role") == "system":
            split += 1
        history[:] = [*remaining[:split], rebuilt[0], *remaining[split:]]
        await self._emit(on_event, CompactionFinished(summary_chars=len(summary)))
        return True

    # -- usage / cost ---------------------------------------------------------------

    def _usage_cost(self, model: str, usage: dict[str, Any] | None) -> tuple[int, int, float, bool]:
        """Input/output tokens, known cost in USD, and whether usage is incomplete."""
        if not usage:
            return 0, 0, 0.0, True
        in_tok, out_tok = _usage_tokens(usage)
        incomplete = bool(usage.get("incomplete"))
        if usage.get("cost_usd") is not None:
            return in_tok, out_tok, float(usage["cost_usd"]), incomplete
        if self._catalog is None:
            self._catalog = self.ctx.catalog or Catalog.default()
        try:
            pricing = self._catalog.get(model).pricing
        except (ModelNotFoundError, AmbiguousModelError):
            return in_tok, out_tok, 0.0, True
        cost = (in_tok * pricing.prompt + out_tok * pricing.completion) / 1e6
        return in_tok, out_tok, cost, incomplete

    def _turn_cost(self, completed: CompletedMessage) -> tuple[int, int, float, bool]:
        """Input/output tokens, known cost, and completeness for one turn."""
        return self._usage_cost(self.model, completed.usage)

    # -- persistence ------------------------------------------------------------------

    def _persist_assistant(
        self, completed: CompletedMessage, in_tok: int, out_tok: int, cost: float, incomplete: bool
    ) -> None:
        usage = {"input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost}
        if incomplete:
            usage["incomplete"] = True
        self._persist_message(completed.as_message(), usage)

    def _persist_message(
        self,
        message: ChatMessage | dict[str, Any],
        usage: dict[str, Any] | None = None,
        *,
        derived: bool = False,
    ) -> None:
        if self.session is not None and self.store is not None:
            self.store.append_message(
                self.session,
                dict(message),
                usage=usage,
                memory_generation=(
                    self._request_generation if derived or message.get("role") != "user" else None
                ),
            )

    def _persist_partial(self, history: list[ChatMessage]) -> None:
        """On cancellation, keep whatever partial assistant turn exists."""
        partial = self._partial
        self._partial = None
        if self.memory_generation() != self._request_generation:
            return
        if partial is None:
            return
        history.append(partial.as_message())
        in_tok, out_tok, cost, incomplete = self._turn_cost(partial)
        self._persist_message(
            {**partial.as_message(), "incomplete": True},
            {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": cost,
                **({"incomplete": True} if incomplete else {}),
            },
        )

    # -- events -------------------------------------------------------------------------

    @staticmethod
    async def _emit(on_event: OnEvent | None, event: AgentEvent) -> None:
        if on_event is None:
            return
        result = on_event(event)
        if inspect.isawaitable(result):
            await result
