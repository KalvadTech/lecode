"""Subagent dispatch: run a child agent loop and hand back its final text.

A subagent gets a lean tool registry (everything the parent has except
``task`` — no recursion), a permission
checker narrowed by the agent's overlay, fresh todos, and its own
conversation. The parent's provider is reused (``ctx.extras["provider"]``,
installed by the runner); progress is reported through the
``subagent_events`` callback the TUI installs.

``SubagentStart``/``SubagentEnd`` hooks are observational — verdicts are
ignored. A run is bounded by ``SUBAGENT_TIMEOUT_S`` and its final text is
capped at ``SUBAGENT_RESPONSE_CAP``. Cancelling the parent turn cancels the
child with it (the child is awaited inside the parent's tool dispatch).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lecode.agent.prompts import build_system_prompt
from lecode.agent.runner import AgentRunner, ToolCall, ToolResult
from lecode.agent.tools.base import ToolContext, ToolRegistry
from lecode.hooks import SUBAGENT_END, SUBAGENT_START, build_envelope, dispatch_event
from lecode.providers.openai_compat import ProviderError

if TYPE_CHECKING:
    from lecode.agent.runner import AgentEvent, RunResult
    from lecode.context.agents import AgentRegistry
    from lecode.hooks import HookDispatcher

#: Hard wall-clock bound for one subagent run.
SUBAGENT_TIMEOUT_S = 300.0

#: Cap on the final text handed back to the parent (chars).
SUBAGENT_RESPONSE_CAP = 32 * 1024

#: Tools never handed to a subagent (no recursion, no user interaction).
CHILD_EXCLUDED_TOOLS = frozenset({"task", "ask_user"})

#: ``ctx.extras`` keys the subagent machinery reads.
PROVIDER_EXTRA = "provider"
HOOKS_EXTRA = "hooks"
REGISTRY_EXTRA = "registry"
AGENTS_EXTRA = "agents"
MEMORY_EXTRA = "memory"
SUBAGENT_EVENTS_EXTRA = "subagent_events"

#: Clip for the SubagentEnd hook envelope's result content.
_END_CONTENT_CLIP = 2000


@dataclass(frozen=True)
class SubagentProgress:
    """One child runner event, tagged with the run's stable identity.

    ``run_id`` distinguishes two concurrent runs of the same agent;
    ``description`` is the human label the caller supplied (task description).
    """

    run_id: str
    agent: str
    description: str
    event: AgentEvent


#: ``on_event(progress)`` — one call per child runner event.
OnSubagentEvent = Callable[[SubagentProgress], Any]


class SubagentError(Exception):
    """A subagent could not run (unknown agent, no provider, timeout, …)."""


@dataclass(frozen=True)
class SubagentOutcome:
    """What a finished subagent run hands back to its caller."""

    agent: str
    text: str
    turns: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    run_id: str = ""
    description: str = ""


def child_registry(parent: ToolRegistry) -> ToolRegistry:
    """The parent's tools minus recursion (hook wrappers ride along)."""
    tools = [parent.get(name) for name in parent.names() if name not in CHILD_EXCLUDED_TOOLS]
    return ToolRegistry([tool for tool in tools if tool is not None])


async def _fire_hook(
    ctx: ToolContext,
    event: str,
    agent: str,
    *,
    prompt: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """Dispatch a Subagent hook event when handlers exist; verdicts ignored."""
    hooks: HookDispatcher | None = ctx.extras.get(HOOKS_EXTRA)
    if hooks is None:
        return
    handlers = hooks.handlers.get(event, [])
    if not handlers:
        return
    envelope = build_envelope(
        event, ctx.cwd, session=ctx.session, agent=agent, prompt=prompt, result=result
    )
    await dispatch_event(event, envelope, handlers)


def _persist_agent_run(
    ctx: ToolContext,
    *,
    run_id: str,
    agent: str,
    description: str,
    prompt: str,
    status: str,
    answer: str,
    error: str | None,
    trail: dict[str, dict[str, Any]],
    trail_order: list[str],
    duration_s: float,
    result: RunResult | None,
) -> None:
    """Append the run's bounded activity trail to the parent session.

    Fail-open: activity is display-only, so a storage failure must not break
    the run or mask its outcome.
    """
    store = ctx.session_store
    session = ctx.session
    if store is None or session is None:
        return
    run: dict[str, Any] = {
        "run_id": run_id,
        "agent": agent,
        "description": description,
        "prompt": prompt,
        "status": status,
        "answer": answer or error or "",
        "tool_calls": [trail[call_id] for call_id in trail_order],
        "duration_s": duration_s,
    }
    if result is not None:
        totals = result.usage_totals
        run.update(
            turns=result.turns,
            input_tokens=totals.input_tokens,
            output_tokens=totals.output_tokens,
            cost_usd=totals.cost_usd,
        )
    with contextlib.suppress(Exception):
        store.record_agent_run(session, run)


async def run_subagent(
    ctx: ToolContext,
    parent_registry: ToolRegistry,
    agents: AgentRegistry,
    *,
    name: str,
    prompt: str,
    description: str = "",
    on_event: OnSubagentEvent | None = None,
) -> SubagentOutcome:
    """Run subagent ``name`` on ``prompt``; returns its final text and usage.

    ``description`` labels the run for the UI and the persisted activity
    record; empty falls back to a prompt preview. Raises :class:`SubagentError`
    for unknown agents, a missing provider, timeouts, and provider failures;
    ``asyncio.CancelledError`` propagates so a cancelled parent turn takes the
    child down with it.
    """
    available = [a.name for a in agents.subagents()]
    agent = agents.get(name)
    if agent is None or name not in available:
        raise SubagentError(
            f"unknown subagent: {name} (available: {', '.join(available) or 'none'})"
        )
    provider = ctx.extras.get(PROVIDER_EXTRA)
    if provider is None:
        raise SubagentError("no provider available for subagents")

    run_id = uuid.uuid4().hex[:8]
    label = description or prompt[:60]

    checker = ctx.permission_checker
    if agent.overlay is not None:
        checker = checker.for_agent(agent.overlay)
    # Fresh extras: the child runner installs its own "conversation" key —
    # sharing the parent's dict would clobber the parent's seam.
    # read_paths stays shared so child reads feed the parent's edit guard.
    extras: dict[str, Any] = {PROVIDER_EXTRA: provider}
    for key in (HOOKS_EXTRA, MEMORY_EXTRA):
        if ctx.extras.get(key) is not None:
            extras[key] = ctx.extras[key]
    child_ctx = dataclasses.replace(
        ctx,
        permission_checker=checker,
        session=None,
        session_store=None,
        todos=[],
        extras=extras,
        question_callback=None,  # children decide themselves; only the parent asks
    )

    system_prompt = build_system_prompt(ctx.config, ctx.cwd, extra=agent.body or None)
    # Pierre reviews the user's task, not subagent side-quests.
    child_config = ctx.config.model_copy(
        update={"pierre": ctx.config.pierre.model_copy(update={"enabled": False})}
    )
    child = AgentRunner(provider, child_registry(parent_registry), child_ctx, config=child_config)
    child.model = agent.model or ctx.config.agent.subagent_model or ctx.config.llm.model

    # Bounded activity trail for the persisted record (tool calls paired by id).
    trail: dict[str, dict[str, Any]] = {}
    trail_order: list[str] = []

    def forward(event: AgentEvent) -> Any:
        if isinstance(event, ToolCall):
            trail[event.id] = {
                "name": event.name,
                "args": event.arguments,
                "result": "",
                "is_error": False,
            }
            trail_order.append(event.id)
        elif isinstance(event, ToolResult):
            entry = trail.setdefault(
                event.id,
                {"name": event.name, "args": "", "result": "", "is_error": False},
            )
            entry["result"] = event.content
            entry["is_error"] = event.is_error
        if on_event is None:
            return None
        return on_event(SubagentProgress(run_id=run_id, agent=name, description=label, event=event))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    started_at = time.monotonic()
    await _fire_hook(ctx, SUBAGENT_START, name, prompt=prompt)
    result: RunResult | None = None
    error: str | None = None
    cancelled = False
    try:
        async with asyncio.timeout(SUBAGENT_TIMEOUT_S):
            result = await child.run(messages, on_event=forward)
    except asyncio.CancelledError:
        cancelled = True
        raise
    except TimeoutError as e:
        error = f"subagent '{name}' timed out after {SUBAGENT_TIMEOUT_S:.0f}s"
        raise SubagentError(error) from e
    except ProviderError as e:
        error = str(e)
        raise SubagentError(error) from e
    finally:
        _persist_agent_run(
            ctx,
            run_id=run_id,
            agent=name,
            description=label,
            prompt=prompt,
            status="cancelled" if cancelled else ("ok" if result is not None else "error"),
            answer=result.final_text if result is not None else "",
            error=error,
            trail=trail,
            trail_order=trail_order,
            duration_s=time.monotonic() - started_at,
            result=result,
        )
        end_content = result.final_text[:_END_CONTENT_CLIP] if result is not None else error
        await _fire_hook(
            ctx,
            SUBAGENT_END,
            name,
            result={"content": end_content or "cancelled", "is_error": result is None},
        )

    text = result.final_text
    if len(text) > SUBAGENT_RESPONSE_CAP:
        text = text[:SUBAGENT_RESPONSE_CAP] + "\n… (truncated)"
    totals = result.usage_totals
    return SubagentOutcome(
        agent=name,
        text=text,
        turns=result.turns,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        cost_usd=totals.cost_usd,
        run_id=run_id,
        description=label,
    )
