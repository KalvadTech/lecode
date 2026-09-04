"""The ``advisor`` tool: a second "expert" model consulted mid-task.

The agent asks a (usually stronger) model for strategic guidance. The
conversation context is serialized compactly (role + text, tool calls as
one-line summaries) and truncated head/tail to ``[advisor].context_limit_kb``.
A per-session uses counter (``[advisor].max_uses``) bounds the budget; in
``handoff`` mode the call is routed to the human inline instead of a model.

Seams (all via ``ctx.extras``): the provider under ``provider`` and the live
conversation under ``conversation`` are installed by
:class:`~lecode.agent.runner.AgentRunner`; the handoff callback under
``advisor_handoff`` is installed by the TUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.context.resources import load_text

if TYPE_CHECKING:
    from lecode.config.models import AdvisorConfig
    from lecode.providers.types import CompletedMessage

#: ``ctx.extras`` keys the advisor reads (installed by the runner / the TUI).
PROVIDER_EXTRA = "provider"
CONVERSATION_EXTRA = "conversation"
HANDOFF_EXTRA = "advisor_handoff"

#: Head/tail split of the context budget; the middle is elided.
HEAD_FRACTION = 0.2
TAIL_FRACTION = 0.6

#: Clip for one-line tool-call argument summaries.
_ARGS_CLIP = 80


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return str(content or "").strip()


def serialize_message(message: dict[str, Any]) -> str:
    """One compact line: ``role: text [tool(args)]``."""
    role = str(message.get("role", "unknown"))
    parts = [_text_of(message)]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name", "?")
        args = _clip(str(function.get("arguments", "")), _ARGS_CLIP)
        parts.append(f"[{name}({args})]")
    text = " ".join(part for part in parts if part)
    return f"{role}: {text}" if text else f"{role}:"


def truncate_context(messages: list[dict[str, Any]], limit_kb: int) -> str:
    """Serialize ``messages``, truncated head/tail to the KB budget.

    Fits whole serialized lines: the first ~20% of the budget from the head,
    the last ~60% from the tail, with an elision marker in between. Output
    stays within the budget plus one marker line.
    """
    budget = max(1, limit_kb) * 1024
    lines = [serialize_message(message) for message in messages]
    if sum(len(line.encode()) + 1 for line in lines) <= budget:
        return "\n".join(lines)

    def take(candidates: Any, line_budget: int) -> list[str]:
        taken: list[str] = []
        used = 0
        for line in candidates:
            size = len(line.encode()) + 1
            if used + size > line_budget:
                break
            taken.append(line)
            used += size
        return taken

    head = take(lines, int(budget * HEAD_FRACTION))
    tail = take(reversed(lines[len(head) :]), int(budget * TAIL_FRACTION))
    tail.reverse()
    elided = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"[… {elided} messages elided …]", *tail])


async def _complete(provider: Any, messages: list[dict[str, Any]], model: str) -> CompletedMessage:
    """One non-streaming call on the duck-typed provider seam."""
    complete = getattr(provider, "complete", None)
    if complete is not None:
        return await complete(messages, model=model)
    from lecode.providers.types import collect

    return await collect(provider.stream_chat(messages, model=model))


class AdvisorTool(Tool):
    """Ask the advisor model (or the human, in handoff mode) for guidance."""

    def __init__(self) -> None:
        super().__init__(
            name="advisor",
            description=(
                "Ask the advisor — a second, expert model — for strategic guidance "
                "mid-task (approach, trade-offs, risks). Use sparingly: the number "
                "of advisor calls per session is limited."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The strategic question for the advisor.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional files/topic the advisor should focus on.",
                    },
                },
                "required": ["question"],
            },
        )
        #: Calls made this session (reset on session switch).
        self.uses = 0

    def reset_uses(self) -> None:
        self.uses = 0

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        config = ctx.config.advisor
        if not config.enabled:
            return ToolResult(
                "error: the advisor is disabled — enable with /advisor on "
                "or [advisor] enabled = true",
                is_error=True,
            )
        if self.uses >= config.max_uses:
            return ToolResult(
                f"error: advisor budget exhausted ({self.uses}/{config.max_uses})",
                is_error=True,
            )
        question = str(args.get("question") or "").strip()
        if not question:
            return ToolResult("error: advisor needs a question", is_error=True)
        focus = str(args.get("focus") or "").strip() or None
        self.uses += 1
        if config.mode == "handoff":
            return await self._handoff(question, focus, ctx)
        return await self._ask_model(question, focus, ctx, config)

    async def _handoff(self, question: str, focus: str | None, ctx: ToolContext) -> ToolResult:
        """Handoff mode: the human answers inline instead of a model."""
        callback = ctx.extras.get(HANDOFF_EXTRA)
        if callback is None:
            return ToolResult(
                "error: advisor handoff is unavailable here (needs the interactive TUI)",
                is_error=True,
            )
        answer = await callback(question, focus)
        if not answer:
            return ToolResult("error: advisor handoff declined (no guidance given)", is_error=True)
        return ToolResult(f"Advisor (human): {answer}")

    async def _ask_model(
        self, question: str, focus: str | None, ctx: ToolContext, config: AdvisorConfig
    ) -> ToolResult:
        provider = ctx.extras.get(PROVIDER_EXTRA)
        if provider is None:
            return ToolResult("error: no provider available for the advisor", is_error=True)
        model = config.model or ctx.config.llm.model
        context = truncate_context(
            ctx.extras.get(CONVERSATION_EXTRA) or [], config.context_limit_kb
        )
        prompt = load_text("prompts", "advisor.md", cwd=ctx.cwd)
        user = f"## Conversation so far\n\n{context or '(empty)'}\n\n## Question\n\n{question}"
        if focus:
            user += f"\n\nFocus: {focus}"
        try:
            completed = await _complete(
                provider,
                [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                model,
            )
        except Exception as e:
            return ToolResult(f"error: advisor call failed: {e}", is_error=True)
        answer = (completed.content or "").strip()
        if not answer:
            return ToolResult("error: advisor returned an empty response", is_error=True)
        return ToolResult(f"Advisor: {answer}")


def make_tool() -> Tool:
    return AdvisorTool()
