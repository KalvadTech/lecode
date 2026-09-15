"""Prompt chaining: brainstorm → plan → code → review with auto-transitions.

Each phase is a separate agent run with its own opinionated prompt
(``data/prompts/chain/<phase>.md``, overridable through the resources
layers). Every phase's output feeds the next phase's prompt as accumulated
context; the chain auto-advances as each phase completes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lecode.context.resources import load_text
from lecode.providers.openai_compat import ProviderError

#: The chain's phases, in order.
PHASES = ("brainstorm", "plan", "code", "review")


@dataclass(frozen=True)
class ChainResult:
    """The chain's per-phase outputs, in order."""

    topic: str
    phases: list[tuple[str, str]]  # (phase name, phase output)

    @property
    def final(self) -> str:
        """The last phase's output (the review)."""
        return self.phases[-1][1] if self.phases else ""


async def run_chain(
    runner_factory: Callable[[], Any],
    topic: str,
    *,
    phases: tuple[str, ...] = PHASES,
    system_prompt: str | None = None,
    cwd: Path | None = None,
    on_phase: Callable[[str, str], None] | None = None,
) -> ChainResult:
    """Run the chain over ``topic``.

    ``runner_factory`` builds a fresh runner per phase — each phase is its
    own conversation, chained only through the accumulated context text.
    ``on_phase(phase, output)`` fires as each phase completes.
    """
    outputs: list[tuple[str, str]] = []
    context: list[str] = []
    generation: int | None = None
    for phase in phases:
        template = load_text("prompts", f"chain/{phase}.md", cwd=cwd)
        prompt = template.replace("{topic}", topic).strip()
        if context:
            prompt += "\n\n# Prior phases\n\n" + "\n\n".join(context)
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        runner = runner_factory()
        if generation is None:
            generation = runner.memory_generation()
        result = await runner.run(messages, expected_generation=generation)
        if result.stop_reason == "context_overflow":
            raise ProviderError(
                f"chain paused during {phase}: context cannot safely fit", retryable=False
            )
        outputs.append((phase, result.final_text))
        context.append(f"## {phase}\n\n{result.final_text}")
        if on_phase is not None:
            on_phase(phase, result.final_text)
    return ChainResult(topic=topic, phases=outputs)
