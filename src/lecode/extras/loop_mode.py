"""Plan-file loop mode (``--loop`` / ``/loop``).

Each iteration hands the agent the remaining ``- [ ]`` items of a markdown
checklist plan; the agent completes one and marks it ``- [x]`` in the file.
An optional verification command (``--loop-cmd``, e.g. the test suite) runs
after every iteration and its pass/fail output feeds the next prompt.

Exit codes: 0 when the plan is complete, 3 when the iteration budget runs
out, 1 on error (unreadable plan file; provider failures propagate to the
caller, which maps them to 1).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lecode.extras.proc import run_proc

#: Default iteration budget (``--max-iterations``).
DEFAULT_MAX_ITERATIONS = 20

#: Timeout for the per-iteration verification command.
LOOP_CMD_TIMEOUT_S = 300.0

#: How much verification output feeds back into the next prompt (tail).
CMD_OUTPUT_TAIL = 2000

PLAN_ITEM_RE = re.compile(r"^\s*[-*]\s+\[(?P<mark>[ xX])\]\s+(?P<text>.+?)\s*$", re.MULTILINE)

EXIT_DONE = 0
EXIT_ERROR = 1
EXIT_MAX_ITERATIONS = 3


def unfinished_items(plan_text: str) -> list[str]:
    """The text of every ``- [ ]`` checklist item, in file order."""
    return [m.group("text") for m in PLAN_ITEM_RE.finditer(plan_text) if m.group("mark") == " "]


def loop_session_name(now: datetime | None = None) -> str:
    """``loop-YYYYMMDD-HHMMSS`` — the auto-name for loop sessions."""
    return f"loop-{(now or datetime.now()):%Y%m%d-%H%M%S}"


@dataclass(frozen=True)
class LoopResult:
    """How the loop ended: exit code, iterations spent, items remaining."""

    exit_code: int
    iterations: int
    remaining: list[str] = field(default_factory=list)
    stop_reason: str = "done"  # "done" | "max_iterations" | "error"
    error: str = ""


def _iteration_prompt(plan_path: Path, remaining: list[str], cmd_report: str | None) -> str:
    items = "\n".join(f"- {item}" for item in remaining)
    prompt = (
        f"Continue executing the plan in {plan_path}. Pick the next unfinished "
        "item, complete it, then mark it done in the file (change `- [ ]` to "
        "`- [x]`). Do not stop until the item is fully done.\n\n"
        f"Remaining items:\n{items}"
    )
    if cmd_report:
        prompt += f"\n\n{cmd_report}"
    return prompt


async def _verification_report(loop_cmd: str, cwd: Path) -> str:
    """Run the verification command; a pass/fail paragraph for the next prompt."""
    result = await run_proc(["bash", "-c", loop_cmd], cwd=cwd, timeout=LOOP_CMD_TIMEOUT_S)
    if result.timed_out:
        status = f"timed out after {LOOP_CMD_TIMEOUT_S:.0f}s"
    elif result.exit_code == 0:
        status = "passed"
    else:
        status = f"failed (exit code {result.exit_code})"
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if len(output) > CMD_OUTPUT_TAIL:
        output = "…" + output[-CMD_OUTPUT_TAIL:]
    return f"The verification command `{loop_cmd}` {status}.\nOutput:\n{output or '(no output)'}"


async def run_plan_loop(
    run_iteration: Callable[[str], Awaitable[str]],
    plan_path: Path,
    *,
    loop_cmd: str | None = None,
    cwd: Path | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_progress: Callable[[str], None] | None = None,
    on_text: Callable[[str], None] | None = None,
) -> LoopResult:
    """Iterate ``run_iteration(prompt)`` over the plan until done or out of budget.

    ``run_iteration`` is the only agent seam: one prompt in, the iteration's
    final text out. ``on_progress`` gets one line per milestone (progress
    goes to stderr in the CLI, to the feed in the TUI); ``on_text`` gets each
    iteration's final text (stdout in the CLI). Cancellation propagates.
    """
    plan_path = Path(plan_path)
    iterations = 0
    cmd_report: str | None = None
    while True:
        try:
            plan_text = plan_path.read_text(encoding="utf-8")
        except OSError as e:
            return LoopResult(
                EXIT_ERROR,
                iterations,
                stop_reason="error",
                error=f"cannot read plan file: {e}",
            )
        remaining = unfinished_items(plan_text)
        if not remaining:
            if on_progress is not None:
                on_progress(f"plan complete after {iterations} iteration(s)")
            return LoopResult(EXIT_DONE, iterations)
        if iterations >= max_iterations:
            return LoopResult(EXIT_MAX_ITERATIONS, iterations, remaining, "max_iterations")
        iterations += 1
        if on_progress is not None:
            on_progress(f"iteration {iterations}: {len(remaining)} item(s) remaining")
        final_text = await run_iteration(_iteration_prompt(plan_path, remaining, cmd_report))
        if on_text is not None:
            on_text(final_text)
        if loop_cmd:
            cmd_report = await _verification_report(loop_cmd, cwd or plan_path.parent)
