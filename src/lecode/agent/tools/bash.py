"""The ``bash`` tool: shell commands with timeouts and truncation.

Commands are first passed through ``rtk rewrite``, which swaps supported
commands for their token-optimized rtk proxies (``git status`` → ``rtk git
status``); the original command runs unchanged when rtk has no equivalent
(fail-open). Runs via ``/bin/sh -c`` with stderr merged into stdout. Output
over the cap is truncated head/tail and the full text is saved to
``<config_dir>/overflow/<uuid>.log`` with a pointer line.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path

from lecode.agent.tools.base import Tool, ToolContext, ToolResult
from lecode.extras.rtk import rewrite_command

DEFAULT_TIMEOUT_S = 120.0
MAX_TIMEOUT_S = 600.0
MAX_OUTPUT_BYTES = 60_000


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the process *group* — ``sh -c`` children must not survive."""
    import os
    import signal

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def _run_shell(
    command: str, cwd: Path, timeout: float, idle_timeout: float, max_bytes: int
) -> tuple[bytes, int, bool, bool]:
    """Run a shell command; returns (output, exit_code, timed_out, idle_killed)."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        command,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group so _kill_tree works
    )
    buffer = bytearray()
    deadline = time.monotonic() + timeout
    idle_deadline = time.monotonic() + idle_timeout
    timed_out = idle_killed = False

    try:
        while True:
            wait = min(deadline, idle_deadline) - time.monotonic()
            if wait <= 0:
                timed_out = time.monotonic() >= deadline
                idle_killed = not timed_out
                _kill_tree(proc)
                break
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(65536), timeout=wait)
            except TimeoutError:
                timed_out = time.monotonic() >= deadline
                idle_killed = not timed_out
                _kill_tree(proc)
                break
            if not chunk:  # EOF: process exited and pipes drained
                break
            buffer += chunk
            idle_deadline = time.monotonic() + idle_timeout
    except asyncio.CancelledError:
        # Turn aborted (Ctrl-C): never leave the child running.
        _kill_tree(proc)
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        raise

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    exit_code = proc.returncode if proc.returncode is not None else -1
    return bytes(buffer), exit_code, timed_out, idle_killed


def _save_overflow(text: str) -> Path:
    from lecode.config.loader import config_dir

    overflow_dir = config_dir() / "overflow"
    overflow_dir.mkdir(parents=True, exist_ok=True)
    path = overflow_dir / f"{uuid.uuid4().hex}.log"
    path.write_text(text, encoding="utf-8")
    return path


class BashTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="bash",
            description=(
                "Run a shell command (/bin/sh -c). Supported commands are "
                "rewritten to token-optimized rtk proxies when available; "
                "output is truncated head/tail (full output saved to a file)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {
                        "type": "number",
                        "description": (
                            f"Seconds (default {DEFAULT_TIMEOUT_S}, max {MAX_TIMEOUT_S})"
                        ),
                    },
                    "idle_timeout": {
                        "type": "number",
                        "description": "Kill after this many seconds without output",
                    },
                },
                "required": ["command"],
            },
        )

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = str(args["command"])
        timeout = min(MAX_TIMEOUT_S, float(args.get("timeout") or DEFAULT_TIMEOUT_S))
        idle = float(args.get("idle_timeout") or ctx.config.agent.tool_idle_timeout_s)

        command = await rewrite_command(command)
        try:
            output, exit_code, timed_out, idle_killed = await _run_shell(
                command, ctx.cwd, timeout, idle, MAX_OUTPUT_BYTES
            )
        except OSError as e:
            return ToolResult(f"error: {e}", is_error=True)

        text = output.decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT_BYTES:
            full_path = _save_overflow(text)
            half = MAX_OUTPUT_BYTES // 2
            text = (
                text[:half]
                + f"\n… [output truncated; full output saved to {full_path}] …\n"
                + text[-half:]
            )

        notes: list[str] = []
        if timed_out:
            notes.append(f"timed out after {timeout}s")
        if idle_killed:
            notes.append(f"killed: no output for {idle}s")
        if exit_code != 0:
            notes.append(f"exit code {exit_code}")
        suffix = f"\n[{'; '.join(notes)}]" if notes else ""
        return ToolResult(
            (text.rstrip("\n") or "(no output)") + suffix,
            is_error=timed_out or idle_killed or exit_code != 0,
        )


def make_tool() -> Tool:
    return BashTool()
