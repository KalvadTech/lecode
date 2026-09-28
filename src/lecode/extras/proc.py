"""The one shared async subprocess wrapper.

Used by the ``grep`` (rg), ``find_files`` (fd), and ``bash`` (rtk) tools and,
later, lifecycle hooks. Kills on timeout and caps captured output head/tail.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_OUTPUT = 1_000_000  # bytes

TRUNCATION_MARKER = "\n… [output truncated: {skipped} bytes elided] …\n"


def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill the whole process group — ``sh -c`` children must not survive."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


@dataclass(frozen=True)
class ProcResult:
    exit_code: int  # -1 when killed after timeout
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


async def _read_capped(stream: asyncio.StreamReader, max_bytes: int) -> tuple[str, bool]:
    """Drain a pipe while retaining the reported prefix and suffix."""
    head = bytearray()
    tail = bytearray()
    total = 0
    half = max_bytes // 2
    while chunk := await stream.read(65536):
        total += len(chunk)
        head.extend(chunk[: max_bytes - len(head)])
        tail.extend(chunk)
        if half and len(tail) > half:
            del tail[:-half]
    if total <= max_bytes:
        return bytes(head).decode("utf-8", errors="replace"), False
    return (
        bytes(head[:half]).decode("utf-8", errors="replace")
        + TRUNCATION_MARKER.format(skipped=total - half - len(tail))
        + bytes(tail).decode("utf-8", errors="replace"),
        True,
    )


async def run_proc(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    input: str | None = None,
    timeout: float = 30.0,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT,
) -> ProcResult:
    """Run ``argv``, capturing stdout/stderr with timeout and output caps."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=max_output_bytes * 2,
        start_new_session=True,  # own process group so _kill works on trees
    )

    async def feed_input() -> None:
        if proc.stdin is not None:
            try:
                proc.stdin.write(input.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Match communicate(): a child may close stdin early.
            finally:
                proc.stdin.close()

    tasks = [
        asyncio.create_task(_read_capped(proc.stdout, max_output_bytes)),
        asyncio.create_task(_read_capped(proc.stderr, max_output_bytes)),
        asyncio.create_task(feed_input()),
    ]

    async def communicate():
        stdout, stderr, _ = await asyncio.gather(*tasks)
        await proc.wait()
        return stdout, stderr

    communication = asyncio.create_task(communicate())
    timed_out = False
    try:
        results = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout)
    except TimeoutError:
        timed_out = True
        _kill(proc)
        try:
            results = await asyncio.wait_for(asyncio.shield(communication), timeout=5.0)
        except TimeoutError:
            results = [("", False), ("", False)]
    except BaseException:
        _kill(proc)
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        raise
    finally:
        communication.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(communication, *tasks, return_exceptions=True)
    (stdout, out_truncated), (stderr, err_truncated) = results[:2]
    exit_code = proc.returncode if proc.returncode is not None else -1
    return ProcResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=out_truncated or err_truncated,
    )
