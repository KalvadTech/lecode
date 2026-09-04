"""The one shared async subprocess wrapper.

Used by the ``grep`` (rg), ``find_files`` (fd), and ``bash`` (rtk) tools and,
later, lifecycle hooks. Kills on timeout and caps captured output head/tail.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_OUTPUT = 1_000_000  # bytes

TRUNCATION_MARKER = "\n… [output truncated: {skipped} bytes elided] …\n"


@dataclass(frozen=True)
class ProcResult:
    exit_code: int  # -1 when killed after timeout
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


def _cap(data: bytes, max_bytes: int) -> tuple[str, bool]:
    """Decode with head/tail keeping; the middle is elided past the cap."""
    if len(data) <= max_bytes:
        return data.decode("utf-8", errors="replace"), False
    half = max_bytes // 2
    head = data[:half]
    tail = data[-half:]
    skipped = len(data) - len(head) - len(tail)
    text = (
        head.decode("utf-8", errors="replace")
        + TRUNCATION_MARKER.format(skipped=skipped)
        + tail.decode("utf-8", errors="replace")
    )
    return text, True


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
    )
    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input.encode() if input is not None else None),
            timeout=timeout,
        )
    except TimeoutError:
        timed_out = True
        proc.kill()
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except TimeoutError:
            stdout_b, stderr_b = b"", b""
    stdout, out_truncated = _cap(stdout_b or b"", max_output_bytes)
    stderr, err_truncated = _cap(stderr_b or b"", max_output_bytes)
    exit_code = proc.returncode if proc.returncode is not None else -1
    return ProcResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=out_truncated or err_truncated,
    )
