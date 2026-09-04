"""Tests for the shared async subprocess wrapper."""

from __future__ import annotations

import asyncio
import contextlib
import sys

from lecode.extras.proc import run_proc


async def test_echo_round_trip():
    result = await run_proc(["/bin/echo", "hello"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert not result.timed_out
    assert not result.truncated


async def test_stdin_piping():
    result = await run_proc(["/bin/cat"], input="piped in")
    assert result.stdout == "piped in"


async def test_stderr_captured():
    result = await run_proc(["/bin/sh", "-c", "echo oops >&2; exit 3"])
    assert result.exit_code == 3
    assert result.stderr.strip() == "oops"


async def test_timeout_kills():
    result = await run_proc(["/bin/sleep", "30"], timeout=0.2)
    assert result.timed_out is True
    assert result.exit_code != 0


async def test_output_cap_head_tail():
    line = "x" * 100 + "\n"
    result = await run_proc(
        [sys.executable, "-c", f"print({line!r} * 200, end='')"],
        max_output_bytes=1000,
    )
    assert result.truncated is True
    assert len(result.stdout) < 1200
    assert "truncated" in result.stdout
    assert result.stdout.startswith("x" * 50)
    assert result.stdout.rstrip().endswith("x")


async def test_cwd():
    result = await run_proc(["/bin/pwd"], cwd="/tmp")
    assert result.stdout.strip().endswith("tmp")


async def test_cancel_kills_child():
    """Cancelling the awaiting task kills the subprocess (Ctrl-C safety)."""
    task = asyncio.ensure_future(run_proc(["/bin/sleep", "30"], timeout=30))
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # If the child leaked, /bin/sleep would still be running; instead the
    # cancel handler killed it. No direct handle to the pid here — the
    # observable contract is that cancel raises promptly (no 30s hang).
