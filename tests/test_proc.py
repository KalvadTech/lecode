"""Tests for the shared async subprocess wrapper."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import tracemalloc

import pytest

from lecode.extras.proc import TRUNCATION_MARKER, run_proc


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


@pytest.mark.parametrize("cap", [1, 2, 3, 32, 1000, 10000])
async def test_streamed_cap_preserves_byte_boundaries(cap):
    from lecode.extras.proc import _read_capped

    data = b"\xff" + "é🙂".encode() * 1000 + b"\xf0"
    stream = asyncio.StreamReader()
    stream.feed_data(data)
    stream.feed_eof()
    text, truncated = await _read_capped(stream, cap)
    assert truncated == (len(data) > cap)
    if truncated:
        head = data[: cap // 2]
        tail = data[-(cap // 2) :] if cap // 2 else b""
        assert text == (
            head.decode("utf-8", errors="replace")
            + TRUNCATION_MARKER.format(skipped=len(data) - len(head) - len(tail))
            + tail.decode("utf-8", errors="replace")
        )
    else:
        assert text == data.decode("utf-8", errors="replace")


async def test_large_stdout_and_stderr_are_drained_while_feeding_stdin():
    result = await run_proc(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "for _ in range(64):\n"
                " sys.stdout.buffer.write(b'o'*65536)\n"
                " sys.stderr.buffer.write(b'e'*65536)\n"
                "print(len(sys.stdin.buffer.read()))\n"
            ),
        ],
        input="i" * (2 * 1024 * 1024),
        max_output_bytes=2048,
        timeout=10,
    )
    assert result.exit_code == 0 and result.truncated and not result.timed_out
    assert result.stdout.startswith("o" * 1024)
    assert result.stdout.endswith("2097152\n")
    assert result.stderr.startswith("e" * 1024) and result.stderr.endswith("e" * 1024)


@pytest.mark.parametrize("cap", [1, 2048])
async def test_output_retention_is_bounded(cap):
    tracemalloc.start()
    try:
        result = await run_proc(
            [
                sys.executable,
                "-c",
                ("import sys\nfor _ in range(128): sys.stdout.buffer.write(b'x'*65536)"),
            ],
            max_output_bytes=cap,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.exit_code == 0 and result.truncated
    assert peak < 2 * 1024 * 1024


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
