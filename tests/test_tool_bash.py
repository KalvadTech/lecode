"""Tests for the bash tool."""

from __future__ import annotations

import asyncio
import contextlib
import os
import stat
import subprocess

from lecode.agent.tools import bash
from lecode.agent.tools.bash import MAX_OUTPUT_BYTES
from lecode.extras import rtk


async def test_echo(tool_ctx):
    result = await bash.make_tool().run({"command": "echo hello"}, tool_ctx)
    assert not result.is_error
    assert "hello" in result.content


async def test_nonzero_exit_reported(tool_ctx):
    result = await bash.make_tool().run({"command": "exit 7"}, tool_ctx)
    assert result.is_error
    assert "exit code 7" in result.content


async def test_stderr_merged(tool_ctx):
    result = await bash.make_tool().run({"command": "echo err >&2"}, tool_ctx)
    assert "err" in result.content


async def test_timeout(tool_ctx):
    result = await bash.make_tool().run({"command": "sleep 30", "timeout": 0.3}, tool_ctx)
    assert result.is_error
    assert "timed out" in result.content


async def test_idle_timeout(tool_ctx):
    result = await bash.make_tool().run(
        {"command": "echo start; sleep 30", "timeout": 30, "idle_timeout": 0.3}, tool_ctx
    )
    assert result.is_error
    assert "no output" in result.content
    assert "start" in result.content


async def test_truncation_and_overflow_file(tool_ctx, tmp_path):
    command = "yes repeated-output-line | head -c 200000"
    result = await bash.make_tool().run({"command": command}, tool_ctx)
    assert "truncated" in result.content
    overflow_dir = tmp_path / "cfg" / "overflow"
    files = list(overflow_dir.glob("*.log"))
    assert len(files) == 1
    assert files[0].read_text().count("repeated-output-line") > 1000
    assert len(result.content) < MAX_OUTPUT_BYTES + 500


async def test_rtk_compaction_applied(tool_ctx, tmp_path, monkeypatch):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\ncat | tr 'a-z' 'A-Z'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(rtk, "RTK_PATH", str(fake))
    result = await bash.make_tool().run({"command": "echo hello"}, tool_ctx)
    assert "HELLO" in result.content


async def test_rtk_failure_fails_open(tool_ctx, tmp_path, monkeypatch):
    fake = tmp_path / "rtk"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(rtk, "RTK_PATH", str(fake))
    result = await bash.make_tool().run({"command": "echo untouched"}, tool_ctx)
    assert "untouched" in result.content


async def test_runs_in_cwd(tool_ctx, tmp_path):
    result = await bash.make_tool().run({"command": "pwd"}, tool_ctx)
    assert os.path.realpath(result.content.splitlines()[0]) == os.path.realpath(tmp_path)


async def test_cancel_kills_child(tool_ctx):
    """Ctrl-C mid-bash-tool kills the subprocess instead of leaking it."""
    tool = bash.make_tool()
    task = asyncio.ensure_future(tool.run({"command": "sleep 30"}, tool_ctx))
    await asyncio.sleep(0.2)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    out = subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True).stdout
    assert out == b""
