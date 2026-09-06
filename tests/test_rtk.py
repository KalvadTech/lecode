"""Tests for rtk command rewriting (fail-open semantics)."""

from __future__ import annotations

import os
import stat

from lecode.extras import rtk
from lecode.extras.rtk import rewrite_command


def _make_rtk(tmp_path, body: str) -> str:
    path = tmp_path / "rtk"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


async def test_rewrite_applied(tmp_path):
    # Real rtk prints the rewritten command with no trailing newline.
    binary = _make_rtk(tmp_path, 'printf %s "rtk ls -la"')
    assert await rewrite_command("ls -la", rtk_path=binary) == "rtk ls -la"


async def test_no_equivalent_keeps_original(tmp_path):
    # Real rtk: empty stdout, exit 1 (exit code must not matter).
    binary = _make_rtk(tmp_path, "exit 1")
    assert await rewrite_command("mycmd --flag", rtk_path=binary) == "mycmd --flag"


async def test_rewrite_success_exit_code_ignored(tmp_path):
    # rtk 0.46 exits 3 on a successful rewrite — non-empty stdout wins.
    binary = _make_rtk(tmp_path, 'echo "rtk git status"; exit 3')
    assert await rewrite_command("git status", rtk_path=binary) == "rtk git status"


async def test_missing_binary_fails_open():
    assert await rewrite_command("ls", rtk_path="/nonexistent/rtk") == "ls"


async def test_none_binary_fails_open(monkeypatch):
    # No rtk on PATH at all: resolution finds nothing.
    monkeypatch.setattr(rtk, "default_path", lambda: None)
    assert await rewrite_command("ls") == "ls"


async def test_timeout_fails_open(tmp_path):
    binary = _make_rtk(tmp_path, "sleep 10")
    assert await rewrite_command("ls", rtk_path=binary) == "ls"


async def test_empty_input_passthrough():
    assert await rewrite_command("") == ""
    assert await rewrite_command("   ") == "   "


def test_default_path_resolution():
    # default_path() resolves rtk on PATH per call; present or None, never raises.
    path = rtk.default_path()
    assert path is None or os.path.exists(path)
