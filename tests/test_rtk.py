"""Tests for rtk output compaction (fail-open semantics)."""

from __future__ import annotations

import os
import stat

from lecode.extras import rtk
from lecode.extras.rtk import compact_output


def _make_rtk(tmp_path, body: str) -> str:
    path = tmp_path / "rtk"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


async def test_rewrite_applied(tmp_path):
    binary = _make_rtk(tmp_path, "cat | tr 'a-z' 'A-Z'")
    assert await compact_output("hello", rtk_path=binary) == "HELLO"


async def test_missing_binary_fails_open():
    assert await compact_output("original", rtk_path="/nonexistent/rtk") == "original"


async def test_none_binary_fails_open():
    assert await compact_output("original", rtk_path=None) == "original"


async def test_timeout_fails_open(tmp_path):
    binary = _make_rtk(tmp_path, "sleep 10")
    original = "some long output"
    assert await compact_output(original, rtk_path=binary) == original


async def test_nonzero_exit_fails_open(tmp_path):
    binary = _make_rtk(tmp_path, "exit 1")
    assert await compact_output("original", rtk_path=binary) == "original"


async def test_empty_input_passthrough():
    assert await compact_output("") == ""


def test_module_level_path_resolution():
    # RTK_PATH is resolved once at import; either present or None, never raises.
    assert rtk.RTK_PATH is None or os.path.exists(rtk.RTK_PATH)
