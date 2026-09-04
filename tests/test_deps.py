"""Tests for the startup dependency check (fd / rg / rtk)."""

from __future__ import annotations

import os
import stat

import pytest

from lecode.deps import find_missing_binaries, format_missing_error


def _make_fake_bin(dirpath, name: str) -> str:
    path = os.path.join(str(dirpath), name)
    with open(path, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_all_present(tmp_path):
    for name in ("fd", "rg", "rtk"):
        _make_fake_bin(tmp_path, name)
    assert find_missing_binaries(path=str(tmp_path)) == []


def test_all_missing_on_empty_path(tmp_path):
    missing = find_missing_binaries(path=str(tmp_path))
    assert {m.name for m in missing} == {"fd", "rg", "rtk"}


def test_partial_missing(tmp_path):
    _make_fake_bin(tmp_path, "fd")
    _make_fake_bin(tmp_path, "rg")
    missing = find_missing_binaries(path=str(tmp_path))
    assert [m.name for m in missing] == ["rtk"]
    assert "rtk" in missing[0].install_hint


def test_error_message_mentions_every_missing_binary():
    missing = find_missing_binaries(path="/nonexistent-dir-xyz")
    msg = format_missing_error(missing)
    for name in ("fd", "rg", "rtk"):
        assert name in msg
    assert "no fallback" in msg


@pytest.mark.parametrize("name", ["fd", "rg", "rtk"])
def test_real_environment_has_binary(name):
    """Sanity check on the dev machine; skipped implicitly by CI shims if absent."""
    import shutil

    if shutil.which(name) is None:
        pytest.skip(f"{name} not installed on this machine")
