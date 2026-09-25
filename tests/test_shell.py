"""Tests for user shell detection (extras/shell.py)."""

from __future__ import annotations

import os
import stat

from lecode.extras.shell import FALLBACK_SHELL, user_shell


def _make_executable(path, body="#!/bin/sh\nexit 0\n"):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_falls_back_to_sh_when_shell_unset(monkeypatch):
    monkeypatch.delenv("SHELL", raising=False)
    assert user_shell() == FALLBACK_SHELL


def test_falls_back_to_sh_when_shell_empty(monkeypatch):
    monkeypatch.setenv("SHELL", "")
    assert user_shell() == FALLBACK_SHELL


def test_falls_back_when_shell_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SHELL", str(tmp_path / "nope"))
    assert user_shell() == FALLBACK_SHELL


def test_falls_back_when_shell_not_executable(monkeypatch, tmp_path):
    plain = tmp_path / "not-a-shell"
    plain.write_text("text\n")
    monkeypatch.setenv("SHELL", str(plain))
    assert user_shell() == FALLBACK_SHELL


def test_absolute_shell_path_is_used(monkeypatch, tmp_path):
    shell = _make_executable(tmp_path / "myshell")
    monkeypatch.setenv("SHELL", str(shell))
    assert user_shell() == str(shell)


def test_bare_shell_name_resolves_on_path(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shell = _make_executable(bin_dir / "myshell")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("SHELL", "myshell")
    assert user_shell() == str(shell)
