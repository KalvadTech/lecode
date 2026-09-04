"""Tests for the fixed statusline rendering and git-branch lookup."""

from __future__ import annotations

import subprocess

import pytest

from lecode.config.models import Config
from lecode.tui.statusline import (
    SPINNER_FRAMES,
    CachedBranch,
    StatusLineState,
    StatusState,
    context_meter,
    format_cost,
    git_branch,
    human_tokens,
    render_statusline,
)
from lecode.tui.themes import load_theme


@pytest.fixture
def theme():
    return load_theme("default", Config())


@pytest.fixture
def state(tmp_path):
    return StatusState(
        session_name="my-session",
        agent="default",
        model="openai/gpt-5-mini",
        cwd=tmp_path / "myproj",
        git_branch="main",
        context_used=84000,
        context_window=200000,
        input_tokens=1234,
        output_tokens=400,
        cost_usd=0.0123,
    )


def test_full_layout_contains_every_segment(state, theme):
    plain = render_statusline(state, theme, width=200).plain
    for segment in (
        "my-session",
        "default",
        "openai/gpt-5-mini",
        "myproj:main",
        "ctx ▓▓▓░░ 42%",
        "↑1.2k ↓0.4k",
        "$0.0123",
        "ready",
    ):
        assert segment in plain
    assert " · " in plain


def test_branch_omitted_when_none(state, theme):
    state.git_branch = None
    plain = render_statusline(state, theme).plain
    assert "myproj" in plain
    assert "myproj:" not in plain


def test_meter_percents():
    assert context_meter(0, 200000) == ("░░░░░", 0)
    assert context_meter(84000, 200000) == ("▓▓▓░░", 42)
    assert context_meter(200000, 200000) == ("▓▓▓▓▓", 100)
    assert context_meter(300000, 200000) == ("▓▓▓▓▓", 100)


def test_human_tokens():
    assert human_tokens(42) == "42"
    assert human_tokens(400) == "0.4k"
    assert human_tokens(1234) == "1.2k"
    assert human_tokens(2_500_000) == "2.5M"


def test_format_cost():
    assert format_cost(0.0123) == "$0.0123"
    assert format_cost(0.9999) == "$0.9999"
    assert format_cost(1.0) == "$1.00"
    assert format_cost(1.234) == "$1.23"


def test_spinner_frames_cycle(state, theme):
    for frame in (0, 3, 10, 13):
        state.spinner_frame = frame
        state.state = StatusLineState.RUNNING
        plain = render_statusline(state, theme).plain
        assert SPINNER_FRAMES[frame % len(SPINNER_FRAMES)] in plain
        assert "ready" not in plain


def test_awaiting_approval(state, theme):
    state.state = StatusLineState.AWAITING_APPROVAL
    assert "awaiting approval" in render_statusline(state, theme, width=200).plain


def test_queued_and_steered_suffixes(state, theme):
    state.queued = 2
    state.steered = 1
    plain = render_statusline(state, theme).plain
    assert "+2q" in plain
    assert "+1s" in plain
    state.queued = 0
    state.steered = 0
    plain = render_statusline(state, theme).plain
    assert "+2q" not in plain
    assert "+1s" not in plain


def test_wide_width_no_truncation(state, theme):
    text = render_statusline(state, theme, width=200)
    assert "…" not in text.plain


def test_narrow_width_truncates_cwd_first(state, theme):
    text = render_statusline(state, theme, width=100)
    assert len(text.plain) <= 100
    assert "myproj:main" not in text.plain
    assert "my-session" in text.plain
    assert "…" in text.plain


def test_narrower_width_then_truncates_session(state, theme):
    text = render_statusline(state, theme, width=90)
    assert len(text.plain) <= 90
    assert "my-session" not in text.plain


async def test_git_branch_in_repo(tmp_path):
    subprocess.run(["git", "init", "-b", "feature"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert await git_branch(tmp_path) == "feature"


async def test_git_branch_outside_repo(tmp_path):
    assert await git_branch(tmp_path) is None


async def test_cached_branch_ttl(monkeypatch, tmp_path):
    calls = []

    async def fake(cwd):
        calls.append(str(cwd))
        return "main"

    monkeypatch.setattr("lecode.tui.statusline.git_branch", fake)
    now = [100.0]
    cache = CachedBranch(clock=lambda: now[0])  # default TTL is 5s
    assert await cache.get(tmp_path) == "main"
    assert await cache.get(tmp_path) == "main"
    assert len(calls) == 1
    now[0] += 4.0
    assert await cache.get(tmp_path) == "main"
    assert len(calls) == 1
    now[0] += 2.0
    assert await cache.get(tmp_path) == "main"
    assert len(calls) == 2


async def test_cached_branch_caches_none(monkeypatch, tmp_path):
    calls = []

    async def fake(cwd):
        calls.append(str(cwd))
        return None

    monkeypatch.setattr("lecode.tui.statusline.git_branch", fake)
    now = [50.0]
    cache = CachedBranch(ttl_s=5.0, clock=lambda: now[0])
    assert await cache.get(tmp_path) is None
    assert await cache.get(tmp_path) is None
    assert len(calls) == 1
