"""Tests for the fixed three-line statusline and git-info lookup."""

from __future__ import annotations

import subprocess

import pytest

from lecode.tui.statusline import (
    SPINNER_FRAMES,
    CachedGitInfo,
    GitInfo,
    StatusLineState,
    StatusState,
    context_meter,
    format_cost,
    git_info,
    human_tokens,
    render_statusline,
)
from lecode.tui.themes import THEME


@pytest.fixture
def theme():
    return THEME


@pytest.fixture
def state(tmp_path):
    return StatusState(
        session_name="my-session",
        agent="default",
        model="openai/gpt-5-mini",
        cwd=tmp_path / "myproj",
        git=GitInfo(branch="main", commit="abc1234", diff="±2 +10 -3"),
        context_used=84000,
        context_window=200000,
        input_tokens=1234,
        output_tokens=400,
        cost_usd=0.0123,
    )


def test_line1_folder_git(state, theme):
    line1 = render_statusline(state, theme, width=200).plain.splitlines()[0]
    assert line1 == "dir: myproj · commit: abc1234 · branch: main · diff: ±2 +10 -3"


def test_line1_omits_missing_git_fields(state, theme):
    state.git = GitInfo(branch="main")
    line1 = render_statusline(state, theme, width=200).plain.splitlines()[0]
    assert line1 == "dir: myproj · branch: main"
    state.git = None
    line1 = render_statusline(state, theme, width=200).plain.splitlines()[0]
    assert line1 == "dir: myproj"


def test_line2_model_cost_context(state, theme):
    line2 = render_statusline(state, theme, width=200).plain.splitlines()[1]
    assert line2 == "model: openai/gpt-5-mini · cost: $0.0123 · ctx: ▓▓▓░░ 84.0k/200.0k 42%"


def test_line3_session_agent_tokens_state(state, theme):
    line3 = render_statusline(state, theme, width=200).plain.splitlines()[2]
    assert line3 == "session: my-session · agent: default · in: 1.2k · out: 0.4k · ready"


def test_renders_exactly_three_lines(state, theme):
    assert len(render_statusline(state, theme, width=200).plain.splitlines()) == 3


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
        line3 = render_statusline(state, theme).plain.splitlines()[2]
        assert SPINNER_FRAMES[frame % len(SPINNER_FRAMES)] in line3
        assert "ready" not in line3


def test_awaiting_approval(state, theme):
    state.state = StatusLineState.AWAITING_APPROVAL
    line3 = render_statusline(state, theme, width=200).plain.splitlines()[2]
    assert "awaiting approval" in line3


def test_queued_and_steered_suffixes(state, theme):
    state.queued = 2
    state.steered = 1
    line3 = render_statusline(state, theme).plain.splitlines()[2]
    assert "+2q" in line3
    assert "+1s" in line3
    state.queued = 0
    state.steered = 0
    line3 = render_statusline(state, theme).plain.splitlines()[2]
    assert "+2q" not in line3
    assert "+1s" not in line3


def test_wide_width_no_truncation(state, theme):
    text = render_statusline(state, theme, width=200)
    assert "…" not in text.plain


def test_narrow_width_truncates_each_line(state, theme):
    text = render_statusline(state, theme, width=30)
    lines = text.plain.splitlines()
    assert len(lines) == 3
    assert all(len(line) <= 30 for line in lines)


async def test_git_info_in_repo(tmp_path):
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
    info = await git_info(tmp_path)
    assert info is not None
    assert info.branch == "feature"
    assert info.commit is not None and len(info.commit) == 7
    assert info.diff is None  # clean tree


async def test_git_info_dirty_tree(tmp_path):
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
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
    (tmp_path / "a.txt").write_text("hello\n")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True, capture_output=True)
    info = await git_info(tmp_path)
    assert info is not None
    assert info.diff == "±1 +1"


async def test_git_info_outside_repo(tmp_path):
    assert await git_info(tmp_path) is None


async def test_cached_git_info_ttl(monkeypatch, tmp_path):
    calls = []

    async def fake(cwd):
        calls.append(str(cwd))
        return GitInfo(branch="main")

    monkeypatch.setattr("lecode.tui.statusline.git_info", fake)
    now = [100.0]
    cache = CachedGitInfo(clock=lambda: now[0])  # default TTL is 5s
    assert (await cache.get(tmp_path)).branch == "main"
    assert (await cache.get(tmp_path)).branch == "main"
    assert len(calls) == 1
    now[0] += 4.0
    await cache.get(tmp_path)
    assert len(calls) == 1
    now[0] += 2.0
    await cache.get(tmp_path)
    assert len(calls) == 2


async def test_cached_git_info_caches_none(monkeypatch, tmp_path):
    calls = []

    async def fake(cwd):
        calls.append(str(cwd))
        return None

    monkeypatch.setattr("lecode.tui.statusline.git_info", fake)
    now = [50.0]
    cache = CachedGitInfo(ttl_s=5.0, clock=lambda: now[0])
    assert await cache.get(tmp_path) is None
    assert await cache.get(tmp_path) is None
    assert len(calls) == 1
