"""Tests for AGENTS.md / CLAUDE.md context collection."""

from __future__ import annotations

import pytest

from lecode.context.agents_md import collect, find_git_root, render


@pytest.fixture
def project(tmp_path):
    """root(.git)/sub with context files at both levels."""
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    sub = root / "sub"
    sub.mkdir()
    (root / "AGENTS.md").write_text("root agents")
    (sub / "AGENTS.md").write_text("sub agents")
    (sub / "CLAUDE.md").write_text("sub claude")
    return root, sub


def test_walk_order_global_then_root_then_cwd(tmp_path, monkeypatch, project):
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    (global_dir / "AGENTS.md").write_text("global agents")
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(global_dir))

    _root, sub = project
    entries = collect(start=sub)
    contents = [c for _, c in entries]
    assert contents == ["global agents", "root agents", "sub agents", "sub claude"]


def test_dedup_when_global_is_on_walk_path(tmp_path, project):
    root, sub = project
    entries = collect(start=sub, config_dir=root)
    paths = [p for p, _ in entries]
    assert paths.count(root / "AGENTS.md") == 1


def test_size_cap(project):
    _, sub = project
    big = sub / "CLAUDE.md"
    big.write_text("x" * 1000)
    entries = collect(start=sub, config_dir=sub / "nonexistent", max_bytes=100)
    for _, content in entries:
        assert len(content) <= 100


def test_no_git_repo_walks_cwd_only(tmp_path):
    parent = tmp_path / "parent"
    cwd = parent / "cwd"
    cwd.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("parent agents")  # not collected: above cwd
    (cwd / "AGENTS.md").write_text("cwd agents")
    entries = collect(start=cwd, config_dir=tmp_path / "nonexistent")
    assert [c for _, c in entries] == ["cwd agents"]


def test_render_includes_path_headers(project):
    root, sub = project
    entries = collect(start=sub, config_dir=sub / "nonexistent")
    text = render(entries)
    assert f"## {root / 'AGENTS.md'}" in text
    assert "root agents" in text
    assert "sub claude" in text


def test_find_git_root(tmp_path):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_git_root(deep) is None
    (tmp_path / ".git").mkdir()
    assert find_git_root(deep) == tmp_path
