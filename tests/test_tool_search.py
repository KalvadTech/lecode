"""Tests for grep (rg), find_files (fd), list_dir, todo_write."""

from __future__ import annotations

import shutil

import pytest

from lecode.agent.tools import find_files, grep, list_dir, todo_write

needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="rg not installed")
needs_fd = pytest.mark.skipif(shutil.which("fd") is None, reason="fd not installed")


@pytest.fixture
def sample_tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def alpha():\n    pass\n")
    (tmp_path / "src" / "b.py").write_text("def beta():\n    alpha()\n")
    (tmp_path / "README.md").write_text("# alpha project\n")
    return tmp_path


# -- grep (real rg) ---------------------------------------------------------------


@needs_rg
async def test_grep_matches(tool_ctx, sample_tree):
    result = await grep.make_tool().run({"pattern": "alpha"}, tool_ctx)
    assert "src/a.py:1: def alpha():" in result.content
    assert result.metadata["match_count"] == 3


@needs_rg
async def test_grep_glob_filter(tool_ctx, sample_tree):
    result = await grep.make_tool().run({"pattern": "alpha", "glob": "*.md"}, tool_ctx)
    assert "README.md" in result.content
    assert "src/a.py" not in result.content


@needs_rg
async def test_grep_ignore_case_and_context(tool_ctx, sample_tree):
    result = await grep.make_tool().run(
        {"pattern": "ALPHA", "ignore_case": True, "context": 1, "path": "src"}, tool_ctx
    )
    assert "a.py:1: def alpha():" in result.content
    assert "a.py-2-     pass" in result.content  # context line rendered with dashes


@needs_rg
async def test_grep_no_matches(tool_ctx, sample_tree):
    result = await grep.make_tool().run({"pattern": "zzzzz"}, tool_ctx)
    assert result.content == "no matches"
    assert not result.is_error


@needs_rg
async def test_grep_max_results_cap(tool_ctx, tmp_path):
    (tmp_path / "many.txt").write_text("\n".join(f"hit {i}" for i in range(20)))
    result = await grep.make_tool().run({"pattern": "hit", "max_results": 5}, tool_ctx)
    assert result.metadata["match_count"] == 20
    assert "more matches" in result.content


# -- find_files (real fd) ----------------------------------------------------------


@needs_fd
async def test_find_files_glob(tool_ctx, sample_tree):
    result = await find_files.make_tool().run({"pattern": "*.py"}, tool_ctx)
    assert "src/a.py" in result.content
    assert "src/b.py" in result.content
    assert "README.md" not in result.content


@needs_fd
async def test_find_files_type_directory(tool_ctx, sample_tree):
    result = await find_files.make_tool().run({"pattern": "*", "type": "directory"}, tool_ctx)
    assert "src" in result.content
    assert "a.py" not in result.content


@needs_fd
async def test_find_files_none(tool_ctx, sample_tree):
    result = await find_files.make_tool().run({"pattern": "*.rs"}, tool_ctx)
    assert result.content == "no files found"


# -- list_dir -----------------------------------------------------------------------


async def test_list_dir_dirs_first(tool_ctx, sample_tree):
    result = await list_dir.make_tool().run({}, tool_ctx)
    lines = result.content.splitlines()
    assert lines[0] == "src/"
    assert "README.md" in lines


async def test_list_dir_not_a_directory(tool_ctx):
    result = await list_dir.make_tool().run({"path": "nope"}, tool_ctx)
    assert result.is_error


# -- todo_write ---------------------------------------------------------------------


async def test_todo_write_renders_and_stores(tool_ctx):
    todos = [
        {"title": "plan", "status": "done"},
        {"title": "code", "status": "in_progress"},
        {"title": "test", "status": "pending"},
    ]
    result = await todo_write.make_tool().run({"todos": todos}, tool_ctx)
    assert "[x] plan" in result.content
    assert "[~] code" in result.content
    assert "[ ] test" in result.content
    assert tool_ctx.todos == todos


async def test_todo_write_validates(tool_ctx):
    bad = await todo_write.make_tool().run({"todos": [{"title": "x", "status": "bogus"}]}, tool_ctx)
    assert bad.is_error
    assert "invalid status" in bad.content
