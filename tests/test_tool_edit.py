"""Tests for write and edit tools."""

from __future__ import annotations

from lecode.agent.tools import edit, write


async def _write(ctx, path, content):
    return await write.make_tool().run({"path": path, "content": content}, ctx)


# -- write ---------------------------------------------------------------------


async def test_write_creates_parents_and_reports_bytes(tool_ctx, tmp_path):
    result = await _write(tool_ctx, "deep/nested/f.txt", "hello")
    assert not result.is_error
    assert "5 bytes" in result.content
    assert (tmp_path / "deep/nested/f.txt").read_text() == "hello"


async def test_write_overwrites_atomically(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.txt", "first")
    await _write(tool_ctx, "f.txt", "second")
    assert (tmp_path / "f.txt").read_text() == "second"
    assert not list(tmp_path.glob("*.tmp"))  # no tmp litter


# -- edit: guard ----------------------------------------------------------------


async def test_edit_refuses_unread_file(tool_ctx, tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "x = 1", "new_string": "x = 2"}, tool_ctx
    )
    assert result.is_error
    assert "not been read" in result.content


async def test_edit_force_bypasses_guard(tool_ctx, tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "x = 1", "new_string": "x = 2", "force": True}, tool_ctx
    )
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "x = 2\n"


async def test_edit_allowed_after_write(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "x = 1\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "x = 1", "new_string": "x = 2"}, tool_ctx
    )
    assert not result.is_error


# -- edit: fuzzy ------------------------------------------------------------------


async def test_fuzzy_exact(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "def f():\n    return 1\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "return 1", "new_string": "return 2"}, tool_ctx
    )
    assert not result.is_error
    assert "return 2" in (tmp_path / "f.py").read_text()


async def test_fuzzy_whitespace_normalized(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "def f():\n        return    1\n")
    result = await edit.make_tool().run(
        {
            "path": "f.py",
            "old_string": "def f():\n    return 1",
            "new_string": "def f():\n    return 2",
        },
        tool_ctx,
    )
    assert not result.is_error
    assert "return 2" in (tmp_path / "f.py").read_text()


async def test_fuzzy_zero_occurrences(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "x = 1\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "nope", "new_string": "y"}, tool_ctx
    )
    assert result.is_error
    assert "not found" in result.content


async def test_fuzzy_two_occurrences_error(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "x = 1\ny\nx = 1\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "x = 1", "new_string": "x = 2"}, tool_ctx
    )
    assert result.is_error
    assert "2 times" in result.content


async def test_fuzzy_replace_all(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "a a a\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "old_string": "a", "new_string": "b", "replace_all": True}, tool_ctx
    )
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "b b b\n"


# -- edit: crc ----------------------------------------------------------------------


async def _anchored(ctx, path) -> list[str]:
    from lecode.agent.tools.read import make_tool as read_tool

    result = await read_tool().run({"path": path, "with_anchors": True}, ctx)
    return result.content.splitlines()


async def test_crc_replace_span(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "one\ntwo\nthree\nfour\n")
    lines = await _anchored(tool_ctx, "f.py")
    start = lines[1].split("\t")[0]  # anchor of "two"
    end = lines[2].split("\t")[0]  # anchor of "three"
    result = await edit.make_tool().run(
        {
            "path": "f.py",
            "engine": "crc",
            "start_anchor": start,
            "end_anchor": end,
            "new_string": "TWO-THREE",
        },
        tool_ctx,
    )
    assert not result.is_error
    assert (tmp_path / "f.py").read_text() == "one\nTWO-THREE\nfour\n"


async def test_crc_anchor_mismatch(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "one\ntwo\n")
    lines = await _anchored(tool_ctx, "f.py")
    stale = lines[0].split("\t")[0]
    (tmp_path / "f.py").write_text("CHANGED\ntwo\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "engine": "crc", "start_anchor": stale, "new_string": "x"},
        tool_ctx,
    )
    assert result.is_error
    assert "file changed" in result.content


async def test_crc_invalid_anchor(tool_ctx, tmp_path):
    await _write(tool_ctx, "f.py", "one\n")
    result = await edit.make_tool().run(
        {"path": "f.py", "engine": "crc", "start_anchor": "bogus", "new_string": "x"}, tool_ctx
    )
    assert result.is_error
    assert "invalid start_anchor" in result.content
