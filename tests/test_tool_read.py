"""Tests for the read tool."""

from __future__ import annotations

from lecode.agent.tools.read import line_anchor, make_tool


async def test_basic_read_line_numbered(tool_ctx, tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n")
    result = await make_tool().run({"path": "a.py"}, tool_ctx)
    assert not result.is_error
    assert "1\tone" in result.content
    assert "3\tthree" in result.content


async def test_pagination(tool_ctx, tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
    tool = make_tool()
    first = await tool.run({"path": "big.txt", "offset": 1, "limit": 4}, tool_ctx)
    assert "4\tline4" in first.content
    assert "6 more lines" in first.content
    second = await tool.run({"path": "big.txt", "offset": 5, "limit": 4}, tool_ctx)
    assert "5\tline5" in second.content
    assert "8\tline8" in second.content


async def test_repeat_read_guard(tool_ctx, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    tool = make_tool()
    await tool.run({"path": "a.py"}, tool_ctx)
    again = await tool.run({"path": "a.py"}, tool_ctx)
    assert "already read, file unchanged" in again.content
    # touch -> changed note
    (tmp_path / "a.py").write_text("x = 2\n")
    changed = await tool.run({"path": "a.py"}, tool_ctx)
    assert "file changed" in changed.content


async def test_missing_file(tool_ctx):
    result = await make_tool().run({"path": "nope.py"}, tool_ctx)
    assert result.is_error
    assert "no such file" in result.content


async def test_directory_is_error(tool_ctx):
    result = await make_tool().run({"path": "."}, tool_ctx)
    assert result.is_error
    assert "directory" in result.content


async def test_image_note(tool_ctx, tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    result = await make_tool().run({"path": "pic.png"}, tool_ctx)
    assert not result.is_error
    assert result.metadata.get("image") is True
    assert "image file" in result.content


async def test_anchors_round_trip(tool_ctx, tmp_path):
    (tmp_path / "a.py").write_text("alpha\n  beta\ngamma\n")
    result = await make_tool().run({"path": "a.py", "with_anchors": True}, tool_ctx)
    lines = result.content.splitlines()
    assert lines[0].startswith(line_anchor(1, "alpha") + "\t")
    assert lines[1].startswith(line_anchor(2, "  beta") + "\t")


async def test_read_marks_path_for_edit_guard(tool_ctx, tmp_path):
    (tmp_path / "a.py").write_text("x\n")
    await make_tool().run({"path": "a.py"}, tool_ctx)
    assert str(tmp_path / "a.py") in tool_ctx.read_paths


def test_anchor_format():
    anchor = line_anchor(12, "  hello world  ")
    assert anchor.startswith("12:")
    assert len(anchor.split(":")[1]) == 2
