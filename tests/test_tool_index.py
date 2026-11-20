"""Tests for the index tool (tree-sitter skeletons)."""

from __future__ import annotations

from lecode.agent.tools.index import make_tool


async def test_python_skeleton(tool_ctx, tmp_path):
    (tmp_path / "mod.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "class Foo:\n"
        "    def bar(self, x: int) -> str:\n"
        "        return str(x)\n"
        "\n"
        "\n"
        "def main() -> None:\n"
        "    pass\n"
    )
    result = await make_tool().run({"path": "mod.py"}, tool_ctx)
    assert not result.is_error
    header, *lines = result.content.splitlines()
    assert header == "mod.py — python, 11 lines"
    assert "import os (L1)" in lines
    assert "from pathlib import Path (L2)" in lines
    assert "class Foo (L5-7)" in lines
    assert "  def bar(self, x: int) -> str (L6-7)" in lines
    assert "def main() -> None (L10-11)" in lines


async def test_python_multiline_signature_collapsed(tool_ctx, tmp_path):
    (tmp_path / "m.py").write_text("def f(\n    a: int,\n    b: str,\n) -> None:\n    pass\n")
    result = await make_tool().run({"path": "m.py"}, tool_ctx)
    assert "def f( a: int, b: str, ) -> None (L1-5)" in result.content


async def test_rust_skeleton(tool_ctx, tmp_path):
    (tmp_path / "lib.rs").write_text(
        "use std::io;\n"
        "\n"
        "pub struct Config {\n"
        "    pub name: String,\n"
        "}\n"
        "\n"
        "impl Config {\n"
        "    pub fn new(name: &str) -> Self {\n"
        "        Self { name: name.into() }\n"
        "    }\n"
        "}\n"
    )
    result = await make_tool().run({"path": "lib.rs"}, tool_ctx)
    assert not result.is_error
    assert "use std::io (L1)" in result.content
    assert "pub struct Config (L3-5)" in result.content
    assert "impl Config (L7-11)" in result.content
    assert "  pub fn new(name: &str) -> Self (L8-10)" in result.content


async def test_javascript_arrow_const(tool_ctx, tmp_path):
    (tmp_path / "app.js").write_text(
        "import { x } from './x.js';\n"
        "const helper = () => 42;\n"
        "const plain = 1;\n"
        "export function run() {\n"
        "  return helper();\n"
        "}\n"
    )
    result = await make_tool().run({"path": "app.js"}, tool_ctx)
    assert not result.is_error
    assert "import { x } from './x.js' (L1)" in result.content
    assert "const helper = () => 42 (L2)" in result.content
    assert "plain" not in result.content
    assert "function run() (L4-6)" in result.content


async def test_elixir_skeleton(tool_ctx, tmp_path):
    (tmp_path / "foo.ex").write_text(
        "defmodule Foo do\n"
        "  import Enum\n"
        "\n"
        "  def bar(x), do: x\n"
        "\n"
        "  defp baz do\n"
        "    1\n"
        "  end\n"
        "end\n"
    )
    result = await make_tool().run({"path": "foo.ex"}, tool_ctx)
    assert not result.is_error
    assert "defmodule Foo (L1-9)" in result.content
    assert "  import Enum (L2)" in result.content
    assert "  def bar(x), do: x (L4)" in result.content
    assert "  defp baz (L6-8)" in result.content


async def test_unsupported_extension(tool_ctx, tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n")
    result = await make_tool().run({"path": "notes.txt"}, tool_ctx)
    assert not result.is_error
    assert "unsupported file type" in result.content


async def test_missing_file(tool_ctx):
    result = await make_tool().run({"path": "nope.py"}, tool_ctx)
    assert result.is_error
    assert "no such file" in result.content


async def test_directory_is_error(tool_ctx):
    result = await make_tool().run({"path": "."}, tool_ctx)
    assert result.is_error
    assert "is a directory" in result.content


async def test_empty_file(tool_ctx, tmp_path):
    (tmp_path / "empty.py").write_text("")
    result = await make_tool().run({"path": "empty.py"}, tool_ctx)
    assert not result.is_error
    assert "(no definitions found)" in result.content


async def test_item_cap(tool_ctx, tmp_path):
    defs = "".join(f"def f{i}():\n    pass\n\n" for i in range(250))
    (tmp_path / "many.py").write_text(defs)
    result = await make_tool().run({"path": "many.py"}, tool_ctx)
    assert not result.is_error
    assert "more items" in result.content
