"""The core tool set and its registry assembly."""

from __future__ import annotations

from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


def core_tools() -> list[Tool]:
    """The thirteen core tools."""
    from lecode.agent.tools import (
        ask_user,
        background,
        bash,
        edit,
        find_files,
        grep,
        list_dir,
        read,
        todo_write,
        write,
    )

    return [
        read.make_tool(),
        write.make_tool(),
        edit.make_tool(),
        bash.make_tool(),
        grep.make_tool(),
        find_files.make_tool(),
        list_dir.make_tool(),
        todo_write.make_tool(),
        ask_user.make_tool(),
        *background.make_tools(),
    ]


__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolResult", "core_tools"]
