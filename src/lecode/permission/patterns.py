"""Target extraction and rule matching for the permission checker.

MCP tool names are accepted in both spellings — ``mcp:<server>:<tool>`` and
``mcp__<server>__<tool>`` — and normalized to the colon form for matching.
Tools from the read-equivalent servers (Exa, context7, grep.app) count as
read-class in every permission mode.
"""

from __future__ import annotations

import fnmatch
import json
import re
from typing import Any

from lecode.config.models import PermissionRule

#: MCP servers whose tools are read-equivalent (web search/fetch, docs lookup).
READ_EQUIV_MCP_SERVERS = frozenset({"exa", "context7", "grep-app"})


def mcp_parts(tool_name: str) -> tuple[str, str] | None:
    """Split an MCP tool name into (server, tool); None for non-MCP tools."""
    if tool_name.startswith("mcp:"):
        parts = tool_name.split(":", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            return parts[1], parts[2]
        return None
    if tool_name.startswith("mcp__"):
        rest = tool_name[5:]
        if "__" in rest:
            server, tool = rest.split("__", 1)
            if server and tool:
                return server, tool
    return None


def normalize_mcp_name(tool_name: str) -> str:
    """Canonical ``mcp:<server>:<tool>`` form (underscores in server → dashes)."""
    parts = mcp_parts(tool_name)
    if parts is None:
        return tool_name
    return f"mcp:{parts[0].replace('_', '-')}:{parts[1]}"


def is_read_equiv_mcp(tool_name: str) -> bool:
    parts = mcp_parts(tool_name)
    if parts is None:
        return False
    return parts[0].replace("_", "-") in READ_EQUIV_MCP_SERVERS


def target_of(tool_name: str, args: dict[str, Any]) -> str:
    """The string permission rules match against for a tool call."""
    parts = mcp_parts(tool_name)
    if parts is not None:
        return f"mcp:{parts[0].replace('_', '-')}:{parts[1]}"
    if tool_name == "bash":
        return str(args.get("command", ""))
    if tool_name in ("read", "write", "edit", "list_dir"):
        return str(args.get("path", ""))
    if tool_name in ("grep", "find_files"):
        return str(args.get("path") or args.get("pattern") or "")
    return json.dumps(args, sort_keys=True, default=str)


def rule_matches(rule: PermissionRule, target: str) -> bool:
    """Glob (fnmatch) or regex (re.search) match; invalid regex never matches."""
    if rule.kind == "regex":
        try:
            return re.search(rule.pattern, target) is not None
        except re.error:
            return False
    return fnmatch.fnmatch(target, rule.pattern)


def canonical_args(args: dict[str, Any]) -> str:
    """Stable serialization for doom-loop identity comparison."""
    return json.dumps(args, sort_keys=True, default=str)
