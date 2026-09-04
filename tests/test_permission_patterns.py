"""Tests for target extraction, MCP normalization, and rule matching."""

from __future__ import annotations

import pytest

from lecode.config.models import PermissionRule
from lecode.permission.patterns import (
    is_read_equiv_mcp,
    mcp_parts,
    normalize_mcp_name,
    rule_matches,
    target_of,
)


def test_target_bash_is_command():
    assert target_of("bash", {"command": "rm -rf /"}) == "rm -rf /"


def test_target_file_tools_are_paths():
    for tool in ("read", "write", "edit", "list_dir"):
        assert target_of(tool, {"path": "src/a.py"}) == "src/a.py"


def test_target_grep_prefers_path_then_pattern():
    assert target_of("grep", {"path": "src/", "pattern": "foo"}) == "src/"
    assert target_of("grep", {"pattern": "foo"}) == "foo"
    assert target_of("find_files", {"pattern": "*.py"}) == "*.py"


def test_target_unknown_tool_is_canonical_json():
    assert target_of("weird", {"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mcp:exa:web_search", ("exa", "web_search")),
        ("mcp__exa__web_search", ("exa", "web_search")),
        ("mcp__context7__get-library-docs", ("context7", "get-library-docs")),
        ("mcp:grep-app:search", ("grep-app", "search")),
        ("read", None),
        ("mcp:broken", None),
    ],
)
def test_mcp_parts(name, expected):
    assert mcp_parts(name) == expected


def test_mcp_normalization():
    assert normalize_mcp_name("mcp__grep_app__search") == "mcp:grep-app:search"
    assert normalize_mcp_name("mcp:exa:web_search") == "mcp:exa:web_search"
    assert normalize_mcp_name("read") == "read"


def test_mcp_target_is_canonical():
    assert target_of("mcp__exa__web_search", {"q": "x"}) == "mcp:exa:web_search"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mcp:exa:web_search", True),
        ("mcp__exa__web_search", True),
        ("mcp:context7:get-docs", True),
        ("mcp:grep-app:search", True),
        ("mcp__grep_app__search", True),
        ("mcp:filesystem:read_file", False),
        ("read", False),
    ],
)
def test_read_equiv_mcp_servers(name, expected):
    assert is_read_equiv_mcp(name) == expected


def test_glob_matching():
    assert rule_matches(PermissionRule(pattern="git *"), "git status")
    assert rule_matches(PermissionRule(pattern="src/**"), "src/a/b.py")
    assert not rule_matches(PermissionRule(pattern="git *"), "hg status")


def test_regex_matching():
    assert rule_matches(PermissionRule(pattern=r"^rm\s+-rf", kind="regex"), "rm -rf x")
    assert not rule_matches(PermissionRule(pattern=r"^rm\s+-rf", kind="regex"), "rm x")


def test_invalid_regex_never_matches():
    assert not rule_matches(PermissionRule(pattern="([", kind="regex"), "anything")
