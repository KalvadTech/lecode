"""Tests for the language-server registry and [lsp.servers] overrides."""

from __future__ import annotations

import pytest

from lecode.config.models import Config, LspServerOverride
from lecode.lsp.registry import BUILTIN_SERVERS, server_for_file


@pytest.mark.parametrize(
    ("filename", "language", "binary"),
    [
        ("app.py", "python", "pyright-langserver"),
        ("types.pyi", "python", "pyright-langserver"),
        ("main.ts", "typescript", "typescript-language-server"),
        ("component.tsx", "typescript", "typescript-language-server"),
        ("main.js", "javascript", "typescript-language-server"),
        ("mod.jsx", "javascript", "typescript-language-server"),
        ("lib.rs", "rust", "rust-analyzer"),
        ("main.go", "go", "gopls"),
        ("main.c", "c", "clangd"),
        ("main.cpp", "cpp", "clangd"),
        ("init.lua", "lua", "lua-language-server"),
        ("deploy.sh", "shell", "bash-language-server"),
    ],
)
def test_builtin_file_mapping(filename, language, binary):
    spec = server_for_file(f"/repo/src/{filename}", Config())
    assert spec is not None
    assert spec.language == language
    assert spec.command[0] == binary


@pytest.mark.parametrize("filename", ["README.md", "data.json", "archive.xyz", "Makefile"])
def test_unknown_filetype_returns_none(filename):
    assert server_for_file(f"/repo/{filename}", Config()) is None


def test_override_command_wins():
    config = Config()
    config.lsp.servers["python"] = LspServerOverride(command=["basedpyright-langserver", "--stdio"])
    spec = server_for_file("/repo/app.py", config)
    assert spec is not None
    assert spec.command == ["basedpyright-langserver", "--stdio"]
    assert "*.py" in spec.file_patterns  # patterns kept from the built-in
    assert "pyproject.toml" in spec.root_markers  # markers kept too


def test_override_file_patterns_wins():
    config = Config()
    config.lsp.servers["python"] = LspServerOverride(file_patterns=["*.pyw"])
    assert server_for_file("/repo/app.py", config) is None
    spec = server_for_file("/repo/app.pyw", config)
    assert spec is not None
    assert spec.command[0] == "pyright-langserver"  # command kept from the built-in


def test_override_defines_new_language():
    config = Config()
    config.lsp.servers["zig"] = LspServerOverride(command=["zls"], file_patterns=["*.zig"])
    spec = server_for_file("/repo/main.zig", config)
    assert spec is not None
    assert spec.command == ["zls"]
    assert spec.root_markers == (".git",)  # default markers


def test_incomplete_new_language_is_ignored():
    config = Config()
    config.lsp.servers["zig"] = LspServerOverride(command=["zls"])  # no patterns
    assert server_for_file("/repo/main.zig", config) is None


def test_builtins_have_root_markers():
    for spec in BUILTIN_SERVERS:
        assert spec.root_markers, spec.language
