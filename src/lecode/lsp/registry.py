"""The built-in language-server registry, with ``[lsp.servers]`` overrides.

Each language maps to a command (argv), the file patterns it serves
(fnmatch over the file name), and root markers: the nearest ancestor
directory holding one of them becomes the server's project root (else the
agent's cwd). ``[lsp.servers.<lang>]`` config entries override the command
and/or file patterns of a built-in entry, or define a new language.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

from lecode.config.models import Config


@dataclass(frozen=True)
class ServerSpec:
    language: str
    command: list[str]
    file_patterns: tuple[str, ...]
    root_markers: tuple[str, ...] = field(default=(".git",))


BUILTIN_SERVERS: tuple[ServerSpec, ...] = (
    ServerSpec(
        "python",
        ["pyright-langserver", "--stdio"],
        ("*.py", "*.pyi"),
        ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", ".git"),
    ),
    ServerSpec(
        "typescript",
        ["typescript-language-server", "--stdio"],
        ("*.ts", "*.tsx"),
        ("tsconfig.json", "package.json", ".git"),
    ),
    ServerSpec(
        "javascript",
        ["typescript-language-server", "--stdio"],
        ("*.js", "*.jsx", "*.mjs", "*.cjs"),
        ("package.json", ".git"),
    ),
    ServerSpec("rust", ["rust-analyzer"], ("*.rs",), ("Cargo.toml", ".git")),
    ServerSpec("go", ["gopls"], ("*.go",), ("go.mod", "go.work", ".git")),
    ServerSpec("c", ["clangd"], ("*.c", "*.h"), ("compile_commands.json", ".git")),
    ServerSpec(
        "cpp",
        ["clangd"],
        ("*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh"),
        ("compile_commands.json", ".git"),
    ),
    ServerSpec("lua", ["lua-language-server"], ("*.lua",), (".luarc.json", ".git")),
    ServerSpec("shell", ["bash-language-server", "start"], ("*.sh", "*.bash"), (".git",)),
)


def all_servers(config: Config) -> list[ServerSpec]:
    """Built-ins with ``[lsp.servers]`` overrides applied (same order)."""
    overrides = config.lsp.servers
    merged: list[ServerSpec] = []
    for spec in BUILTIN_SERVERS:
        override = overrides.get(spec.language)
        if override is None:
            merged.append(spec)
            continue
        merged.append(
            ServerSpec(
                spec.language,
                list(override.command) or spec.command,
                tuple(override.file_patterns) or spec.file_patterns,
                spec.root_markers,
            )
        )
    for language, override in overrides.items():
        if language in {spec.language for spec in BUILTIN_SERVERS}:
            continue
        if override.command and override.file_patterns:
            merged.append(
                ServerSpec(
                    language,
                    list(override.command),
                    tuple(override.file_patterns),
                )
            )
    return merged


def server_for_file(path: str, config: Config) -> ServerSpec | None:
    """The server spec matching ``path``'s file name, if any."""
    name = path.rsplit("/", 1)[-1]
    for spec in all_servers(config):
        if any(fnmatch.fnmatch(name, pattern) for pattern in spec.file_patterns):
            return spec
    return None
