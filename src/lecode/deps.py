"""Startup verification of mandatory external binaries.

lecode hard-depends on three external programs:

- ``fd``    — powers the ``find_files`` tool
- ``rg``    — (ripgrep) powers the ``grep`` tool
- ``rtk``   — compacts ``bash`` tool output (``rtk rewrite``)

There is no auto-download and no fallback code path: if any of them is
missing, startup aborts with install instructions and exit code 2.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

#: Binaries that must exist on PATH, with per-binary install hints.
REQUIRED_BINARIES: dict[str, str] = {
    "fd": "brew install fd  |  cargo install fd-find  |  apt install fd-find",
    "rg": "brew install ripgrep  |  cargo install ripgrep  |  apt install ripgrep",
    "rtk": "see https://github.com/rtk-ai/rtk for install instructions",
}


@dataclass(frozen=True)
class MissingBinary:
    name: str
    install_hint: str


def find_missing_binaries(path: str | None = None) -> list[MissingBinary]:
    """Return the required binaries not found on ``path`` (defaults to ``$PATH``)."""
    missing: list[MissingBinary] = []
    for name, hint in REQUIRED_BINARIES.items():
        if shutil.which(name, path=path) is None:
            missing.append(MissingBinary(name=name, install_hint=hint))
    return missing


def format_missing_error(missing: list[MissingBinary]) -> str:
    """Render the startup error shown when required binaries are absent."""
    lines = [
        "lecode requires external binaries that were not found on PATH:",
        "",
    ]
    for m in missing:
        lines.append(f"  ✗ {m.name:<4} install: {m.install_hint}")
    lines += [
        "",
        "lecode does not auto-download binaries and has no fallback code paths.",
        "Install the missing tools and try again.",
    ]
    return "\n".join(lines)
