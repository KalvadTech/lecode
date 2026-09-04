"""Collection of project-context files (``AGENTS.md`` / ``CLAUDE.md``).

Order: the global ``~/.config/lecode/AGENTS.md`` first, then every
``AGENTS.md`` and ``CLAUDE.md`` in each directory from the git root down to
the cwd. If no git root is found, only the cwd is walked. Each file is
size-capped on read.
"""

from __future__ import annotations

from pathlib import Path

#: Maximum bytes read from a single context file.
MAX_FILE_BYTES = 64 * 1024

#: File names collected from each directory, in order.
CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")


def find_git_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the nearest directory containing ``.git``."""
    current = start.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _read_capped(path: Path, max_bytes: int) -> str:
    with path.open("rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def collect(
    start: Path | None = None,
    config_dir: Path | None = None,
    max_bytes: int = MAX_FILE_BYTES,
) -> list[tuple[Path, str]]:
    """Collect context files, global first, then git root → cwd.

    Returns an ordered, de-duplicated list of ``(path, content)`` pairs.
    ``config_dir`` defaults to the lecode global config directory.
    """
    if config_dir is None:
        from lecode.config.loader import config_dir as _config_dir

        config_dir = _config_dir()
    start = (start or Path.cwd()).resolve()

    entries: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.is_file() and path not in seen:
            seen.add(path)
            entries.append((path, _read_capped(path, max_bytes)))

    add(config_dir / "AGENTS.md")

    root = find_git_root(start) or start
    try:
        rel = start.relative_to(root)
        dirs = [root] + [root.joinpath(*rel.parts[: i + 1]) for i in range(len(rel.parts))]
    except ValueError:  # pragma: no cover - root is always an ancestor of start
        dirs = [start]
    for directory in dirs:
        for name in CONTEXT_FILENAMES:
            add(directory / name)

    return entries


def render(entries: list[tuple[Path, str]]) -> str:
    """Concatenate collected entries with a header per source path."""
    return "\n\n".join(f"## {path}\n\n{content.strip()}" for path, content in entries)
