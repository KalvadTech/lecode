"""Loading of embedded static data with user overrides.

The ``prompts`` kind ships inside the package under ``lecode/data/`` and is
read via ``importlib.resources``. Prompts can be overridden or extended in
the global config dir (``~/.config/lecode/prompts/``) or the project dir
(``.lecode/prompts/``). Precedence: embedded < global < project.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

KINDS = ("prompts",)

#: Directory inside a project that holds lecode-local overrides.
PROJECT_DIR_NAME = ".lecode"


def _embedded_root(kind: str) -> Traversable:
    return resources.files("lecode.data").joinpath(kind)


def _global_root(kind: str) -> Path:
    from lecode.config.loader import config_dir

    return config_dir() / kind


def _project_root(kind: str, cwd: Path | None = None) -> Path | None:
    """Nearest ``.lecode/<kind>/`` from ``cwd`` up to the git root."""
    from lecode.context.agents_md import find_git_root

    start = (cwd or Path.cwd()).resolve()
    stop = find_git_root(start) or start
    current = start
    while True:
        candidate = current / PROJECT_DIR_NAME / kind
        if candidate.is_dir():
            return candidate
        if current == stop:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown resource kind '{kind}'; expected one of {KINDS}")


def load_text(kind: str, name: str, cwd: Path | None = None) -> str:
    """Load a resource by relative name, highest-precedence layer first.

    ``name`` is a path relative to the kind dir, extension included
    (e.g. ``load_text("prompts", "personas/reviewer.md")``).
    """
    _check_kind(kind)
    for root in (_project_root(kind, cwd), _global_root(kind)):
        if root is not None:
            candidate = root / name
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
    embedded = _embedded_root(kind)
    for part in name.split("/"):
        embedded = embedded.joinpath(part)
    if embedded.is_file():
        return embedded.read_text("utf-8")
    raise FileNotFoundError(f"resource not found: {kind}/{name}")


def _walk_embedded(root: Traversable, prefix: str, out: set[str]) -> None:
    for child in root.iterdir():
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            _walk_embedded(child, rel + "/", out)
        else:
            out.add(rel)


def _walk_fs(root: Path, prefix: str, out: set[str]) -> None:
    for child in sorted(root.iterdir()):
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            _walk_fs(child, rel + "/", out)
        elif child.is_file():
            out.add(rel)


def list_available(kind: str, cwd: Path | None = None) -> list[str]:
    """List resource names across all layers (later layers shadow earlier)."""
    _check_kind(kind)
    names: set[str] = set()
    _walk_embedded(_embedded_root(kind), "", names)
    for root in (_global_root(kind), _project_root(kind, cwd)):
        if root is not None and root.is_dir():
            _walk_fs(root, "", names)
    return sorted(names)
