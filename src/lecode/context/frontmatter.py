"""YAML frontmatter parsing for markdown resource files (skills, agents).

Frontmatter is a ``---``-delimited YAML mapping at the very top of the file.
:func:`split_frontmatter` raises :class:`ValueError` on any malformed input so
callers can warn-and-skip with a single ``except ValueError``.
"""

from __future__ import annotations

from typing import Any

import yaml


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``---``-delimited YAML frontmatter from the markdown body.

    Returns ``(frontmatter, body)``; no frontmarker marker → ``({}, text)``.
    Raises :class:`ValueError` for unterminated, invalid, or non-mapping YAML.
    """
    if not (text.startswith("---\n") or text.startswith("---\r\n")):
        return {}, text
    lines = text.splitlines(keepends=True)
    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        raise ValueError("unterminated frontmatter (missing closing '---')")
    raw = "".join(lines[1:closing])
    try:
        data = yaml.safe_load(raw) if raw.strip() else {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid frontmatter YAML: {e}") from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return data, "".join(lines[closing + 1 :])
