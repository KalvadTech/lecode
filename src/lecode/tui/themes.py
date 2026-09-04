"""Theme loading and resolution for the TUI.

A theme is a JSON file with a ``name`` and a ``colors`` mapping over the
nine semantic color slots (see :data:`COLOR_KEYS`). Resolution follows the
standard resource precedence (embedded ``data/themes/`` < global config dir
< project ``.lecode/themes/``) via :mod:`lecode.context.resources`, then the
``[colors]`` config table overrides individual slots last. Missing color
keys inherit from the ``default`` theme; an unknown theme name falls back
to ``default`` with a warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from lecode.config.models import Config
from lecode.context.resources import list_available, load_text

log = logging.getLogger(__name__)

#: The nine semantic color slots every theme defines.
COLOR_KEYS = (
    "accent",
    "text",
    "muted",
    "error",
    "warning",
    "success",
    "thinking",
    "tool",
    "permission",
)

#: Theme used as fallback for unknown names and for missing color keys.
DEFAULT_THEME = "default"


@dataclass(frozen=True)
class Theme:
    """A resolved theme: a name plus one hex color per semantic slot."""

    name: str
    accent: str
    text: str
    muted: str
    error: str
    warning: str
    success: str
    thinking: str
    tool: str
    permission: str


def _load_colors(name: str, cwd: Path | None = None) -> dict[str, str] | None:
    """Load the color mapping for a theme, or ``None`` if it does not exist."""
    try:
        raw = load_text("themes", f"{name}.json", cwd=cwd)
    except FileNotFoundError:
        return None
    data = json.loads(raw)
    colors = data.get("colors", {})
    return {k: v for k, v in colors.items() if k in COLOR_KEYS}


def _embedded_default_colors() -> dict[str, str]:
    """The bundled default theme colors — the base for missing-key inheritance."""
    raw = resources.files("lecode.data").joinpath("themes/default.json").read_text("utf-8")
    return dict(json.loads(raw)["colors"])


def load_theme(name: str, config: Config, cwd: Path | None = None) -> Theme:
    """Resolve ``name`` into a :class:`Theme` with config overrides applied."""
    colors = _load_colors(name, cwd)
    if colors is None:
        if name != DEFAULT_THEME:
            log.warning("unknown theme %r; falling back to %r", name, DEFAULT_THEME)
        name = DEFAULT_THEME
        colors = _load_colors(DEFAULT_THEME, cwd) or {}
    colors = {**_embedded_default_colors(), **colors}
    colors.update({k: v for k, v in config.colors.items() if k in COLOR_KEYS})
    return Theme(name=name, **{k: colors[k] for k in COLOR_KEYS})


def list_themes(cwd: Path | None = None) -> list[str]:
    """List theme names across all layers (embedded, global, project)."""
    names = list_available("themes", cwd=cwd)
    return sorted(n.removesuffix(".json") for n in names if n.endswith(".json"))
