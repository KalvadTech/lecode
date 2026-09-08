"""The single, fixed lecode theme.

There is exactly one theme, ``kalvad`` — dark and purple-dominant,
following Kalvad's brand. The former theming system (JSON theme files,
``.lecode/themes/`` overrides, the ``[colors]`` config table) was removed:
no selection, no overrides, no theme files.
"""

from __future__ import annotations

from dataclasses import dataclass

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


#: Picker-dropdown panel colors, part of the one fixed theme.
PICKER_MENU_BG = "#1c162b"
PICKER_MENU_SELECTED_BG = "#35264f"

#: The one and only theme: Kalvad purple on dark.
THEME = Theme(
    name="kalvad",
    accent="#a78bfa",  # violet — primary brand purple
    text="#ece7f7",  # lavender-white
    muted="#8a80a3",  # grayed purple
    error="#f0647e",
    warning="#e0a458",
    success="#6fd3a7",
    thinking="#6e6392",  # dim purple
    tool="#c4b5fd",  # light violet
    permission="#e879f9",  # fuchsia — stands out inside the purple family
)
