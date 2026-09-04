"""Tests for the single hardcoded theme (tui/themes.py)."""

from __future__ import annotations

from lecode.tui.themes import COLOR_KEYS, THEME


def test_theme_is_kalvad():
    assert THEME.name == "kalvad"


def test_theme_palette_is_locked():
    assert {key: getattr(THEME, key) for key in COLOR_KEYS} == {
        "accent": "#a78bfa",
        "text": "#ece7f7",
        "muted": "#8a80a3",
        "error": "#f0647e",
        "warning": "#e0a458",
        "success": "#6fd3a7",
        "thinking": "#6e6392",
        "tool": "#c4b5fd",
        "permission": "#e879f9",
    }
