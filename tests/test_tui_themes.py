"""Tests for theme loading, precedence, inheritance and config overrides."""

from __future__ import annotations

import json
import logging

import pytest

from lecode.config.models import Config
from lecode.tui.themes import COLOR_KEYS, list_themes, load_theme

EXPECTED_THEMES = {
    "default",
    "dark",
    "light",
    "monokai",
    "solarized-dark",
    "solarized-light",
    "dracula",
    "nord",
    "gruvbox",
    "catppuccin-mocha",
    "tokyo-night",
    "one-dark",
    "rose-pine",
    "everforest",
    "kanagawa",
    "ayu",
    "minimal",
}


@pytest.fixture
def global_dir(tmp_path, monkeypatch):
    d = tmp_path / "global"
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(d))
    return d


def _write_theme(path, name, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "colors": colors}), encoding="utf-8")


def _override_colors(**kwargs):
    base = {k: "#123456" for k in COLOR_KEYS}
    base.update(kwargs)
    return base


@pytest.mark.parametrize("name", sorted(EXPECTED_THEMES))
def test_all_bundled_themes_load(name, tmp_path):
    theme = load_theme(name, Config(), cwd=tmp_path)
    assert theme.name == name
    for key in COLOR_KEYS:
        value = getattr(theme, key)
        assert value.startswith("#") and len(value) == 7


def test_unknown_theme_falls_back_to_default_with_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="lecode.tui.themes"):
        theme = load_theme("no-such-theme", Config(), cwd=tmp_path)
    assert theme.name == "default"
    assert "no-such-theme" in caplog.text
    assert theme.accent == load_theme("default", Config(), cwd=tmp_path).accent


def test_global_overrides_embedded(global_dir, tmp_path):
    _write_theme(global_dir / "themes" / "default.json", "default", _override_colors())
    theme = load_theme("default", Config(), cwd=tmp_path)
    assert theme.accent == "#123456"


def test_project_overrides_global(global_dir, tmp_path):
    (tmp_path / ".git").mkdir()
    _write_theme(global_dir / "themes" / "default.json", "default", _override_colors())
    _write_theme(
        tmp_path / ".lecode" / "themes" / "default.json",
        "default",
        _override_colors(accent="#abcdef"),
    )
    theme = load_theme("default", Config(), cwd=tmp_path)
    assert theme.accent == "#abcdef"
    assert theme.text == "#123456"


def test_config_colors_override_last(global_dir, tmp_path):
    (tmp_path / ".git").mkdir()
    _write_theme(
        tmp_path / ".lecode" / "themes" / "default.json",
        "default",
        _override_colors(accent="#abcdef"),
    )
    config = Config(colors={"accent": "#000001"})
    theme = load_theme("default", config, cwd=tmp_path)
    assert theme.accent == "#000001"


def test_missing_color_keys_inherit_from_default(global_dir, tmp_path):
    (tmp_path / ".git").mkdir()
    _write_theme(tmp_path / ".lecode" / "themes" / "partial.json", "partial", {"accent": "#ff0000"})
    theme = load_theme("partial", Config(), cwd=tmp_path)
    default = load_theme("default", Config(), cwd=tmp_path)
    assert theme.name == "partial"
    assert theme.accent == "#ff0000"
    assert theme.text == default.text
    assert theme.error == default.error


def test_unknown_config_color_keys_ignored(tmp_path):
    config = Config(colors={"not-a-slot": "#ffffff"})
    theme = load_theme("default", config, cwd=tmp_path)
    assert theme.name == "default"


def test_list_themes_includes_all_bundled(tmp_path):
    assert set(list_themes(cwd=tmp_path)) >= EXPECTED_THEMES


def test_list_themes_merges_layers(global_dir, tmp_path):
    (tmp_path / ".git").mkdir()
    _write_theme(global_dir / "themes" / "mine.json", "mine", _override_colors())
    _write_theme(tmp_path / ".lecode" / "themes" / "proj.json", "proj", _override_colors())
    names = list_themes(cwd=tmp_path)
    assert "mine" in names
    assert "proj" in names
    assert "default" in names
    assert names == sorted(names)
