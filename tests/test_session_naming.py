"""Tests for session naming rules."""

from __future__ import annotations

import re

import pytest

from lecode.session import (
    SessionStore,
    auto_name,
    sanitize_title,
    unique_name,
    validate_name,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    return SessionStore()


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_empty_and_whitespace_rejected(name):
    assert validate_name(name) is not None
    assert validate_name(None) is not None


@pytest.mark.parametrize("name", ["a/b", "a\\b", ".hidden", "x" * 65, "bad\ttab"])
def test_invalid_names_rejected(name):
    assert validate_name(name) is not None


@pytest.mark.parametrize("name", ["demo", "My Session 2", "with spaces ok", "x" * 64, "résumé-✓"])
def test_valid_names_accepted(name):
    assert validate_name(name) is None


def test_error_messages_are_specific():
    assert "empty" in validate_name("")
    assert "64" in validate_name("x" * 65)
    assert "dot" in validate_name(".x")
    assert "separator" in validate_name("a/b")
    assert "control" in validate_name("a\x01b")


def test_unique_name_suffixes_duplicates(store):
    store.create("demo", cwd="/tmp")
    assert unique_name("demo", store) == "demo-2"
    store.create("demo-2", cwd="/tmp")
    assert unique_name("demo", store) == "demo-3"
    assert unique_name("fresh", store) == "fresh"


def test_unique_name_strips_and_validates(store):
    assert unique_name("  padded  ", store) == "padded"
    with pytest.raises(ValueError):
        unique_name("   ", store)


def test_unique_name_suffix_respects_length_cap(store):
    store.create("x" * 64, cwd="/tmp")
    candidate = unique_name("x" * 64, store)
    assert len(candidate) == 64
    assert candidate.endswith("-2")


def test_auto_name_format():
    assert re.fullmatch(r"session-\d{8}-\d{6}", auto_name())


def test_auto_name_collision_suffixes(store):
    store.create(auto_name(), cwd="/tmp")
    assert auto_name(store) == auto_name() + "-2"


def test_sanitize_title_keeps_plain_titles():
    assert sanitize_title("Fix login bug") == "Fix login bug"


def test_sanitize_title_takes_first_line_and_strips():
    assert sanitize_title("Fix login bug.\n\nlonger explanation here.") == "Fix login bug"
    assert sanitize_title('  "Session titles"  ') == "Session titles"


def test_sanitize_title_truncates_to_max_length():
    title = sanitize_title("A long and winding title " * 5)
    assert title is not None
    assert len(title) <= 64
    assert validate_name(title or "") is None


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", ".hidden", "bad/title", "tab\there"])
def test_sanitize_title_rejects_unusable_output(raw):
    assert sanitize_title(raw) is None
