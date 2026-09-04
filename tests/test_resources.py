"""Tests for prompt resource loading and override precedence."""

from __future__ import annotations

import pytest

from lecode.context.resources import list_available, load_text

EXPECTED_PERSONAS = {
    "default",
    "reviewer",
    "architect",
    "teacher",
    "pair",
    "security",
    "performance",
    "debugger",
    "refactorer",
    "documenter",
    "tester",
    "devops",
    "data",
    "frontend",
    "minimal",
    "concise",
}


@pytest.fixture
def global_dir(tmp_path, monkeypatch):
    d = tmp_path / "global"
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(d))
    return d


def test_load_embedded_resource(tmp_path):
    text = load_text("prompts", "minimal.md", cwd=tmp_path)
    assert "coding agent" in text


def test_global_overrides_embedded(global_dir, tmp_path):
    (global_dir / "prompts").mkdir(parents=True)
    (global_dir / "prompts" / "minimal.md").write_text("global override")
    assert load_text("prompts", "minimal.md", cwd=tmp_path) == "global override"


def test_project_overrides_global(global_dir, tmp_path):
    (global_dir / "prompts").mkdir(parents=True)
    (global_dir / "prompts" / "minimal.md").write_text("global override")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".lecode" / "prompts").mkdir(parents=True)
    (tmp_path / ".lecode" / "prompts" / "minimal.md").write_text("project override")
    assert load_text("prompts", "minimal.md", cwd=tmp_path) == "project override"


def test_missing_resource_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text("prompts", "nope.md", cwd=tmp_path)


def test_unknown_kind_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown resource kind"):
        load_text("bogus", "x", cwd=tmp_path)


def test_list_available_merges_layers(global_dir, tmp_path):
    (global_dir / "prompts").mkdir(parents=True)
    (global_dir / "prompts" / "my-prompt.md").write_text("{}")
    names = list_available("prompts", cwd=tmp_path)
    assert "minimal.md" in names
    assert "my-prompt.md" in names


def test_all_personas_load(tmp_path):
    for persona in EXPECTED_PERSONAS:
        text = load_text("prompts", f"personas/{persona}.md", cwd=tmp_path)
        assert len(text.strip()) > 0
    listed = set(list_available("prompts", cwd=tmp_path))
    assert {f"personas/{p}.md" for p in EXPECTED_PERSONAS} <= listed


def test_minimal_prompt_is_short(tmp_path):
    text = load_text("prompts", "minimal.md", cwd=tmp_path)
    assert len(text.split()) < 400
