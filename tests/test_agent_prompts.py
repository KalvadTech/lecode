"""Tests for system prompt assembly."""

from __future__ import annotations

import pytest

from lecode.agent.prompts import build_system_prompt
from lecode.config.models import Config


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tmp git repo; the global config dir is isolated from the real one."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_minimal_default(repo):
    prompt = build_system_prompt(Config(), repo)
    assert prompt.startswith("You are lecode")
    assert "rich system prompt" not in prompt


def test_agents_md_included(repo):
    (repo / "AGENTS.md").write_text("project law: always test", encoding="utf-8")
    prompt = build_system_prompt(Config(), repo)
    assert "project law: always test" in prompt
    assert str(repo / "AGENTS.md") in prompt


def test_agents_md_walks_git_root_to_cwd(repo):
    (repo / "AGENTS.md").write_text("root rules", encoding="utf-8")
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("pkg rules", encoding="utf-8")
    prompt = build_system_prompt(Config(), sub)
    assert prompt.index("root rules") < prompt.index("pkg rules")


def test_custom_replaces_base(repo):
    config = Config()
    config.llm.system_prompt.custom = "CUSTOM PROMPT"
    (repo / "AGENTS.md").write_text("still appended", encoding="utf-8")
    prompt = build_system_prompt(config, repo)
    assert prompt.startswith("CUSTOM PROMPT")
    assert "You are lecode" not in prompt
    assert "still appended" in prompt  # context walk survives a custom base


def test_rich_with_persona(repo):
    config = Config()
    config.llm.system_prompt.style = "rich"
    config.llm.system_prompt.persona = "reviewer"
    prompt = build_system_prompt(config, repo)
    assert "rich system prompt" in prompt
    assert "review" in prompt.lower()  # persona snippet appended


def test_persona_ignored_in_minimal_style(repo):
    config = Config()
    config.llm.system_prompt.persona = "reviewer"
    prompt = build_system_prompt(config, repo)
    assert "rich system prompt" not in prompt


def test_memory_and_extra_seams(repo):
    prompt = build_system_prompt(Config(), repo, memory_text="MEMORY", extra="SKILLS")
    assert prompt.endswith("MEMORY\n\nSKILLS")


def test_no_context_files_omits_section(repo):
    prompt = build_system_prompt(Config(), repo)
    assert "##" not in prompt
