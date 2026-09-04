"""Tests for the skills system (SKILL.md pack discovery and parsing)."""

from __future__ import annotations

import pytest

from lecode.context.skills import (
    SkillRegistry,
    load_skills,
    project_skills_dir,
    skill_commands,
)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Isolated global skills dir + a tmp project dir (no real home access)."""
    global_dir = tmp_path / "global-skills"
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(global_dir))
    project = tmp_path / "project"
    project.mkdir()
    return tmp_path, global_dir, project


def write_skill(root, dirname, frontmatter: str, body: str = "Do the thing.") -> None:
    pack = root / dirname
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")


def test_empty_dirs_fail_open(dirs):
    _, _, project = dirs
    registry = load_skills(cwd=project)
    assert registry.list() == []
    assert registry.warnings == []
    assert registry.render_listing() == ""


def test_discovery_project_and_global(dirs):
    _, global_dir, project = dirs
    write_skill(global_dir, "pdf", "description: Extract text from PDFs")
    write_skill(project / ".agents" / "skills", "review", "description: Code review checklist")

    registry = load_skills(cwd=project)
    assert [s.name for s in registry.list()] == ["pdf", "review"]


def test_project_wins_on_collision(dirs):
    _, global_dir, project = dirs
    write_skill(global_dir, "pdf", "description: global pdf", body="global body")
    write_skill(project / ".agents" / "skills", "pdf", "description: project pdf", body="proj body")

    registry = load_skills(cwd=project)
    skill = registry.get("pdf")
    assert len(registry) == 1
    assert skill.description == "project pdf"
    assert registry.render_skill("pdf") == "proj body"


def test_name_defaults_to_directory(dirs):
    _, _, project = dirs
    write_skill(project / ".agents" / "skills", "my-skill", "description: named by dir")
    registry = load_skills(cwd=project)
    assert registry.get("my-skill") is not None


def test_frontmatter_name_overrides_directory(dirs):
    _, _, project = dirs
    write_skill(project / ".agents" / "skills", "dirname", "name: fancy\ndescription: d")
    registry = load_skills(cwd=project)
    assert registry.get("fancy") is not None
    assert registry.get("dirname") is None


def test_missing_description_skipped_with_warning(dirs):
    _, _, project = dirs
    write_skill(project / ".agents" / "skills", "nodesc", "register_cmd: true")
    registry = load_skills(cwd=project)
    assert registry.list() == []
    assert any("description" in w for w in registry.warnings)


def test_no_frontmatter_skipped(dirs):
    _, _, project = dirs
    pack = project / ".agents" / "skills" / "plain"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text("just markdown, no frontmatter\n", encoding="utf-8")
    registry = load_skills(cwd=project)
    assert registry.list() == []
    assert len(registry.warnings) == 1


def test_invalid_yaml_skipped_with_warning(dirs):
    _, _, project = dirs
    pack = project / ".agents" / "skills" / "broken"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text("---\nkey: [unclosed\n---\nbody\n", encoding="utf-8")
    registry = load_skills(cwd=project)
    assert registry.list() == []
    assert any("YAML" in w for w in registry.warnings)


def test_unterminated_frontmatter_skipped(dirs):
    _, _, project = dirs
    pack = project / ".agents" / "skills" / "open"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text("---\ndescription: never closed\n", encoding="utf-8")
    registry = load_skills(cwd=project)
    assert registry.list() == []
    assert any("unterminated" in w for w in registry.warnings)


def test_render_listing_format(dirs):
    _, _, project = dirs
    write_skill(project / ".agents" / "skills", "a-skill", "description: Alpha")
    write_skill(project / ".agents" / "skills", "b-skill", "description: Beta")
    listing = load_skills(cwd=project).render_listing()
    assert listing.startswith("## Available skills")
    assert "- `a-skill` — Alpha" in listing
    assert "- `b-skill` — Beta" in listing


def test_render_skill_unknown_returns_none(dirs):
    _, _, project = dirs
    registry = load_skills(cwd=project)
    assert registry.render_skill("nope") is None


def test_skill_commands_only_register_cmd(dirs):
    _, _, project = dirs
    skills = project / ".agents" / "skills"
    write_skill(skills, "plain", "description: not registered")
    write_skill(skills, "cmd", "description: registered\nregister_cmd: true", body="CMD BODY")
    registry = load_skills(cwd=project)

    commands = skill_commands(registry)
    assert list(commands) == ["cmd"]
    assert commands["cmd"] == {"name": "cmd", "description": "registered", "body": "CMD BODY"}


def test_skill_commands_cmd_info_as_help(dirs):
    _, _, project = dirs
    write_skill(
        project / ".agents" / "skills",
        "cmd",
        'description: desc\nregister_cmd: true\ncmd_info: "usage: /cmd <arg>"',
    )
    commands = skill_commands(load_skills(cwd=project))
    assert commands["cmd"]["description"] == "usage: /cmd <arg>"


def test_skill_commands_invalid_slug_skipped(dirs):
    _, _, project = dirs
    write_skill(
        project / ".agents" / "skills",
        "dir",
        "name: Not A Slug!\ndescription: d\nregister_cmd: true",
    )
    registry = load_skills(cwd=project)
    commands = skill_commands(registry)
    assert commands == {}
    assert any("not a valid command slug" in w for w in registry.warnings)


def test_project_skills_dir_found_from_subdirectory(dirs):
    tmp_path, _, _ = dirs
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".agents" / "skills").mkdir(parents=True)
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    assert project_skills_dir(sub) == repo / ".agents" / "skills"


def test_non_skill_entries_ignored(dirs):
    _, _, project = dirs
    skills = project / ".agents" / "skills"
    skills.mkdir(parents=True)
    (skills / "README.md").write_text("not a pack", encoding="utf-8")
    (skills / "empty-dir").mkdir()
    registry = load_skills(cwd=project)
    assert registry.list() == []
    assert registry.warnings == []


def test_empty_registry_helpers():
    registry = SkillRegistry()
    assert len(registry) == 0
    assert registry.get("x") is None
    assert skill_commands(registry) == {}
