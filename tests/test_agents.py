"""Tests for custom user-defined agents (markdown + frontmatter)."""

from __future__ import annotations

import pytest

from lecode.context.agents import AgentRegistry, load_agents, parse_mentions


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Isolated global config dir + a tmp project dir (no real home access)."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    global_dir = tmp_path / "cfg" / "agents"
    project = tmp_path / "project"
    project.mkdir()
    return tmp_path, global_dir, project


def write_agent(root, name: str, frontmatter: str, body: str = "Agent prompt.") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")


def test_builtins_present_without_files(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    assert {"build", "plan", "explore"} <= set(registry.names())
    assert registry.get("build").mode == "primary"
    assert registry.get("plan").mode == "primary"
    assert registry.get("explore").mode == "subagent"


def test_builtin_overlays(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    assert registry.overlay_for("build") is None  # full access, no narrowing
    assert registry.overlay_for("plan").mode == "readonly"
    assert registry.overlay_for("explore").mode == "readonly"


def test_loading_global_and_project(dirs):
    _, global_dir, project = dirs
    write_agent(global_dir, "globalone", "description: from global")
    write_agent(project / ".lecode" / "agents", "projone", "description: from project")

    registry = load_agents(cwd=project)
    assert registry.get("globalone").description == "from global"
    assert registry.get("projone").description == "from project"


def test_project_wins_on_collision(dirs):
    _, global_dir, project = dirs
    write_agent(global_dir, "same", "description: global version")
    write_agent(project / ".lecode" / "agents", "same", "description: project version")

    registry = load_agents(cwd=project)
    assert registry.get("same").description == "project version"


def test_user_file_overrides_builtin(dirs):
    _, _, project = dirs
    write_agent(
        project / ".lecode" / "agents",
        "plan",
        "description: custom plan\nmode: primary",
        body="My own planner.",
    )
    registry = load_agents(cwd=project)
    plan = registry.get("plan")
    assert plan.description == "custom plan"
    assert plan.body == "My own planner."
    assert plan.overlay is None  # user file replaces the built-in wholesale
    assert plan.builtin is False


def test_defaults(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "plain", "description: minimal agent")
    agent = load_agents(cwd=project).get("plain")
    assert agent.mode == "all"
    assert agent.hidden is False
    assert agent.model is None
    assert agent.temperature is None
    assert agent.overlay is None
    assert agent.color is None


def test_full_frontmatter(dirs):
    _, _, project = dirs
    write_agent(
        project / ".lecode" / "agents",
        "fancy",
        "description: full\nmode: primary\nmodel: openai/gpt-5\ntemperature: 0.2\n"
        "hidden: false\ncolor: cyan",
    )
    agent = load_agents(cwd=project).get("fancy")
    assert agent.mode == "primary"
    assert agent.model == "openai/gpt-5"
    assert agent.temperature == pytest.approx(0.2)
    assert agent.color == "cyan"


def test_permission_overlay_mapping(dirs):
    _, _, project = dirs
    write_agent(
        project / ".lecode" / "agents",
        "guard",
        "description: guarded\npermission:\n  mode: readonly\n  denied_tools: [bash]\n"
        "  rules:\n    allow:\n      write: ['docs/*']",
    )
    overlay = load_agents(cwd=project).overlay_for("guard")
    assert overlay.mode == "readonly"
    assert overlay.denied_tools == ("bash",)
    rule = overlay.extra_rules.allow["write"][0]
    assert rule.pattern == "docs/*"


def test_invalid_permission_mode_skipped(dirs):
    _, _, project = dirs
    write_agent(
        project / ".lecode" / "agents",
        "bad",
        "description: bad\npermission:\n  mode: godmode",
    )
    registry = load_agents(cwd=project)
    assert registry.get("bad") is None
    assert any("godmode" in w for w in registry.warnings)


def test_permission_not_a_mapping_skipped(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "bad", "description: bad\npermission: yes")
    registry = load_agents(cwd=project)
    assert registry.get("bad") is None
    assert any("permission" in w for w in registry.warnings)


def test_missing_description_skipped(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "nodesc", "mode: primary")
    registry = load_agents(cwd=project)
    assert registry.get("nodesc") is None
    assert any("description" in w for w in registry.warnings)


def test_invalid_mode_skipped(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "badmode", "description: d\nmode: supreme")
    registry = load_agents(cwd=project)
    assert registry.get("badmode") is None
    assert any("invalid mode" in w for w in registry.warnings)


def test_invalid_temperature_skipped(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "badtemp", "description: d\ntemperature: hot")
    registry = load_agents(cwd=project)
    assert registry.get("badtemp") is None
    assert any("temperature" in w for w in registry.warnings)


def test_invalid_yaml_skipped(dirs):
    _, _, project = dirs
    root = project / ".lecode" / "agents"
    root.mkdir(parents=True)
    (root / "broken.md").write_text("---\nkey: [unclosed\n---\nbody\n", encoding="utf-8")
    registry = load_agents(cwd=project)
    assert registry.get("broken") is None
    assert any("YAML" in w for w in registry.warnings)


def test_primaries_ordering(dirs):
    _, _, project = dirs
    agents = project / ".lecode" / "agents"
    write_agent(agents, "zeta", "description: z\nmode: primary")
    write_agent(agents, "alpha", "description: a\nmode: primary")
    write_agent(agents, "sub", "description: s\nmode: subagent")

    primaries = load_agents(cwd=project).primaries()
    assert [a.name for a in primaries] == ["build", "plan", "alpha", "zeta"]


def test_primaries_exclude_hidden(dirs):
    _, _, project = dirs
    write_agent(
        project / ".lecode" / "agents", "secret", "description: h\nmode: primary\nhidden: true"
    )
    registry = load_agents(cwd=project)
    assert [a.name for a in registry.primaries()] == ["build", "plan"]
    assert "secret" not in [a.name for a in registry.visible()]
    assert registry.get("secret") is not None  # still loadable by name


def test_subagents_listing(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "helper", "description: h\nmode: all")
    write_agent(project / ".lecode" / "agents", "sub", "description: s\nmode: subagent")
    registry = load_agents(cwd=project)
    assert [a.name for a in registry.subagents()] == ["explore", "helper", "sub"]


def test_cycle_wraps_around(dirs):
    _, _, project = dirs
    write_agent(project / ".lecode" / "agents", "extra", "description: e\nmode: primary")
    registry = load_agents(cwd=project)
    assert registry.cycle("build") == "plan"
    assert registry.cycle("plan") == "extra"
    assert registry.cycle("extra") == "build"  # wraps


def test_cycle_unknown_current_returns_first(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    assert registry.cycle("nonexistent") == "build"


def test_parse_mentions_known_and_cleaned(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    names, cleaned = parse_mentions("@explore find the auth module", registry)
    assert names == ["explore"]
    assert cleaned == "find the auth module"


def test_parse_mentions_unknown_left_untouched(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    names, cleaned = parse_mentions("@nobody do something", registry)
    assert names == []
    assert cleaned == "@nobody do something"


def test_parse_mentions_longest_prefix_wins(dirs):
    _, _, project = dirs
    agents = project / ".lecode" / "agents"
    write_agent(agents, "plan", "description: short")
    write_agent(agents, "planner", "description: long")
    registry = load_agents(cwd=project)
    names, cleaned = parse_mentions("@planner sketch it out", registry)
    assert names == ["planner"]
    assert cleaned == "sketch it out"


def test_parse_mentions_trailing_punctuation(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    names, cleaned = parse_mentions("@plan, then @explore the repo", registry)
    assert names == ["plan", "explore"]
    assert cleaned == "then the repo"


def test_parse_mentions_name_char_boundary(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    # "planx" is not an agent; "@planx" must not match "plan"
    names, cleaned = parse_mentions("@planx go", registry)
    assert names == []
    assert cleaned == "@planx go"


def test_empty_dirs_fail_open(dirs):
    _, _, project = dirs
    registry = load_agents(cwd=project)
    assert registry.warnings == []
    assert isinstance(registry, AgentRegistry)
