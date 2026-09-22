"""Tests for the runtime assembly helper."""

from __future__ import annotations

import pytest
from tests.test_worktree import git_sync, make_repo_sync

from lecode.agent.builder import build_runtime, refresh_system_prompt
from lecode.agent.tools import core_tools
from lecode.config.models import Config
from lecode.memory import MemoryStore, memory_root
from lecode.permission import Decision


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_runtime(cwd):
    runtime = build_runtime(Config(), cwd)
    expected = [
        *(t.name for t in core_tools()),
        "lsp_diagnostics",
        "memory_correct",
        "memory_edit",
        "memory_forget",
        "memory_list",
        "memory_read",
        "memory_recall",
        "memory_search",
        "memory_write",
        "task",
    ]
    assert runtime.registry.names() == sorted(expected)
    assert runtime.ctx.auto_approve is False
    assert runtime.system_prompt.startswith("You are lecode")
    assert "- general: General-purpose coding subagent" in runtime.system_prompt
    assert "- explore: Fast read-only" in runtime.system_prompt


def test_subagent_discovery_respects_overrides(cwd):
    from tests.test_agents import write_agent

    agents = cwd / ".lecode" / "agents"
    write_agent(agents, "general", "description: Custom primary\nmode: primary")
    write_agent(agents, "secret", "description: Hidden\nmode: subagent\nhidden: true")
    write_agent(agents, "helper", "description: Custom helper\nmode: subagent")
    runtime = build_runtime(Config(), cwd)
    assert "- general:" not in runtime.system_prompt
    assert "- secret:" not in runtime.system_prompt
    assert "- helper: Custom helper" in runtime.system_prompt


def test_runtime_rebinding_releases_lazy_session_fact_connection(cwd):
    from pathlib import Path

    from lecode.memory.facts import FactStore
    from lecode.session.storage import SessionStore

    path = memory_root(cwd) / "facts.sqlite3"
    facts = FactStore(path)
    facts.add("legacy", source_id="old", source_seq=1)
    facts.close()
    sessions = SessionStore(cwd / "cfg")
    session = sessions.create("existing", cwd)
    sessions.append_message(session, {"role": "user", "content": "evidence"})
    sessions.source_snapshot(session.id, 1, 1, project_root=cwd)
    runtime = build_runtime(Config(), cwd, store=sessions, session=session)
    runtime.close()
    assert not Path(str(path) + "-wal").exists()


def test_read_only_mode_denies_writes(cwd):
    runtime = build_runtime(Config(), cwd, mode="readonly")
    checker = runtime.ctx.permission_checker
    assert checker.check("write", {"file_path": "x.txt"}).decision == Decision.DENY
    assert checker.check("bash", {"command": "ls"}).decision == Decision.DENY
    assert checker.check("read", {"file_path": "x.txt"}).decision == Decision.ALLOW


def test_auto_approve(cwd):
    runtime = build_runtime(Config(), cwd, auto_approve=True)
    assert runtime.ctx.auto_approve is True


def test_allowed_tools_filter(cwd):
    runtime = build_runtime(Config(), cwd, allowed_tools=["read", "grep"])
    assert runtime.registry.names() == ["grep", "read"]


def test_unknown_allowed_tools_are_ignored(cwd):
    runtime = build_runtime(Config(), cwd, allowed_tools=["read", "nonexistent"])
    assert runtime.registry.names() == ["read"]


def test_session_grants_loaded(cwd, tmp_path):
    from lecode.session.storage import SessionStore

    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("grants", cwd)
    store.grant_permission(session, "bash", "git status")
    runtime = build_runtime(Config(), cwd, session=session, store=store)
    check = runtime.ctx.permission_checker.check("bash", {"command": "git status"})
    assert check.decision == Decision.ALLOW
    assert "session grant" in check.reason


def test_session_runtime_installs_workers(cwd, tmp_path):
    from lecode.session.storage import SessionStore

    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("workers", cwd)
    runtime = build_runtime(Config(), cwd, session=session, store=store)
    assert "workers" in runtime.registry.names()
    assert runtime.ctx.extras["workers"].session is session


def test_workers_schema_exposes_reviewed_integration_controls(cwd, tmp_path):
    from lecode.session.storage import SessionStore

    store = SessionStore(config_dir=tmp_path / "cfg")
    session = store.create("workers", cwd)
    runtime = build_runtime(Config(), cwd, session=session, store=store)
    actions = runtime.registry.get("workers").parameters["properties"]["action"]["enum"]
    assert {"review", "integrate", "cleanup", "recover"} <= set(actions)


def test_agent_name_applies_overlay(cwd):
    runtime = build_runtime(Config(), cwd, agent_name="plan")
    checker = runtime.ctx.permission_checker
    assert checker.check("write", {"file_path": "x.txt"}).decision == Decision.DENY
    assert checker.check("bash", {"command": "ls"}).decision == Decision.DENY
    assert checker.check("read", {"file_path": "x.txt"}).decision == Decision.ALLOW


def test_agent_name_prepends_body(cwd):
    runtime = build_runtime(Config(), cwd, agent_name="plan")
    assert "planning mode" in runtime.system_prompt
    assert runtime.system_prompt.startswith("You are lecode")  # base still first


def test_unknown_agent_name_ignored(cwd):
    runtime = build_runtime(Config(), cwd, agent_name="nope")
    checker = runtime.ctx.permission_checker
    assert checker.check("write", {"file_path": "x.txt"}).decision == Decision.ALLOW
    assert "planning mode" not in runtime.system_prompt


def test_agent_name_none_disables(cwd):
    runtime = build_runtime(Config(), cwd, agent_name=None)
    assert "planning mode" not in runtime.system_prompt


def test_skills_listing_in_system_prompt(cwd):
    pack = cwd / ".agents" / "skills" / "review"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text(
        "---\ndescription: Code review checklist\n---\nReview carefully.\n", encoding="utf-8"
    )
    runtime = build_runtime(Config(), cwd)
    assert "## Available skills" in runtime.system_prompt
    assert "- `review` — Code review checklist" in runtime.system_prompt
    assert runtime.skills.get("review") is not None


def test_agent_body_before_skills_listing(cwd):
    pack = cwd / ".agents" / "skills" / "s"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text("---\ndescription: a skill\n---\nbody\n", encoding="utf-8")
    runtime = build_runtime(Config(), cwd, agent_name="plan")
    prompt = runtime.system_prompt
    assert prompt.index("planning mode") < prompt.index("## Available skills")


def test_runtime_exposes_registries(cwd):
    runtime = build_runtime(Config(), cwd)
    assert runtime.agents.get("build") is not None
    assert len(runtime.skills) == 0


def test_refresh_system_prompt_rereads_context_and_memory(cwd):
    runtime = build_runtime(Config(), cwd)
    (cwd / "AGENTS.md").write_text("# Project rules\n\nAlways run ruff.\n", encoding="utf-8")
    runtime.ctx.extras["memory"].write_long_term("Remember the alpaca.")

    refresh_system_prompt(runtime)

    assert "Always run ruff." in runtime.system_prompt
    assert "Remember the alpaca." in runtime.system_prompt


def test_refresh_system_prompt_keeps_agent_body_and_skills(cwd):
    pack = cwd / ".agents" / "skills" / "review"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text(
        "---\ndescription: Code review checklist\n---\nReview carefully.\n", encoding="utf-8"
    )
    runtime = build_runtime(Config(), cwd, agent_name="plan")

    refresh_system_prompt(runtime)

    assert runtime.system_prompt.startswith("You are lecode")
    assert "planning mode" in runtime.system_prompt
    assert "## Available skills" in runtime.system_prompt


def test_runtime_shares_memory_and_facts_but_keeps_checkout_context(cwd):
    from lecode.memory.facts import FactStore

    repo = make_repo_sync(cwd / "repo")
    nested = repo / "nested"
    nested.mkdir()
    linked = cwd / "linked"
    git_sync(repo, "worktree", "add", "-b", "feature", str(linked))
    (repo / "AGENTS.md").write_text("main checkout instructions")
    (linked / "AGENTS.md").write_text("linked checkout instructions")
    MemoryStore(memory_root(nested)).write_long_term("migrated nested notes")
    runtime = build_runtime(Config(), nested)
    child_checkout = build_runtime(Config(), linked)

    assert runtime.ctx.project_root == child_checkout.ctx.project_root == repo
    assert runtime.ctx.cwd == nested
    assert child_checkout.ctx.cwd == linked
    assert runtime.ctx.extras["memory"].root == memory_root(repo)
    assert "migrated nested notes" in child_checkout.system_prompt
    assert "linked checkout instructions" in child_checkout.system_prompt
    assert "main checkout instructions" not in child_checkout.system_prompt
    facts = runtime.ctx.extras["facts"]
    assert isinstance(facts, FactStore)
    assert not (memory_root(repo) / "facts.sqlite3").exists()
    fact = facts.add("shared fact", source_id="s", source_seq=1)
    assert child_checkout.ctx.extras["facts"].get(fact.id) == fact
    runtime.ctx.extras["memory"].write_long_term("fresh project notes")
    refresh_system_prompt(child_checkout)
    assert "fresh project notes" in child_checkout.system_prompt
    facts.close()
    child_checkout.ctx.extras["facts"].close()


def test_runtime_explicit_project_and_scope_override(cwd):
    project = cwd / "project"
    checkout = cwd / "checkout"
    checkout.mkdir()
    MemoryStore(memory_root(project)).write_long_term("explicit project")
    runtime = build_runtime(Config(), checkout, project_root=project, scope="lecode/feature")
    assert runtime.ctx.project_root == project
    assert runtime.ctx.scope == "lecode/feature"
    assert runtime.ctx.extras["memory"].root == memory_root(project)
    assert "explicit project" in runtime.system_prompt


def test_disabled_memory_never_creates_store_or_migrates(cwd):
    project, checkout = cwd / "project", cwd / "checkout"
    checkout.mkdir()
    MemoryStore(memory_root(checkout)).write_long_term("legacy")
    config = Config()
    config.memory.enabled = False
    runtime = build_runtime(config, checkout, project_root=project)
    refresh_system_prompt(runtime)
    assert "memory" not in runtime.ctx.extras
    assert "facts" not in runtime.ctx.extras
    assert not any(name.startswith("memory_") for name in runtime.registry.names())
    assert "legacy" not in runtime.system_prompt
    assert not memory_root(project).exists()
    assert not list((cwd / "cfg").rglob("*.sqlite3"))


def test_registered_recall_is_read_class(cwd):
    runtime = build_runtime(Config(), cwd, mode="readonly")
    assert runtime.ctx.permission_checker.check("memory_recall", {}).decision == Decision.ALLOW
    assert runtime.registry.get("memory_recall") is not None
