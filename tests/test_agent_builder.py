"""Tests for the runtime assembly helper."""

from __future__ import annotations

import pytest

from lecode.agent.builder import build_runtime
from lecode.agent.tools import core_tools
from lecode.config.models import Config
from lecode.permission import Decision


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    return tmp_path


def test_default_runtime(cwd):
    runtime = build_runtime(Config(), cwd)
    expected = [
        *(t.name for t in core_tools()),
        "advisor",
        "lsp_diagnostics",
        "memory_edit",
        "memory_read",
        "memory_search",
        "memory_write",
        "task",
    ]
    assert runtime.registry.names() == sorted(expected)
    assert runtime.ctx.auto_approve is False
    assert runtime.system_prompt.startswith("You are lecode")


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
    assert checker.check("write", {"file_path": "x.txt"}).decision == Decision.ASK
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
