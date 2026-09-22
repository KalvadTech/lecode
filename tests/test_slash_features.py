"""Tests for the phase-11 handlers: init/tutor/review/notifications/prompt/
compress/editsys."""

from __future__ import annotations

import shlex
import shutil

import pytest
from tests.test_tui_app import make_app
from tests.test_worker_controls import commit, review_heads
from tests.test_worktree import make_repo

from lecode.config.models import PermissionRule, PermissionRuleSet
from lecode.context.agents import AgentDefinition, AgentRegistry
from lecode.permission.checker import AgentOverlay, Deny


async def test_agent_focus_command_targets_persistent_worker(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "done"}])
    manager = app.worker_manager
    assert manager is not None
    worker = await manager.start(
        app.runtime.ctx,
        agent="explore",
        prompt="inspect",
        origin="human",
        background=True,
    )
    await manager.wait(worker.id)
    await app.handle_command("/agent 1 focus")
    assert app._focused_worker_id == worker.id
    assert "composer focused on @explore" in out.getvalue()
    await manager.shutdown()


async def test_agent_submit_wakes_root_once(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "done"}, {"text": "integrated"}])
    manager = app.worker_manager
    assert manager is not None
    worker = await manager.start(
        app.runtime.ctx,
        agent="explore",
        prompt="inspect",
        origin="human",
        background=True,
    )
    await manager.wait(worker.id)
    await app.handle_command("/agent 1 submit")
    assert app._worker_wake_task is not None
    await app._worker_wake_task
    assert app._turn_task is not None
    await app._turn_task
    assert any(
        message["content"] == "Worker updates are available."
        for message in provider.requests[-1]["messages"]
    )
    wake_task = app._worker_wake_task
    await app.handle_command("/agent 1 submit")
    assert app._worker_wake_task is wake_task
    assert len(provider.requests) == 2
    await manager.shutdown()


@pytest.mark.parametrize(
    "action", ["send hello", "stop", "resume hello", "submit", "cleanup", "recover"]
)
async def test_agent_mutations_use_workers_permission_gate(tmp_path, monkeypatch, action):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "done"}])
    manager = app.worker_manager
    worker = await manager.start(app.runtime.ctx, agent="explore", prompt="Inspect", origin="human")
    await manager.wait(worker.id)
    app.runtime.ctx.permission_checker = app.runtime.ctx.permission_checker.for_agent(
        AgentOverlay(denied_tools=("workers",))
    )
    try:
        await app.handle_command(f"/agent 1 {action}")
        assert "denied" in out.getvalue()
        assert worker.state == "completed"
        assert manager.pending(worker.id) == []
    finally:
        await manager.shutdown()


async def test_agent_review_integrate_cleanup_and_retained_transcript(tmp_path, monkeypatch):
    await make_repo(tmp_path)
    (tmp_path / ".git/info/exclude").write_text("/cfg/\n/global-skills/\n")
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "Implemented"}])
    app.runtime.ctx.extras["agents"] = AgentRegistry(
        {"writer": AgentDefinition(name="writer", description="Write", body="", mode="subagent")}
    )
    manager = app.worker_manager
    worker = await manager.start(
        app.runtime.ctx, agent="writer", prompt="Implement", origin="human"
    )
    await manager.wait(worker.id)
    head = commit(worker.cwd)
    reviewed = await review_heads(app.runtime, worker)
    manager.confirm = lambda _: True
    try:
        await app.handle_command("/agent 1 inspect")
        assert head in out.getvalue() and "+worker change" in out.getvalue()
        await app.handle_command(f"/agent 1 integrate {head}")
        assert "integrate WORKER_HASH PARENT_HASH" in out.getvalue()
        assert not (tmp_path / "change.txt").exists()
        await app.handle_command(f"/agent 1 integrate {head} {reviewed['reviewed_parent_head']}")
        assert (tmp_path / "change.txt").exists(), out.getvalue()
        assert (tmp_path / "change.txt").read_text() == "worker change\n"
        await app.handle_command("/agent 1 cleanup")
        assert not worker.cwd.exists()
        await app.handle_command("/agent 1")
        assert app.detail_run_id == worker.id
        assert "Implemented" in str(app._roster_text())
    finally:
        await manager.shutdown()


async def test_agent_recover_warns_and_requires_human_confirmation(tmp_path, monkeypatch):
    await make_repo(tmp_path)
    (tmp_path / ".git/info/exclude").write_text("/cfg/\n/global-skills/\n")
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "Implemented"}])
    app.runtime.ctx.extras["agents"] = AgentRegistry(
        {"writer": AgentDefinition(name="writer", description="Write", body="", mode="subagent")}
    )
    manager = app.worker_manager
    worker = await manager.start(
        app.runtime.ctx, agent="writer", prompt="Implement", origin="human"
    )
    await manager.wait(worker.id)
    shutil.rmtree(worker.cwd)
    questions = []

    def confirm(question):
        questions.append(question)
        return len(questions) > 1

    manager.confirm = confirm
    try:
        await app.handle_command("/agent 1 recover")
        assert not worker.cwd.exists() and "declined" in out.getvalue()
        await app.handle_command("/agent 1 recover")
        assert worker.cwd.is_dir()
        assert worker.id in questions[0] and "unrecoverable" in questions[0]
    finally:
        await manager.shutdown()


@pytest.mark.parametrize(
    "restriction",
    [
        "ancestor-deny",
        "ancestor-ask",
        "root-deny-workers",
        "root-ask-workers",
        "root-deny-bash",
        "root-ask-bash",
        "root-mode",
        "both",
        "allowed",
    ],
)
async def test_nested_agent_integration_keeps_live_ancestor_and_root_permissions(
    tmp_path, monkeypatch, restriction
):
    await make_repo(tmp_path)
    (tmp_path / ".git/info/exclude").write_text("/cfg/\n/global-skills/\n")
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "done"}] * 2)
    app.runtime.ctx.extras["agents"] = AgentRegistry(
        {"writer": AgentDefinition(name="writer", description="Write", body="", mode="subagent")}
    )
    base = app.runtime.ctx.permission_checker
    approvals = []

    async def reject(name, args, reason, **kwargs):
        approvals.append((name, reason))
        return Deny()

    app.runtime.ctx.approval_callback = reject
    if restriction in {"ancestor-deny", "both"}:
        app.runtime.ctx.permission_checker = base.for_agent(AgentOverlay(denied_tools=("bash",)))
    elif restriction == "ancestor-ask":
        app.runtime.ctx.permission_checker = base.for_agent(
            AgentOverlay(extra_rules=PermissionRuleSet(ask={"bash": [PermissionRule(pattern="*")]}))
        )
    manager = app.worker_manager
    parent = await manager.start(app.runtime.ctx, agent="writer", prompt="Parent", origin="human")
    await manager.wait(parent.id)
    nested = manager._runtime(parent)
    child = await manager.start(nested.ctx, agent="writer", prompt="Child")
    await manager.wait(child.id)
    head = commit(child.cwd)
    reviewed = await review_heads(nested, child)
    assert parent.depth == 1 and child.depth == 2
    # Persisted child grants cannot override an ancestor's Deny or Ask.
    for worker in (parent, child):
        manager._runtime(worker).ctx.session_perms.grant("bash", "*")
        manager.store.grant_permission(worker.session, "bash", "*")
    marker = tmp_path.parent / f"{tmp_path.name}-validation"
    app.config.worktree.validation = [f"printf validated > {shlex.quote(str(marker))}"]
    # Simulate a root agent switch while retaining the real cached supervisor.
    app.runtime.ctx.permission_checker = base.for_agent(AgentOverlay(denied_tools=("write",)))
    if restriction.startswith("root-deny-") or restriction == "both":
        tool = "workers" if restriction == "both" else restriction.removeprefix("root-deny-")
        app.runtime.ctx.permission_checker = base.for_agent(AgentOverlay(denied_tools=(tool,)))
    elif restriction.startswith("root-ask-"):
        tool = restriction.removeprefix("root-ask-")
        app.runtime.ctx.permission_checker = base.for_agent(
            AgentOverlay(extra_rules=PermissionRuleSet(ask={tool: [PermissionRule(pattern="*")]}))
        )
    elif restriction == "root-mode":
        app.set_permission_mode("readonly")
    try:
        await app.handle_command(
            f"/agent {child.id} integrate {head} {reviewed['reviewed_parent_head']}"
        )
        if restriction == "allowed":
            assert marker.read_text() == "validated", out.getvalue()
            assert (parent.cwd / "change.txt").read_text() == "worker change\n"
        else:
            assert "denied" in out.getvalue(), out.getvalue()
            assert not marker.exists()
            assert not (parent.cwd / "change.txt").exists()
        assert bool(approvals) is ("ask" in restriction)
        if approvals and approvals[0][0] == "bash":
            assert child.id in approvals[0][1] and str(child.cwd) in approvals[0][1]
        assert not (tmp_path / "change.txt").exists()
    finally:
        await manager.shutdown()


# -- /init -------------------------------------------------------------------------


async def test_init_creates_agents_md(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/init")
    path = tmp_path / "AGENTS.md"
    assert path.is_file()
    text = path.read_text()
    assert "# AGENTS.md" in text
    assert "Build & test" in text
    assert "wrote" in out.getvalue()


async def test_init_refuses_to_overwrite(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# existing\n")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/init")
    assert "already exists" in out.getvalue()
    assert (tmp_path / "AGENTS.md").read_text() == "# existing\n"


# -- /tutor ------------------------------------------------------------------------


async def test_tutor_lists_topics(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tutor")
    rendered = out.getvalue()
    assert "usage: /tutor <topic>" in rendered
    for topic in ("permissions", "worktrees", "mcp"):
        assert topic in rendered


async def test_tutor_answers_a_topic(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tutor permissions")
    assert "readonly" in out.getvalue()


async def test_tutor_unknown_topic(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tutor quantum")
    assert "unknown topic: quantum" in out.getvalue()


# -- /review ------------------------------------------------------------------------


async def test_review_clean_repo(tmp_path, monkeypatch):
    await make_repo(tmp_path)
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "ok"}])
    await app.handle_command("/review")
    assert "nothing to review" in out.getvalue()
    assert provider.requests == []


async def test_review_submits_diff_with_reviewer_persona(tmp_path, monkeypatch):
    await make_repo(tmp_path)
    (tmp_path / "file.txt").write_text("changed content\n")
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "review done"}])
    await app.handle_command("/review")
    await app._turn_task
    request = provider.requests[-1]
    prompt = request["messages"][-1]["content"]
    assert "meticulous code reviewer" in prompt  # the reviewer persona
    assert "changed content" in prompt  # the diff body


async def test_review_listed_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "review done"}])
    await app.handle_command("/review a.py")
    await app._turn_task
    prompt = provider.requests[-1]["messages"][-1]["content"]
    assert "meticulous code reviewer" in prompt
    assert "### a.py" in prompt
    assert "x = 1" in prompt


async def test_review_outside_git_repo(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "unused"}])
    await app.handle_command("/review")
    assert "not a git repository" in out.getvalue()
    assert provider.requests == []


# -- /notifications -------------------------------------------------------------------


async def test_notifications_shows_state(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])  # make_app disables sounds
    await app.handle_command("/notifications")
    rendered = out.getvalue()
    assert "notifications: off" in rendered
    assert "volume" in rendered and "approval" in rendered


async def test_notifications_toggle(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    assert app.config.notifications.enabled is False
    await app.handle_command("/notifications on")
    assert app.config.notifications.enabled is True
    await app.handle_command("/notifications")
    assert "notifications: on" in out.getvalue()
    await app.handle_command("/notifications bogus")
    assert "usage: /notifications" in out.getvalue()


# -- /prompt -------------------------------------------------------------------------


async def test_prompt_prints_assembled_system_prompt(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# Project\n\nAlways run the tests.\n")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/prompt")
    rendered = out.getvalue()
    assert "You are lecode" in rendered
    assert "Always run the tests." in rendered  # AGENTS.md made it in


# -- /compress -------------------------------------------------------------------------


async def test_compress_aliases_compact(tmp_path, monkeypatch):
    from lecode.slash.handlers import cmd_compact

    app, _, out = make_app(tmp_path, monkeypatch, [])
    assert app.commands.get("compress").handler is cmd_compact
    await app.handle_command("/compress")
    assert "not enough history to compact" in out.getvalue()


# -- /editsys ---------------------------------------------------------------------------


async def _fake_editor(tmp_path, monkeypatch, body: str) -> None:
    script = tmp_path / "editor.sh"
    script.write_text(f"#!/bin/sh\n{body}\n")
    monkeypatch.setenv("EDITOR", f"/bin/sh {script}")

    async def run_directly(func, **kwargs):
        func()

    monkeypatch.setattr("lecode.tui.input.run_in_terminal", run_directly)


async def test_editsys_saves_session_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    await _fake_editor(tmp_path, monkeypatch, "printf 'CUSTOM SYSTEM PROMPT' > \"$1\"")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/editsys")
    assert app.config.llm.system_prompt.custom == "CUSTOM SYSTEM PROMPT"
    assert app.runtime.system_prompt == "CUSTOM SYSTEM PROMPT"
    assert app._history[0]["content"] == "CUSTOM SYSTEM PROMPT"
    assert "overridden for this session" in out.getvalue()


async def test_editsys_unchanged_is_noop(tmp_path, monkeypatch):
    await _fake_editor(tmp_path, monkeypatch, "true")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    original = app.runtime.system_prompt
    await app.handle_command("/editsys")
    assert app.runtime.system_prompt == original
    assert app.config.llm.system_prompt.custom is None
    assert "unchanged" in out.getvalue()


async def test_editsys_preserves_edits_without_freezing_managed_facts(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    app.config.llm.system_prompt.custom = "My base prompt"
    record = app.store.append_message(app.session, {"role": "user", "content": "old fact"})
    ref = app.store.source_snapshot(
        app.session.id, record.seq, record.seq, project_root=tmp_path
    ).ref
    facts = app.runtime.ctx.extras["facts"]
    fact = facts.remember("old fact", ref, sessions=app.store, project_root=tmp_path)
    app.reload_history()
    assert "old fact" in app.runtime.system_prompt
    await _fake_editor(tmp_path, monkeypatch, "printf '\\nMy edit' >> \"$1\"")
    await app.handle_command("/editsys")
    facts.forget(fact.id)
    app.reload_history()
    assert "My edit" in app.runtime.system_prompt
    assert "old fact" not in app.runtime.system_prompt
    app.runtime.close()


# -- /doctor ---------------------------------------------------------------------------


def _patch_doctor_provider(monkeypatch, origin="live", count=2):
    """Patch the deferred fetch_catalog import used by /doctor."""
    from tests.fakes import sample_catalog

    import lecode.cli as cli
    from lecode.providers.live import LoadedCatalog

    monkeypatch.setattr(
        cli,
        "fetch_catalog",
        lambda config, api_key=None: LoadedCatalog(sample_catalog(), origin, count),
    )


async def test_doctor_all_sections(tmp_path, monkeypatch):
    """Every section renders with a mark; a healthy setup ends in 'all good'."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _patch_doctor_provider(monkeypatch)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/doctor")
    rendered = out.getvalue()
    for needle in (
        "fd:",
        "rg:",
        "rtk:",
        "config:",
        "provider:",
        "model:",
        "connectivity: reachable — 2 models",
        "mcp:",
        "session:",
        "memory:",
        "hooks:",
        "lsp:",
        "telemetry:",
        "permissions:",
        "tools:",
    ):
        assert needle in rendered, f"missing {needle!r}:\n{rendered}"
    # no API key in the test env → one warning, not "all good"
    assert "doctor: 1 issue(s)" in rendered


async def test_doctor_missing_binary_fails(tmp_path, monkeypatch):
    import shutil

    _patch_doctor_provider(monkeypatch)
    real_which = shutil.which

    def which(name, path=None):
        return None if name == "fd" else real_which(name, path=path)

    monkeypatch.setattr(shutil, "which", which)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/doctor")
    rendered = out.getvalue()
    assert "✗ fd: MISSING" in rendered
    assert "install:" in rendered


async def test_doctor_unreachable_provider(tmp_path, monkeypatch):
    _patch_doctor_provider(monkeypatch, origin="empty", count=0)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/doctor")
    rendered = out.getvalue()
    assert "✗ connectivity: catalog fetch failed" in rendered


async def test_doctor_memory_path_keeps_durable_project_root(tmp_path, monkeypatch):
    _patch_doctor_provider(monkeypatch)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    root = app.runtime.ctx.extras["memory"].root
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    app.set_cwd(checkout)
    await app.handle_command("/doctor")
    assert f"memory: {root}" in out.getvalue()
