"""Production worker controls through the registered, permission-gated tools."""

import asyncio
import json
import shlex
import shutil
import subprocess
import sys

import pytest
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.agent.tools.base import ToolResult
from lecode.agent.tools.bash import BashTool
from lecode.config.models import Config
from lecode.context.agents import AgentDefinition, AgentRegistry
from lecode.extras.proc import ProcResult
from lecode.extras.subagents import SubagentError
from lecode.permission.checker import AgentOverlay
from lecode.session.storage import SessionStore


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(cwd, text="worker change\n"):
    (cwd / "change.txt").write_text(text)
    git(cwd, "add", "change.txt")
    git(cwd, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "change")
    return git(cwd, "rev-parse", "HEAD")


async def dispatch(runtime, action, **args):
    _, result = await runtime.registry.dispatch_result(
        "control", "workers", json.dumps({"action": action, **args}), runtime.ctx
    )
    return result


async def review_heads(runtime, worker):
    result = await dispatch(runtime, "review", id=worker.id)
    assert not result.is_error, result.content
    return result.metadata


@pytest.fixture
async def workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "skills"))
    cwd = tmp_path / "repo"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    git(cwd, "init", "-b", "main")
    commit(cwd, "base\n")
    config = Config()
    config.memory.enabled = config.lsp.enabled = config.pierre.enabled = False
    store = SessionStore(tmp_path / "config")
    runtime = build_runtime(
        config,
        cwd,
        session=store.create("main", cwd),
        store=store,
        auto_approve=True,
        agent_registry=AgentRegistry(
            {
                "writer": AgentDefinition(
                    name="writer", description="Write code", body="", mode="subagent"
                )
            }
        ),
    )
    runtime.ctx.extras["provider"] = FakeProvider([{"text": "done"}] * 12)
    try:
        yield runtime
    finally:
        await runtime.ctx.extras["workers"].shutdown()


async def start(runtime):
    _, result = await runtime.registry.dispatch_result(
        "assignment", "task", '{"agent":"writer","prompt":"Implement the assignment"}', runtime.ctx
    )
    assert not result.is_error, result.content
    return runtime.ctx.extras["workers"].get(result.metadata["worker_id"])


async def test_registered_review_returns_exact_diff_and_commit(workflow):
    worker = await start(workflow)
    head = commit(worker.cwd)
    result = await dispatch(workflow, "review", id=worker.id)
    assert not result.is_error, result.content
    assert result.metadata["reviewed_head"] == head
    parent_head = git(workflow.ctx.cwd, "rev-parse", "HEAD")
    assert result.metadata["reviewed_parent_head"] == parent_head
    assert f"reviewed_head {head} and reviewed_parent_head {parent_head}" in result.content
    assert "-base" in result.content and "+worker change" in result.content
    assert "main" in result.content and "Implement the assignment" in result.content
    assert (
        "review this exact diff against assignment then integrate reviewed_head" in result.content
    )


@pytest.mark.parametrize("approved", [True, False, None])
async def test_empty_validation_requires_explicit_human_confirmation(workflow, approved):
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    questions = []
    if approved is not None:

        def confirm(question):
            questions.append(question)
            return approved

        workflow.ctx.extras["workers"].confirm = confirm
    result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert result.is_error is (approved is not True), result.content
    assert (workflow.ctx.cwd / "change.txt").read_text() == (
        "worker change\n" if approved else "base\n"
    )
    if approved is not None:
        assert len(questions) == 1
        assert worker.id in questions[0] and "@writer" in questions[0]
        assert str(worker.cwd) in questions[0] and "validation" in questions[0]
    else:
        assert "unavailable" in result.content


async def test_stale_review_never_merges(workflow):
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    commit(worker.cwd, "unreviewed\n")
    workflow.ctx.extras["workers"].confirm = lambda _: True
    result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert result.is_error and "re-review" in result.content
    assert (workflow.ctx.cwd / "change.txt").read_text() == "base\n"


async def test_parent_rewind_requires_review_without_refreshing_hash(workflow):
    base = git(workflow.ctx.cwd, "rev-parse", "HEAD")
    parent_head = commit(workflow.ctx.cwd, "parent progress\n")
    worker = await start(workflow)
    head = commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    assert reviewed == {"reviewed_head": head, "reviewed_parent_head": parent_head}
    git(workflow.ctx.cwd, "reset", "--keep", base)
    workflow.ctx.config.worktree.validation = ["touch must-not-run"]
    for _ in range(2):
        result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
        assert result.is_error and "destination HEAD changed since review" in result.content
        assert git(workflow.ctx.cwd, "rev-parse", "HEAD") == base
        assert git(worker.cwd, "rev-parse", "HEAD") == head
        assert not (worker.cwd / "must-not-run").exists()


@pytest.mark.parametrize("deny", [False, True])
async def test_configured_validation_runs_in_child_through_permission_gate(workflow, deny):
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    script = "from pathlib import Path; assert Path('change.txt').read_text() == 'worker change\\n'"
    workflow.ctx.config.worktree.validation = [
        f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    ]
    if deny:
        workflow.ctx.permission_checker = workflow.ctx.permission_checker.for_agent(
            AgentOverlay(denied_tools=("bash",))
        )
    result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert result.is_error is deny, result.content
    assert (workflow.ctx.cwd / "change.txt").read_text() == (
        "base\n" if deny else "worker change\n"
    )
    if deny:
        assert "denied" in result.content


async def test_failed_validation_never_merges(workflow):
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    workflow.ctx.config.worktree.validation = ["exit 7"]
    result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert result.is_error and "7" in result.content
    assert (workflow.ctx.cwd / "change.txt").read_text() == "base\n"


async def test_worker_overlay_constrains_validation(workflow):
    workflow.ctx.extras["agents"] = AgentRegistry(
        {
            "writer": AgentDefinition(
                name="writer",
                description="Write",
                body="",
                mode="subagent",
                overlay=AgentOverlay(denied_tools=("bash",)),
            )
        }
    )
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    workflow.ctx.config.worktree.validation = ["touch must-not-run"]
    result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert result.is_error and "denied by agent overlay: bash" in result.content
    assert not (worker.cwd / "must-not-run").exists()
    assert (workflow.ctx.cwd / "change.txt").read_text() == "base\n"


async def test_cleanup_refuses_dirty_and_unmerged_then_retains_transcript(workflow):
    worker = await start(workflow)
    (worker.cwd / "change.txt").write_text("dirty\n")
    dirty = await dispatch(workflow, "cleanup", id=worker.id)
    assert dirty.is_error and "uncommitted" in dirty.content
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    unmerged = await dispatch(workflow, "cleanup", id=worker.id)
    assert unmerged.is_error and "not integrated" in unmerged.content
    workflow.ctx.extras["workers"].confirm = lambda _: True
    integrated = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert not integrated.is_error, integrated.content
    cleaned = await dispatch(workflow, "cleanup", id=worker.id)
    assert not cleaned.is_error, cleaned.content
    assert not worker.cwd.exists()
    manager = workflow.ctx.extras["workers"]
    assert manager.get(worker.id).session_id == worker.session_id
    assert manager.store.load_for_model(worker.session)[-1]["content"] == "done"


@pytest.mark.parametrize("approve", [False, True, None])
async def test_missing_checkout_recovery_needs_human_confirmation(workflow, approve):
    worker = await start(workflow)
    head = commit(worker.cwd)
    shutil.rmtree(worker.cwd)
    questions = []
    if approve is not None:

        async def confirm(question):
            questions.append(question)
            return approve

        workflow.ctx.extras["workers"].confirm = confirm
    result = await dispatch(workflow, "recover", id=worker.id)
    assert result.is_error is (approve is not True), result.content
    assert worker.cwd.exists() is (approve is True)
    if approve:
        assert git(worker.cwd, "rev-parse", "HEAD") == head
    if approve is not None:
        assert worker.id in questions[0] and str(worker.cwd) in questions[0]
        assert "uncommitted" in questions[0] and "unrecoverable" in questions[0]


@pytest.mark.parametrize("action", ["send", "resume"])
async def test_write_followup_reconciles_parent_commits(workflow, action):
    worker = await start(workflow)
    (workflow.ctx.cwd / "parent.txt").write_text("parent progress\n")
    git(workflow.ctx.cwd, "add", "parent.txt")
    git(
        workflow.ctx.cwd,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "parent",
    )
    result = await dispatch(workflow, action, id=worker.id, text="Continue")
    assert not result.is_error, result.content
    await workflow.ctx.extras["workers"].wait(worker.id)
    assert (worker.cwd / "parent.txt").read_text() == "parent progress\n"


def supervisor_runtime(runtime, worker):
    return runtime.ctx.extras["workers"]._runtime(worker)


async def test_nested_integration_requires_immediate_supervisor_and_pinned_target(workflow):
    parent = await start(workflow)
    nested = supervisor_runtime(workflow, parent)
    child = await start(nested)
    commit(child.cwd, "nested change\n")
    review = await dispatch(nested, "inspect", id=child.id)
    assert not review.is_error and parent.worktree.branch in review.content
    manager = workflow.ctx.extras["workers"]
    manager.confirm = lambda _: True
    denied = await dispatch(workflow, "integrate", id=child.id, **review.metadata)
    assert denied.is_error and "immediate supervisor" in denied.content
    result = await dispatch(nested, "integrate", id=child.id, **review.metadata)
    assert not result.is_error, result.content
    assert (parent.cwd / "change.txt").read_text() == "nested change\n"
    assert (workflow.ctx.cwd / "change.txt").read_text() == "base\n"


async def test_maintenance_reserves_workspace_against_send_resume_and_child_start(workflow):
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    entered, release = asyncio.Event(), asyncio.Event()

    async def confirm(_):
        entered.set()
        await release.wait()
        return True

    manager = workflow.ctx.extras["workers"]
    manager.confirm = confirm
    integration = asyncio.create_task(dispatch(workflow, "integrate", id=worker.id, **reviewed))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
        for action in ("send", "resume"):
            result = await dispatch(workflow, action, id=worker.id, text="Race")
            assert result.is_error and "maintenance" in result.content
        nested = supervisor_runtime(workflow, worker)
        _, result = await nested.registry.dispatch_result(
            "nested", "task", '{"agent":"writer","prompt":"Race"}', nested.ctx
        )
        assert result.is_error and "maintenance" in result.content
        assert manager.pending(worker.id) == []
    finally:
        release.set()
        result = await integration
    assert not result.is_error, result.content


async def test_missing_followup_fails_without_implicit_recreation(workflow):
    worker = await start(workflow)
    shutil.rmtree(worker.cwd)
    result = await dispatch(workflow, "send", id=worker.id, text="Continue")
    assert not result.is_error, result.content
    with pytest.raises(SubagentError, match="checkout missing"):
        await workflow.ctx.extras["workers"].wait(worker.id)
    assert not worker.cwd.exists()


async def test_dirty_review_requires_checkpoint_before_reviewed_head(workflow):
    worker = await start(workflow)
    (worker.cwd / "change.txt").write_text("uncommitted work\n")
    (worker.cwd / "new.txt").write_text("new work\n")
    review = await dispatch(workflow, "review", id=worker.id)
    assert not review.is_error, review.content
    assert "+uncommitted work" in review.content and "new.txt" in review.content
    assert "OWN branch" in review.content and "reviewed_head" not in review.metadata


@pytest.mark.parametrize(
    "args",
    [
        {"action": "cleanup", "discard": True},
        {"action": "recover", "recreate": True},
        {"action": "integrate", "reviewed_head": "main"},
        {"action": "integrate", "reviewed_head": "a" * 40},
        {"action": "integrate", "reviewed_parent_head": "b" * 40},
        {"action": "integrate", "reviewed_head": "a" * 40, "reviewed_parent_head": "main"},
        {"action": "integrate", "reviewed_head": "a" * 40, "reviewed_parent_head": "b" * 39},
        {"action": "integrate", "reviewed_head": "a" * 40, "reviewed_parent_head": 42},
        {"action": "integrate", "reviewed_head": "a" * 40, "allow_unvalidated": True},
        {"action": "integrate", "reviewed_head": "a" * 40, "validation": ["true"]},
        {"action": "resume", "text": 3},
    ],
)
async def test_control_arguments_cannot_override_human_or_validation_gates(workflow, args):
    worker = await start(workflow)
    result = await dispatch(workflow, id=worker.id, **args)
    assert result.is_error
    assert worker.cwd.is_dir()


async def test_validation_hooks_are_fresh_for_each_child_and_block_execution(workflow, tmp_path):
    workers = [await start(workflow), await start(workflow)]
    reviews = []
    for worker in workers:
        commit(worker.cwd)
        reviews.append(await review_heads(workflow, worker))
    log = tmp_path / "validation-hooks.jsonl"
    script = (
        "import json,sys; from pathlib import Path; data=json.load(sys.stdin); "
        f"p=Path({str(log)!r}); "
        "p.open('a').write(json.dumps(data)+'\\n'); "
        "print(json.dumps({'verdict':'deny','reason':'validation hook blocked'}))"
    )
    workflow.ctx.config.hooks = {
        "PreToolUse": [f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"]
    }
    workflow.ctx.config.worktree.validation = ["touch should-not-exist"]
    for worker, reviewed in zip(workers, reviews, strict=True):
        result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
        assert result.is_error and "validation hook blocked" in result.content
        assert not (worker.cwd / "should-not-exist").exists()
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [entry["cwd"] for entry in entries] == [str(w.cwd) for w in workers]
    assert [entry["session"]["id"] for entry in entries] == [w.session_id for w in workers]
    assert all(entry["tool"]["args"] == {"command": "touch should-not-exist"} for entry in entries)
    _, root_result = await workflow.registry.dispatch_result(
        "root", "bash", '{"command":"pwd"}', workflow.ctx
    )
    assert not root_result.is_error and str(workflow.ctx.cwd) in root_result.content
    assert len(log.read_text().splitlines()) == 2


async def test_readonly_inspection_respects_workers_deny(workflow):
    worker = await start(workflow)
    commit(worker.cwd)
    workflow.ctx.permission_checker = workflow.ctx.permission_checker.for_child(read_only=True)
    review = await dispatch(workflow, "inspect", id=worker.id)
    assert not review.is_error, review.content
    workflow.ctx.permission_checker = workflow.ctx.permission_checker.for_agent(
        AgentOverlay(denied_tools=("workers",))
    )
    denied = await dispatch(workflow, "inspect", id=worker.id)
    assert denied.is_error and "denied" in denied.content


async def test_child_creation_reserves_retained_parent_against_cleanup(workflow, tmp_path):
    parent = await start(workflow)
    nested = supervisor_runtime(workflow, parent)
    entered, release = tmp_path / "checkout-entered", tmp_path / "checkout-release"
    hook = workflow.ctx.cwd / ".git/hooks/post-checkout"
    hook.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(entered))}\n"
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done\n"
    )
    hook.chmod(0o755)
    creation = asyncio.create_task(
        nested.registry.dispatch_result(
            "child", "task", '{"agent":"writer","prompt":"Nested"}', nested.ctx
        )
    )
    try:
        async with asyncio.timeout(3):
            while not entered.exists():
                await asyncio.sleep(0.01)
        result = await dispatch(workflow, "cleanup", id=parent.id)
        assert result.is_error and "idle" in result.content
        assert parent.cwd.is_dir()
    finally:
        release.touch()
        await creation


async def test_cleanup_preserves_retained_child_destination(workflow):
    parent = await start(workflow)
    child = await start(supervisor_runtime(workflow, parent))
    result = await dispatch(workflow, "cleanup", id=parent.id)
    assert result.is_error and "child workspaces first" in result.content
    assert parent.cwd.is_dir() and child.cwd.is_dir()
    child_cleanup = await dispatch(workflow, "cleanup", id=child.id)
    assert not child_cleanup.is_error, child_cleanup.content
    parent_cleanup = await dispatch(workflow, "cleanup", id=parent.id)
    assert not parent_cleanup.is_error, parent_cleanup.content


async def test_truncated_diff_cannot_be_presented_as_exact_review(workflow):
    worker = await start(workflow)
    commit(worker.cwd, "large change\n" * 100_000)
    review = await dispatch(workflow, "inspect", id=worker.id)
    assert review.is_error and "truncated" in review.content
    assert "reviewed_head" not in review.metadata


@pytest.mark.parametrize(
    "tool_result",
    [
        ToolResult("looks successful"),
        ToolResult("tool error", is_error=True, metadata={"proc_result": ProcResult(0, "", "")}),
    ],
)
async def test_validation_never_infers_success_without_successful_exit(
    workflow, monkeypatch, tool_result
):
    worker = await start(workflow)
    commit(worker.cwd)
    reviewed = await review_heads(workflow, worker)
    workflow.ctx.config.worktree.validation = ["true"]

    async def result_only(args, ctx):
        return tool_result

    monkeypatch.setattr(BashTool, "run", result_only)
    result = await dispatch(workflow, "integrate", id=worker.id, **reviewed)
    assert result.is_error and "validation failed" in result.content
    assert (workflow.ctx.cwd / "change.txt").read_text() == "base\n"
