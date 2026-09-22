"""Tests for git worktree isolation: the manager, the commands, --worktree.

All git operations happen in throwaway repos under tmp_path; commits use
``-c user.email/name`` flags and repos get a local identity so no global
git config is needed.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from os.path import realpath
from pathlib import Path

import pytest
from tests.test_tui_app import FakeTui, _name_prompt, make_app
from typer.testing import CliRunner

from lecode.cli import app as cli_app
from lecode.extras.proc import ProcResult, run_proc
from lecode.extras.worktree import WorktreeError, WorktreeManager

runner = CliRunner()

_COMMIT = ["-c", "user.email=t@example.com", "-c", "user.name=test", "commit"]


async def git(cwd, *args):
    result = await run_proc(["git", *args], cwd=cwd, timeout=30)
    assert result.exit_code == 0, result.stderr
    return result.stdout.strip()


async def commit_all(cwd, message):
    await git(cwd, "add", "-A")
    await git(cwd, *_COMMIT, "-m", message)


async def make_repo(path):
    """A git repo at ``path`` with one commit on ``main``."""
    path.mkdir(parents=True, exist_ok=True)
    await git(path, "init", "-b", "main")
    await git(path, "config", "user.email", "t@example.com")
    await git(path, "config", "user.name", "test")
    (path / "file.txt").write_text("base\n", encoding="utf-8")
    await git(path, "add", ".")
    await git(path, *_COMMIT, "-m", "init")
    return path


def git_sync(cwd, *args):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def make_repo_sync(path):
    path.mkdir(parents=True, exist_ok=True)
    git_sync(path, "init", "-b", "main")
    git_sync(path, "config", "user.email", "t@example.com")
    git_sync(path, "config", "user.name", "test")
    (path / "file.txt").write_text("base\n", encoding="utf-8")
    git_sync(path, "add", ".")
    git_sync(path, *_COMMIT, "-m", "init")
    return path


# -- WorktreeManager -------------------------------------------------------------


async def test_create(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    assert info.name == "feat"
    assert info.branch == "lecode/feat"
    assert info.path == repo / ".lecode" / "worktrees" / "feat"
    assert info.path.is_dir()
    assert (info.path / "file.txt").read_text() == "base\n"
    await git(repo, "show-ref", "--verify", "refs/heads/lecode/feat")


async def test_create_invalid_name(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    with pytest.raises(WorktreeError, match="invalid worktree name"):
        await manager.create("bad name")


async def test_create_duplicate_refused(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    await manager.create("feat")
    with pytest.raises(WorktreeError, match="already exists"):
        await manager.create("feat")


async def test_create_existing_branch_refused(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    await git(repo, "branch", "lecode/taken")
    manager = WorktreeManager(repo)
    with pytest.raises(WorktreeError, match="branch already exists"):
        await manager.create("taken")


async def test_create_excludes_worktree_dir(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    await manager.create("feat")
    exclude = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert "/.lecode/worktrees/" in exclude
    assert await git(repo, "status", "--porcelain") == ""


async def test_discover_not_a_repo(tmp_path):
    with pytest.raises(WorktreeError, match="not a git repository"):
        await WorktreeManager.discover(tmp_path)


async def test_discover_from_subdir(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    subdir = repo / "a" / "b"
    subdir.mkdir(parents=True)
    manager = await WorktreeManager.discover(subdir)
    assert manager.repo_root == repo


async def test_status_ahead_and_dirty(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    (info.path / "new.txt").write_text("work\n", encoding="utf-8")
    await commit_all(info.path, "worktree commit")
    status = await manager.status("feat")
    assert status.ahead == 1
    assert status.behind == 0
    assert status.dirty is False
    (info.path / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")
    assert (await manager.status("feat")).dirty is True


async def test_status_unknown_worktree(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    with pytest.raises(WorktreeError, match="no such worktree"):
        await manager.status("ghost")


async def test_merge_clean(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    await commit_all(info.path, "add feature")
    result = await manager.merge_back("feat")
    assert result.merged is True
    assert result.conflicts == []
    assert "lecode/feat" in result.message
    assert (repo / "feature.txt").read_text() == "feature\n"


async def test_merge_conflict_reports_files(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    (info.path / "file.txt").write_text("worktree change\n", encoding="utf-8")
    await commit_all(info.path, "worktree edit")
    (repo / "file.txt").write_text("main change\n", encoding="utf-8")
    await commit_all(repo, "main edit")
    result = await manager.merge_back("feat")
    assert result.merged is False
    assert result.conflicts == ["file.txt"]
    assert "merge conflicts" in result.message


async def test_exit_clean(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    await manager.exit_worktree("feat")
    assert not info.path.exists()
    await git(repo, "show-ref", "--verify", "refs/heads/lecode/feat")  # branch kept


async def test_exit_dirty_refused_then_forced(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    (info.path / "dirty.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(WorktreeError, match="uncommitted changes"):
        await manager.exit_worktree("feat")
    assert info.path.exists()
    await manager.exit_worktree("feat", force=True)
    assert not info.path.exists()


async def test_exit_deletes_branch(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    await manager.create("feat")
    await manager.exit_worktree("feat", delete_branch=True)
    result = await run_proc(
        ["git", "show-ref", "--verify", "refs/heads/lecode/feat"], cwd=repo, timeout=30
    )
    assert result.exit_code != 0


# -- worker worktrees: create/attach/inspect ------------------------------------------


async def test_discover_from_linked_worktree_returns_main_root(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    subdir = info.path / "a" / "b"
    subdir.mkdir(parents=True)
    found = await WorktreeManager.discover(subdir)
    assert found.repo_root == repo


async def test_create_worker_pins_base_and_writes_sidecar(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    base = await git(repo, "rev-parse", "HEAD")
    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    await commit_all(repo, "later")
    destination_subdir = repo / "nested"
    destination_subdir.mkdir()
    info = await manager.create_worker(
        "worker", base_commit=base, dest_path=destination_subdir, dest_branch="main"
    )
    assert info.branch == "lecode/worker"
    assert await git(info.path, "rev-parse", "HEAD") == base
    assert not (info.path / "later.txt").exists()
    assert manager.read_sidecar("worker") == {
        "name": "worker",
        "path": str(info.path),
        "branch": "lecode/worker",
        "base_commit": base,
        "dest_path": str(repo),
        "dest_branch": "main",
        "dest_common_dir": str(repo / ".git"),
        "repo_identity": manager._identity(repo / ".git"),
        "dest_identity": manager._identity(repo),
        "checkout_identity": manager._identity(info.path),
        "integrated_at": None,
        "integrated_head": None,
    }
    assert (repo / ".lecode" / "worktrees" / "worker.json").is_file()


async def test_attach_existing_worktree(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    created = await manager.create("feat")
    assert await manager.attach("feat") == created


async def test_attach_missing_worktree_raises(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    with pytest.raises(WorktreeError, match="no such worktree"):
        await manager.attach("ghost")


async def test_inspect_reports_state(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create("feat")
    state = await manager.inspect("feat")
    assert state.present is True
    assert state.dirty is False
    assert state.merge_in_progress is False
    assert state.sidecar is None

    (info.path / "dirty.txt").write_text("x\n", encoding="utf-8")
    assert (await manager.inspect("feat")).dirty is True

    (info.path / "dirty.txt").unlink()
    (info.path / "file.txt").write_text("worker\n", encoding="utf-8")
    await commit_all(info.path, "worker edit")
    (repo / "file.txt").write_text("main\n", encoding="utf-8")
    await commit_all(repo, "main edit")
    merge = await run_proc(["git", "merge", "--no-edit", "main"], cwd=info.path, timeout=30)
    assert merge.exit_code != 0
    assert (await manager.inspect("feat")).merge_in_progress is True

    absent = await manager.inspect("ghost")
    assert absent.present is False
    assert absent.info.name == "ghost"
    assert absent.dirty is False
    assert absent.merge_in_progress is False


async def worker_repo(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    manager = WorktreeManager(repo)
    info = await manager.create_worker(
        "worker", base_commit="HEAD", dest_path=repo, dest_branch="main"
    )
    (info.path / "feature.txt").write_text("feature\n")
    await commit_all(info.path, "worker feature")
    return repo, manager, info


async def validate_git(cmd, cwd):
    assert cmd == "check"
    return await run_proc(["git", "diff", "--exit-code", "HEAD"], cwd=cwd)


async def integrate_head(manager, info, *, reviewed_parent_head=None, **kwargs):
    if reviewed_parent_head is None:
        reviewed_parent_head = await git(
            manager.read_sidecar(info.name)["dest_path"], "rev-parse", "HEAD"
        )
    return await manager.integrate(
        info.name,
        reviewed_head=await git(info.path, "rev-parse", "HEAD"),
        reviewed_parent_head=reviewed_parent_head,
        validation=["check"],
        validation_runner=validate_git,
        **kwargs,
    )


async def test_discover_unrelated_linked_relative_subdir(tmp_path, monkeypatch):
    repo = await make_repo(tmp_path / "repo")
    linked = tmp_path / "linked"
    await git(repo, "worktree", "add", "-b", "parent", str(linked))
    (linked / "sub").mkdir()
    monkeypatch.chdir(tmp_path)
    manager = await WorktreeManager.discover(Path("linked/sub"))
    assert manager.repo_root == linked
    info = await manager.create_worker(
        "child", base_commit="HEAD", dest_path="linked/sub", dest_branch="parent"
    )
    assert manager.read_sidecar("child")["dest_path"] == str(linked)
    assert await git(linked, "status", "--porcelain") == ""
    (info.path / "feature").write_text("child")
    await commit_all(info.path, "child")
    assert (await WorktreeManager.discover(info.path)).repo_root == linked
    main_head = await git(repo, "rev-parse", "HEAD")
    assert (await integrate_head(manager, info)).merged
    assert await git(repo, "rev-parse", "HEAD") == main_head
    assert (linked / "feature").read_text() == "child"


async def test_reconcile_merges_parent_and_retains_dirty_worker(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    worker_head = await git(info.path, "rev-parse", "HEAD")
    (repo / "parent.txt").write_text("parent")
    await commit_all(repo, "parent progress")
    parent_head = await git(repo, "rev-parse", "HEAD")
    (info.path / "dirty.txt").write_text("unfinished")
    assert (await manager.reconcile(info.name)).dirty
    assert await git(info.path, "rev-parse", "HEAD") == worker_head
    (info.path / "dirty.txt").unlink()
    # Parent dirt is not copied into the worker and must not be lost either.
    (repo / "uncommitted").write_text("parent unfinished")
    state = await manager.reconcile(info.name)
    assert not state.dirty
    assert await git(info.path, "rev-parse", "HEAD^1") == worker_head
    assert await git(info.path, "rev-parse", "HEAD^2") == parent_head
    assert not (info.path / "uncommitted").exists()
    assert (repo / "uncommitted").read_text() == "parent unfinished"


async def test_reconcile_conflict_left_visible(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    (repo / "file.txt").write_text("parent")
    await commit_all(repo, "parent edit")
    (info.path / "file.txt").write_text("worker")
    await commit_all(info.path, "worker edit")
    head = await git(info.path, "rev-parse", "HEAD")
    with pytest.raises(WorktreeError, match="merge"):
        await manager.reconcile(info.name)
    assert (await manager.reconcile(info.name)).merge_in_progress
    assert await git(info.path, "rev-parse", "HEAD") == head
    assert "<<<<<<<" in (info.path / "file.txt").read_text()
    assert (repo / "file.txt").read_text() == "parent"


@pytest.mark.parametrize("keep_branch", [True, False])
async def test_missing_worker_explicit_recovery(tmp_path, keep_branch):
    repo, manager, info = await worker_repo(tmp_path)
    head = await git(info.path, "rev-parse", "HEAD")
    sidecar = manager.read_sidecar(info.name)
    (info.path / "lost").write_text("uncommitted")
    shutil.rmtree(info.path)
    if not keep_branch:
        await git(repo, "update-ref", "-d", f"refs/heads/{info.branch}")
    with pytest.raises(WorktreeError, match="Missing uncommitted content cannot be recovered"):
        await manager.reconcile(info.name)
    assert not info.path.exists()
    assert manager.read_sidecar(info.name) == sidecar
    state = await manager.reconcile(info.name, recreate=True)
    assert state.present and not state.dirty
    expected = head if keep_branch else manager.read_sidecar(info.name)["base_commit"]
    assert await git(info.path, "rev-parse", "HEAD") == expected
    assert not (info.path / "lost").exists()
    assert manager.read_sidecar(info.name) == {
        **sidecar,
        "checkout_identity": manager._identity(info.path),
    }
    assert await WorktreeManager(repo).attach(info.name) == info


@pytest.mark.parametrize("change", ["directory", "repo", "branch", "detached"])
async def test_reconcile_refuses_replaced_worker(tmp_path, change):
    repo, manager, info = await worker_repo(tmp_path)
    if change in {"directory", "repo"}:
        await git(repo, "worktree", "remove", str(info.path))
        if change == "repo":
            await make_repo(info.path)
        else:
            info.path.mkdir()
    elif change == "branch":
        await git(info.path, "switch", "-c", "unrelated")
        assert (await WorktreeManager.discover(info.path)).repo_root == repo
    else:
        await git(info.path, "switch", "--detach")
    with pytest.raises(WorktreeError):
        await manager.reconcile(info.name, recreate=True)
    with pytest.raises(WorktreeError):
        await manager.cleanup_worker(info.name, discard=True)
    assert info.path.is_dir()


async def test_explicit_recreation_repins_checkout_inode(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    sidecar = manager.read_sidecar(info.name)
    (info.path / "unfinished").write_text("retained original work")
    saved = tmp_path / "saved-worker"
    info.path.rename(saved)
    manager = WorktreeManager(repo)
    assert not (await manager.inspect(info.name)).present
    with pytest.raises(WorktreeError, match="Missing uncommitted content cannot be recovered"):
        await manager.reconcile(info.name)
    assert manager.read_sidecar(info.name) == sidecar
    assert not info.path.exists()
    recovered = await manager.reconcile(info.name, recreate=True)
    identity = manager._identity(info.path)
    assert recovered.present and not recovered.dirty
    assert identity != sidecar["checkout_identity"]
    assert recovered.sidecar == {**sidecar, "checkout_identity": identity}
    assert (saved / "unfinished").read_text() == "retained original work"
    assert not (info.path / "unfinished").exists()
    assert await manager.attach(info.name) == info
    assert (await integrate_head(manager, info)).merged
    await manager.cleanup_worker(info.name)
    assert manager.read_sidecar(info.name)["checkout_identity"] == identity


@pytest.mark.parametrize("action", ["inspect", "attach", "reconcile", "integrate", "cleanup"])
async def test_worker_inode_replacement_with_same_git_pointer_refused(tmp_path, action):
    repo, manager, info = await worker_repo(tmp_path)
    sidecar = manager.read_sidecar(info.name)
    (info.path / "unfinished").write_text("irreplaceable work")
    saved = tmp_path / "saved-worker"
    info.path.rename(saved)
    info.path.mkdir()
    (info.path / ".git").write_text((saved / ".git").read_text())
    await git(info.path, "restore", ".")
    assert await git(info.path, "status", "--porcelain") == ""
    assert await git(info.path, "branch", "--show-current") == info.branch
    manager = WorktreeManager(repo)
    with pytest.raises(WorktreeError, match="checkout identity changed"):
        if action == "integrate":
            await integrate_head(manager, info)
        elif action == "cleanup":
            await manager.cleanup_worker(info.name, discard=True)
        elif action == "reconcile":
            await manager.reconcile(info.name, recreate=True)
        else:
            await getattr(manager, action)(info.name)
    assert info.path.is_dir()
    assert (saved / "unfinished").read_text() == "irreplaceable work"
    assert manager.read_sidecar(info.name) == sidecar


async def test_integrate_rejects_parent_rewind_before_validation(tmp_path):
    repo = await make_repo(tmp_path / "repo")
    base = await git(repo, "rev-parse", "HEAD")
    (repo / "unreviewed-parent-change").write_text("parent work")
    await commit_all(repo, "parent progress")
    reviewed_parent = await git(repo, "rev-parse", "HEAD")
    manager = WorktreeManager(repo)
    info = await manager.create_worker(
        "worker", base_commit="HEAD", dest_path=repo, dest_branch="main"
    )
    (info.path / "feature").write_text("reviewed feature")
    await commit_all(info.path, "worker feature")
    candidate = await git(info.path, "rev-parse", "HEAD")
    await git(repo, "reset", "--hard", base)

    async def never_run(cmd, cwd):
        pytest.fail("stale parent review must be rejected before validation")

    with pytest.raises(WorktreeError, match="re-review"):
        await manager.integrate(
            info.name,
            reviewed_head=candidate,
            reviewed_parent_head=reviewed_parent,
            validation=["check"],
            validation_runner=never_run,
        )
    assert await git(repo, "rev-parse", "HEAD") == base
    assert not (repo / "unreviewed-parent-change").exists()


@pytest.mark.parametrize("already_merged", [False, True])
async def test_integrate_parent_advance_always_requires_rereview(tmp_path, already_merged):
    repo, manager, info = await worker_repo(tmp_path)
    reviewed_parent = await git(repo, "rev-parse", "HEAD")
    (repo / "parent-progress").write_text("parent progress")
    await commit_all(repo, "parent progress")
    target = await git(repo, "rev-parse", "HEAD")
    if already_merged:
        await manager.reconcile(info.name)
    candidate = await git(info.path, "rev-parse", "HEAD")

    async def never_run(cmd, cwd):
        pytest.fail("a mismatched parent review must never reach validation")

    with pytest.raises(WorktreeError, match="re-review"):
        await manager.integrate(
            info.name,
            reviewed_head=candidate,
            reviewed_parent_head=reviewed_parent,
            validation=["check"],
            validation_runner=never_run,
        )
    assert await git(repo, "rev-parse", "HEAD") == target
    assert await manager._ancestor(target, await git(info.path, "rev-parse", "HEAD"))
    assert manager.read_sidecar(info.name)["integrated_head"] is None
    assert (await integrate_head(manager, info)).merged


@pytest.mark.parametrize("reviewed_parent", ["HEAD", "a" * 39, "A" * 40, "g" * 64, None])
async def test_integrate_requires_exact_parent_hash(tmp_path, reviewed_parent):
    _repo, manager, info = await worker_repo(tmp_path)
    with pytest.raises(WorktreeError, match="reviewed_parent_head must be the exact full"):
        await manager.integrate(
            info.name,
            reviewed_head=await git(info.path, "rev-parse", "HEAD"),
            reviewed_parent_head=reviewed_parent,
            validation=["check"],
            validation_runner=validate_git,
        )


@pytest.mark.parametrize("replace", ["worker", "parent"])
async def test_validation_rejects_checkout_replacement_before_next_command(tmp_path, replace):
    repo = await make_repo(tmp_path / "repo")
    parent = tmp_path / "parent"
    await git(repo, "worktree", "add", "-b", "parent", str(parent))
    manager = WorktreeManager(repo)
    info = await manager.create_worker(
        "worker", base_commit="HEAD", dest_path=parent, dest_branch="parent"
    )
    (info.path / "feature").write_text("feature")
    await commit_all(info.path, "worker feature")
    candidate = await git(info.path, "rev-parse", "HEAD")
    parent_head = await git(parent, "rev-parse", "HEAD")
    sidecar = manager.read_sidecar(info.name)
    path = info.path if replace == "worker" else parent
    saved = tmp_path / "saved-checkout"
    calls = []

    async def validate(cmd, cwd):
        calls.append(cmd)
        assert cmd == "first", "must stop before validating a replacement checkout"
        (path / "unfinished").write_text("retained work")
        path.rename(saved)
        path.mkdir()
        (path / ".git").write_text((saved / ".git").read_text())
        await git(path, "restore", ".")
        return ProcResult(0, "", "")

    with pytest.raises(WorktreeError, match="checkout identity changed"):
        await manager.integrate(
            info.name,
            reviewed_head=candidate,
            reviewed_parent_head=parent_head,
            validation=["first", "second"],
            validation_runner=validate,
        )
    assert calls == ["first"]
    assert await git(parent, "rev-parse", "HEAD") == parent_head
    assert (saved / "unfinished").read_text() == "retained work"
    assert manager.read_sidecar(info.name) == sidecar


@pytest.mark.parametrize("change", ["dirty", "missing", "replaced", "branch", "detached"])
async def test_integrate_refuses_changed_parent(tmp_path, change):
    repo = await make_repo(tmp_path / "repo")
    parent = tmp_path / "parent"
    await git(repo, "worktree", "add", "-b", "parent", str(parent))
    manager = WorktreeManager(repo)
    info = await manager.create_worker(
        "worker", base_commit="HEAD", dest_path=parent, dest_branch="parent"
    )
    (info.path / "feature").write_text("feature")
    await commit_all(info.path, "feature")
    target = await git(parent, "rev-parse", "HEAD")
    if change == "dirty":
        (parent / "file.txt").write_text("uncommitted")
    elif change in {"missing", "replaced"}:
        # Keep the original inode alive so replacement cannot reuse it.
        parent.rename(tmp_path / "old-parent")
        if change == "replaced":
            parent.mkdir()
            (parent / ".git").write_text((tmp_path / "old-parent" / ".git").read_text())
            await git(parent, "restore", ".")
    elif change == "branch":
        await git(parent, "switch", "-c", "different")
    else:
        await git(parent, "switch", "--detach")
    with pytest.raises(WorktreeError):
        await integrate_head(manager, info, reviewed_parent_head=target)
    assert await git(repo, "rev-parse", "parent") == target


async def test_integrate_rejects_stale_review_and_requires_merge_rereview(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    old_head = await git(info.path, "rev-parse", "HEAD")
    (info.path / "extra").write_text("extra")
    await commit_all(info.path, "extra work")
    with pytest.raises(WorktreeError, match="re-review"):
        await manager.integrate(
            info.name,
            reviewed_head=old_head,
            reviewed_parent_head=await git(repo, "rev-parse", "HEAD"),
            validation=["check"],
            validation_runner=validate_git,
        )
    (repo / "parent").write_text("progress")
    await commit_all(repo, "parent progress")
    target = await git(repo, "rev-parse", "HEAD")
    with pytest.raises(WorktreeError, match=r"destination merged.*re-review"):
        await integrate_head(manager, info)
    assert await git(repo, "rev-parse", "HEAD") == target
    assert await git(info.path, "rev-parse", "HEAD^2") == target
    candidate = await git(info.path, "rev-parse", "HEAD")
    assert (await integrate_head(manager, info)).merged
    assert await git(repo, "rev-parse", "HEAD") == candidate
    assert manager.read_sidecar(info.name)["integrated_head"] == candidate


@pytest.mark.parametrize(
    "change",
    [
        "fail",
        "timeout",
        "worker-dirty",
        "worker-head",
        "target-head",
        "target-dirty",
        "target-switch",
        "merge-state",
    ],
)
async def test_integration_validation_cannot_change_candidate_or_target(tmp_path, change):
    repo, manager, info = await worker_repo(tmp_path)
    before = await git(repo, "rev-parse", "HEAD")
    candidate = await git(info.path, "rev-parse", "HEAD")
    calls = []

    async def validate(cmd, cwd):
        calls.append((cmd, cwd))
        if len(calls) > 1:
            return ProcResult(0, "", "")
        if change == "fail":
            return ProcResult(1, "", "failed test")
        if change == "timeout":
            return ProcResult(0, "", "timeout", timed_out=True)
        if change.startswith("worker"):
            (cwd / "mutation").write_text("validation mutation")
            if change == "worker-head":
                await commit_all(cwd, "mutation")
        elif change in {"target-head", "target-dirty"}:
            (repo / "mutation").write_text("parent mutation")
            if change == "target-head":
                await commit_all(repo, "parent mutation")
        elif change == "target-switch":
            await git(repo, "switch", "-c", "other")
        else:
            # A real --no-commit merge, with a clean index but MERGE_HEAD set.
            await git(repo, "switch", "-c", "other")
            await git(repo, *_COMMIT, "--allow-empty", "-m", "diverge")
            await git(repo, "switch", "main")
            await git(cwd, "merge", "--no-ff", "--no-commit", "other")
        return ProcResult(0, "", "")

    with pytest.raises(WorktreeError):
        await manager.integrate(
            info.name,
            reviewed_head=candidate,
            reviewed_parent_head=before,
            validation=["first", "second"],
            validation_runner=validate,
        )
    assert calls[0] == ("first", info.path)
    assert len(calls) == 1
    assert not await manager._ancestor(candidate, await git(repo, "rev-parse", "main"))
    if change != "target-head":
        assert await git(repo, "rev-parse", "main") == before
    assert manager.read_sidecar(info.name)["integrated_head"] is None


async def test_integration_empty_checks_require_explicit_human_gate(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    candidate = await git(info.path, "rev-parse", "HEAD")
    parent_head = await git(repo, "rev-parse", "HEAD")

    async def never_run(cmd, cwd):
        pytest.fail("empty validation must not call a runner")

    for checks in ([], [" "], [1]):
        with pytest.raises(WorktreeError):
            await manager.integrate(
                info.name,
                reviewed_head=candidate,
                reviewed_parent_head=parent_head,
                validation=checks,
                validation_runner=never_run,
            )
    result = await manager.integrate(
        info.name,
        reviewed_head=candidate,
        reviewed_parent_head=parent_head,
        validation=[],
        validation_runner=never_run,
        allow_unvalidated=True,
    )
    assert result.merged
    assert await git(repo, "rev-parse", "HEAD") == candidate


async def test_nested_integration_cleanup_checks_current_pinned_target(tmp_path):
    repo, manager, parent = await worker_repo(tmp_path)
    root_head = await git(repo, "rev-parse", "HEAD")
    child = await manager.create_worker(
        "child", base_commit=parent.branch, dest_path=parent.path, dest_branch=parent.branch
    )
    (child.path / "nested").write_text("nested")
    await commit_all(child.path, "nested work")
    with pytest.raises(WorktreeError, match="not integrated"):
        await manager.cleanup_worker(child.name)
    candidate = await git(child.path, "rev-parse", "HEAD")
    await integrate_head(manager, child)
    assert await git(parent.path, "rev-parse", "HEAD") == candidate
    assert await git(repo, "rev-parse", "HEAD") == root_head
    (child.path / "after").write_text("extra commit after integration")
    await commit_all(child.path, "after integration")
    with pytest.raises(WorktreeError, match="not integrated"):
        await manager.cleanup_worker(child.name)
    assert child.path.exists()
    await integrate_head(manager, child)
    sidecar = manager.read_sidecar(child.name)
    await manager.cleanup_worker(child.name)
    assert not child.path.exists()
    assert manager.read_sidecar(child.name) == sidecar
    assert (parent.path / "after").read_text() == "extra commit after integration"


async def test_cleanup_dirty_requires_explicit_discard(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    await integrate_head(manager, info)
    (info.path / "unfinished").write_text("dirty")
    with pytest.raises(WorktreeError, match="uncommitted"):
        await manager.cleanup_worker(info.name)
    await manager.cleanup_worker(info.name, discard=True)
    assert not info.path.exists()
    assert manager.read_sidecar(info.name) is not None
    assert (repo / "feature.txt").read_text() == "feature\n"


async def test_integration_lock_serializes_tasks_and_cancellation(tmp_path):
    repo, _manager, info = await worker_repo(tmp_path)
    candidate = await git(info.path, "rev-parse", "HEAD")
    parent_head = await git(repo, "rev-parse", "HEAD")
    entered = asyncio.Event()
    hold = asyncio.Event()

    async def validate(cmd, cwd):
        entered.set()
        await hold.wait()
        return await validate_git(cmd, cwd)

    async def integrate():
        return await WorktreeManager(repo).integrate(
            info.name,
            reviewed_head=candidate,
            reviewed_parent_head=parent_head,
            validation=["check"],
            validation_runner=validate,
        )

    first = asyncio.create_task(integrate())
    await asyncio.wait_for(entered.wait(), 5)
    (lock_path,) = (repo / ".git" / "lecode-integration-locks").iterdir()
    inode = lock_path.stat().st_ino
    pending = asyncio.create_task(integrate())
    await asyncio.sleep(0.15)
    assert not pending.done()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert lock_path.stat().st_ino == inode
    assert await git(repo, "rev-parse", "HEAD") != candidate
    hold.set()
    assert (await asyncio.wait_for(integrate(), 5)).merged
    assert lock_path.stat().st_ino == inode
    assert await git(repo, "rev-parse", "HEAD") == candidate


async def test_integration_lock_serializes_processes(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    candidate = await git(info.path, "rev-parse", "HEAD")
    parent_head = await git(repo, "rev-parse", "HEAD")
    script = """
import asyncio, sys
from lecode.extras.worktree import WorktreeManager
from lecode.extras.proc import ProcResult
async def validate(cmd, cwd):
    print("ready", flush=True)
    await asyncio.to_thread(sys.stdin.readline)
    return ProcResult(0, "", "")
asyncio.run(WorktreeManager(sys.argv[1]).integrate(
    "worker", reviewed_head=sys.argv[2], reviewed_parent_head=sys.argv[3],
    validation=["check"], validation_runner=validate))
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(repo),
        candidate,
        parent_head,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    task = None
    try:
        assert await asyncio.wait_for(proc.stdout.readline(), 10) == b"ready\n"
        (lock_path,) = (repo / ".git" / "lecode-integration-locks").iterdir()
        inode = lock_path.stat().st_ino
        task = asyncio.create_task(integrate_head(manager, info))
        await asyncio.sleep(0.15)
        assert not task.done()
        assert await git(repo, "rev-parse", "HEAD") != candidate
        proc.stdin.write(b"continue\n")
        await proc.stdin.drain()
        _, stderr = await asyncio.wait_for(proc.communicate(), 10)
        assert proc.returncode == 0, stderr
        with pytest.raises(WorktreeError, match="re-review"):
            await asyncio.wait_for(task, 10)
        assert (await integrate_head(manager, info)).merged
        assert lock_path.stat().st_ino == inode
        assert await git(repo, "rev-parse", "HEAD") == candidate
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_sibling_integrations_serialize_by_destination(tmp_path):
    repo, manager, first = await worker_repo(tmp_path)
    second = await manager.create_worker(
        "second", base_commit="HEAD", dest_path=repo, dest_branch="main"
    )
    (second.path / "second").write_text("second feature")
    await commit_all(second.path, "second feature")
    first_head = await git(first.path, "rev-parse", "HEAD")
    second_head = await git(second.path, "rev-parse", "HEAD")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def validate(cmd, cwd):
        entered.set()
        await release.wait()
        return await validate_git(cmd, cwd)

    first_task = asyncio.create_task(
        manager.integrate(
            first.name,
            reviewed_head=first_head,
            reviewed_parent_head=await git(repo, "rev-parse", "HEAD"),
            validation=["check"],
            validation_runner=validate,
        )
    )
    second_task = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        second_task = asyncio.create_task(integrate_head(WorktreeManager(repo), second))
        await asyncio.sleep(0.15)
        assert not second_task.done()
        release.set()
        assert (await asyncio.wait_for(first_task, 5)).merged
        with pytest.raises(WorktreeError, match="re-review"):
            await asyncio.wait_for(second_task, 5)
        assert await git(repo, "rev-parse", "HEAD") == first_head
        assert await git(second.path, "rev-parse", "HEAD^1") == second_head
        assert await git(second.path, "rev-parse", "HEAD^2") == first_head
        assert (await integrate_head(manager, second)).merged
        assert (repo / "second").read_text() == "second feature"
        assert (repo / "feature.txt").read_text() == "feature\n"
    finally:
        for task in (first_task, second_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_cleanup_rejects_switched_pinned_parent_even_if_other_branch_contains_worker(
    tmp_path,
):
    repo, manager, info = await worker_repo(tmp_path)
    await integrate_head(manager, info)
    await git(repo, "switch", "-c", "other")
    with pytest.raises(WorktreeError, match="branch changed"):
        await manager.cleanup_worker(info.name)
    assert info.path.exists()
    await git(repo, "switch", "main")
    await manager.cleanup_worker(info.name)
    assert not info.path.exists()


async def test_missing_switched_registration_is_not_recreated(tmp_path):
    repo, manager, info = await worker_repo(tmp_path)
    await git(info.path, "switch", "-c", "unrelated")
    shutil.rmtree(info.path)
    with pytest.raises(WorktreeError, match="registration has changed branch"):
        await manager.reconcile(info.name, recreate=True)
    assert "unrelated" in await git(repo, "worktree", "list", "--porcelain")


# -- /worktree /wt-merge /wt-exit commands ------------------------------------------


async def make_repo_app(tmp_path, monkeypatch):
    """make_app with tmp_path itself as a git repo."""
    await make_repo(tmp_path)
    return make_app(tmp_path, monkeypatch, [])


async def test_worktree_command_switches_cwd(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    memory = app.runtime.ctx.extras["memory"]
    memory.write_long_term("shared across checkouts")
    await app.handle_command("/worktree feat")
    expected = tmp_path / ".lecode" / "worktrees" / "feat"
    assert realpath(app.runtime.ctx.cwd) == realpath(expected)
    assert realpath(app.status.cwd) == realpath(expected)
    assert app._worktree.branch == "lecode/feat"
    assert "branch lecode/feat" in out.getvalue()
    assert app.runtime.ctx.project_root == tmp_path
    assert app.runtime.ctx.scope == "lecode/feat"
    assert app.runtime.ctx.extras["memory"] is memory
    from lecode.agent.builder import refresh_system_prompt

    assert "shared across checkouts" in refresh_system_prompt(app.runtime)


async def test_worktree_command_twice_refused(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/worktree feat")
    await app.handle_command("/worktree other")
    assert "already in worktree 'feat'" in out.getvalue()


async def test_worktree_command_not_a_repo(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/worktree feat")
    assert "not a git repository" in out.getvalue()


async def test_wt_commands_require_worktree(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/wt-merge")
    await app.handle_command("/wt-exit")
    rendered = out.getvalue()
    assert rendered.count("not in a worktree") == 2


async def test_wt_exit_restores_cwd(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/worktree feat")
    await app.handle_command("/wt-exit")
    assert realpath(app.runtime.ctx.cwd) == realpath(tmp_path)
    assert app._worktree is None
    assert "left worktree 'feat'" in out.getvalue()
    assert app.runtime.ctx.project_root == tmp_path
    assert app.runtime.ctx.scope == str(tmp_path)


async def test_wt_merge_command(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/worktree feat")
    wt_path = app._worktree.path
    (wt_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    await commit_all(wt_path, "add feature")
    await app.handle_command("/wt-merge")
    assert "merged lecode/feat into main" in out.getvalue()
    assert (tmp_path / "feature.txt").read_text() == "feature\n"


async def test_wt_merge_command_reports_conflicts(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/worktree feat")
    wt_path = app._worktree.path
    (wt_path / "file.txt").write_text("worktree change\n", encoding="utf-8")
    await commit_all(wt_path, "worktree edit")
    (tmp_path / "file.txt").write_text("main change\n", encoding="utf-8")
    await commit_all(tmp_path, "main edit")
    await app.handle_command("/wt-merge")
    rendered = out.getvalue()
    assert "merge conflicts" in rendered
    assert "file.txt" in rendered


async def test_wt_exit_dirty_refused(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/worktree feat")
    (app._worktree.path / "dirty.txt").write_text("x\n", encoding="utf-8")
    await app.handle_command("/wt-exit")
    assert "uncommitted changes" in out.getvalue()
    await app.handle_command("/wt-exit --force")
    assert "left worktree 'feat'" in out.getvalue()
    assert realpath(app.runtime.ctx.cwd) == realpath(tmp_path)


# -- the --worktree CLI flag ----------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolated cwd + config dir; deps check, provider, prompt and TuiApp faked."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lecode.cli.check_dependencies", lambda: None)
    monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: object())
    FakeTui.instances = []
    monkeypatch.setattr("lecode.cli.TuiApp", FakeTui)
    return tmp_path


def test_cli_worktree_flag_switches_cwd(cli_env, monkeypatch):
    make_repo_sync(cli_env)
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("wt-session"))
    result = runner.invoke(cli_app, ["--worktree", "feat"])
    assert result.exit_code == 0, result.output
    expected = cli_env / ".lecode" / "worktrees" / "feat"
    assert expected.is_dir()
    session = FakeTui.instances[0].session
    assert realpath(session.meta.cwd) == realpath(expected)
    ctx = FakeTui.instances[0].runtime.ctx
    assert ctx.project_root == cli_env
    assert ctx.scope == "lecode/feat"
    assert "worktree kept at" in result.output
    assert "git merge lecode/feat" in result.output


def test_cli_worktree_flag_not_a_repo(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("x"))
    result = runner.invoke(cli_app, ["--worktree", "feat"])
    assert result.exit_code == 2
    assert "not a git repository" in result.output
