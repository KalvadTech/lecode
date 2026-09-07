"""Tests for git worktree isolation: the manager, the commands, --worktree.

All git operations happen in throwaway repos under tmp_path; commits use
``-c user.email/name`` flags and repos get a local identity so no global
git config is needed.
"""

from __future__ import annotations

import subprocess
from os.path import realpath

import pytest
from tests.test_tui_app import FakeTui, _name_prompt, make_app
from typer.testing import CliRunner

from lecode.cli import app as cli_app
from lecode.extras.proc import run_proc
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


# -- /worktree /wt-merge /wt-exit commands ------------------------------------------


async def make_repo_app(tmp_path, monkeypatch):
    """make_app with tmp_path itself as a git repo."""
    await make_repo(tmp_path)
    return make_app(tmp_path, monkeypatch, [])


async def test_worktree_command_switches_cwd(tmp_path, monkeypatch):
    app, _, out = await make_repo_app(tmp_path, monkeypatch)
    await app.handle_command("/worktree feat")
    expected = tmp_path / ".lecode" / "worktrees" / "feat"
    assert realpath(app.runtime.ctx.cwd) == realpath(expected)
    assert realpath(app.status.cwd) == realpath(expected)
    assert app._worktree.branch == "lecode/feat"
    assert "branch lecode/feat" in out.getvalue()


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
    assert "worktree kept at" in result.output
    assert "git merge lecode/feat" in result.output


def test_cli_worktree_flag_not_a_repo(cli_env, monkeypatch):
    monkeypatch.setattr("lecode.cli.prompt_session_name", _name_prompt("x"))
    result = runner.invoke(cli_app, ["--worktree", "feat"])
    assert result.exit_code == 2
    assert "not a git repository" in result.output
