"""Git worktree isolation for lecode sessions.

Layout: worktrees live under ``<repo>/.lecode/worktrees/<name>`` with a
branch ``lecode/<name>`` created from the current HEAD. The worktree root
is appended to the repo's local ``.git/info/exclude`` (best effort) so the
main checkout's status stays clean. All git access goes through
:func:`~lecode.extras.proc.run_proc`; every failure raises
:class:`WorktreeError` — never a traceback.

Merge semantics: ``merge_back`` runs a plain ``git merge --no-edit`` of the
worktree branch into the main checkout's current branch. On conflict the
merge is left in progress (standard git flow) and the conflicted paths are
reported — no auto-resolution, ``git merge --abort`` stays available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lecode.extras.proc import run_proc

#: Per-git-command timeout.
GIT_TIMEOUT_S = 30.0

#: Worktree directory, relative to the repo root.
WORKTREE_ROOT = ".lecode/worktrees"

#: Branch namespace for worktree branches.
BRANCH_PREFIX = "lecode/"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorktreeError(Exception):
    """A git or worktree operation failed cleanly."""


@dataclass(frozen=True)
class WorktreeInfo:
    """One lecode worktree: directory + backing branch."""

    name: str
    path: Path
    branch: str


@dataclass(frozen=True)
class MergeResult:
    """The outcome of a merge-back: conflicts listed, never auto-resolved."""

    merged: bool
    conflicts: list[str]
    message: str = ""


@dataclass(frozen=True)
class WorktreeStatus:
    """Ahead/behind vs the main checkout's branch, plus dirty state."""

    info: WorktreeInfo
    ahead: int
    behind: int
    dirty: bool


class WorktreeManager:
    """Worktree operations over one repository root."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root)

    @classmethod
    async def discover(cls, cwd: Path | str) -> WorktreeManager:
        """The manager for the git repository containing ``cwd``."""
        result = await run_proc(
            ["git", "rev-parse", "--show-toplevel"], cwd=cwd, timeout=GIT_TIMEOUT_S
        )
        if result.exit_code != 0:
            raise WorktreeError(f"not a git repository: {cwd}")
        return cls(Path(result.stdout.strip()))

    async def _git(self, *args: str, cwd: Path | None = None) -> str:
        """Run git, returning stdout; non-zero exits become WorktreeError."""
        result = await run_proc(["git", *args], cwd=cwd or self.repo_root, timeout=GIT_TIMEOUT_S)
        if result.exit_code != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise WorktreeError(f"git {args[0]} failed: {detail or f'exit {result.exit_code}'}")
        return result.stdout.strip()

    def _info(self, name: str) -> WorktreeInfo:
        return WorktreeInfo(
            name=name,
            path=self.repo_root / WORKTREE_ROOT / name,
            branch=f"{BRANCH_PREFIX}{name}",
        )

    def _require(self, name: str) -> WorktreeInfo:
        info = self._info(name)
        if not info.path.is_dir():
            raise WorktreeError(f"no such worktree: {name} (expected at {info.path})")
        return info

    def _exclude_worktree_root(self) -> None:
        """Best-effort: keep the worktree dir out of the main checkout's status."""
        try:
            exclude = self.repo_root / ".git" / "info" / "exclude"
            if not exclude.parent.is_dir():
                return  # .git is a file (linked worktree) — nothing to do
            existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
            line = f"/{WORKTREE_ROOT}/"
            if line not in existing:
                with exclude.open("a", encoding="utf-8") as f:
                    f.write(f"{line}\n")
        except OSError:
            pass

    async def _current_branch(self) -> str:
        return await self._git("rev-parse", "--abbrev-ref", "HEAD")

    async def create(self, name: str) -> WorktreeInfo:
        """Create ``.lecode/worktrees/<name>`` on a fresh ``lecode/<name>`` branch."""
        if not _NAME_RE.match(name):
            raise WorktreeError(f"invalid worktree name: {name!r} (letters, digits, . _ -)")
        info = self._info(name)
        if info.path.exists():
            raise WorktreeError(f"worktree already exists: {info.path}")
        branch_ref = await run_proc(
            ["git", "show-ref", "--verify", f"refs/heads/{info.branch}"],
            cwd=self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        if branch_ref.exit_code == 0:
            raise WorktreeError(f"branch already exists: {info.branch}")
        self._exclude_worktree_root()
        await self._git("worktree", "add", "-b", info.branch, str(info.path))
        return info

    async def status(self, name: str) -> WorktreeStatus:
        """Ahead/behind vs the main checkout's branch, plus dirty state."""
        info = self._require(name)
        porcelain = await self._git("status", "--porcelain", cwd=info.path)
        base = await self._current_branch()
        counts = await self._git("rev-list", "--left-right", "--count", f"{base}...{info.branch}")
        behind, ahead = counts.split()
        return WorktreeStatus(
            info=info, ahead=int(ahead), behind=int(behind), dirty=bool(porcelain)
        )

    async def merge_back(self, name: str) -> MergeResult:
        """Merge ``lecode/<name>`` into the main checkout's current branch.

        On conflict the merge is left in progress and the conflicted paths
        are returned; nothing is auto-resolved or aborted.
        """
        info = self._require(name)
        base = await self._current_branch()
        result = await run_proc(
            ["git", "merge", "--no-edit", info.branch],
            cwd=self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        if result.exit_code == 0:
            return MergeResult(
                merged=True, conflicts=[], message=f"merged {info.branch} into {base}"
            )
        conflicts = await self._conflicted_files()
        if conflicts:
            return MergeResult(
                merged=False,
                conflicts=conflicts,
                message="merge conflicts — resolve them and commit, or: git merge --abort",
            )
        return MergeResult(
            merged=False, conflicts=[], message=result.stderr.strip() or "merge failed"
        )

    async def _conflicted_files(self) -> list[str]:
        result = await run_proc(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]

    async def exit_worktree(
        self, name: str, *, delete_branch: bool = False, force: bool = False
    ) -> WorktreeInfo:
        """Remove the worktree (+ optionally the branch).

        A worktree with uncommitted changes refuses unless ``force``.
        """
        info = self._require(name)
        if not force:
            porcelain = await self._git("status", "--porcelain", cwd=info.path)
            if porcelain:
                raise WorktreeError(
                    f"worktree '{name}' has uncommitted changes (use force to discard)"
                )
        args = ["worktree", "remove", *(["--force"] if force else []), str(info.path)]
        await self._git(*args)
        if delete_branch:
            await self._git("branch", "-D", info.branch)
        return info
