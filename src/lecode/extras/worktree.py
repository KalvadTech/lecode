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

Worker worktrees: ``create_worker`` pins a base commit and a destination
(path + branch) in a sidecar under ``.lecode/worktrees/<name>.json``;
``integrate`` validates the worker tree and merges it into that pinned
destination. ``discover`` returns the *main* repository root even when
called from inside a linked worktree, so worktrees and sidecars agree
across restarts.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import json
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lecode.extras.proc import run_proc

#: Per-git-command timeout.
GIT_TIMEOUT_S = 30.0

#: Validation output kept in the returned :class:`IntegrationResult`.
_VALIDATION_OUTPUT_LIMIT = 4000

#: Worktree directory, relative to the repo root.
WORKTREE_ROOT = ".lecode/worktrees"

#: Branch namespace for worktree branches.
BRANCH_PREFIX = "lecode/"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _clip_validation(text: str) -> str:
    """Keep validation output bounded; the tail usually holds the failure."""
    if len(text) <= _VALIDATION_OUTPUT_LIMIT:
        return text
    return "[… output clipped …]\n" + text[-_VALIDATION_OUTPUT_LIMIT:]


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


@dataclass(frozen=True)
class WorktreeInspection:
    """A worker worktree's on-disk state plus its sidecar, if any."""

    present: bool
    info: WorktreeInfo
    dirty: bool
    merge_in_progress: bool
    sidecar: dict[str, Any] | None


@dataclass(frozen=True)
class IntegrationResult:
    """The outcome of integrating a worker into its pinned destination.

    ``status`` is one of ``integrated``, ``paused``, ``blocked``,
    ``conflict``, or ``validation_failed``.
    """

    status: str
    detail: str
    conflicts: list[str]
    validation_output: str


class WorktreeManager:
    """Worktree operations over one repository root."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()

    @classmethod
    async def discover(cls, cwd: Path | str) -> WorktreeManager:
        """The manager for the *main* repository containing ``cwd``.

        Inside a linked worktree ``--show-toplevel`` names the worktree, not
        the repository that owns it; ``--git-common-dir`` names the main
        ``.git``, whose parent is the root we want.
        """
        start = Path(cwd).expanduser().resolve()
        common = await run_proc(
            ["git", "rev-parse", "--git-common-dir"], cwd=cwd, timeout=GIT_TIMEOUT_S
        )
        if common.exit_code == 0:
            git_dir = Path(common.stdout.strip())
            if not git_dir.is_absolute():
                git_dir = start / git_dir
            git_dir = git_dir.resolve()
            if git_dir.name == ".git":
                return cls(git_dir.parent)
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

    async def _common_dir(self, cwd: Path) -> Path:
        """Return a resolved common git directory for a working tree."""
        result = await run_proc(
            ["git", "rev-parse", "--git-common-dir"], cwd=cwd, timeout=GIT_TIMEOUT_S
        )
        if result.exit_code != 0:
            raise WorktreeError(f"not a git repository: {cwd}")
        common = Path(result.stdout.strip())
        return (common if common.is_absolute() else cwd / common).resolve()

    async def _commit(self, revision: str, *, cwd: Path) -> str:
        """Resolve one revision to a commit, without accepting arbitrary refs later."""
        return await self._git("rev-parse", "--verify", f"{revision}^{{commit}}", cwd=cwd)

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

    async def _check_new(self, name: str) -> WorktreeInfo:
        """Validate ``name`` and that neither its directory nor branch exist."""
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
        return info

    async def create(self, name: str) -> WorktreeInfo:
        """Create ``.lecode/worktrees/<name>`` on a fresh ``lecode/<name>`` branch."""
        info = await self._check_new(name)
        self._exclude_worktree_root()
        await self._git("worktree", "add", "-b", info.branch, str(info.path))
        return info

    def _sidecar_path(self, name: str) -> Path:
        return self.repo_root / WORKTREE_ROOT / f"{name}.json"

    def write_sidecar(self, name: str, data: dict[str, Any]) -> None:
        """Atomically write ``.lecode/worktrees/<name>.json``."""
        if not _NAME_RE.match(name):
            raise WorktreeError(f"invalid worktree name: {name!r} (letters, digits, . _ -)")
        path = self._sidecar_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def read_sidecar(self, name: str) -> dict[str, Any] | None:
        """The worker's sidecar, or ``None`` when it was never written."""
        if not _NAME_RE.match(name):
            raise WorktreeError(f"invalid worktree name: {name!r} (letters, digits, . _ -)")
        try:
            text = self._sidecar_path(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            raise WorktreeError(f"cannot read sidecar for '{name}': {e}") from e
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise WorktreeError(f"invalid sidecar for '{name}': {e}") from e
        if not isinstance(data, dict):
            raise WorktreeError(f"invalid sidecar for '{name}': expected an object")
        return data

    def _validated_sidecar(self, name: str, info: WorktreeInfo) -> dict[str, Any] | None:
        """Return a sidecar only when its fixed identity fields are safe to use."""
        data = self.read_sidecar(name)
        if data is None:
            return None
        expected_path = info.path.resolve()
        try:
            path = Path(data["path"])
            dest_path = Path(data["dest_path"])
            common_dir = Path(data["dest_common_dir"])
            valid = (
                data["name"] == name
                and isinstance(data["path"], str)
                and path.is_absolute()
                and path == path.resolve() == expected_path
                and data["branch"] == info.branch
                and isinstance(data["base_commit"], str)
                and isinstance(data["dest_path"], str)
                and dest_path.is_absolute()
                and dest_path == dest_path.resolve()
                and isinstance(data["dest_branch"], str)
                and bool(data["dest_branch"])
                and isinstance(data["dest_common_dir"], str)
                and common_dir.is_absolute()
                and common_dir == common_dir.resolve()
                and (
                    data.get("integrated_at") is None or isinstance(data.get("integrated_at"), str)
                )
                and (
                    data.get("integrated_head") is None
                    or isinstance(data.get("integrated_head"), str)
                )
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise WorktreeError(f"invalid sidecar for '{name}': unsafe worker destination")
        return data

    async def _destination_path(
        self, dest_path: Path | str, dest_branch: str, *, expected_common: Path
    ) -> Path:
        """Normalize and verify a destination is this repository's checkout."""
        if not isinstance(dest_branch, str) or not dest_branch:
            raise WorktreeError("invalid destination branch")
        branch = await run_proc(
            ["git", "check-ref-format", "--branch", dest_branch],
            cwd=self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        if branch.exit_code != 0:
            raise WorktreeError(f"invalid destination branch: {dest_branch!r}")
        path = Path(dest_path).expanduser().resolve()
        if not path.is_dir():
            raise WorktreeError(f"destination missing: {path}")
        root = await run_proc(
            ["git", "rev-parse", "--show-toplevel"], cwd=path, timeout=GIT_TIMEOUT_S
        )
        if root.exit_code != 0:
            raise WorktreeError(f"destination is not a git repository: {path}")
        destination = Path(root.stdout.strip()).resolve()
        if await self._common_dir(destination) != expected_common:
            raise WorktreeError(f"destination is not in this repository: {destination}")
        return destination

    async def create_worker(
        self,
        name: str,
        *,
        base_commit: str,
        dest_path: Path | str,
        dest_branch: str,
    ) -> WorktreeInfo:
        """Create a worker worktree pinned to ``base_commit`` and a destination.

        The branch is ``lecode/<name>``; the pinned destination and base are
        recorded in the sidecar so integration resumes deterministically.
        """
        info = await self._check_new(name)
        common_dir = await self._common_dir(self.repo_root)
        destination = await self._destination_path(
            dest_path, dest_branch, expected_common=common_dir
        )
        base = await self._commit(base_commit, cwd=self.repo_root)
        self._exclude_worktree_root()
        await self._git("worktree", "add", "-b", info.branch, str(info.path), base)
        self.write_sidecar(
            name,
            {
                "name": name,
                "path": str(info.path),
                "branch": info.branch,
                "base_commit": base,
                "dest_path": str(destination),
                "dest_branch": dest_branch,
                "dest_common_dir": str(common_dir),
                "integrated_at": None,
                "integrated_head": None,
            },
        )
        return info

    async def attach(self, name: str) -> WorktreeInfo:
        """Resume an existing worker: directory and branch must both exist."""
        info = self._info(name)
        if not info.path.is_dir():
            raise WorktreeError(f"no such worktree: {name} (expected at {info.path})")
        branch_ref = await run_proc(
            ["git", "show-ref", "--verify", f"refs/heads/{info.branch}"],
            cwd=self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        if branch_ref.exit_code != 0:
            raise WorktreeError(f"worktree '{name}' has no branch {info.branch}")
        return info

    async def inspect(self, name: str) -> WorktreeInspection:
        """Best-effort worker state; absent worktrees never raise."""
        info = self._info(name)
        sidecar = self.read_sidecar(name)
        if not info.path.is_dir():
            return WorktreeInspection(
                present=False, info=info, dirty=False, merge_in_progress=False, sidecar=sidecar
            )
        porcelain = await self._git("status", "--porcelain", cwd=info.path)
        merge_head = await run_proc(
            ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"],
            cwd=info.path,
            timeout=GIT_TIMEOUT_S,
        )
        return WorktreeInspection(
            present=True,
            info=info,
            dirty=bool(porcelain),
            merge_in_progress=merge_head.exit_code == 0,
            sidecar=sidecar,
        )

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

    async def _conflicted_files(self, cwd: Path | None = None) -> list[str]:
        result = await run_proc(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=cwd or self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]

    async def _destination_problem(
        self, dest_path: Path, dest_branch: str, common_dir: Path
    ) -> str | None:
        """Check the pinned checkout without mutating it."""
        if not dest_path.is_dir():
            raise WorktreeError(f"destination missing: {dest_path}")
        root = await run_proc(
            ["git", "rev-parse", "--show-toplevel"], cwd=dest_path, timeout=GIT_TIMEOUT_S
        )
        if root.exit_code != 0:
            raise WorktreeError(f"destination is not a git repository: {dest_path}")
        if Path(root.stdout.strip()).resolve() != dest_path:
            raise WorktreeError(f"destination path was replaced: {dest_path}")
        if await self._common_dir(dest_path) != common_dir:
            raise WorktreeError(f"destination repository was replaced: {dest_path}")
        ref = await run_proc(
            ["git", "show-ref", "--verify", f"refs/heads/{dest_branch}"],
            cwd=dest_path,
            timeout=GIT_TIMEOUT_S,
        )
        if ref.exit_code != 0:
            raise WorktreeError(f"destination branch missing: {dest_branch}")
        branch = await self._git("rev-parse", "--abbrev-ref", "HEAD", cwd=dest_path)
        if branch != dest_branch:
            return f"destination is on '{branch}', expected '{dest_branch}'"
        if await self._git("status", "--porcelain", cwd=dest_path):
            return "destination has uncommitted changes"
        return None

    async def _worker_state(self, info: WorktreeInfo, common_dir: Path) -> tuple[str, bool]:
        """Return the worker HEAD and dirty state after proving its identity."""
        if not info.path.is_dir():
            raise WorktreeError(f"worker path missing: {info.path}")
        if await self._common_dir(info.path) != common_dir:
            raise WorktreeError(f"worker repository was replaced: {info.path}")
        ref = await run_proc(
            ["git", "show-ref", "--verify", f"refs/heads/{info.branch}"],
            cwd=info.path,
            timeout=GIT_TIMEOUT_S,
        )
        if ref.exit_code != 0:
            raise WorktreeError(f"worker branch missing: {info.branch}")
        branch = await self._git("rev-parse", "--abbrev-ref", "HEAD", cwd=info.path)
        if branch != info.branch:
            raise WorktreeError(
                f"worker branch changed: expected '{info.branch}', found '{branch}'"
            )
        return (
            await self._commit("HEAD", cwd=info.path),
            bool(await self._git("status", "--porcelain", cwd=info.path)),
        )

    @asynccontextmanager
    async def _integration_lock(self, common_dir: Path, dest_path: Path, dest_branch: str):
        """Hold a persistent per-destination flock for the whole integration."""
        if not common_dir.is_dir():
            raise WorktreeError(f"destination git directory missing: {common_dir}")
        digest = hashlib.sha256(f"{dest_path}\0{dest_branch}".encode()).hexdigest()
        lock = (common_dir / f"lecode-integrate-{digest}.lock").open("a+")
        try:
            await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    async def _is_ancestor(self, ancestor: str, descendant: str, *, cwd: Path) -> bool:
        result = await run_proc(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=cwd,
            timeout=GIT_TIMEOUT_S,
        )
        return result.exit_code == 0

    async def integrate(
        self,
        name: str,
        *,
        validation: list[str],
        validation_runner: Callable[[str, Path], Any] | None = None,
        allow_unvalidated: bool = False,
        reviewed_head: str | None = None,
    ) -> IntegrationResult:
        """Validate a pinned candidate, then fast-forward its destination only."""
        if not validation and not allow_unvalidated:
            return IntegrationResult("blocked", "validation required", [], "")
        if validation and validation_runner is None:
            raise WorktreeError("validation runner required")
        if any(not isinstance(cmd, str) or not cmd for cmd in validation):
            raise WorktreeError("validation commands must be nonempty strings")

        info = self._require(name)
        sidecar = self._validated_sidecar(name, info)
        if sidecar is None:
            return IntegrationResult("paused", "no destination recorded", [], "")
        dest_path = Path(sidecar["dest_path"])
        dest_branch = sidecar["dest_branch"]
        common_dir = Path(sidecar["dest_common_dir"])
        if common_dir != await self._common_dir(self.repo_root):
            raise WorktreeError("worker sidecar belongs to a different repository")

        async with self._integration_lock(common_dir, dest_path, dest_branch):
            # Re-read after acquiring the stable lock so a replaced sidecar cannot retarget us.
            if self._validated_sidecar(name, info) != sidecar:
                raise WorktreeError(f"sidecar changed while integrating '{name}'")
            problem = await self._destination_problem(dest_path, dest_branch, common_dir)
            if problem is not None:
                return IntegrationResult("paused", problem, [], "")
            worker_head, worker_dirty = await self._worker_state(info, common_dir)
            if worker_dirty:
                return IntegrationResult("blocked", "uncommitted changes", [], "")
            if reviewed_head is not None and worker_head != reviewed_head:
                raise WorktreeError("worker head differs from reviewed head")
            base_commit = await self._commit(sidecar["base_commit"], cwd=info.path)
            if not await self._is_ancestor(base_commit, worker_head, cwd=info.path):
                raise WorktreeError("worker branch no longer descends from its pinned base")
            dest_head = await self._commit("HEAD", cwd=dest_path)

            if not await self._is_ancestor(dest_head, worker_head, cwd=info.path):
                merge = await run_proc(
                    ["git", "merge", "--no-edit", dest_head], cwd=info.path, timeout=GIT_TIMEOUT_S
                )
                if merge.exit_code != 0:
                    conflicts = await self._conflicted_files(cwd=info.path)
                    if conflicts:
                        return IntegrationResult(
                            "conflict",
                            f"conflicts merging {dest_branch} into {info.branch}",
                            conflicts,
                            "",
                        )
                    return IntegrationResult(
                        "blocked", merge.stderr.strip() or "merge failed", [], ""
                    )
            candidate, worker_dirty = await self._worker_state(info, common_dir)
            if worker_dirty:
                return IntegrationResult("blocked", "uncommitted changes", [], "")

            validation_output = ""
            for cmd in validation:
                result = validation_runner(cmd, info.path)  # type: ignore[misc]
                if inspect.isawaitable(result):
                    result = await result
                try:
                    exit_code, output = result
                except (TypeError, ValueError) as error:
                    raise WorktreeError(
                        "validation runner must return (exit_code, output)"
                    ) from error
                if not isinstance(exit_code, int) or not isinstance(output, str):
                    raise WorktreeError("validation runner must return (int, str)")
                validation_output += output
                if exit_code != 0:
                    return IntegrationResult(
                        "validation_failed",
                        f"validation failed: {cmd}",
                        [],
                        _clip_validation(validation_output),
                    )
            validation_output = _clip_validation(validation_output)

            problem = await self._destination_problem(dest_path, dest_branch, common_dir)
            if problem is not None:
                return IntegrationResult("paused", problem, [], validation_output)
            if await self._commit("HEAD", cwd=dest_path) != dest_head:
                return IntegrationResult(
                    "paused", "destination changed during validation", [], validation_output
                )
            current_worker, worker_dirty = await self._worker_state(info, common_dir)
            if worker_dirty or current_worker != candidate:
                return IntegrationResult(
                    "blocked", "worker changed during validation", [], validation_output
                )
            if not await self._is_ancestor(dest_head, candidate, cwd=info.path):
                raise WorktreeError("validated candidate does not contain the pinned destination")

            merge = await run_proc(
                ["git", "merge", "--ff-only", candidate], cwd=dest_path, timeout=GIT_TIMEOUT_S
            )
            if merge.exit_code != 0:
                return IntegrationResult(
                    "blocked", merge.stderr.strip() or "fast-forward failed", [], validation_output
                )
            integrated_head = await self._commit("HEAD", cwd=dest_path)
            if integrated_head != candidate:
                return IntegrationResult(
                    "blocked", "destination changed while fast-forwarding", [], validation_output
                )
            sidecar["integrated_at"] = datetime.now(UTC).isoformat()
            sidecar["integrated_head"] = integrated_head
            self.write_sidecar(name, sidecar)
            return IntegrationResult(
                "integrated", f"fast-forwarded {dest_branch} to {candidate}", [], validation_output
            )

    async def cleanup_worker(self, name: str, *, discard: bool = False) -> WorktreeInfo:
        """Remove an integrated worker, or explicitly discard one."""
        info = self._require(name)
        if discard:
            await self._git("worktree", "remove", "--force", str(info.path))
            await self._git("branch", "-D", info.branch)
            return info

        sidecar = self._validated_sidecar(name, info)
        if sidecar is None or sidecar.get("integrated_at") is None:
            raise WorktreeError(
                f"worktree '{name}' is not integrated (use discard=True to discard)"
            )
        dest_path = Path(sidecar["dest_path"])
        dest_branch = sidecar["dest_branch"]
        common_dir = Path(sidecar["dest_common_dir"])
        problem = await self._destination_problem(dest_path, dest_branch, common_dir)
        if problem is not None:
            raise WorktreeError(problem)
        _worker_head, worker_dirty = await self._worker_state(info, common_dir)
        if worker_dirty:
            raise WorktreeError(
                f"worktree '{name}' has uncommitted changes (use discard=True to discard)"
            )
        dest_head = await self._commit("HEAD", cwd=dest_path)
        if not await self._is_ancestor(info.branch, dest_head, cwd=dest_path):
            raise WorktreeError(
                f"worktree '{name}' is not fully integrated (use discard=True to discard)"
            )
        await self._git("worktree", "remove", str(info.path))
        await self._git("branch", "-d", info.branch, cwd=dest_path)
        return info

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
