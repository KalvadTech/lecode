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

Worker worktrees pin their immediate parent in a durable sidecar. Reconcile
merges committed parent progress only into clean workers. Integration requires
exact reviewed worker and parent commits and caller-supplied, policy-checked validation.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lecode.extras.proc import ProcResult, run_proc

type ValidationRunner = Callable[[str, Path], Awaitable[ProcResult]]

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


@dataclass(frozen=True)
class WorktreeInspection:
    """A worker worktree's on-disk state plus its sidecar, if any."""

    present: bool
    info: WorktreeInfo
    dirty: bool
    merge_in_progress: bool
    sidecar: dict[str, Any] | None


class WorktreeManager:
    """Worktree operations over one repository root."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()

    @classmethod
    async def discover(cls, cwd: Path | str) -> WorktreeManager:
        """Find the checkout, or the owner of a managed worktree's sidecars.

        An unrelated linked checkout remains its own root, never the main
        checkout merely because it shares a common Git directory.
        """
        start = Path(cwd).expanduser().resolve()
        result = await run_proc(
            ["git", "rev-parse", "--show-toplevel"], cwd=start, timeout=GIT_TIMEOUT_S
        )
        if result.exit_code != 0:
            raise WorktreeError(f"not a git repository: {cwd}")
        root = Path(result.stdout.strip()).resolve()
        manager = cls(root)
        if root.parent.name == "worktrees" and root.parent.parent.name == ".lecode":
            owner = root.parents[2]
            branch = await manager._git("rev-parse", "--abbrev-ref", "HEAD")
            managed = (owner / WORKTREE_ROOT / f"{root.name}.json").is_file()
            if (managed or branch == f"{BRANCH_PREFIX}{root.name}") and (
                await manager._common_dir(owner) == await manager._common_dir(root)
            ):
                manager = cls(owner)
        return manager

    async def _git(self, *args: str, cwd: Path | None = None) -> str:
        """Run git, returning stdout; non-zero exits become WorktreeError."""
        try:
            result = await run_proc(
                ["git", *args], cwd=cwd or self.repo_root, timeout=GIT_TIMEOUT_S
            )
        except OSError as error:
            raise WorktreeError(f"cannot run git at {cwd or self.repo_root}: {error}") from error
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

    @staticmethod
    def _identity(path: Path) -> list[int]:
        try:
            stat = path.stat()
        except OSError as error:
            raise WorktreeError(f"cannot identify checkout: {path}: {error}") from error
        return [stat.st_dev, stat.st_ino]

    async def _commit(self, revision: str, *, cwd: Path) -> str:
        """Resolve one revision to a commit, without accepting arbitrary refs later."""
        return await self._git(
            "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}", cwd=cwd
        )

    def _info(self, name: str) -> WorktreeInfo:
        if not _NAME_RE.fullmatch(name):
            raise WorktreeError(f"invalid worktree name: {name!r} (letters, digits, . _ -)")
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

    async def _exclude_worktree_root(self) -> None:
        """Best-effort: keep the worktree dir out of the main checkout's status."""
        try:
            exclude = await self._common_dir(self.repo_root) / "info" / "exclude"
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
        if self._sidecar_path(name).exists():
            raise WorktreeError(f"worker sidecar already exists: {name}; use reconcile")
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
        await self._exclude_worktree_root()
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
                and bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", data["base_commit"]))
                and isinstance(data["dest_path"], str)
                and dest_path.is_absolute()
                and dest_path == dest_path.resolve()
                and isinstance(data["dest_branch"], str)
                and bool(data["dest_branch"])
                and isinstance(data["dest_common_dir"], str)
                and common_dir.is_absolute()
                and common_dir == common_dir.resolve()
                and all(
                    key not in data
                    or (
                        isinstance(data[key], list)
                        and len(data[key]) == 2
                        and all(type(n) is int for n in data[key])
                    )
                    for key in ("repo_identity", "dest_identity", "checkout_identity")
                )
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
        actual_branch = await self._git("symbolic-ref", "-q", "HEAD", cwd=destination)
        if actual_branch != f"refs/heads/{dest_branch}":
            raise WorktreeError(f"destination branch changed: expected {dest_branch}")
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
        await self._exclude_worktree_root()
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
                "repo_identity": self._identity(common_dir),
                "dest_identity": self._identity(destination),
                "checkout_identity": self._identity(info.path),
                "integrated_at": None,
                "integrated_head": None,
            },
        )
        return info

    def _worker_data(self, name: str) -> tuple[WorktreeInfo, dict[str, Any]]:
        info = self._info(name)
        data = self._validated_sidecar(name, info)
        if data is None:
            raise WorktreeError(f"worker '{name}' has no pinned sidecar")
        if data["dest_branch"] == info.branch or Path(data["dest_path"]) == info.path:
            raise WorktreeError("worker cannot be its own destination")
        return info, data

    @asynccontextmanager
    async def _worker_lock(self, name: str) -> AsyncIterator[tuple[WorktreeInfo, dict[str, Any]]]:
        """One stable inode per canonical repository + destination branch.

        Nonblocking flock also serializes separate opens within this process.
        Polling is cancellable, unlike a blocking flock in a background thread.
        """
        info, data = self._worker_data(name)
        common = await self._common_dir(self.repo_root)
        if common != Path(data["dest_common_dir"]):
            raise WorktreeError("worker repository identity changed")
        if "repo_identity" in data and self._identity(common) != data["repo_identity"]:
            raise WorktreeError("worker repository identity changed")
        key = hashlib.sha256(data["dest_branch"].encode()).hexdigest()
        locks = common / "lecode-integration-locks"
        try:
            locks.mkdir(exist_ok=True)
            lock = (locks / f"{key}.lock").open("a")
        except OSError as error:
            raise WorktreeError(f"cannot lock worker destination: {error}") from error
        with lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
            try:
                _, current = self._worker_data(name)
                if any(
                    current.get(k) != data.get(k)
                    for k in (
                        "path",
                        "branch",
                        "base_commit",
                        "dest_path",
                        "dest_branch",
                        "dest_common_dir",
                        "repo_identity",
                        "dest_identity",
                        "checkout_identity",
                    )
                ):
                    raise WorktreeError("worker sidecar changed while waiting for destination")
                yield info, current
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                # Never unlink: waiters must continue locking this same inode.

    async def _registration(self, path: Path) -> dict[str, str] | None:
        listing = await self._git("worktree", "list", "--porcelain", "-z")
        for entry in listing.split("\0\0"):
            fields = dict(field.partition(" ")[::2] for field in entry.split("\0") if field)
            if fields.get("worktree") == str(path):
                return fields
        return None

    async def _checkout(
        self, path: Path, branch: str, common: Path, identity: list[int] | None = None
    ) -> None:
        """Reject missing, replaced, detached, switched, or unregistered checkouts."""
        if not path.is_dir() or path.is_symlink():
            raise WorktreeError(f"checkout missing or replaced: {path}")
        if identity is not None and self._identity(path) != identity:
            raise WorktreeError(f"checkout identity changed: {path}")
        root = Path(await self._git("rev-parse", "--show-toplevel", cwd=path)).resolve()
        if root != path or await self._common_dir(path) != common:
            raise WorktreeError(f"checkout identity changed: {path}")
        actual = await self._git("symbolic-ref", "-q", "HEAD", cwd=path)
        if actual != f"refs/heads/{branch}":
            raise WorktreeError(f"checkout branch changed: expected {branch} at {path}")
        registration = await self._registration(path)
        if registration is None or registration.get("branch") != actual:
            raise WorktreeError(f"checkout is not registered: {path}")
        git_dir = Path(await self._git("rev-parse", "--absolute-git-dir", cwd=path))
        if git_dir != common:
            try:
                backlink = Path((git_dir / "gitdir").read_text().strip()).resolve()
            except OSError as error:
                raise WorktreeError(f"cannot identify linked checkout: {path}") from error
            if backlink != path / ".git":
                raise WorktreeError(f"linked checkout identity changed: {path}")
        if identity is not None and self._identity(path) != identity:
            raise WorktreeError(f"checkout identity changed: {path}")

    async def _destination(self, data: dict[str, Any]) -> Path:
        path = Path(data["dest_path"])
        await self._checkout(
            path, data["dest_branch"], Path(data["dest_common_dir"]), data.get("dest_identity")
        )
        if "dest_identity" in data and self._identity(path) != data["dest_identity"]:
            raise WorktreeError(f"destination checkout identity changed: {path}")
        if (
            "repo_identity" in data
            and self._identity(Path(data["dest_common_dir"])) != data["repo_identity"]
        ):
            raise WorktreeError("destination repository identity changed")
        return path

    async def _state(self, path: Path) -> tuple[bool, bool]:
        dirty = bool(await self._git("status", "--porcelain", "--untracked-files=all", cwd=path))
        git_dir = Path(await self._git("rev-parse", "--absolute-git-dir", cwd=path))
        busy = any(
            (git_dir / marker).exists()
            for marker in (
                "MERGE_HEAD",
                "CHERRY_PICK_HEAD",
                "REVERT_HEAD",
                "rebase-merge",
                "rebase-apply",
                "sequencer",
            )
        )
        return dirty, busy

    async def _clean(self, path: Path) -> None:
        dirty, busy = await self._state(path)
        if dirty or busy:
            raise WorktreeError(
                f"checkout has uncommitted changes or an operation in progress: {path}"
            )

    async def _ancestor(self, ancestor: str, descendant: str) -> bool:
        result = await run_proc(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=self.repo_root,
            timeout=GIT_TIMEOUT_S,
        )
        if result.exit_code not in (0, 1):
            raise WorktreeError(f"cannot verify commit ancestry: {result.stderr.strip()}")
        return result.exit_code == 0

    async def _merge_destination(self, info: WorktreeInfo, target: str) -> None:
        head = await self._commit("HEAD", cwd=info.path)
        if not await self._ancestor(target, head):
            await self._git("merge", "--ff", "--no-edit", "--no-autostash", target, cwd=info.path)

    async def reconcile(self, name: str, *, recreate: bool = False) -> WorktreeInspection:
        """Check an idle worker and merge its parent's committed progress if clean.

        The caller must stop/join the worker first. Dirty/in-progress work is
        retained and returned unchanged. Conflicting merges raise WorktreeError
        and remain in the worker for resolution. Missing checkouts require
        explicit recreation; missing uncommitted content cannot be recovered.
        """
        async with self._worker_lock(name) as (info, data):
            destination = await self._destination(data)
            if not info.path.exists():
                if recreate is not True:
                    raise WorktreeError(
                        f"worker '{name}' checkout missing; use recreate=True explicitly. "
                        "Missing uncommitted content cannot be recovered."
                    )
                # Remove only this absent checkout's stale registration, not
                # other worktrees' metadata or retained sidecar/session data.
                registration = await self._registration(info.path)
                if registration is not None:
                    if registration.get("branch") != f"refs/heads/{info.branch}":
                        raise WorktreeError("missing worker registration has changed branch")
                    await self._git("worktree", "remove", str(info.path))
                ref = await run_proc(
                    ["git", "show-ref", "--verify", f"refs/heads/{info.branch}"],
                    cwd=self.repo_root,
                    timeout=GIT_TIMEOUT_S,
                )
                if ref.exit_code == 0:
                    await self._git("worktree", "add", str(info.path), info.branch)
                else:
                    base = await self._commit(data["base_commit"], cwd=self.repo_root)
                    await self._git("worktree", "add", "-b", info.branch, str(info.path), base)
                data = {**data, "checkout_identity": self._identity(info.path)}
                self.write_sidecar(name, data)
            await self._checkout(
                info.path,
                info.branch,
                Path(data["dest_common_dir"]),
                data.get("checkout_identity", []),
            )
            dirty, busy = await self._state(info.path)
            if not dirty and not busy:
                target = await self._commit("HEAD", cwd=destination)
                await self._merge_destination(info, target)
            return await self.inspect(name)

    async def integrate(
        self,
        name: str,
        *,
        reviewed_head: str,
        reviewed_parent_head: str,
        validation: list[str],
        validation_runner: ValidationRunner,
        allow_unvalidated: bool = False,
    ) -> MergeResult:
        """Validate the exact reviewed candidate, then fast-forward its pinned parent.

        The parent model must inspect the actual diff and approve both full HEAD
        hashes. Any parent mismatch requires re-review, even if the worker already
        contains that parent. This mechanical gate cannot establish review honesty. The caller
        owns worker idleness and must route validation_runner(cmd, worker_path)
        through ToolRegistry; there is deliberately no default shell runner.
        Only an explicit human decision may set allow_unvalidated=True.
        """
        if not isinstance(validation, list) or any(
            not isinstance(cmd, str) or not cmd.strip() for cmd in validation
        ):
            raise WorktreeError("validation must be a list of nonempty commands")
        checks = tuple(validation)
        if not checks and allow_unvalidated is not True:
            raise WorktreeError("no validation checks: explicit human approval required")
        for field, value in (
            ("reviewed_head", reviewed_head),
            ("reviewed_parent_head", reviewed_parent_head),
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
                raise WorktreeError(f"{field} must be the exact full reviewed commit hash")
        async with self._worker_lock(name) as (info, data):
            destination = await self._destination(data)
            common = Path(data["dest_common_dir"])
            await self._checkout(info.path, info.branch, common, data.get("checkout_identity", []))
            await self._clean(destination)
            await self._clean(info.path)
            if await self._commit("HEAD", cwd=info.path) != reviewed_head:
                raise WorktreeError("worker HEAD changed: re-review the actual diff and new HEAD")
            target = await self._commit("HEAD", cwd=destination)
            if target != reviewed_parent_head:
                if await self._ancestor(reviewed_parent_head, target):
                    await self._merge_destination(info, target)
                raise WorktreeError("destination HEAD changed since review: re-review required")
            await self._merge_destination(info, target)
            if await self._commit("HEAD", cwd=info.path) != reviewed_head:
                raise WorktreeError(
                    "destination merged into worker: re-review the actual diff and new HEAD"
                )

            async def check_review() -> None:
                if self._worker_data(name)[1] != data:
                    raise WorktreeError("worker sidecar changed during validation")
                await self._destination(data)
                await self._checkout(
                    info.path, info.branch, common, data.get("checkout_identity", [])
                )
                await self._clean(info.path)
                await self._clean(destination)
                if await self._commit("HEAD", cwd=info.path) != reviewed_head:
                    raise WorktreeError("worker HEAD changed during validation: re-review required")
                if await self._commit("HEAD", cwd=destination) != reviewed_parent_head:
                    raise WorktreeError(
                        "destination HEAD changed during validation: retry and re-review"
                    )
                if self._identity(info.path) != data.get("checkout_identity"):
                    raise WorktreeError(f"checkout identity changed: {info.path}")
                if "dest_identity" in data and self._identity(destination) != data["dest_identity"]:
                    raise WorktreeError(f"destination checkout identity changed: {destination}")

            await check_review()
            for cmd in checks:
                try:
                    result = await validation_runner(cmd, info.path)
                except Exception as error:
                    raise WorktreeError(f"validation failed ({cmd}): {error}") from error
                if not isinstance(result, ProcResult) or result.exit_code != 0 or result.timed_out:
                    detail = (
                        result.stderr or result.stdout
                        if isinstance(result, ProcResult)
                        else "invalid runner result"
                    )
                    raise WorktreeError(f"validation failed ({cmd}): {detail}")
                await check_review()
            await self._git("merge", "--ff-only", "--no-autostash", reviewed_head, cwd=destination)
            self.write_sidecar(
                name,
                {
                    **data,
                    "integrated_head": reviewed_head,
                    "integrated_at": datetime.now(UTC).isoformat(),
                },
            )
            return MergeResult(True, [], f"integrated {reviewed_head} into {data['dest_branch']}")

    async def cleanup_worker(self, name: str, *, discard: bool = False) -> WorktreeInfo:
        """Remove a clean, integrated worker; retain its sidecar and session data.

        discard=True is an explicit human data-loss decision, never inferred
        from tool auto-approval. Identity checks cannot be bypassed by discard.
        """
        async with self._worker_lock(name) as (info, data):
            await self._checkout(
                info.path,
                info.branch,
                Path(data["dest_common_dir"]),
                data.get("checkout_identity", []),
            )
            destination = await self._destination(data)
            if discard is not True:
                await self._clean(info.path)
                head = await self._commit("HEAD", cwd=info.path)
                target = await self._commit("HEAD", cwd=destination)
                if not await self._ancestor(head, target):
                    raise WorktreeError("worker HEAD is not integrated into its pinned destination")
            await self._git(
                "worktree", "remove", *(["--force"] if discard is True else []), str(info.path)
            )
            await self._git(
                "branch", "-D" if discard is True else "-d", info.branch, cwd=destination
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
        data = self._validated_sidecar(name, info)
        common = Path(data["dest_common_dir"]) if data else await self._common_dir(self.repo_root)
        await self._checkout(
            info.path, info.branch, common, data.get("checkout_identity", []) if data else None
        )
        return info

    async def inspect(self, name: str) -> WorktreeInspection:
        """Best-effort worker state; absent worktrees never raise."""
        info = self._info(name)
        sidecar = self._validated_sidecar(name, info)
        if not info.path.is_dir():
            return WorktreeInspection(
                present=False, info=info, dirty=False, merge_in_progress=False, sidecar=sidecar
            )
        if sidecar is not None:
            await self._checkout(
                info.path,
                info.branch,
                Path(sidecar["dest_common_dir"]),
                sidecar.get("checkout_identity", []),
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
