"""Control persisted worker runs from their supervising conversation."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import replace
from typing import Any

from lecode.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from lecode.agent.tools.bash import BashTool
from lecode.extras.proc import ProcResult, run_proc
from lecode.extras.subagents import SubagentError
from lecode.extras.workers import WORKER_CURRENT_EXTRA, WORKER_EXTRA
from lecode.extras.worktree import WorktreeError, WorktreeManager
from lecode.hooks import apply_hooks, dispatcher_from_config
from lecode.permission import SessionPermissions

HUMAN_CONTROL_EXTRA = "worker_human_control"
_HUMAN_ROOT_EXTRA = "worker_human_root_context"

_ACTIONS = (
    "list",
    "send",
    "stop",
    "resume",
    "submit",
    "question",
    "inspect",
    "review",
    "integrate",
    "cleanup",
    "recover",
)


class WorkersTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="workers",
            description=(
                "List and control existing delegated workers; this tool does not create workers. "
                "First call task(agent='general', prompt=..., run_in_background=True) to create "
                "a coding worker. Use the actual returned worker_id (or an id from action='list') "
                "for action='send' follow-ups; never invent an id or use an agent name as an id. "
                "Workers can manage descendants only. "
                "After a write worker completes, inspect/review its exact diff against the "
                "assignment, then explicitly integrate its reviewed_head and reviewed_parent_head. "
                "Only the immediate "
                "supervisor integrates; configured validation runs before merging. "
                "Cleanup removes only clean integrated workspaces, retaining transcripts. "
                "Recover recreates missing checkouts only with human confirmation. "
                "Run inspect, review, integrate, cleanup, and recover separately after sibling "
                "tools complete; task delegation can still run concurrently."
            ),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": list(_ACTIONS)},
                    "id": {
                        "type": "string",
                        "description": "Existing worker_id returned by task or workers list.",
                    },
                    "text": {"type": "string"},
                    "interrupt": {"type": "boolean"},
                    "tree": {"type": "boolean"},
                    "reviewed_head": {"type": "string"},
                    "reviewed_parent_head": {"type": "string"},
                },
                "required": ["action"],
            },
        )

    @staticmethod
    def _descendants(manager: Any, parent_id: str) -> set[str]:
        ids = set()
        pending = [parent_id]
        while pending:
            parent = pending.pop()
            for child in manager.children(parent):
                ids.add(child.id)
                pending.append(child.id)
        return ids

    def _validate(self, args: dict[str, Any]) -> str | None:
        action = args.get("action")
        allowed = {
            "list": {"action"},
            "send": {"action", "id", "text", "interrupt"},
            "stop": {"action", "id", "tree"},
            "resume": {"action", "id", "text"},
            "submit": {"action", "id"},
            "question": {"action", "text"},
            "inspect": {"action", "id"},
            "review": {"action", "id"},
            "integrate": {"action", "id", "reviewed_head", "reviewed_parent_head"},
            "cleanup": {"action", "id"},
            "recover": {"action", "id"},
        }
        if action not in _ACTIONS or set(args) - allowed[action]:
            return "invalid workers action or arguments"
        if action not in {"list", "question"} and not isinstance(args.get("id"), str):
            return f"workers {action} needs an id"
        if action in {"send", "question"} and not isinstance(args.get("text"), str):
            return f"workers {action} needs text"
        if not isinstance(args.get("interrupt", False), bool) or not isinstance(
            args.get("tree", False), bool
        ):
            return "interrupt and tree must be booleans"
        if "text" in args and not isinstance(args["text"], str):
            return "text must be a string"
        if action == "integrate":
            for field in ("reviewed_head", "reviewed_parent_head"):
                if not isinstance(args.get(field), str) or not re.fullmatch(
                    r"[0-9a-f]{40}|[0-9a-f]{64}", args[field]
                ):
                    return f"{field} must be the exact full reviewed commit hash"
        return None

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        error = self._validate(args)
        if error is not None:
            return ToolResult(f"error: {error}", is_error=True)
        manager = ctx.extras.get(WORKER_EXTRA)
        if manager is None:
            return ToolResult("error: workers are unavailable in this context", is_error=True)
        action = args["action"]
        current = ctx.extras.get(WORKER_CURRENT_EXTRA)
        if action == "list":
            workers = (
                manager.list()
                if current is None
                else [
                    worker
                    for worker in manager.list()
                    if worker.id in self._descendants(manager, current)
                ]
            )
            if not workers:
                return ToolResult(
                    "(no workers) Create one with task(agent='general', prompt=..., "
                    "run_in_background=True), then use its returned worker_id."
                )
            return ToolResult(
                "\n".join(
                    f"{worker.id}  {worker.state}  {worker.agent}  {worker.description}"
                    for worker in workers
                )
            )
        id = args.get("id")
        if action == "question":
            if current is None:
                return ToolResult("error: the main agent has no parent", is_error=True)
            try:
                manager.ask_parent(current, args["text"])
            except (KeyError, SubagentError) as e:
                return ToolResult(f"error: {e}", is_error=True)
            return ToolResult("question sent to parent")
        if current is not None and id not in self._descendants(manager, current):
            return ToolResult("error: workers can manage descendants only", is_error=True)
        if not any(worker.id == id for worker in manager.list()):
            return ToolResult(
                f"error: unknown worker id: {id}. Use workers(action='list') to find existing "
                "ids, or task(agent='general', prompt=..., run_in_background=True) to create "
                "a worker. Send to its actual returned worker_id, not an invented id.",
                is_error=True,
            )
        try:
            if (
                action == "integrate"
                and current is None
                and ctx.extras.get(HUMAN_CONTROL_EXTRA) is True
            ):
                worker = manager.get(id)
                if worker.parent_id is not None:
                    # Root dispatch already approved this human control. Preserve the
                    # live supervisor policy, including every ancestor and its grants.
                    supervisor = manager._runtime(manager.get(worker.parent_id)).ctx
                    supervisor = replace(
                        supervisor,
                        extras={
                            **supervisor.extras,
                            HUMAN_CONTROL_EXTRA: True,
                            _HUMAN_ROOT_EXTRA: ctx,
                        },
                    )
                    _, result = await supervisor.extras["registry"].dispatch_result(
                        "human-worker-control", "workers", json.dumps(args), supervisor
                    )
                    return result
            if action in {"inspect", "review", "integrate", "cleanup", "recover"}:
                async with manager.maintain_workspace(id) as worker:
                    if action == "integrate" and worker.parent_id != current:
                        raise WorktreeError("only the immediate supervisor can integrate a worker")
                    if worker.worktree is None:
                        raise WorktreeError("worker has no isolated write workspace")
                    worktrees = await WorktreeManager.discover(manager.cwd)
                    state = await worktrees.inspect(worker.worktree.name)
                    if state.info != worker.worktree or worker.cwd != state.info.path:
                        raise WorktreeError("worker workspace identity changed")
                    if action == "cleanup":
                        if any(
                            child.worktree is not None and child.worktree.path.exists()
                            for child in manager.children(id)
                        ):
                            raise WorktreeError("cleanup retained child workspaces first")
                        await worktrees.cleanup_worker(worker.worktree.name)
                        return ToolResult(f"worker {id} workspace cleaned; transcript retained")
                    if action == "recover":
                        await self._confirm(
                            manager,
                            worker,
                            "Recreate the missing checkout from retained committed history? "
                            "Old uncommitted data is unrecoverable.",
                        )
                        await worktrees.reconcile(worker.worktree.name, recreate=True)
                        return ToolResult(
                            f"worker {id} checkout recovered; inspect before resuming"
                        )
                    if action == "integrate":
                        checks = list(manager.config.worktree.validation)
                        if not checks:
                            await self._confirm(
                                manager,
                                worker,
                                "No validation checks are configured. Integrate this reviewed "
                                "commit without validation?",
                            )

                        async def validate(command, path):
                            return await self._validation_command(
                                manager, worker, ctx, command, path
                            )

                        result = await worktrees.integrate(
                            worker.worktree.name,
                            reviewed_head=args["reviewed_head"],
                            reviewed_parent_head=args["reviewed_parent_head"],
                            validation=checks,
                            validation_runner=validate,
                            allow_unvalidated=not checks,
                        )
                        return ToolResult(result.message, is_error=not result.merged)
                    if not state.present:
                        return ToolResult(
                            f"Worker {id} (@{worker.agent}) checkout missing: {worker.cwd}. "
                            "Use /agent <id> recover with human confirmation."
                        )
                    return await self._review(manager, worker, worktrees, state)
            if action == "send":
                message_id = await manager.send(
                    id,
                    args["text"],
                    bool(args.get("interrupt")),
                    from_human=ctx.extras.get(HUMAN_CONTROL_EXTRA) is True,
                )
                return ToolResult(f"worker {id} message {message_id} queued")
            if action == "stop":
                await manager.stop(id, bool(args.get("tree")))
                return ToolResult(f"worker {id} stopped")
            if action == "resume":
                await manager.resume(id, args.get("text"))
                return ToolResult(f"worker {id} resumed")
            note = await manager.submit(id)
            return ToolResult(
                f"worker {id} submitted",
                metadata={"notification_id": note["id"], "notification": note},
            )
        except (KeyError, RuntimeError, SubagentError, WorktreeError) as e:
            return ToolResult(f"error: {e}", is_error=True)

    @staticmethod
    async def _confirm(manager, worker, question):
        if manager.confirm is None:
            raise WorktreeError("human confirmation unavailable in this context")
        answer = manager.confirm(
            f"Worker {worker.id} (@{worker.agent}), cwd={worker.cwd}: {question}"
        )
        if inspect.isawaitable(answer):
            answer = await answer
        if answer is not True:
            raise WorktreeError("human confirmation declined")

    @staticmethod
    async def _validation_command(manager, worker, supervisor, command, path):
        """Fresh child policy and hooks, with no shell path outside tool dispatch."""
        if "bash" not in supervisor.extras["registry"].names():
            raise WorktreeError("validation bash is unavailable to the supervisor")
        definition = supervisor.extras["agents"].get(worker.agent)
        if definition is None:
            raise WorktreeError("worker agent definition is unavailable")
        grants = SessionPermissions(manager.store.load_grants(worker.session))
        hooks, _ = dispatcher_from_config(manager.config, path, session=worker.session)
        ctx = replace(
            supervisor,
            cwd=path,
            config=manager.config,
            session=worker.session,
            session_store=manager.store,
            session_perms=grants,
            permission_checker=supervisor.permission_checker.for_child(
                definition.overlay,
                cwd=path,
                session_perms=grants,
            ),
            extras={**supervisor.extras, "hooks": hooks, WORKER_CURRENT_EXTRA: worker.id},
        )
        if supervisor.approval_callback is not None:

            async def approve(name, args, reason):
                answer = supervisor.approval_callback(
                    name, args, f"Worker {worker.id} (@{worker.agent}), cwd={path}: {reason}"
                )
                return await answer if inspect.isawaitable(answer) else answer

            ctx.approval_callback = approve
        tool = BashTool()
        run = tool.run

        async def exact_command(args, context):
            if args != {"command": command}:
                return ToolResult("validation command rewritten by hook; refused", is_error=True)
            root = supervisor.extras.get(_HUMAN_ROOT_EXTRA)
            if root is not None:
                if "bash" not in root.extras["registry"].names():
                    raise WorktreeError("validation bash is unavailable to the current root")
                # A root agent switch can replace its checker. Gate the live root
                # separately without rebuilding or widening the supervisor chain.
                root_tool = BashTool()

                async def execute(approved_args, _):
                    return await run(approved_args, context)

                async def approve_root(name, args, reason):
                    answer = root.approval_callback(
                        name, args, f"Worker {worker.id} (@{worker.agent}), cwd={path}: {reason}"
                    )
                    return await answer if inspect.isawaitable(answer) else answer

                root_tool.run = execute
                _, result = await ToolRegistry([root_tool]).dispatch_result(
                    "human-worker-validation",
                    "bash",
                    json.dumps(args),
                    replace(
                        root,
                        approval_callback=approve_root if root.approval_callback else None,
                    ),
                )
                return result
            return await run(args, context)

        tool.run = exact_command
        registry = ToolRegistry([tool])
        ctx.extras["registry"] = registry
        if hooks is not None:
            apply_hooks(registry, hooks)
        _, result = await registry.dispatch_result(
            "worker-validation", "bash", json.dumps({"command": command}), ctx
        )
        proc = result.metadata.get("proc_result")
        if result.is_error and (
            not isinstance(proc, ProcResult) or (proc.exit_code == 0 and not proc.timed_out)
        ):
            raise WorktreeError(result.content)
        if not isinstance(proc, ProcResult):
            raise WorktreeError("validation bash returned no structured process exit result")
        return proc

    async def _review(self, manager, worker, worktrees, state) -> ToolResult:
        data = state.sidecar
        if data is None:
            raise WorktreeError("worker has no pinned parent sidecar")
        head = await worktrees._git("rev-parse", "HEAD", cwd=worker.cwd)
        parent = f"refs/heads/{data['dest_branch']}"
        target = await worktrees._git("rev-parse", "--verify", parent, cwd=worker.cwd)
        diff = await run_proc(
            ["git", "diff", "--no-ext-diff", "--no-textconv", target, head, "--"], cwd=worker.cwd
        )
        if diff.truncated:
            raise WorktreeError(
                "review diff truncated; inspect the full diff before approving HEAD"
            )
        if diff.exit_code != 0 or diff.timed_out:
            raise WorktreeError(f"cannot read review diff: {diff.stderr}")
        assignment = "\n".join(
            item["text"] for item in manager.store.load_events(worker.session, "worker_inbox")
        )
        content = (
            f"Worker {worker.id} (@{worker.agent}), cwd={worker.cwd}\n"
            f"Assignment:\n{assignment}\n"
            f"Pinned parent: {data['dest_branch']} at {data['dest_path']}\n"
            f"Base: {data['base_commit']}\nParent HEAD: {target}\nWorker HEAD: {head}\n"
            f"Committed diff (parent HEAD to worker HEAD):\n{diff.stdout or '(no changes)'}\n"
        )
        if state.dirty or state.merge_in_progress:
            pending = await worktrees._git(
                "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--", cwd=worker.cwd
            )
            untracked = await worktrees._git(
                "ls-files", "--others", "--exclude-standard", cwd=worker.cwd
            )
            return ToolResult(
                content + f"Uncommitted diff:\n{pending}\nUntracked paths:\n{untracked}\n"
                "Commit/checkpoint changes on the worker's OWN branch using bash, resolve any "
                "merge, then review again before integration. Never commit the user root."
            )
        return ToolResult(
            content + "review this exact diff against assignment then integrate reviewed_head "
            f"{head} and reviewed_parent_head {target}. "
            "If either HEAD changes, review again; never refresh a hash without review. "
            "If changes are needed, send feedback to the worker and review again.",
            metadata={"reviewed_head": head, "reviewed_parent_head": target},
        )


def make_tool() -> Tool:
    return WorkersTool()
