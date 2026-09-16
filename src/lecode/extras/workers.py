"""Persistent child runners. Runner boundary wiring is deliberately external.

``consume`` may only be called at a safe conversation boundary. Only the
supervisor releases its lease, once all unfinished tools are blocked on workers.
Inbox durability does not make external effects atomic.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lecode.agent.builder import build_runtime
from lecode.agent.runner import AgentRunner, LlmResponse, RunResult, UsageTotals
from lecode.agent.tools.base import ToolContext
from lecode.config.models import Config
from lecode.extras.subagents import SUBAGENT_EVENTS_EXTRA, SubagentError, SubagentProgress
from lecode.extras.worktree import WorktreeError, WorktreeInfo, WorktreeManager
from lecode.hooks import SUBAGENT_END, SUBAGENT_START, build_envelope, dispatch_event
from lecode.session.model import EventRecord, MessageRecord
from lecode.session.storage import Session, SessionStore

WORKER_EXTRA = "workers"
WORKER_CURRENT_EXTRA = "worker_id"
MAX_EXECUTING = 10
MAX_DEPTH = 2


@dataclass
class Worker:
    id: str
    parent_id: str | None
    depth: int
    agent: str
    origin: str
    state: str
    session: Session
    cwd: Path
    description: str = ""
    background: bool = False
    worktree: WorktreeInfo | None = None
    usage_totals: UsageTotals = field(default_factory=UsageTotals)
    usage_incomplete: bool = False
    dispatch_id: str | None = None
    result: RunResult | None = None
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    # Includes terminal hooks and the done callback's follow-up scheduling.
    _dispatch_active: bool = field(default=False, repr=False, compare=False)

    @property
    def session_id(self) -> str:
        return self.session.id

    @property
    def is_active(self) -> bool:
        """Authoritative activity predicate for scheduling and UI consumers."""
        return self._dispatch_active or self.state in {"queued", "running", "waiting"}


class WorkerManager:
    """Own child sessions, tasks, and locks until shutdown.

    ``confirm(question: str) -> bool`` and ``notify(note: dict) -> None`` may
    be synchronous or asynchronous. Notify is observational; parent delivery
    happens through ``boundary``/``consume`` (live) or ``drain_notifications``
    (idle), always after outstanding tool calls have been paired.
    """

    def __init__(
        self,
        config: Config,
        *,
        cwd: Path,
        root_ctx: ToolContext,
        session: Session | None = None,
        store: SessionStore | None = None,
        confirm=None,
        notify=None,
        workspace_guard: Callable[[Worker], Awaitable[None] | None] | None = None,
    ) -> None:
        self.config = config
        self.cwd = Path(cwd)
        self.root_ctx = root_ctx
        self.store = store or root_ctx.session_store or SessionStore()
        self.session = session or root_ctx.session or self.store.create("workers", self.cwd)
        self.confirm = confirm
        self.notify = notify
        #: Called with Worker before reused dispatches, may await or raise.
        #: Required for write worktrees; must validate/reconcile, never relocate.
        self.workspace_guard = workspace_guard or self.reconcile_workspace
        self._maintenance: dict[str, asyncio.Task] = {}
        self._workspace_starts: dict[str, int] = {}
        #: UI observer for durable state/usage changes; never affects execution.
        self.progress = None
        self._workers: dict[str, Worker] = {}
        self._contexts: dict[str, Any] = {}
        self._runtimes: dict[str, Any] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._progress_tasks: set[asyncio.Task] = set()
        self._locks: dict[str, Any] = {}
        self._closed = False
        self._slots = asyncio.Semaphore(MAX_EXECUTING)
        self._leases: set[str] = set()
        self._changed = asyncio.Event()
        self._blocked: set[asyncio.Task] = set()
        self._tool_gates: dict[asyncio.Task, asyncio.Event] = {}
        self._tool_owners: dict[asyncio.Task, str | None] = {}
        self._wait_results: dict[asyncio.Task, str] = {}

    def _signal(self, *_):
        self._changed.set()
        self._changed = asyncio.Event()

    @asynccontextmanager
    async def _blocking(self):
        task = asyncio.current_task()
        self._blocked.add(task)
        self._signal()
        try:
            yield
        finally:
            self._blocked.discard(task)
            self._signal()
            gate = self._tool_gates.get(task)
            if gate is not None and not task.cancelling():
                await gate.wait()

    async def await_tools(self, worker_id: str | None, tasks: list[asyncio.Task]):
        """Keep ordinary tool work leased; yield only when every tool is waiting.

        Wait/stop register actual waits, irrespective of tool name or rewritten
        arguments. Returning tools cannot execute their post-hooks until leased.
        """
        gate = asyncio.Event()
        gate.set()
        for task in tasks:
            self._tool_gates[task] = gate
            self._tool_owners[task] = worker_id
            task.add_done_callback(self._signal)
        try:
            while pending := {task for task in tasks if not task.done()}:
                changed = self._changed
                if worker_id is not None and pending <= self._blocked:
                    gate.clear()
                    async with self.suspend(worker_id):
                        while pending <= self._blocked:
                            await changed.wait()
                            changed = self._changed
                            pending = {task for task in tasks if not task.done()}
                            if not pending:
                                break
                    gate.set()
                else:
                    await changed.wait()
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                self._tool_gates.pop(task, None)
                self._tool_owners.pop(task, None)

    def get(self, id: str) -> Worker:
        return self._workers[id]

    def _workspace_available(self, id: str | None) -> None:
        while id is not None:
            if id in self._maintenance:
                raise WorktreeError(f"worker {id} workspace maintenance is in progress")
            id = self.get(id).parent_id

    @asynccontextmanager
    async def maintain_workspace(self, id: str, *, include_parent: bool = True):
        """Reserve the workspace subtree and its integration destination.

        Only the destination's sole unfinished tool may maintain its child.
        Reconciliation runs under the worker supervisor's existing reservation.
        """
        worker = self.get(id)
        task = asyncio.current_task()
        if include_parent and task in self._tool_owners:
            owner = self._tool_owners[task]
            if any(
                other is not task and not other.done() and other_owner == owner
                for other, other_owner in self._tool_owners.items()
            ):
                raise WorktreeError(
                    "workspace maintenance must run separately after sibling tools complete"
                )
        if not include_parent and self._maintenance.get(id) is task:
            yield worker
            return
        self._workspace_available(id)
        root = self.get(worker.parent_id) if include_parent and worker.parent_id else worker
        pending = [root]
        while pending:
            item = pending.pop()
            own_execution = (
                item.id == worker.parent_id and self._tool_owners.get(task) == item.id
            ) or (not include_parent and item.id == id and self._tasks.get(id) is task)
            if (
                (item.is_active and not own_execution)
                or item.id in self._maintenance
                or self._workspace_starts.get(item.id, 0)
            ):
                raise WorktreeError("workspace maintenance requires idle worker and descendants")
            pending.extend(self.children(item.id))
        self._maintenance[root.id] = task
        try:
            yield worker
        finally:
            del self._maintenance[root.id]

    async def reconcile_workspace(self, worker: Worker) -> None:
        """Default retained write-worker guard; never recreate missing data."""
        if worker.worktree is not None:
            async with self.maintain_workspace(worker.id, include_parent=False):
                manager = await WorktreeManager.discover(self.cwd)
                state = await manager.inspect(worker.worktree.name)
                if state.info != worker.worktree or worker.cwd != state.info.path:
                    raise WorktreeError("worker workspace identity changed")
                await manager.reconcile(worker.worktree.name)

    def list(self) -> list[Worker]:
        return list(self._workers.values())

    def children(self, id: str | None) -> list[Worker]:
        return [w for w in self.list() if w.parent_id == id]

    def load(self) -> list[Worker]:
        """Attach persisted children without scheduling any work; idempotent."""
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        latest = {item["id"]: item for item in self.store.load_events(self.session, "worker")}
        for id, snapshot in latest.items():
            if id in self._workers:
                continue
            data = dict(snapshot)
            child = self.store.open(data.pop("session_id"))
            lock = self.store.acquire_lock(child)
            try:
                data["session"] = child
                data["cwd"] = Path(data["cwd"])
                usages = self.store.load_events(child, "worker_usage_checkpoint")
                if usages:
                    checkpoint = dict(usages[-1])
                    incomplete = checkpoint.pop("incomplete", False)
                    data["usage_incomplete"] = bool(
                        data.get("usage_incomplete")
                        or incomplete
                        or checkpoint.get("usage_incomplete", False)
                    )
                    data["usage_totals"] = {
                        key: value for key, value in checkpoint.items() if key != "dispatch_id"
                    }
                data["usage_totals"] = UsageTotals(**data["usage_totals"])
                data["usage_incomplete"] = bool(
                    data.get("usage_incomplete") or data["usage_totals"].usage_incomplete
                )
                if data.get("worktree"):
                    info = data["worktree"]
                    data["worktree"] = WorktreeInfo(
                        info["name"], Path(info["path"]), info["branch"]
                    )
                if data.get("result"):
                    result = dict(data["result"])
                    result["usage_totals"] = UsageTotals(**result["usage_totals"])
                    data["result"] = RunResult(**result)
                worker = Worker(**data)
                if worker.is_active:
                    worker.state = "interrupted"
                    worker.usage_incomplete = True
            except BaseException:
                if lock is not None:
                    lock.release()
                raise
            self._workers[id] = worker
            self._locks[id] = lock
            self._record_usage(worker)
        return self.list()

    def attach(self, session: Session) -> list[Worker]:
        """Move this root manager to an idle session and hydrate its workers."""
        if any(worker.is_active for worker in self.list()):
            raise RuntimeError("workers are still active")
        for lock in self._locks.values():
            if lock is not None:
                lock.release()
        self.session = session
        self._workers.clear()
        self._contexts.clear()
        self._runtimes.clear()
        self._tasks.clear()
        self._locks.clear()
        return self.load()

    def _parent_context(self, worker):
        if worker.id not in self._contexts:
            self._contexts[worker.id] = (
                self._runtime(self.get(worker.parent_id)).ctx if worker.parent_id else self.root_ctx
            )
        return self._contexts[worker.id]

    def _record(self, worker: Worker) -> None:
        data = {
            key: getattr(worker, key)
            for key in (
                "id",
                "parent_id",
                "depth",
                "agent",
                "origin",
                "state",
                "description",
                "background",
                "usage_incomplete",
                "dispatch_id",
                "error",
                "started_at",
            )
        }
        data.update(
            session_id=worker.session_id,
            cwd=str(worker.cwd),
            usage_totals=asdict(worker.usage_totals),
            result=asdict(worker.result) if worker.result else None,
        )
        if worker.worktree is not None:
            data["worktree"] = {**asdict(worker.worktree), "path": str(worker.worktree.path)}
        self.store.append_event(self.session, "worker", data)
        self._signal()
        if self.progress is not None:
            result = self.progress(worker)
            if inspect.isawaitable(result):
                task = asyncio.create_task(result)
                self._progress_tasks.add(task)
                task.add_done_callback(self._progress_tasks.discard)

    def _record_usage(self, worker):
        # ponytail: scan the root ledger per dispatch; index if session size warrants it.
        recorded = [
            r.data
            for r in self.store.read_records(self.session)
            if isinstance(r, EventRecord)
            and r.kind == "worker_usage"
            and r.data.get("worker_id") == worker.id
        ]
        delta = {
            key: getattr(worker.usage_totals, key)
            - sum(item["usage"].get(key, 0) for item in recorded)
            for key in ("input_tokens", "output_tokens", "cost_usd")
        }
        if (
            recorded
            and not any(delta.values())
            and recorded[-1]["dispatch_id"] == worker.dispatch_id
            and recorded[-1]["incomplete"] == worker.usage_incomplete
        ):
            return
        self.store.append_event(
            self.session,
            "worker_usage",
            {
                "worker_id": worker.id,
                "dispatch_id": worker.dispatch_id,
                "usage": delta,
                "incomplete": worker.usage_incomplete,
            },
        )

    def _agent(self, ctx, name):
        agents = ctx.extras.get("agents")
        available = [a.name for a in agents.subagents()] if agents is not None else []
        if name not in available:
            raise SubagentError(
                f"unknown or ineligible subagent: {name} "
                f"(available: {', '.join(available) or 'none'}). "
                "Create a worker with task(agent=<available name>, prompt=...)."
            )
        return agents.get(name)

    @staticmethod
    def _read_only(ctx, definition):
        return ctx.permission_checker.for_child(definition.overlay, cwd=ctx.cwd).read_only

    async def _workspace(self, ctx, definition, id):
        if self._read_only(ctx, definition):
            return Path(ctx.cwd), None
        try:
            manager = await WorktreeManager.discover(ctx.cwd)
            base = await manager._commit("HEAD", cwd=ctx.cwd)
            branch = await manager._git("rev-parse", "--abbrev-ref", "HEAD", cwd=ctx.cwd)
        except WorktreeError as error:
            raise WorktreeError(
                f"Cannot create a writable worker at runtime cwd {ctx.cwd}: {error}. "
                "Restart the session from a Git repository with a committed HEAD. "
                "A shell cd does not change the session's runtime cwd."
            ) from error
        if branch == "HEAD":
            raise WorktreeError("write workers require an attached branch, not detached HEAD")
        destination = Path(await manager._git("rev-parse", "--show-toplevel", cwd=ctx.cwd))
        dirty_root = await manager._git("status", "--porcelain", cwd=self.cwd)
        dirty_parent = await manager._git("status", "--porcelain", cwd=ctx.cwd)
        if dirty_root or dirty_parent:
            question = (
                f"Worker {id} (@{definition.name}), parent cwd={ctx.cwd}: "
                "Uncommitted changes will not enter the worker's committed-HEAD worktree. Continue?"
            )
            approved = self.confirm(question) if self.confirm is not None else False
            if inspect.isawaitable(approved):
                approved = await approved
            if approved is not True:
                raise WorktreeError("dirty root/parent requires human confirmation")
        info = await manager.create_worker(
            id, base_commit=base, dest_path=destination, dest_branch=branch
        )
        return info.path, info

    async def start(
        self,
        ctx: ToolContext,
        *,
        agent: str,
        prompt: str,
        description: str = "",
        origin: str = "delegated",
        background: bool = False,
    ) -> Worker:
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        definition = self._agent(ctx, agent)
        if ctx.extras.get("provider") is None:
            raise SubagentError("no provider available for workers")
        parent_id = ctx.extras.get(WORKER_CURRENT_EXTRA)
        self._workspace_available(parent_id)
        depth = self.get(parent_id).depth + 1 if parent_id else 1
        if depth > MAX_DEPTH:
            raise SubagentError(f"worker depth exceeds {MAX_DEPTH}")
        id = uuid.uuid4().hex
        if parent_id is not None:
            self._workspace_starts[parent_id] = self._workspace_starts.get(parent_id, 0) + 1
        try:
            cwd, worktree = await self._workspace(ctx, definition, id)
        finally:
            if parent_id is not None:
                self._workspace_starts[parent_id] -= 1
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        session = self.store.create(
            f"worker-{id}",
            cwd,
            agent=agent,
            model=definition.model or self.config.agent.subagent_model or self.config.llm.model,
        )
        self._locks[id] = self.store.acquire_lock(session)
        worker = Worker(
            id,
            parent_id,
            depth,
            agent,
            origin,
            "queued",
            session,
            cwd,
            description=description,
            background=background,
            worktree=worktree,
        )
        self._workers[id] = worker
        self._contexts[id] = ctx
        self._record(worker)
        self._enqueue(worker, prompt)
        self._launch(worker)
        return worker

    def _runtime(self, worker):
        if worker.id not in self._runtimes:
            parent = self._parent_context(worker)
            definition = self._agent(parent, worker.agent)
            config = self.config.model_copy(
                update={
                    "pierre": self.config.pierre.model_copy(update={"enabled": False}),
                    "llm": self.config.llm.model_copy(update={"model": worker.session.meta.model}),
                }
            )
            runtime = build_runtime(
                config,
                worker.cwd,
                session=worker.session,
                store=self.store,
                agent_name=worker.agent,
                agent_registry=parent.extras["agents"],
                allowed_tools=parent.extras["registry"].names(),
                catalog=parent.catalog,
                worker_manager=self,
            )
            runtime.ctx.permission_checker = parent.permission_checker.for_child(
                definition.overlay,
                cwd=worker.cwd,
                session_perms=runtime.ctx.session_perms,
                read_only=self._read_only(parent, definition),
            )
            runtime.ctx.auto_approve = parent.auto_approve
            callback = parent.approval_callback
            if callback is not None:

                async def approval_callback(
                    tool_name,
                    args,
                    reason,
                    *,
                    worker=worker.id,
                    conversation=worker.session.name,
                ):
                    """Keep worker identity with the root approval FIFO."""
                    params = inspect.signature(callback).parameters
                    if "worker" in params or any(
                        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
                    ):
                        result = callback(
                            tool_name,
                            args,
                            reason,
                            worker=worker,
                            conversation=conversation,
                        )
                    else:
                        result = callback(tool_name, args, reason)
                    return await result if inspect.isawaitable(result) else result

                runtime.ctx.approval_callback = approval_callback
            runtime.ctx.extras.update(
                {
                    WORKER_EXTRA: self,
                    WORKER_CURRENT_EXTRA: worker.id,
                    "provider": parent.extras["provider"],
                }
            )
            runtime.registry.unregister("ask_user")
            self._runtimes[worker.id] = runtime
        return self._runtimes[worker.id]

    def _enqueue(self, worker, text):
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        item = {
            "id": uuid.uuid4().hex,
            "text": text,
            "answers": [note["id"] for note in self.questions(worker.id)],
        }
        self.store.append_event(worker.session, "worker_inbox", item)
        self._signal()
        return item["id"]

    def pending(self, id: str) -> list[dict[str, str]]:
        worker = self.get(id)
        consumed = {
            r.usage["worker_inbox_id"]
            for r in self.store.read_records(worker.session)
            if isinstance(r, MessageRecord) and r.usage and "worker_inbox_id" in r.usage
        }
        return [
            item
            for item in self.store.load_events(worker.session, "worker_inbox")
            if item["id"] not in consumed
        ]

    @staticmethod
    def _outstanding(history):
        outstanding = {}
        for message in history:
            for call in message.get("tool_calls") or []:
                outstanding[call["id"]] = call
            if message.get("role") == "tool":
                outstanding.pop(message.get("tool_call_id"), None)
        return outstanding

    def repair_interrupted_tools(self, session: Session, history: list[dict] | None = None) -> None:
        """Pair persisted calls left by an interrupted run without replaying them."""
        outstanding = self._outstanding(self.store.load_for_model(session))
        live = self._outstanding(history) if history is not None else {}
        for call in outstanding.values():
            message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": (
                    "Worker interrupted; tool outcome unknown. Inspect state before retrying."
                ),
            }
            self.store.append_message(session, message)
            if history is not None and call["id"] in live:
                history.append(message)

    @staticmethod
    def _notification_text(note: dict[str, Any]) -> str:
        if note.get("kind") == "question":
            return f"[worker {note['worker_id']} asks] {note['content']}"
        if note.get("kind") == "human_message":
            return f"[human → worker {note['worker_id']}] {note['content']}"
        return f"[worker {note['worker_id']} {note['state']}] {note['content']}"

    def consume(self, id: str | None, history: list[dict]) -> list[dict[str, str]]:
        """Append pending inputs durably and to live history at a safe boundary."""
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        worker = self.get(id) if id is not None else None
        session = worker.session if worker is not None else self.session
        if self._outstanding(history) or self._outstanding(self.store.load_for_model(session)):
            raise RuntimeError("cannot consume inbox with outstanding tool calls")
        items = self.pending(id) if id is not None else []
        for item in items:
            message = {"role": "user", "content": item["text"]}
            self.store.append_message(session, message, usage={"worker_inbox_id": item["id"]})
            history.append(message)
        notes = self.pending_notifications(id)
        for note in notes:
            text = self._notification_text(note)
            self.store.append_message(
                session,
                {"role": "user", "content": text},
                usage={"worker_notification_id": note["id"]},
            )
            history.append({"role": "user", "content": text})
            self.store.append_event(self.session, "worker_notification_ack", {"id": note["id"]})
            items.append({"id": note["id"], "text": text})
        return items

    def pending_notifications(self, parent_id: str | None = None) -> list[dict[str, Any]]:
        """Undelivered notes; the durable message is authoritative, not its ack."""
        # ponytail: scan durable recipients; index delivery IDs if histories grow.
        delivered = {
            id: {
                record.usage["worker_notification_id"]
                for record in self.store.read_records(session)
                if isinstance(record, MessageRecord)
                and record.usage
                and "worker_notification_id" in record.usage
            }
            for id, session in [(None, self.session), *((w.id, w.session) for w in self.list())]
        }
        delivered_anywhere = set().union(*delivered.values())
        notes = []
        for note in self.store.load_events(self.session, "worker_notification"):
            if not note["deliver"] or self._recipient(note) != parent_id:
                continue
            if note.get("kind") == "question":
                if (
                    note in self.questions(note["worker_id"])
                    and note["id"] not in delivered[parent_id]
                ):
                    notes.append(note)
            elif note["id"] not in delivered_anywhere:
                notes.append(note)
        return notes

    def _recipient(self, note):
        parent_id = note["parent_id"]
        while parent_id is not None and not self.get(parent_id).is_active:
            parent_id = self.get(parent_id).parent_id
        return parent_id

    def result_usage(self, task: asyncio.Task) -> dict | None:
        """Mark a foreground result delivered in the same durable tool record."""
        if notification_id := self._wait_results.pop(task, None):
            return {"worker_notification_id": notification_id}
        return None

    async def boundary(
        self,
        id: str | None,
        history: list[dict],
        *,
        completing: bool = False,
        input_ready: Callable[[], bool] | None = None,
    ) -> list[dict[str, str]]:
        """Deliver input and review delegated descendants before a final answer.

        Human-origin branches are independent unless explicitly submitted.
        Wake on each state/input change, so questions can preempt completion.
        """
        while True:
            changed = self._changed
            if id is not None and self.questions(id):
                async with self.suspend(id):
                    while self.questions(id):
                        await changed.wait()
                        changed = self._changed
            items = self.consume(id, history)
            if items or not completing or (input_ready is not None and input_ready()):
                return items
            if not any(worker.is_active for worker in self.descendants(id)):
                return []
            if id is None:
                await changed.wait()
            else:
                async with self.suspend(id):
                    while (
                        any(worker.is_active for worker in self.descendants(id))
                        and not self.pending(id)
                        and not self.pending_notifications(id)
                    ):
                        changed = self._changed
                        await changed.wait()

    def descendants(self, id: str | None) -> list[Worker]:
        """Delegated descendants, excluding independent human-origin branches."""
        pending = [id]
        descendants = []
        while pending:
            children = [w for w in self.children(pending.pop()) if w.origin == "delegated"]
            descendants.extend(children)
            pending.extend(child.id for child in children)
        return descendants

    def stop_message(self, id: str | None, reason: str) -> str:
        message = f"Run stopped: {reason}."
        unresolved = [
            f"{worker.id} ({worker.state})"
            for worker in self.descendants(id)
            if worker.is_active or worker.state != "completed"
        ]
        if unresolved:
            message += " Unresolved delegated workers: " + ", ".join(unresolved)
        return message

    def _launch(self, worker):
        self._workspace_available(worker.id)
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        existing = self._tasks.get(worker.id)
        if existing is not None and not existing.done():
            raise RuntimeError("worker is already active")
        resuming = worker.dispatch_id is not None
        worker.state = "queued"
        worker.error = None
        worker.result = None
        worker.dispatch_id = uuid.uuid4().hex
        worker._dispatch_active = True
        self._record(worker)
        self._tasks[worker.id] = asyncio.create_task(self._execute(worker, resuming=resuming))
        self._tasks[worker.id].add_done_callback(lambda task: self._finished(worker, task))

    def _finished(self, worker, task):
        if self._tasks.get(worker.id) is not task:
            return
        worker._dispatch_active = False
        if not self._closed and worker.state == "completed" and self.pending(worker.id):
            self._launch(worker)
        else:
            self._record(worker)

    async def _execute(self, worker, *, resuming=False):
        try:
            start = asyncio.create_task(self._hook(worker, SUBAGENT_START))
            try:
                await asyncio.shield(start)
            except asyncio.CancelledError:
                await start
                raise
            if worker.state == "stopped":
                raise asyncio.CancelledError
            await self._slots.acquire()
            self._leases.add(worker.id)
            worker.state = "running"
            self._record(worker)

            if not worker.cwd.is_dir():
                raise WorktreeError(f"worker checkout missing: {worker.cwd}")
            if resuming:
                if worker.worktree is not None and self.workspace_guard is None:
                    raise WorktreeError("write worker resume requires a workspace guard")
                if self.workspace_guard is not None:
                    async with (
                        self.maintain_workspace(worker.id, include_parent=False)
                        if worker.worktree is not None
                        else nullcontext()
                    ):
                        guarded = self.workspace_guard(worker)
                        if inspect.isawaitable(guarded):
                            await guarded

            runtime = self._runtime(worker)
            runtime.ctx.config.llm.model = worker.session.meta.model
            runner = AgentRunner(
                runtime.ctx.extras["provider"],
                runtime.registry,
                runtime.ctx,
                session=worker.session,
                store=self.store,
                config=runtime.ctx.config,
                catalog=runtime.ctx.catalog,
            )
            history = self.store.load_for_model(worker.session)
            self.consume(worker.id, history)
            worker.result = await runner.run(
                [{"role": "system", "content": runtime.system_prompt}, *history],
                on_event=lambda event: self._event(worker, event),
            )
            worker.state = "completed" if worker.result.stop_reason == "done" else "failed"
            if worker.state == "failed":
                worker.error = self.stop_message(worker.id, worker.result.stop_reason)
        except asyncio.CancelledError:
            worker.state = "stopped"
            worker.usage_incomplete = True
        except Exception as error:
            worker.state = "failed"
            worker.error = f"{type(error).__name__}: {error}"
            worker.usage_incomplete = True
        finally:
            if worker.id in self._leases:
                self._leases.remove(worker.id)
                self._slots.release()
            worker.usage_incomplete |= any(
                isinstance(record, MessageRecord)
                and record.role == "assistant"
                and record.usage is None
                for record in self.store.read_records(worker.session)
            )
            self._record_usage(worker)
            self._record(worker)
            await self._hook(worker, SUBAGENT_END)

        note = self._notification(worker, submitted=False)
        if (worker.background or worker.origin == "human") and self.notify is not None:
            try:
                result = self.notify(note)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                self.store.append_event(
                    self.session,
                    "worker_notify_error",
                    {"worker_id": worker.id, "error": str(error)},
                )

    async def _hook(self, worker: Worker, event: str) -> None:
        parent = self._parent_context(worker)
        hooks = parent.extras.get("hooks")
        if hooks is None or not hooks.handlers.get(event):
            return
        envelope = build_envelope(
            event,
            parent.cwd,
            session=parent.session,
            agent=worker.agent,
            prompt="\n".join(item["text"] for item in self.pending(worker.id))
            if event == SUBAGENT_START
            else None,
            result={
                "content": (
                    worker.error or (worker.result.final_text if worker.result else worker.state)
                )[:2000],
                "is_error": worker.state != "completed",
            }
            if event == SUBAGENT_END
            else None,
        )
        envelope["worker"] = {
            "id": worker.id,
            "parent_id": worker.parent_id,
            "dispatch_id": worker.dispatch_id,
            "session_id": worker.session_id,
            "cwd": str(worker.cwd),
        }
        await dispatch_event(event, envelope, hooks.handlers[event])

    @asynccontextmanager
    async def suspend(self, worker_id: str):
        """Yield a supervisor lease only while all of its work is blocked.

        Only the worker runner task may suspend itself. Concurrent tool tasks
        cannot release a supervisor that is still executing sibling tools.
        """
        if worker_id not in self._leases or asyncio.current_task() is not self._tasks.get(
            worker_id
        ):
            raise RuntimeError("suspend must be called once by the worker supervisor")
        worker = self.get(worker_id)
        self._leases.remove(worker_id)
        self._slots.release()
        worker.state = "waiting"
        self._record(worker)
        try:
            yield
        except BaseException:
            # Cancellation must not queue behind the children we were awaiting.
            raise
        else:
            await self._slots.acquire()
            self._leases.add(worker_id)
            worker.state = "running"
            self._record(worker)

    async def _event(self, worker, event):
        if isinstance(event, LlmResponse):
            worker.usage_incomplete |= event.usage_incomplete
            old = worker.usage_totals
            worker.usage_totals = UsageTotals(
                old.input_tokens + event.input_tokens,
                old.output_tokens + event.output_tokens,
                old.cost_usd + event.cost_usd,
                event.input_tokens or old.context_tokens,
                usage_incomplete=worker.usage_incomplete,
            )
            self.store.append_event(
                worker.session,
                "worker_usage_checkpoint",
                {
                    "dispatch_id": worker.dispatch_id,
                    **asdict(worker.usage_totals),
                },
            )
            self._record(worker)
        callback = self.root_ctx.extras.get(SUBAGENT_EVENTS_EXTRA)
        if callback is not None:
            result = callback(SubagentProgress(worker.id, worker.agent, worker.description, event))
            if inspect.isawaitable(result):
                await result

    async def wait(self, id: str) -> RunResult:
        worker = self.get(id)
        waiting = False
        async with self._blocking():
            while worker.is_active:
                changed = self._changed
                if asyncio.current_task() in self._tool_gates and any(
                    note.get("kind") == "question"
                    for note in self.pending_notifications(worker.parent_id)
                ):
                    waiting = True
                    break
                await changed.wait()
            result, state, error = worker.result, worker.state, worker.error
            caller = asyncio.current_task()
            if not waiting and caller in self._tool_gates and not worker.background:
                self._wait_results[caller] = f"{worker.dispatch_id}:completed"
        if waiting:
            # Close this tool pair before delivering the question. Completion
            # arrives separately; the child keeps its session and dispatch.
            return RunResult(
                f"Worker {id} needs a parent answer; use workers send to reply.",
                0,
                "waiting",
                UsageTotals(),
            )
        if result is None or state != "completed":
            raise SubagentError(error or f"worker is {state}")
        return result

    async def send(
        self, id: str, text: str, interrupt: bool = False, *, from_human: bool = False
    ) -> str:
        self._workspace_available(id)
        worker = self.get(id)
        message_id = self._enqueue(worker, text)
        if from_human:
            self.store.append_event(
                self.session,
                "worker_notification",
                {
                    "id": uuid.uuid4().hex,
                    "worker_id": worker.id,
                    "parent_id": worker.parent_id,
                    "dispatch_id": worker.dispatch_id,
                    "agent": worker.agent,
                    "origin": "human",
                    "state": worker.state,
                    "kind": "human_message",
                    "content": text,
                    "deliver": True,
                },
            )
        if interrupt and worker.is_active:
            await self.stop(id)
            await self.resume(id)
        elif worker.state == "completed" and (id not in self._tasks or self._tasks[id].done()):
            self._launch(worker)
        return message_id

    async def stop(self, id: str, tree: bool = False) -> None:
        worker = self.get(id)
        finalizing = worker.state in {"completed", "failed", "stopped"}
        worker.state = "stopped"
        self._record(worker)
        task = self._tasks.get(id)
        if task is not None and not task.done():
            # Let a newly queued coroutine enter its lifecycle try/finally.
            await asyncio.sleep(0)
            if not finalizing and not task.cancelling():
                task.cancel()
            async with self._blocking():
                await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        if tree:
            for child in self.children(id):
                await self.stop(child.id, tree=True)

    async def resume(self, id: str, text: str | None = None) -> Worker:
        self._workspace_available(id)
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        worker = self.get(id)
        task = self._tasks.get(id)
        if task is not None and not task.done():
            raise RuntimeError("worker is already active")
        # Finish protocol pairs, not the interrupted external actions. Replaying
        # those actions could duplicate an effect whose outcome is unknown.
        self.repair_interrupted_tools(worker.session)
        if text is not None:
            self._enqueue(worker, text)
        self._launch(worker)
        return worker

    def _notification(self, worker, *, submitted):
        note = {
            "id": f"{worker.dispatch_id}:{'submitted' if submitted else 'completed'}",
            "worker_id": worker.id,
            "parent_id": worker.parent_id,
            "dispatch_id": worker.dispatch_id,
            "agent": worker.agent,
            "origin": worker.origin,
            "state": worker.state,
            "content": worker.error
            or (worker.result.final_text if worker.result else worker.state),
            "deliver": submitted or worker.origin == "delegated",
        }
        existing = self.store.load_events(self.session, "worker_notification")
        is_new = not any(item["id"] == note["id"] for item in existing)
        if is_new:
            self.store.append_event(self.session, "worker_notification", note)
            self._signal()
        return {**note, "new": is_new}

    async def submit(self, id: str) -> dict[str, Any]:
        """Explicitly make a human-origin result available to its parent."""
        worker = self.get(id)
        if worker.origin != "human":
            raise SubagentError("only human workers can be submitted")
        if worker.state != "completed" or worker.result is None:
            raise SubagentError("only completed workers can be submitted")
        return self._notification(worker, submitted=True)

    def ask_parent(self, id: str, text: str) -> dict[str, Any]:
        """Durably deliver a worker question without exposing interactive UI."""
        worker = self.get(id)
        note = {
            "id": uuid.uuid4().hex,
            "worker_id": worker.id,
            "parent_id": worker.parent_id,
            "dispatch_id": worker.dispatch_id,
            "agent": worker.agent,
            "origin": worker.origin,
            "state": worker.state,
            "kind": "question",
            "content": text,
            "deliver": True,
        }
        self.store.append_event(self.session, "worker_notification", note)
        self._signal()
        return note

    def questions(self, id: str) -> list[dict[str, Any]]:
        """Questions stay unresolved until a durable inbox reply is queued."""
        answered = {
            answer
            for item in self.store.load_events(self.get(id).session, "worker_inbox")
            for answer in item.get("answers", [])
        }
        return [
            note
            for note in self.store.load_events(self.session, "worker_notification")
            if note["worker_id"] == id
            and note.get("kind") == "question"
            and note["id"] not in answered
        ]

    def drain_notifications(self, parent_id: str | None = None) -> list[dict[str, Any]]:
        """Durably deliver notes at an idle boundary; use consume for live history."""
        notes = self.pending_notifications(parent_id)
        session = self.get(parent_id).session if parent_id is not None else self.session
        self.consume(parent_id, self.store.load_for_model(session))
        return notes

    async def shutdown(self) -> None:
        self._closed = True
        try:
            for id, task in self._tasks.items():
                if not task.done():
                    await self.stop(id)
            for runtime in self._runtimes.values():
                for key in ("background", "lsp"):
                    resource = runtime.ctx.extras.get(key)
                    if resource is not None:
                        result = resource.shutdown()
                        if inspect.isawaitable(result):
                            await result
        finally:
            for lock in self._locks.values():
                if lock is not None:
                    lock.release()
