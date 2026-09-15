"""Persistent child runners. Runner boundary wiring is deliberately external.

``consume`` may only be called at a safe conversation boundary. ``suspend``
belongs around the supervisor's whole tool batch, never around individual
concurrent tool calls. Inbox durability does not make external effects atomic.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from contextlib import asynccontextmanager
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

    @property
    def session_id(self) -> str:
        return self.session.id


class WorkerManager:
    """Own child sessions, tasks, and locks until shutdown.

    ``confirm(question: str) -> bool`` and ``notify(note: dict) -> None`` may
    be synchronous or asynchronous. Notify is observational; parent delivery
    happens only through ``drain_notifications(parent_id)`` at a safe boundary.
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
    ) -> None:
        self.config = config
        self.cwd = Path(cwd)
        self.root_ctx = root_ctx
        self.store = store or root_ctx.session_store or SessionStore()
        self.session = session or root_ctx.session or self.store.create("workers", self.cwd)
        self.confirm = confirm
        self.notify = notify
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

    def get(self, id: str) -> Worker:
        return self._workers[id]

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
                    data["usage_totals"] = {
                        key: value for key, value in usages[-1].items() if key != "dispatch_id"
                    }
                data["usage_totals"] = UsageTotals(**data["usage_totals"])
                if data.get("worktree"):
                    info = data["worktree"]
                    data["worktree"] = WorktreeInfo(
                        info["name"], Path(info["path"]), info["branch"]
                    )
                if data.get("result"):
                    result = dict(data["result"])
                    result["usage_totals"] = UsageTotals(**result["usage_totals"])
                    data["result"] = RunResult(**result)
                if data["state"] in {"queued", "running", "waiting"}:
                    data["state"] = "interrupted"
                    data["usage_incomplete"] = True
                worker = Worker(**data)
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
        if any(worker.state in {"queued", "running", "waiting"} for worker in self.list()):
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
        if agents is None or name not in {a.name for a in agents.subagents()}:
            raise SubagentError(f"unknown or ineligible subagent: {name}")
        return agents.get(name)

    @staticmethod
    def _read_only(ctx, definition):
        return ctx.permission_checker.for_child(definition.overlay, cwd=ctx.cwd).read_only

    async def _workspace(self, ctx, definition, id):
        if self._read_only(ctx, definition):
            return Path(ctx.cwd), None
        manager = await WorktreeManager.discover(ctx.cwd)
        branch = await manager._git("rev-parse", "--abbrev-ref", "HEAD", cwd=ctx.cwd)
        if branch == "HEAD":
            raise WorktreeError("write workers require an attached branch, not detached HEAD")
        destination = Path(await manager._git("rev-parse", "--show-toplevel", cwd=ctx.cwd))
        base = await manager._git("rev-parse", "HEAD", cwd=ctx.cwd)
        dirty_root = await manager._git("status", "--porcelain", cwd=self.cwd)
        dirty_parent = await manager._git("status", "--porcelain", cwd=ctx.cwd)
        if dirty_root or dirty_parent:
            question = (
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
        depth = self.get(parent_id).depth + 1 if parent_id else 1
        if depth > MAX_DEPTH:
            raise SubagentError(f"worker depth exceeds {MAX_DEPTH}")
        id = uuid.uuid4().hex
        cwd, worktree = await self._workspace(ctx, definition, id)
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
                update={"pierre": self.config.pierre.model_copy(update={"enabled": False})}
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

                async def approval_callback(tool_name, args, reason):
                    """Keep worker identity with the root approval FIFO."""
                    params = inspect.signature(callback).parameters
                    if "worker" in params or any(
                        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
                    ):
                        result = callback(
                            tool_name,
                            args,
                            reason,
                            worker=worker.id,
                            conversation=worker.session.name,
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
        item = {"id": uuid.uuid4().hex, "text": text}
        self.store.append_event(worker.session, "worker_inbox", item)
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
        acknowledged = {
            item["id"] for item in self.store.load_events(self.session, "worker_notification_ack")
        }
        notes = [
            note
            for note in self.store.load_events(self.session, "worker_notification")
            if note["deliver"] and note["parent_id"] == id and note["id"] not in acknowledged
        ]
        for note in notes:
            text = self._notification_text(note)
            self.store.append_message(
                session,
                {"role": "user", "content": text},
                usage={"worker_notification_id": note["id"]},
            )
            self.store.append_event(self.session, "worker_notification_ack", {"id": note["id"]})
            history.append({"role": "user", "content": text})
            items.append({"id": note["id"], "text": text})
        return items

    def _launch(self, worker):
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        existing = self._tasks.get(worker.id)
        if existing is not None and not existing.done():
            raise RuntimeError("worker is already active")
        worker.state = "queued"
        worker.error = None
        worker.result = None
        worker.dispatch_id = uuid.uuid4().hex
        self._record(worker)
        self._tasks[worker.id] = asyncio.create_task(self._execute(worker))
        self._tasks[worker.id].add_done_callback(lambda task: self._finished(worker, task))

    def _finished(self, worker, task):
        if (
            not self._closed
            and self._tasks.get(worker.id) is task
            and worker.state == "completed"
            and self.pending(worker.id)
        ):
            self._launch(worker)

    def _background_descendants(self, id: str) -> list[Worker]:
        pending = [id]
        descendants: list[Worker] = []
        while pending:
            parent = pending.pop()
            children = self.children(parent)
            descendants.extend(children)
            pending.extend(child.id for child in children)
        return [worker for worker in descendants if worker.background]

    async def _execute(self, worker):
        try:
            await self._slots.acquire()
            self._leases.add(worker.id)
            worker.state = "running"
            self._record(worker)

            runtime = self._runtime(worker)
            definition = self._agent(self._parent_context(worker), worker.agent)
            runner = AgentRunner(
                runtime.ctx.extras["provider"],
                runtime.registry,
                runtime.ctx,
                session=worker.session,
                store=self.store,
                config=runtime.ctx.config,
                catalog=runtime.ctx.catalog,
            )
            runner.model = (
                definition.model or self.config.agent.subagent_model or self.config.llm.model
            )
            while True:
                history = self.store.load_for_model(worker.session)
                self.consume(worker.id, history)
                worker.result = await runner.run(
                    [{"role": "system", "content": runtime.system_prompt}, *history],
                    on_event=lambda event: self._event(worker, event),
                )
                active = [
                    child
                    for child in self._background_descendants(worker.id)
                    if child.state in {"queued", "running", "waiting"}
                ]
                if active:
                    # Let queued descendants use this supervisor's lease.
                    async with self.suspend(worker.id):
                        await asyncio.gather(
                            *(self.wait(child.id) for child in active), return_exceptions=True
                        )
                if not self.pending(worker.id):
                    # Background completions are durable notes, consumed by the
                    # next child run rather than racing a final response.
                    acknowledged = {
                        item["id"]
                        for item in self.store.load_events(self.session, "worker_notification_ack")
                    }
                    if not any(
                        note["deliver"]
                        and note["parent_id"] == worker.id
                        and note["id"] not in acknowledged
                        for note in self.store.load_events(self.session, "worker_notification")
                    ):
                        break
            worker.state = "completed"
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

        if worker.background or worker.origin == "human":
            note = self._notification(worker, submitted=False)
            if self.notify is not None:
                try:
                    result = self.notify(note)
                    if inspect.isawaitable(result):
                        await result
                except Exception as error:
                    self.store.append_event(
                        self.session,
                        "worker_notify_error",
                        {
                            "worker_id": worker.id,
                            "error": str(error),
                        },
                    )

    @asynccontextmanager
    async def suspend(self, worker_id: str):
        """Yield the supervisor lease while awaiting its entire tool batch.

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
        finally:
            acquire = asyncio.create_task(self._slots.acquire())
            try:
                await asyncio.shield(acquire)
            except asyncio.CancelledError:
                await asyncio.shield(acquire)
                raise
            self._leases.add(worker_id)
            worker.state = "running"
            self._record(worker)

    async def _event(self, worker, event):
        if isinstance(event, LlmResponse):
            old = worker.usage_totals
            worker.usage_totals = UsageTotals(
                old.input_tokens + event.input_tokens,
                old.output_tokens + event.output_tokens,
                old.cost_usd + event.cost_usd,
                event.input_tokens or old.context_tokens,
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
        task = self._tasks.get(id)
        while task is not None:
            await asyncio.shield(task)
            current = self._tasks.get(id)
            if current is task:
                break
            task = current
        if worker.result is None or worker.state != "completed":
            raise SubagentError(worker.error or f"worker is {worker.state}")
        return worker.result

    async def send(
        self, id: str, text: str, interrupt: bool = False, *, from_human: bool = False
    ) -> str:
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
        if interrupt and worker.state in {"queued", "running", "waiting"}:
            await self.stop(id)
            await self.resume(id)
        elif worker.state == "completed" and (id not in self._tasks or self._tasks[id].done()):
            self._launch(worker)
        return message_id

    async def stop(self, id: str, tree: bool = False) -> None:
        worker = self.get(id)
        worker.state = "stopped"
        self._record(worker)
        task = self._tasks.get(id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if tree:
            for child in self.children(id):
                await self.stop(child.id, tree=True)

    async def resume(self, id: str, text: str | None = None) -> Worker:
        if self._closed:
            raise RuntimeError("worker manager is shut down")
        worker = self.get(id)
        task = self._tasks.get(id)
        if task is not None and not task.done():
            raise RuntimeError("worker is already active")
        # Finish protocol pairs, not the interrupted external actions. Replaying
        # those actions could duplicate an effect whose outcome is unknown.
        for call in self._outstanding(self.store.load_for_model(worker.session)).values():
            self.store.append_message(
                worker.session,
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": (
                        "Worker interrupted; tool outcome unknown. Inspect state before retrying."
                    ),
                },
            )
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
            "content": worker.result.final_text if worker.result else worker.error or worker.state,
            "deliver": submitted or (worker.origin == "delegated" and worker.background),
        }
        existing = self.store.load_events(self.session, "worker_notification")
        is_new = not any(item["id"] == note["id"] for item in existing)
        if is_new:
            self.store.append_event(self.session, "worker_notification", note)
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
        if worker.parent_id is None:
            raise SubagentError("the root worker has no parent")
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
        return note

    def drain_notifications(self, parent_id: str | None = None) -> list[dict[str, Any]]:
        """Acknowledge parent-bound result notes; human results require submit."""
        acknowledged = {
            item["id"] for item in self.store.load_events(self.session, "worker_notification_ack")
        }
        notes = [
            item
            for item in self.store.load_events(self.session, "worker_notification")
            if item["deliver"] and item["parent_id"] == parent_id and item["id"] not in acknowledged
        ]
        for note in notes:
            self.store.append_event(self.session, "worker_notification_ack", {"id": note["id"]})
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
