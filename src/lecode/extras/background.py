"""Background tasks: detached bash commands and subagents.

``bash`` and ``task`` accept ``run_in_background: true`` and hand the work to
the :class:`BackgroundTaskManager` instead of awaiting it inline. Manager
tasks are standalone ``asyncio.Task`` objects — deliberately outside the
runner's tool-dispatch gather, so cancelling the turn (Ctrl-C) never kills
them. Output streams into ``<config_dir>/tasks/<id>.log`` (plus the record's
capped final text); completions queue a notification the runner drains into
the conversation as a synthetic user message and are reported live through
the ``notify`` callback the TUI installs.

Subagent children never see the manager: their context carries a fresh extras
dict (provider/hooks/memory only), so background params and the ``tasks_*``
tools fail cleanly with "unavailable in this context" inside a child.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: ``ctx.extras`` key the manager is installed under.
BACKGROUND_EXTRA = "background"

#: Live (still running) tasks the manager accepts before rejecting starts.
MAX_LIVE_TASKS = 20

#: Cap on the per-task retained output text (same budget as the bash tool).
TAIL_CAP_BYTES = 60_000

#: Grace between SIGTERM and SIGKILL when stopping a task.
STOP_GRACE_S = 5.0

#: Output characters included in a completion notification.
NOTIFY_SNIPPET_CHARS = 4000

#: ``emit(chunk)`` — the body streams raw output chunks through this.
Emit = Callable[[bytes], None]

#: A background body: runs the work, returns ``(final_text, exit_code)``.
BgBody = Callable[[Emit], Awaitable[tuple[str, "int | None"]]]


class BackgroundError(Exception):
    """Unknown task id, task-cap reached, or a stale handle."""


@dataclass
class BackgroundTask:
    """One manager-owned task and its final state."""

    id: str
    kind: str  # "bash" | "agent"
    description: str
    task: asyncio.Task[None]
    log_path: Path
    status: str = "running"  # running | done | failed | stopped
    exit_code: int | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    #: Final text (capped), also mirrored into the log file via ``emit``.
    output: str = ""
    stop_requested: bool = False
    #: SIGTERM/SIGKILL equivalents for the underlying process; ``None`` (agent
    #: tasks) falls back to cancelling the asyncio task.
    term: Callable[[], None] | None = None
    kill: Callable[[], None] | None = None

    @property
    def age_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.created_at


def notification_text(record: BackgroundTask) -> str:
    """The completion line fed back to the model (and the TUI feed)."""
    head = f"[background {record.id} {record.status}"
    if record.exit_code is not None:
        head += f", exit {record.exit_code}"
    head += f"] {record.description}"
    snippet = record.output.strip()
    if len(snippet) > NOTIFY_SNIPPET_CHARS:
        snippet = "…" + snippet[-NOTIFY_SNIPPET_CHARS:]
    return f"{head}\n{snippet}" if snippet else head


async def _await_done(record: BackgroundTask, timeout: float) -> bool:
    """Wait for the record's task without cancelling it; False on timeout."""
    try:
        await asyncio.wait_for(asyncio.shield(record.task), timeout)
    except TimeoutError:
        return False
    except asyncio.CancelledError:
        if not record.task.done():
            raise  # the waiter itself was cancelled — propagate
    return True


class BackgroundTaskManager:
    """Registry + lifecycle for detached tasks (cap, output, stop, shutdown)."""

    def __init__(self, *, notify: Callable[[str], Any] | None = None) -> None:
        self._records: dict[str, BackgroundTask] = {}
        self._counter = 0
        self._pending: list[str] = []
        self._notify_tasks: set[asyncio.Task[Any]] = set()
        #: Optional completion reporter (the TUI installs ``feed.info``).
        self.notify = notify

    # -- inspection -------------------------------------------------------------

    def tasks(self) -> list[BackgroundTask]:
        return list(self._records.values())

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._records.get(task_id)

    def _require(self, task_id: str) -> BackgroundTask:
        record = self._records.get(task_id)
        if record is None:
            raise BackgroundError(f"unknown background task: {task_id}")
        return record

    def output(self, task_id: str, tail_bytes: int = TAIL_CAP_BYTES) -> str:
        """The last ``tail_bytes`` of a task's output, from its log file."""
        record = self._require(task_id)
        try:
            with record.log_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - tail_bytes))
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return record.output

    # -- lifecycle ----------------------------------------------------------------

    def start(
        self,
        kind: str,
        description: str,
        body: BgBody,
        *,
        term: Callable[[], None] | None = None,
        kill: Callable[[], None] | None = None,
    ) -> BackgroundTask:
        """Spawn ``body`` as a detached task; raises at the live-task cap."""
        live = sum(1 for r in self._records.values() if r.status == "running")
        if live >= MAX_LIVE_TASKS:
            raise BackgroundError(
                f"background task limit reached ({MAX_LIVE_TASKS} running); "
                "stop one with tasks_stop or wait for one with tasks_wait"
            )
        self._counter += 1
        task_id = f"bg-{self._counter}"
        from lecode.config.loader import config_dir

        log_dir = config_dir() / "tasks"
        log_dir.mkdir(parents=True, exist_ok=True)
        record = BackgroundTask(
            id=task_id,
            kind=kind,
            description=description,
            task=None,  # type: ignore[arg-type]  # set just below
            log_path=log_dir / f"{task_id}.log",
            term=term,
            kill=kill,
        )

        def emit(chunk: bytes) -> None:
            with record.log_path.open("ab") as fh:
                fh.write(chunk)

        async def run() -> None:
            try:
                text, exit_code = await body(emit)
            except asyncio.CancelledError:
                record.status = "stopped"
                raise
            except Exception as e:
                record.output = f"error: {type(e).__name__}: {e}"
                if record.exit_code is None:
                    record.exit_code = 1
                record.status = "failed"
            else:
                record.output = text
                record.exit_code = exit_code
                if record.stop_requested:
                    record.status = "stopped"
                else:
                    record.status = "done" if not exit_code else "failed"
            finally:
                record.finished_at = time.monotonic()
                self._announce(record)

        record.task = asyncio.ensure_future(run())
        self._records[task_id] = record
        return record

    async def stop(self, task_id: str) -> BackgroundTask:
        """Stop a running task: SIGTERM → grace → SIGKILL (killpg for shells)."""
        record = self._require(task_id)
        if record.status != "running":
            return record
        record.stop_requested = True
        self._terminate(record)
        if not await _await_done(record, STOP_GRACE_S):
            self._kill(record)
            await _await_done(record, 2.0)
        return record

    async def wait(self, task_id: str, timeout: float) -> BackgroundTask:
        """Await completion (up to ``timeout``) and return the record."""
        record = self._require(task_id)
        if record.status == "running":
            await _await_done(record, timeout)
        return record

    async def shutdown(self) -> None:
        """Stop every live task (SIGTERM → grace → SIGKILL); used at teardown."""
        running = [r for r in self._records.values() if r.status == "running"]
        for record in running:
            record.stop_requested = True
            self._terminate(record)
        if not running:
            return
        await asyncio.wait([r.task for r in running], timeout=STOP_GRACE_S)
        stragglers = [r for r in running if not r.task.done()]
        for record in stragglers:
            self._kill(record)
        if stragglers:
            await asyncio.wait([r.task for r in stragglers], timeout=2.0)
            for record in stragglers:
                if not record.task.done():
                    record.task.cancel()
            await asyncio.gather(*(r.task for r in stragglers), return_exceptions=True)

    # -- notifications -------------------------------------------------------------

    def drain_notifications(self) -> list[str]:
        """Pop the completion notifications queued since the last drain."""
        pending, self._pending = self._pending, []
        return pending

    def _announce(self, record: BackgroundTask) -> None:
        message = notification_text(record)
        self._pending.append(message)
        if self.notify is None:
            return
        try:
            result = self.notify(message)
            if inspect.isawaitable(result):
                task = asyncio.ensure_future(result)
                self._notify_tasks.add(task)
                task.add_done_callback(self._notify_tasks.discard)
        except Exception:
            pass  # notifications never break the task

    # -- internals --------------------------------------------------------------------

    def _terminate(self, record: BackgroundTask) -> None:
        if record.term is not None:
            with contextlib.suppress(OSError):
                record.term()
        else:
            record.task.cancel()

    def _kill(self, record: BackgroundTask) -> None:
        if record.kill is not None:
            with contextlib.suppress(OSError):
                record.kill()
        else:
            record.task.cancel()


def start_shell_task(
    manager: BackgroundTaskManager,
    command: str,
    cwd: Path,
    timeout: float,
    idle_timeout: float,
) -> BackgroundTask:
    """Spawn a detached shell command with the bash tool's kill discipline."""
    from lecode.agent.tools.bash import _kill_tree, _run_shell

    proc_slot: dict[str, Any] = {}

    async def body(emit: Emit) -> tuple[str, int | None]:
        output, exit_code, timed_out, idle_killed = await _run_shell(
            command,
            cwd,
            timeout,
            idle_timeout,
            TAIL_CAP_BYTES,
            on_chunk=emit,
            proc_slot=proc_slot,
        )
        text = output.decode("utf-8", errors="replace").rstrip("\n") or "(no output)"
        notes: list[str] = []
        if timed_out:
            notes.append(f"timed out after {timeout}s")
        if idle_killed:
            notes.append(f"killed: no output for {idle_timeout}s")
        if exit_code != 0 and not (timed_out or idle_killed):
            notes.append(f"exit code {exit_code}")
        if notes:
            text += f"\n[{'; '.join(notes)}]"
        return text, exit_code

    def term() -> None:
        proc = proc_slot.get("proc")
        if proc is not None:
            import signal

            os.killpg(proc.pid, signal.SIGTERM)

    def kill() -> None:
        proc = proc_slot.get("proc")
        if proc is not None:
            _kill_tree(proc)

    return manager.start("bash", command, body, term=term, kill=kill)
