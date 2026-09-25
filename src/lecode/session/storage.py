"""Append-only JSONL session storage.

One file per session at ``<config_dir>/sessions/<id>.jsonl``. All mutation is
append; undo/redo/rewind/compaction are expressed as tombstone and event
records, so the full history always survives on disk.

Replay semantics: a tombstone hides the suffix that existed when it was
written — records with ``up_to_seq < seq < tombstone.seq`` — so messages
appended after an undo stay visible. A later ``redo`` event cancels a
tombstone, making its range visible again.

Attach locking: a live process holds an ``flock`` on the sidecar
``<id>.lock`` (see :meth:`SessionStore.acquire_lock`), so a second lecode
cannot attach to the same session. The lock dies with the process — no
stale-lock cleanup is ever needed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from lecode.session.model import (
    EventRecord,
    MessageRecord,
    MetaRecord,
    Record,
    TombstoneRecord,
    parse_record,
)

try:
    import fcntl
except ImportError:  # Windows: best-effort, attach locking disabled
    fcntl = None  # type: ignore[assignment]

#: Caps for one persisted agent-run activity record (bounded trails).
AGENT_RUN_ANSWER_CAP = 32 * 1024
AGENT_RUN_PROMPT_CAP = 2000
AGENT_RUN_RESULT_CAP = 2000
AGENT_RUN_ARGS_CAP = 500
AGENT_RUN_MAX_TOOLS = 100
_TRUNCATED_MARK = "… (truncated)"


def _clip_activity(text: str, cap: int) -> tuple[str, bool]:
    """Clip ``text`` to ``cap`` chars; the flag reports a truncation."""
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n{_TRUNCATED_MARK}", True


def _bounded_agent_run(run: dict[str, Any]) -> dict[str, Any]:
    """Apply the activity caps so one run can never bloat the session file."""
    truncated = False
    data: dict[str, Any] = {
        "run_id": str(run.get("run_id") or ""),
        "agent": str(run.get("agent") or ""),
        "description": str(run.get("description") or ""),
        "status": str(run.get("status") or ""),
    }
    for key, cap in (("prompt", AGENT_RUN_PROMPT_CAP), ("answer", AGENT_RUN_ANSWER_CAP)):
        data[key], clipped = _clip_activity(str(run.get(key) or ""), cap)
        truncated |= clipped
    tools: list[dict[str, Any]] = []
    raw_tools = run.get("tool_calls")
    raw_tools = raw_tools if isinstance(raw_tools, list) else []
    for raw in raw_tools[:AGENT_RUN_MAX_TOOLS]:
        if not isinstance(raw, dict):
            continue
        entry: dict[str, Any] = {
            "name": str(raw.get("name") or ""),
            "is_error": bool(raw.get("is_error")),
        }
        for key, cap in (("args", AGENT_RUN_ARGS_CAP), ("result", AGENT_RUN_RESULT_CAP)):
            entry[key], clipped = _clip_activity(str(raw.get(key) or ""), cap)
            truncated |= clipped
        tools.append(entry)
    truncated |= len(raw_tools) > AGENT_RUN_MAX_TOOLS
    data["tool_calls"] = tools
    for key in ("turns", "input_tokens", "output_tokens"):
        data[key] = int(run.get(key) or 0)
    data["cost_usd"] = float(run.get("cost_usd") or 0.0)
    data["duration_s"] = float(run.get("duration_s") or 0.0)
    data["truncated"] = truncated
    return data


class SessionNotFoundError(KeyError):
    """No session matched the reference."""


class AmbiguousSessionError(KeyError):
    """An id-prefix reference matched more than one session."""

    def __init__(self, ref: str, matches: list[str]) -> None:
        self.ref = ref
        self.matches = matches
        super().__init__(f"ambiguous session '{ref}', matches: {', '.join(matches)}")


class SessionInUseError(Exception):
    """The session is already attached to a live lecode process."""

    def __init__(self, name: str, holder_pid: int | None = None) -> None:
        self.name = name
        self.holder_pid = holder_pid
        detail = f" (pid {holder_pid})" if holder_pid is not None else ""
        super().__init__(f"session '{name}' is already open in another lecode process{detail}")


class SessionLock:
    """An ``flock`` held on the session's ``.lock`` sidecar file.

    The lock is released by closing the fd, and automatically if the process
    dies. The file itself is never unlinked on release — a new opener must
    lock the same inode the holder has locked.
    """

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd: int | None = fd

    def release(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _next_seq(records: list[Record]) -> int:
    return 1 + max(
        (r.seq for r in records if isinstance(r, MessageRecord | EventRecord | TombstoneRecord)),
        default=0,
    )


@dataclass
class Session:
    """Handle for an open session file."""

    meta: MetaRecord
    path: Path
    next_seq: int = 1

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def name(self) -> str:
        return self.meta.name


@dataclass(frozen=True)
class SourceRef:
    """Exact immutable message identity; no inferred sequence boundaries."""

    session_id: str
    seqs: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class SourceSnapshot:
    status: str
    ref: SourceRef | None = None
    messages: tuple[dict[str, Any], ...] = ()


class SessionStore:
    """Create, append to, list, and replay JSONL sessions."""

    def __init__(self, config_dir: Path | None = None, *, fsync: bool = False) -> None:
        if config_dir is None:
            from lecode.config.loader import config_dir as default_config_dir

            config_dir = default_config_dir()
        self.config_dir = Path(config_dir)
        self.sessions_dir = self.config_dir / "sessions"
        self.fsync = fsync
        #: Number of corrupt lines skipped while reading, cumulative.
        self.corrupt_lines = 0
        self._fact_stores: dict[Path, Any] = {}
        self._attached: dict[str, tuple[Session, SessionLock]] = {}
        self.exclusion_reader: Callable[[str], frozenset[int]] = lambda session_id: (
            self._project_facts(session_id).excluded_seqs(session_id)
        )

    def bind_facts(self, project_root: Path, facts: Any) -> None:
        """Bind by durable project identity, never the last runtime's cwd."""
        from lecode.memory.store import resolve_project_root

        root = resolve_project_root(project_root)
        previous = self._fact_stores.get(root)
        if previous is not None and previous is not facts:
            previous.close()
        self._fact_stores[root] = facts

    def _project_facts(self, session_id: str):
        from lecode.memory.facts import FactStore
        from lecode.memory.store import memory_root, resolve_project_root

        session = self.open(session_id)
        root = resolve_project_root(session.meta.cwd)
        if root not in self._fact_stores:
            self._fact_stores[root] = FactStore(
                memory_root(root, self.config_dir) / "facts.sqlite3"
            )
        return self._fact_stores[root]

    def memory_generation(self, session_id: str) -> int:
        return self._project_facts(session_id).generation()

    def close(self) -> None:
        """Release project fact connections owned or bound by this session-store lifetime."""
        for facts in self._fact_stores.values():
            facts.close()
        self._fact_stores.clear()

    # -- creation / opening -------------------------------------------------

    def create(
        self,
        name: str,
        cwd: str | Path,
        model: str | None = None,
        agent: str = "build",
    ) -> Session:
        """Create a new session file with its meta line."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        meta = MetaRecord(
            id=_new_id(),
            name=name,
            cwd=str(cwd),
            created_at=_now(),
            agent=agent,
            model=model,
        )
        path = self.sessions_dir / f"{meta.id}.jsonl"
        path.write_text(meta.model_dump_json() + "\n", encoding="utf-8")
        return Session(meta=meta, path=path, next_seq=1)

    def open(self, session_id: str) -> Session:
        """Open an existing session by id; the latest rename event wins."""
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("invalid session ID")
        path = self.sessions_dir / f"{session_id}.jsonl"
        if path.is_symlink():
            raise ValueError("session symlinks are not allowed")
        if not path.is_file():
            raise SessionNotFoundError(session_id)
        records = self._read_records_at(path)
        meta = next((r for r in records if isinstance(r, MetaRecord)), None)
        if meta is None:
            raise SessionNotFoundError(f"{session_id} (no meta record)")
        renames = [r for r in records if isinstance(r, EventRecord) and r.kind == "rename"]
        if renames and (new_name := renames[-1].data.get("name")):
            meta = meta.model_copy(update={"name": str(new_name)})
        return Session(meta=meta, path=path, next_seq=_next_seq(records))

    def acquire_lock(self, session: Session) -> SessionLock | None:
        """Lock the session against a second live lecode process.

        Returns the held lock (keep it for the process's attachment lifetime),
        or ``None`` on platforms without ``flock``. Raises
        :class:`SessionInUseError` when another process holds the lock.
        """
        if fcntl is None:
            return None
        lock_path = session.path.with_suffix(".lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder_pid: int | None = None
            with contextlib.suppress(OSError, ValueError):
                holder_pid = int(lock_path.read_text(encoding="utf-8").strip())
            os.close(fd)
            raise SessionInUseError(session.name, holder_pid) from None
        if not session.path.is_file():
            os.close(fd)
            raise SessionNotFoundError(session.id)
        # Record our pid for the contention message other processes show.
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
        lock = SessionLock(lock_path, fd)
        self._attached[session.id] = (session, lock)
        self._flush_forgets_locked(session)
        return lock

    def flush_forgets(self, session: Session | str) -> None:
        """Best-effort observability: reuse our attach lock or defer to another holder."""
        session_id = session.id if isinstance(session, Session) else session
        attached = self._attached.get(session_id)
        if attached is not None and attached[1].held:
            self._flush_forgets_locked(attached[0])
            return
        lock = None
        try:
            lock = self.acquire_lock(
                session if isinstance(session, Session) else self.open(session_id)
            )
        except (SessionInUseError, SessionNotFoundError, OSError, ValueError):
            pass  # Missing/locked sessions keep their pending IDs; no aliases or scans.
        finally:
            if lock is not None:
                lock.release()

    def _flush_forgets_locked(self, session: Session) -> None:
        try:
            facts = self._project_facts(session.id)
            pending = facts.pending_forgets(session.id)
            if not pending:
                return
            records = self.read_records(session)
            delivered = {
                r.data.get("fact_id")
                for r in records
                if isinstance(r, EventRecord) and r.kind == "forget"
            }
            session.next_seq = _next_seq(records)
            for fact_id in pending:
                if fact_id not in delivered:
                    self.append_event(session, "forget", {"fact_id": fact_id}, durable=True)
                else:
                    # Retry after append/fsync but before SQLite acknowledgement.
                    with session.path.open("rb") as stream:
                        os.fsync(stream.fileno())
                facts.acknowledge_forget(fact_id, session.id)
        except (OSError, ValueError, sqlite3.Error, SessionNotFoundError):
            # A failed fsync may follow a complete append. Keep the attached writer's
            # sequence in sync even though acknowledgement must wait for a retry.
            with contextlib.suppress(OSError):
                session.next_seq = _next_seq(self.read_records(session))

    def lock_holder(self, session_id: str) -> int | None:
        """Pid of the live process holding this session's lock, or None if free.

        None also on platforms without ``flock`` or when no lock file exists.
        """
        if fcntl is None:
            return None
        lock_path = self.sessions_dir / f"{session_id}.lock"
        if not lock_path.is_file():
            return None
        fd = os.open(lock_path, os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                with contextlib.suppress(OSError, ValueError):
                    return int(lock_path.read_text(encoding="utf-8").strip())
                return -1  # locked but pid unreadable
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        finally:
            os.close(fd)

    # -- appending ----------------------------------------------------------

    def _append(self, session: Session, record: Record, *, durable: bool = False) -> None:
        line = record.model_dump_json().encode("utf-8") + b"\n"
        with session.path.open("a+b") as f:
            if f.tell():
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")  # Preserve a torn record without swallowing the next append.
            f.write(line)
            f.flush()
            if self.fsync or durable:
                os.fsync(f.fileno())
        if isinstance(record, MessageRecord | EventRecord | TombstoneRecord):
            session.next_seq = record.seq + 1

    def append_message(
        self,
        session: Session,
        message: dict[str, Any],
        usage: dict[str, Any] | None = None,
        *,
        memory_generation: int | None = None,
    ) -> MessageRecord:
        record = MessageRecord(
            seq=session.next_seq,
            ts=_now(),
            role=str(message.get("role", "unknown")),
            message=dict(message),
            usage=usage,
            memory_generation=memory_generation,
        )
        self._append(session, record)
        return record

    def append_event(
        self,
        session: Session,
        kind: str,
        data: dict[str, Any] | None = None,
        *,
        durable: bool = False,
    ) -> EventRecord:
        record = EventRecord(seq=session.next_seq, ts=_now(), kind=kind, data=data or {})
        self._append(session, record, durable=durable)
        return record

    def append_tombstone(self, session: Session, up_to_seq: int) -> TombstoneRecord:
        record = TombstoneRecord(seq=session.next_seq, ts=_now(), up_to_seq=up_to_seq)
        self._append(session, record)
        return record

    def record_agent_run(self, session: Session, run: dict[str, Any]) -> EventRecord:
        """Append one bounded agent-run activity record (kind ``agent_run``)."""
        return self.append_event(session, "agent_run", _bounded_agent_run(run))

    # -- reading ------------------------------------------------------------

    def source_snapshot(
        self,
        session_id: str,
        start_seq: int,
        end_seq: int,
        *,
        project_root: Path,
        excluded: frozenset[int] = frozenset(),
        purpose: Literal["working", "durable"] = "working",
    ) -> SourceSnapshot:
        """Capture a bounded visible range by exact ID, never a path or global scan."""
        from lecode.memory.store import resolve_project_root

        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("invalid session ID")
        if (
            type(start_seq) is not int
            or type(end_seq) is not int
            or not 1 <= start_seq <= end_seq < 2**63
            or end_seq - start_seq >= 200
        ):
            raise ValueError("source range must contain at most 200 sequence positions")
        path = self.sessions_dir / f"{session_id}.jsonl"
        if path.is_symlink():
            raise ValueError("session symlinks are not allowed")
        if not path.is_file():
            return SourceSnapshot("missing")
        try:
            raw = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            parsed = [parse_record(json.dumps(record)) for record in raw]
        except (ValueError, UnicodeError, OSError):
            return SourceSnapshot("stale")
        if not parsed or any(record is None for record in parsed):
            return SourceSnapshot("stale")
        records = [record for record in parsed if record is not None]
        meta = records[0]
        if not isinstance(meta, MetaRecord) or meta.id != session_id:
            return SourceSnapshot("stale")
        if any(isinstance(record, MetaRecord) for record in records[1:]):
            return SourceSnapshot("stale")
        if any(type(record.get("seq")) is not int for record in raw[1:]):
            return SourceSnapshot("stale")
        if resolve_project_root(meta.cwd) != resolve_project_root(project_root):
            raise ValueError("source belongs to another project")
        excluded = excluded | self.exclusion_reader(session_id)
        seqs = [r.seq for r in records if not isinstance(r, MetaRecord)]
        if len(seqs) != len(set(seqs)) or seqs != sorted(seqs):
            return SourceSnapshot("stale")
        selected = [
            r for r in records if isinstance(r, MessageRecord) and start_seq <= r.seq <= end_seq
        ]
        if not selected:
            return SourceSnapshot("missing")
        visible = {
            r.seq
            for r in self._visible_messages(
                records, self._active_tombstones(records), purpose=purpose
            )
        }
        if any(r.seq not in visible or r.seq in excluded for r in selected):
            return SourceSnapshot("hidden")
        exact = [r for r in raw if r.get("type") == "message" and start_seq <= r["seq"] <= end_seq]
        digest = hashlib.sha256(
            json.dumps(exact, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        ref = SourceRef(session_id, tuple(r.seq for r in selected), digest)
        return SourceSnapshot("valid", ref, tuple(r.message for r in selected))

    def validate_source(
        self,
        ref: SourceRef,
        *,
        project_root: Path,
        excluded: frozenset[int] = frozenset(),
        purpose: Literal["working", "durable"] = "working",
    ) -> SourceSnapshot:
        """Recheck identity, complete records and current replay visibility."""
        if (
            not ref.seqs
            or any(type(seq) is not int for seq in ref.seqs)
            or tuple(sorted(set(ref.seqs))) != ref.seqs
        ):
            raise ValueError("invalid source sequences")
        snapshot = self.source_snapshot(
            ref.session_id,
            ref.seqs[0],
            ref.seqs[-1],
            project_root=project_root,
            excluded=excluded,
            purpose=purpose,
        )
        if snapshot.status != "valid":
            return snapshot
        assert snapshot.ref is not None
        if snapshot.ref.seqs != ref.seqs:
            return SourceSnapshot("missing")
        if snapshot.ref.digest != ref.digest:
            return SourceSnapshot("stale")
        return snapshot

    def _read_records_at(self, path: Path) -> list[Record]:
        records: list[Record] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = parse_record(line)
                if record is None:
                    self.corrupt_lines += 1
                else:
                    records.append(record)
        return records

    def read_records(self, session: Session) -> list[Record]:
        """All records in file order; corrupt lines are skipped and counted."""
        return self._read_records_at(session.path)

    def list_sessions(self, cwd: Path | str | None = None) -> list[MetaRecord]:
        """All sessions' meta records, most recent first.

        ``cwd`` scopes the listing to sessions created in that folder —
        resume never crosses directories.
        """
        if not self.sessions_dir.is_dir():
            return []
        metas: list[MetaRecord] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    record = parse_record(line)
                    if isinstance(record, MetaRecord):
                        metas.append(record)
                        break
                    if record is None:
                        self.corrupt_lines += 1
                        continue
                    break  # first valid record is not meta: skip file
        if cwd is not None:
            wanted = str(cwd)
            metas = [m for m in metas if m.cwd == wanted]
        return sorted(metas, key=lambda m: (m.created_at, m.id), reverse=True)

    def delete(self, session_id: str) -> None:
        session = self.open(session_id)
        lock = self.acquire_lock(session)
        try:
            session.path.unlink()
        finally:
            if lock is not None:
                lock.release()

    def resolve(self, ref: str | None, cwd: Path | str | None = None) -> MetaRecord:
        """Resolve a reference by id, unique id prefix, exact name, or recency.

        ``None``/``"latest"`` resolve to the most recently created session.
        ``cwd`` restricts the candidates to sessions created in that folder.
        """
        sessions = self.list_sessions(cwd)
        if not sessions:
            raise SessionNotFoundError(ref or "(no sessions)")
        if ref is None or ref == "latest":
            return sessions[0]
        for meta in sessions:
            if meta.id == ref:
                return meta
        prefix = [m for m in sessions if m.id.startswith(ref)]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            raise AmbiguousSessionError(ref, sorted(m.id for m in prefix))
        for meta in sessions:
            if meta.name == ref:
                return meta
        raise SessionNotFoundError(ref)

    def import_session(self, source: str | Path) -> Session:
        """Validate and copy an external JSONL session into the store.

        Id collisions get a fresh id; name collisions get a ``-2`` suffix.
        """
        from lecode.session.naming import unique_name

        records: list[Record] = []
        with Path(source).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = parse_record(line)
                if record is None:
                    self.corrupt_lines += 1
                else:
                    records.append(record)
        if not records or not isinstance(records[0], MetaRecord):
            raise ValueError(f"not a lecode session file (first record must be meta): {source}")

        meta = records[0]
        existing_ids = {m.id for m in self.list_sessions()}
        if meta.id in existing_ids:
            meta = meta.model_copy(update={"id": _new_id()})
        meta = meta.model_copy(update={"name": unique_name(meta.name, self)})

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{meta.id}.jsonl"
        lines = [meta.model_dump_json()] + [r.model_dump_json() for r in records[1:]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return Session(meta=meta, path=path, next_seq=_next_seq(records))

    # -- replay -------------------------------------------------------------

    def _active_tombstones(self, records: list[Record]) -> list[TombstoneRecord]:
        """Tombstones not cancelled by a later ``redo`` event."""
        cancelled = {
            r.data.get("cancels_seq")
            for r in records
            if isinstance(r, EventRecord) and r.kind == "redo"
        }
        return [r for r in records if isinstance(r, TombstoneRecord) and r.seq not in cancelled]

    def _is_hidden(self, record_seq: int, tombstones: list[TombstoneRecord]) -> bool:
        return any(t.up_to_seq < record_seq < t.seq for t in tombstones)

    def _visible_messages(
        self,
        records: list[Record],
        tombstones: list[TombstoneRecord],
        *,
        purpose: Literal["working", "durable"] = "working",
        include_filtered: bool = False,
    ) -> list[MessageRecord]:
        """Messages replay sees: tombstones and the latest ``clear`` applied."""
        if purpose not in {"working", "durable"}:
            raise ValueError("invalid source validation purpose")
        clears = [
            r
            for r in records
            if isinstance(r, EventRecord) and r.kind == "clear" and purpose == "working"
        ]
        clear_from = clears[-1].seq if clears else None
        meta = next((r for r in records if isinstance(r, MetaRecord)), None)
        excluded = self.exclusion_reader(meta.id) if meta and not include_filtered else frozenset()
        generation = self.memory_generation(meta.id) if meta and not include_filtered else 0
        return [
            r
            for r in records
            if isinstance(r, MessageRecord)
            and not self._is_hidden(r.seq, tombstones)
            and (clear_from is None or r.seq > clear_from)
            and r.seq not in excluded
            # ponytail: any forget invalidates older generated content project-wide.
            # Precise transitive lineage can narrow this conservative cutoff later.
            and (
                include_filtered
                or (r.role == "user" and r.memory_generation is None)
                or (r.memory_generation or 0) == generation
            )
        ]

    def visible_messages(self, session: Session) -> list[MessageRecord]:
        """The messages compaction may summarize, matching replay visibility."""
        records = self.read_records(session)
        return self._visible_messages(records, self._active_tombstones(records))

    def compaction_input(self, session: Session) -> tuple[list[MessageRecord], list[MessageRecord]]:
        """Visible content plus original exchange structure to prove filtered turns complete.

        The second list is structural evidence only; filtered content must never be encoded.
        Clear and undo apply to both lists, so cancelled turns cannot be bridged.
        """
        records = self.read_records(session)
        tombstones = self._active_tombstones(records)
        return (
            self._visible_messages(records, tombstones),
            self._visible_messages(records, tombstones, include_filtered=True),
        )

    def refresh_model_history(
        self,
        session: Session,
        history: list[dict[str, Any]],
        *,
        reset: bool = False,
        fresh_request: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Classify cached persisted content using the same replay/visibility rules."""
        records = self.read_records(session)
        replay = self.load_for_model(session)
        summaries = [
            {"role": "system", "content": r.data.get("summary", "")}
            for r in records
            if isinstance(r, EventRecord) and r.kind == "compact"
        ]
        raw = [r.message for r in records if isinstance(r, MessageRecord)]
        if reset:
            base = history[:1] if history and history[0].get("role") == "system" else []
            if base and base[0] in summaries:
                base = []
            rebuilt = [*base, *replay]
            if fresh_request is not None and fresh_request not in raw:
                rebuilt.append(fresh_request)
            return rebuilt
        if any(m in history and m not in replay[:1] for m in summaries):
            extras = [m for m in history if m not in summaries and m not in raw]
            split = 0
            while split < len(extras) and extras[split].get("role") == "system":
                split += 1
            history = [*extras[:split], *replay, *extras[split:]]
        visible = self._visible_messages(records, self._active_tombstones(records))
        visible_seqs = {r.seq for r in visible}
        hidden = [
            {key: value for key, value in r.message.items() if key != "incomplete"}
            for r in records
            if isinstance(r, MessageRecord) and r.seq not in visible_seqs
        ]
        # Preserve independently re-authored identical messages, not hidden copies.
        visible_text = [r.message for r in visible]
        return [m for m in history if m not in hidden or m in visible_text]

    def load_messages(self, session: Session) -> list[MessageRecord]:
        """The logical message history with tombstones applied."""
        records = self.read_records(session)
        tombstones = self._active_tombstones(records)
        return [
            r
            for r in records
            if isinstance(r, MessageRecord) and not self._is_hidden(r.seq, tombstones)
        ]

    def load_events(self, session: Session, kind: str) -> list[dict[str, Any]]:
        """Event records of ``kind`` with tombstones applied, in append order."""
        records = self.read_records(session)
        tombstones = self._active_tombstones(records)
        return [
            dict(r.data)
            for r in records
            if isinstance(r, EventRecord)
            and r.kind == kind
            and not self._is_hidden(r.seq, tombstones)
        ]

    def load_agent_runs(self, session: Session) -> list[dict[str, Any]]:
        """Agent-run activity records with tombstones applied, in append order."""
        return self.load_events(session, "agent_run")

    def undo(self, session: Session) -> TombstoneRecord | None:
        """Hide the last user turn (the user message and everything after it)."""
        visible = self.load_messages(session)
        last_user = next((m for m in reversed(visible) if m.role == "user"), None)
        if last_user is None:
            return None
        return self.append_tombstone(session, up_to_seq=last_user.seq - 1)

    def redo(self, session: Session) -> bool:
        """Cancel the latest tombstone; model-usage bookkeeping cannot consume redo."""
        records = [
            r
            for r in self.read_records(session)
            if not (isinstance(r, EventRecord) and r.kind in {"memory_usage", "forget"})
        ]
        if not records or not isinstance(records[-1], TombstoneRecord):
            return False
        cancelled = {
            r.data.get("cancels_seq")
            for r in records
            if isinstance(r, EventRecord) and r.kind == "redo"
        }
        if records[-1].seq in cancelled:
            return False
        self.append_event(session, "redo", {"cancels_seq": records[-1].seq})
        return True

    def rewind_to(self, session: Session, seq: int) -> TombstoneRecord:
        """Hide everything after ``seq``, leaving a restore point first."""
        self.append_event(session, "restore_point", {"seq": seq})
        return self.append_tombstone(session, up_to_seq=seq)

    # -- compaction ----------------------------------------------------------

    def source_version(
        self,
        session: Session,
        *,
        sync: bool = False,
        include_derivations: bool = True,
        include_worker_events: bool = True,
    ) -> tuple | None:
        """Identity + source bytes; worker control events may be excluded at model boundaries."""
        if not session.path.is_file():
            return None
        with session.path.open("rb") as source:
            if sync:
                os.fsync(source.fileno())
            stat = os.fstat(source.fileno())
            if include_derivations and include_worker_events:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            else:
                hasher = hashlib.sha256()
                for line in source:
                    record = parse_record(line.decode("utf-8"))
                    if isinstance(record, EventRecord):
                        if not include_derivations and record.kind in {"compact", "memory_usage"}:
                            continue
                        if not include_worker_events and record.kind in {
                            "worker",
                            "worker_inbox",
                            "worker_notification",
                            "worker_notification_ack",
                            "worker_usage",
                            "worker_usage_checkpoint",
                        }:
                            continue
                    hasher.update(line)
                digest = hasher.hexdigest()
        return (
            session.id,
            str(session.path),
            stat.st_ino,
            digest,
            self.exclusion_reader(session.id),
            self.memory_generation(session.id),
        )

    def capture_sources(self, session: Session, messages: list[MessageRecord]) -> list[dict]:
        """Full lineage in Phase 3 snapshots, each at most 200 sequence positions."""
        refs: list[dict] = []
        offset = 0
        while offset < len(messages):
            end = offset + 1
            while (
                end < len(messages)
                and messages[end].seq == messages[end - 1].seq + 1
                and messages[end].seq - messages[offset].seq < 200
            ):
                end += 1
            snapshot = self.source_snapshot(
                session.id,
                messages[offset].seq,
                messages[end - 1].seq,
                project_root=Path(session.meta.cwd),
            )
            if snapshot.status != "valid" or snapshot.ref is None:
                raise ValueError("source is no longer valid")
            refs.append(asdict(snapshot.ref))
            offset = end
        return refs

    @staticmethod
    def summary_identity(event: EventRecord) -> dict:
        return {
            "seq": event.seq,
            "digest": hashlib.sha256(event.model_dump_json().encode()).hexdigest(),
        }

    def working_summary(
        self, session: Session, *, excluded: frozenset[int] = frozenset()
    ) -> EventRecord | None:
        """Latest validated working summary. Exclusions are the Phase 5 read seam."""
        records = self.read_records(session)
        events = {r.seq: r for r in records if isinstance(r, EventRecord)}
        compacts = [r for r in events.values() if r.kind == "compact"]
        if not compacts:
            return None
        latest = compacts[-1]
        tombstones = self._active_tombstones(records)
        if any(r.kind == "clear" and r.seq > latest.seq for r in events.values()):
            return None
        try:
            current = latest
            while True:
                if self._summary_intersects_tombstone(current, tombstones):
                    return None
                parent = current.data.get("prior_summary")
                if parent is None:
                    break
                previous = events[parent["seq"]]
                if previous.seq >= current.seq or self.summary_identity(previous) != parent:
                    return None
                current = previous
            refs = latest.data["source_refs"]
            seqs = []
            for data in refs:
                ref = SourceRef(data["session_id"], tuple(data["seqs"]), data["digest"])
                if (
                    ref.session_id != session.id
                    or self.validate_source(
                        ref, project_root=Path(session.meta.cwd), excluded=excluded
                    ).status
                    != "valid"
                ):
                    return None
                seqs.extend(ref.seqs)
            expected = [
                m.seq
                for m in self.visible_messages(session)
                if m.seq < latest.data["keep_from_seq"]
            ]
            if not seqs or seqs != expected:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return latest

    def compact(
        self,
        session: Session,
        summary: str,
        keep_from_seq: int,
        *,
        source_start_seq: int | None = None,
        source_end_seq: int | None = None,
        source_refs: list[dict] | None = None,
        prior_summary: dict | None = None,
        expected_version: tuple | None = None,
        usage: dict | None = None,
        model: str | None = None,
    ) -> EventRecord | None:
        """Atomically append summary + validated lineage after durable source writes."""
        if source_refs is None:
            expected_version = self.source_version(session, sync=True)
            source_refs = self.capture_sources(
                session, [m for m in self.visible_messages(session) if m.seq < keep_from_seq]
            )
        data: dict[str, Any] = {"summary": summary, "keep_from_seq": keep_from_seq}
        data.update(usage=usage, model=model)
        if source_start_seq is not None and source_end_seq is not None:
            data["source_start_seq"] = source_start_seq
            data["source_end_seq"] = source_end_seq
        if source_refs is not None:
            data["source_refs"] = source_refs
            data["prior_summary"] = prior_summary
        if expected_version is not None and self.source_version(session) != expected_version:
            return None
        record = EventRecord(
            seq=_next_seq(self.read_records(session)), ts=_now(), kind="compact", data=data
        )
        self._append(session, record, durable=True)
        return record

    def _summary_intersects_tombstone(
        self, compact: EventRecord, tombstones: list[TombstoneRecord]
    ) -> bool:
        """Whether an active tombstone undoes the compaction or its coverage.

        A compact event inside a tombstoned suffix was itself undone; legacy
        events without a recorded source range can only be checked this way.
        A recorded range intersects when it overlaps the hidden window
        ``(up_to_seq, tombstone.seq)``.
        """
        if self._is_hidden(compact.seq, tombstones):
            return True
        start = compact.data.get("source_start_seq")
        end = compact.data.get("source_end_seq")
        if start is None or end is None:
            return False
        return any(int(start) < t.seq and int(end) > t.up_to_seq for t in tombstones)

    def load_for_model(self, session: Session) -> list[dict[str, Any]]:
        """What gets replayed into the model context.

        The latest compact event contributes a leading system message with the
        summary plus the kept tail, but only while no active tombstone undoes
        it or intersects its covered range. A later ``clear`` event supersedes
        the compaction: everything before it is hidden and no summary is
        injected. Visibility otherwise matches :meth:`visible_messages`.
        """
        records = self.read_records(session)
        tombstones = self._active_tombstones(records)
        compacts = [r for r in records if isinstance(r, EventRecord) and r.kind == "compact"]
        clears = [r for r in records if isinstance(r, EventRecord) and r.kind == "clear"]
        clear_from = clears[-1].seq if clears else None
        keep_from: int | None = None
        out: list[dict[str, Any]] = []
        if compacts and (clear_from is None or compacts[-1].seq > clear_from):
            latest = compacts[-1]
            valid = self.working_summary(session) is not None
            if valid:
                keep_from = int(latest.data.get("keep_from_seq", 0))
                out.append({"role": "system", "content": str(latest.data.get("summary", ""))})
        if clear_from is not None and (keep_from is None or clear_from >= keep_from):
            keep_from = clear_from
        for r in self._visible_messages(records, tombstones):
            if keep_from is not None and r.seq < keep_from:
                continue
            out.append({key: value for key, value in r.message.items() if key != "incomplete"})
        return out

    # -- permission grants ----------------------------------------------------

    def grant_permission(self, session: Session, tool: str, pattern: str) -> EventRecord:
        """Persist a session-scoped "allow always" grant."""
        return self.append_event(session, "permission_grant", {"tool": tool, "pattern": pattern})

    def load_grants(self, session: Session) -> list[tuple[str, str]]:
        """All persisted (tool, pattern) permission grants."""
        return [
            (str(r.data.get("tool", "")), str(r.data.get("pattern", "")))
            for r in self.read_records(session)
            if isinstance(r, EventRecord) and r.kind == "permission_grant"
        ]
