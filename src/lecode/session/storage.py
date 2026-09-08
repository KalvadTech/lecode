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
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def _latest_rename_name(records: list[Record]) -> str | None:
    """The last rename event's name, or ``None`` if the session was never renamed."""
    for record in reversed(records):
        if (
            isinstance(record, EventRecord)
            and record.kind == "rename"
            and (name := record.data.get("name"))
        ):
            return str(name)
    return None


@dataclass
class Session:
    """Handle for an open session file."""

    meta: MetaRecord
    path: Path
    next_seq: int = 1
    #: Transient: this process auto-named the session and may auto-title it
    #: from the first user message. Never persisted; explicit names stay.
    auto_title: bool = False

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def name(self) -> str:
        return self.meta.name


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

    # -- creation / opening -------------------------------------------------

    def create(
        self,
        name: str,
        cwd: str | Path,
        model: str | None = None,
        agent: str = "build",
        *,
        auto_title: bool = False,
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
        return Session(meta=meta, path=path, next_seq=1, auto_title=auto_title)

    def open(self, session_id: str) -> Session:
        """Open an existing session by id; the latest rename event wins."""
        path = self.sessions_dir / f"{session_id}.jsonl"
        if not path.is_file():
            raise SessionNotFoundError(session_id)
        records = self._read_records_at(path)
        meta = next((r for r in records if isinstance(r, MetaRecord)), None)
        if meta is None:
            raise SessionNotFoundError(f"{session_id} (no meta record)")
        if (new_name := _latest_rename_name(records)) is not None:
            meta = meta.model_copy(update={"name": new_name})
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
        # Record our pid for the contention message other processes show.
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
        return SessionLock(lock_path, fd)

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

    def _append(self, session: Session, record: Record) -> None:
        line = record.model_dump_json()
        with session.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())
        if isinstance(record, MessageRecord | EventRecord | TombstoneRecord):
            session.next_seq = record.seq + 1

    def append_message(
        self,
        session: Session,
        message: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> MessageRecord:
        record = MessageRecord(
            seq=session.next_seq,
            ts=_now(),
            role=str(message.get("role", "unknown")),
            message=dict(message),
            usage=usage,
        )
        self._append(session, record)
        return record

    def append_event(
        self, session: Session, kind: str, data: dict[str, Any] | None = None
    ) -> EventRecord:
        record = EventRecord(seq=session.next_seq, ts=_now(), kind=kind, data=data or {})
        self._append(session, record)
        return record

    def rename(self, session: Session, name: str) -> EventRecord:
        """Append a rename event and update the live handle's name in one step.

        Callers are responsible for validating/deduplicating ``name``.
        """
        record = self.append_event(session, "rename", {"name": name})
        session.meta.name = name
        return record

    def append_tombstone(self, session: Session, up_to_seq: int) -> TombstoneRecord:
        record = TombstoneRecord(seq=session.next_seq, ts=_now(), up_to_seq=up_to_seq)
        self._append(session, record)
        return record

    # -- reading ------------------------------------------------------------

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

        The latest ``rename`` event's name wins, so listings, name-based
        resume and deduplication all see renames. ``cwd`` scopes the listing
        to sessions created in that folder, resume never crosses directories.
        """
        if not self.sessions_dir.is_dir():
            return []
        metas: list[MetaRecord] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            meta: MetaRecord | None = None
            records: list[Record] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    record = parse_record(line)
                    if record is None:
                        self.corrupt_lines += 1
                        continue
                    if meta is None:
                        if not isinstance(record, MetaRecord):
                            break  # first valid record is not meta: skip file
                        meta = record
                    records.append(record)
            if meta is None:
                continue
            if (renamed := _latest_rename_name(records)) is not None:
                meta = meta.model_copy(update={"name": renamed})
            metas.append(meta)
        if cwd is not None:
            wanted = str(cwd)
            metas = [m for m in metas if m.cwd == wanted]
        return sorted(metas, key=lambda m: (m.created_at, m.id), reverse=True)

    def delete(self, session_id: str) -> None:
        path = self.sessions_dir / f"{session_id}.jsonl"
        if not path.is_file():
            raise SessionNotFoundError(session_id)
        path.unlink()
        path.with_suffix(".lock").unlink(missing_ok=True)

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

    def load_messages(self, session: Session) -> list[MessageRecord]:
        """The logical message history with tombstones applied."""
        records = self.read_records(session)
        tombstones = self._active_tombstones(records)
        return [
            r
            for r in records
            if isinstance(r, MessageRecord) and not self._is_hidden(r.seq, tombstones)
        ]

    def undo(self, session: Session) -> TombstoneRecord | None:
        """Hide the last user turn (the user message and everything after it)."""
        visible = self.load_messages(session)
        last_user = next((m for m in reversed(visible) if m.role == "user"), None)
        if last_user is None:
            return None
        return self.append_tombstone(session, up_to_seq=last_user.seq - 1)

    def redo(self, session: Session) -> bool:
        """Cancel the most recent tombstone — only while it is the last record."""
        records = self.read_records(session)
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

    def compact(self, session: Session, summary: str, keep_from_seq: int) -> EventRecord:
        """Record a compaction: summary + first kept message seq."""
        return self.append_event(
            session, "compact", {"summary": summary, "keep_from_seq": keep_from_seq}
        )

    def load_for_model(self, session: Session) -> list[dict[str, Any]]:
        """What gets replayed into the model context.

        The latest compact event (if any) contributes a leading system message
        with the summary plus the kept tail; tombstones still apply. A later
        ``clear`` event supersedes the compaction: everything before it is
        hidden and no summary is injected.
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
            keep_from = int(latest.data.get("keep_from_seq", 0))
            out.append({"role": "system", "content": str(latest.data.get("summary", ""))})
        if clear_from is not None and (keep_from is None or clear_from >= keep_from):
            keep_from = clear_from
        for r in records:
            if not isinstance(r, MessageRecord):
                continue
            if self._is_hidden(r.seq, tombstones):
                continue
            if keep_from is not None and r.seq < keep_from:
                continue
            out.append(dict(r.message))
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
