"""Project facts as untrusted evidence, with source-linked revision history."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from lecode.session.storage import SessionStore, SourceRef


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    revision: int


@dataclass(frozen=True)
class Provenance:
    fact_id: str
    revision: int
    source_id: str
    source_seq: int
    created_at: str


class RevisionConflict(ValueError):
    """The fact changed since the caller last read it."""


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text without NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"{name} must be valid UTF-8 text") from e
    return value.strip()


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer from {minimum} to {2**63 - 1}")


class FactStore:
    """Lazy SQLite storage. Each instance owns its connection; writers serialize."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5)
            try:
                connection.execute("PRAGMA busy_timeout = 5000")
                deadline = time.monotonic() + 5
                while True:
                    try:
                        connection.execute("PRAGMA journal_mode = WAL")
                        break
                    except sqlite3.OperationalError as error:
                        if (
                            error.sqlite_errorcode & 0xFF
                            not in (
                                sqlite3.SQLITE_BUSY,
                                sqlite3.SQLITE_LOCKED,
                            )
                            or time.monotonic() >= deadline
                        ):
                            raise
                        time.sleep(0.01)
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS facts (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL CHECK(length(trim(text)) > 0),
                        revision INTEGER NOT NULL CHECK(revision > 0)
                    );
                    CREATE TABLE IF NOT EXISTS revisions (
                        fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                        revision INTEGER NOT NULL CHECK(revision > 0),
                        text TEXT NOT NULL CHECK(length(trim(text)) > 0),
                        PRIMARY KEY (fact_id, revision)
                    );
                    CREATE TABLE IF NOT EXISTS provenance (
                        fact_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        source_id TEXT NOT NULL CHECK(length(trim(source_id)) > 0),
                        source_seq INTEGER NOT NULL CHECK(source_seq >= 0),
                        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                        PRIMARY KEY (fact_id, revision),
                        FOREIGN KEY (fact_id, revision)
                            REFERENCES revisions(fact_id, revision) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS exclusions (
                        source_id TEXT NOT NULL CHECK(length(trim(source_id)) > 0),
                        source_seq INTEGER NOT NULL CHECK(source_seq >= 0),
                        PRIMARY KEY (source_id, source_seq)
                    );
                    CREATE TABLE IF NOT EXISTS source_refs (
                        fact_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        session_id TEXT NOT NULL,
                        seqs TEXT NOT NULL,
                        digest TEXT NOT NULL,
                        PRIMARY KEY (fact_id, revision),
                        FOREIGN KEY (fact_id, revision)
                            REFERENCES revisions(fact_id, revision) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS forgotten (fact_id TEXT PRIMARY KEY);
                    CREATE TABLE IF NOT EXISTS forget_events (
                        fact_id TEXT NOT NULL REFERENCES forgotten(fact_id),
                        session_id TEXT NOT NULL,
                        PRIMARY KEY (fact_id, session_id)
                    );
                    CREATE TABLE IF NOT EXISTS exclusion_generation (
                        id INTEGER PRIMARY KEY CHECK(id = 1), value INTEGER NOT NULL
                    );
                    INSERT OR IGNORE INTO exclusion_generation
                        SELECT 1, count(*) FROM exclusions;
                    CREATE TRIGGER IF NOT EXISTS exclusion_insert AFTER INSERT ON exclusions
                    BEGIN
                        UPDATE exclusion_generation SET value = value + 1 WHERE id = 1;
                    END;
                    CREATE TRIGGER IF NOT EXISTS forgotten_insert AFTER INSERT ON forgotten
                    BEGIN
                        UPDATE exclusion_generation SET value = value + 1 WHERE id = 1;
                    END;
                    PRAGMA user_version = 6;
                    COMMIT;
                """
                    if connection.execute("PRAGMA user_version").fetchone()[0] < 6
                    else ""
                )
            except BaseException:
                connection.close()
                raise
            self._connection = connection
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @contextmanager
    def _write(self):
        connection = self._connect()
        if connection.in_transaction:
            yield connection
        else:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                yield connection

    def generation(self) -> int:
        """Monotonic exclusion epoch, including content-free forget retry markers."""
        if self._connection is None and not self.path.exists():
            return 0
        return (
            self._connect()
            .execute("SELECT value FROM exclusion_generation WHERE id = 1")
            .fetchone()[0]
        )

    @contextmanager
    def guard_generation(self, expected: int | None):
        """Serialize a synchronous external mutation against forget.

        The DB lock protects the pre-write epoch check, not rollback of external files.
        Never await while holding this guard.
        """
        with self._write():
            if expected is not None and expected != self.generation():
                raise ValueError("memory exclusions changed; write discarded")
            yield

    def _check_excluded(self, source_id: str, seqs) -> None:
        if self.excluded_seqs(source_id).intersection(seqs):
            raise ValueError("source is excluded")

    def is_excluded(self, fact_id: str) -> bool:
        """Any contributing revision can invalidate a fact, not only its latest source."""
        for source in self.provenance(fact_id):
            if source.source_seq in self.excluded_seqs(source.source_id):
                return True
        if self._connection is None and not self.path.exists():
            return False
        return any(
            self.excluded_seqs(session_id).intersection(json.loads(seqs))
            for session_id, seqs in self._connect()
            .execute("SELECT session_id, seqs FROM source_refs WHERE fact_id = ?", (fact_id,))
            .fetchall()
        )

    def forget(self, fact_id: str) -> bool:
        """Exclude every contributing source and purge one fact in one transaction.

        False is an idempotent retry. SessionStore delivers queued content-free markers;
        no JSONL append or source-session scan participates in this transaction.
        """
        fact_id = _text(fact_id, "fact_id")
        with self._write() as connection:
            if self.get(fact_id) is None:
                if connection.execute(
                    "SELECT 1 FROM forgotten WHERE fact_id = ?", (fact_id,)
                ).fetchone():
                    return False
                raise KeyError(f"unknown fact: {fact_id}")
            sources = {(p.source_id, p.source_seq) for p in self.provenance(fact_id)}
            for session_id, seqs in connection.execute(
                "SELECT session_id, seqs FROM source_refs WHERE fact_id = ?", (fact_id,)
            ):
                sources.update((session_id, seq) for seq in json.loads(seqs))
            connection.executemany("INSERT OR IGNORE INTO exclusions VALUES (?, ?)", sources)
            connection.execute("INSERT INTO forgotten VALUES (?)", (fact_id,))
            connection.executemany(
                "INSERT INTO forget_events VALUES (?, ?)",
                ((fact_id, session_id) for session_id in sorted({sid for sid, _ in sources})),
            )
            # Explicit deletes also cover the pre-foreign-key legacy schema.
            for table in ("source_refs", "provenance", "revisions"):
                connection.execute(f"DELETE FROM {table} WHERE fact_id = ?", (fact_id,))
            connection.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return True

    def pending_forgets(self, session_id: str) -> list[str]:
        """Content-free operation IDs awaiting this exact source session's marker."""
        if self._connection is None and not self.path.exists():
            return []
        return [
            row[0]
            for row in self._connect().execute(
                "SELECT fact_id FROM forget_events WHERE session_id = ? ORDER BY fact_id",
                (session_id,),
            )
        ]

    def pending_forget_sessions(self, fact_id: str) -> list[str]:
        if self._connection is None and not self.path.exists():
            return []
        return [
            row[0]
            for row in self._connect().execute(
                "SELECT session_id FROM forget_events WHERE fact_id = ? ORDER BY session_id",
                (fact_id,),
            )
        ]

    def acknowledge_forget(self, fact_id: str, session_id: str) -> None:
        """Only after the session marker is durable; exclusions and retry IDs remain."""
        with self._write() as connection:
            connection.execute(
                "DELETE FROM forget_events WHERE fact_id = ? AND session_id = ?",
                (fact_id, session_id),
            )

    def remember(
        self,
        text: str,
        ref: SourceRef,
        *,
        sessions: SessionStore,
        project_root: Path,
        expected_generation: int | None = None,
        deduplicate: bool = False,
        expected_version: tuple | None = None,
    ) -> Fact:
        """Atomically add evidence with its validated, explicit persisted snapshot."""
        sessions.bind_facts(project_root, self)
        with self._write():
            if expected_generation is not None and expected_generation != self.generation():
                raise ValueError("memory exclusions changed; remember discarded")
            if (
                expected_version is not None
                and sessions.source_version(sessions.open(ref.session_id)) != expected_version
            ):
                raise ValueError("session sources changed; remember discarded")
            snapshot = sessions.validate_source(
                ref, project_root=project_root, excluded=self.excluded_seqs(ref.session_id)
            )
            if snapshot.status != "valid":
                raise ValueError(f"source is {snapshot.status}")
            if deduplicate:
                key = " ".join(_text(text, "text").casefold().split())
                # Exact normalized text, including old revisions, never semantic inference.
                # ponytail: linear revision scan; index normalized text if large stores need it.
                for fact_id, prior in self._connect().execute(
                    "SELECT fact_id, text FROM revisions"
                ):
                    if " ".join(prior.casefold().split()) == key:
                        existing = self.get(fact_id)
                        assert existing is not None
                        return existing
            fact = self.add(text, source_id=ref.session_id, source_seq=ref.seqs[0])
            self.attach_source(
                fact.id, fact.revision, ref, sessions=sessions, project_root=project_root
            )
            return fact

    def correct(
        self,
        fact_id: str,
        text: str,
        *,
        expected_revision: int,
        ref: SourceRef,
        sessions: SessionStore,
        project_root: Path,
        expected_generation: int | None = None,
    ) -> Fact:
        """Correct a selected revision using real, currently visible evidence."""
        sessions.bind_facts(project_root, self)
        with self._write():
            if expected_generation is not None and expected_generation != self.generation():
                raise ValueError("memory exclusions changed; correction discarded")
            snapshot = sessions.validate_source(
                ref, project_root=project_root, excluded=self.excluded_seqs(ref.session_id)
            )
            if snapshot.status != "valid":
                raise ValueError(f"source is {snapshot.status}")
            if not any(
                message.get("role") in {"user", "assistant"}
                and (
                    (isinstance(content := message.get("content"), str) and content.strip())
                    or (
                        isinstance(content, list)
                        and any(
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and isinstance(part.get("text"), str)
                            and part["text"].strip()
                            for part in content
                        )
                    )
                )
                for message in snapshot.messages
            ):
                raise ValueError("correction requires persisted user or assistant text evidence")
            fact = self.revise(
                fact_id,
                text,
                expected_revision=expected_revision,
                source_id=ref.session_id,
                source_seq=ref.seqs[0],
            )
            self.attach_source(
                fact.id, fact.revision, ref, sessions=sessions, project_root=project_root
            )
            return fact

    def add(self, text: str, *, source_id: str, source_seq: int) -> Fact:
        """Add once per normalized content + source, returning the current fact."""
        text = _text(text, "text")
        source_id = _text(source_id, "source_id")
        _integer(source_seq, "source_seq")
        identity = json.dumps([text, source_id, source_seq], ensure_ascii=False)
        fact_id = hashlib.sha256(identity.encode()).hexdigest()
        with self._write() as connection:
            self._check_excluded(source_id, (source_seq,))
            if self.is_excluded(fact_id):
                raise ValueError("fact source is excluded")
            connection.execute("INSERT OR IGNORE INTO facts VALUES (?, ?, 1)", (fact_id, text))
            connection.execute("INSERT OR IGNORE INTO revisions VALUES (?, 1, ?)", (fact_id, text))
            connection.execute(
                "INSERT OR IGNORE INTO provenance (fact_id, revision, source_id, source_seq) "
                "VALUES (?, 1, ?, ?)",
                (fact_id, source_id, source_seq),
            )
            row = connection.execute(
                "SELECT id, text, revision FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()
            return Fact(*row)

    def revise(
        self, fact_id: str, text: str, *, expected_revision: int, source_id: str, source_seq: int
    ) -> Fact:
        """Replace a fact atomically, or raise without changing its history."""
        fact_id = _text(fact_id, "fact_id")
        text = _text(text, "text")
        source_id = _text(source_id, "source_id")
        _integer(source_seq, "source_seq")
        _integer(expected_revision, "expected_revision", 1)
        with self._write() as connection:
            self._check_excluded(source_id, (source_seq,))
            if self.is_excluded(fact_id):
                raise ValueError("fact source is excluded")
            current = self.get(fact_id)
            if current is None:
                raise KeyError(f"unknown fact: {fact_id}")
            if current.revision != expected_revision:
                raise RevisionConflict(
                    f"expected revision {expected_revision}, found {current.revision}"
                )
            revision = expected_revision + 1
            connection.execute(
                "UPDATE facts SET text = ?, revision = ? WHERE id = ?", (text, revision, fact_id)
            )
            connection.execute("INSERT INTO revisions VALUES (?, ?, ?)", (fact_id, revision, text))
            connection.execute(
                "INSERT INTO provenance (fact_id, revision, source_id, source_seq) "
                "VALUES (?, ?, ?, ?)",
                (fact_id, revision, source_id, source_seq),
            )
            return Fact(fact_id, text, revision)

    def get(self, fact_id: str) -> Fact | None:
        fact_id = _text(fact_id, "fact_id")
        if self._connection is None and not self.path.exists():
            return None
        row = (
            self._connect()
            .execute("SELECT id, text, revision FROM facts WHERE id = ?", (fact_id,))
            .fetchone()
        )
        return Fact(*row) if row is not None else None

    def provenance(self, fact_id: str) -> list[Provenance]:
        fact_id = _text(fact_id, "fact_id")
        if self._connection is None and not self.path.exists():
            return []
        rows = self._connect().execute(
            "SELECT fact_id, revision, source_id, source_seq, created_at "
            "FROM provenance WHERE fact_id = ? ORDER BY revision",
            (fact_id,),
        )
        return [Provenance(*row) for row in rows]

    def search(self, query: str, *, limit: int = 20, offset: int = 0) -> list[Fact]:
        """Bounded literal substring search of project facts (unverified data)."""
        query = _text(query, "query")
        _integer(limit, "limit", 1)
        _integer(offset, "offset")
        if self._connection is None and not self.path.exists():
            return []
        return [
            Fact(*row)
            for row in self._connect().execute(
                "SELECT id, text, revision FROM facts WHERE instr(lower(text), lower(?)) > 0 "
                "ORDER BY id LIMIT ? OFFSET ?",
                (query, min(limit, 50), offset),
            )
        ]

    def list(self, *, limit: int = 50, offset: int = 0) -> list[Fact]:
        """Bounded ID-ordered inspection; callers must validate evidence before using it."""
        _integer(limit, "limit", 1)
        _integer(offset, "offset")
        if self._connection is None and not self.path.exists():
            return []
        return [
            Fact(*row)
            for row in self._connect().execute(
                "SELECT id, text, revision FROM facts ORDER BY id LIMIT ? OFFSET ?",
                (min(limit, 50), offset),
            )
        ]

    def excluded_seqs(self, session_id: str) -> frozenset[int]:
        """Read existing exclusions without adding a forget operation."""
        if self._connection is None and not self.path.exists():
            return frozenset()
        return frozenset(
            row[0]
            for row in self._connect().execute(
                "SELECT source_seq FROM exclusions WHERE source_id = ?", (session_id,)
            )
        )

    def source(self, fact_id: str, revision: int | None = None) -> SourceRef | None:
        fact = self.get(fact_id)
        if fact is None:
            return None
        row = (
            self._connect()
            .execute(
                "SELECT session_id, seqs, digest FROM source_refs "
                "WHERE fact_id = ? AND revision = ?",
                (fact_id, fact.revision if revision is None else revision),
            )
            .fetchone()
        )
        return SourceRef(row[0], tuple(json.loads(row[1])), row[2]) if row else None

    def attach_source(
        self,
        fact_id: str,
        revision: int,
        ref: SourceRef,
        *,
        sessions: SessionStore,
        project_root: Path,
    ) -> None:
        """Attach once to the matching revision, only after validating actual storage."""
        _integer(revision, "revision", 1)
        sessions.bind_facts(project_root, self)
        with self._write() as connection:
            snapshot = sessions.validate_source(
                ref, project_root=project_root, excluded=self.excluded_seqs(ref.session_id)
            )
            if snapshot.status != "valid":
                raise ValueError(f"source is {snapshot.status}")
            if self.is_excluded(fact_id):
                raise ValueError("fact source is excluded")
            row = connection.execute(
                "SELECT source_id, source_seq FROM provenance WHERE fact_id = ? AND revision = ?",
                (fact_id, revision),
            ).fetchone()
            if row is None or row[0] != ref.session_id or row[1] not in ref.seqs:
                raise ValueError("source does not match revision provenance")
            existing = self.source(fact_id, revision)
            if existing is not None and existing != ref:
                raise ValueError("source reference is immutable")
            connection.execute(
                "INSERT OR IGNORE INTO source_refs VALUES (?, ?, ?, ?, ?)",
                (fact_id, revision, ref.session_id, json.dumps(ref.seqs), ref.digest),
            )
