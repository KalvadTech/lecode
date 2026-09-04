"""Project-scoped persistent memory store (zerostack lineage).

Layout under ``<config_dir>/memory/<project-slug>/``::

    MEMORY.md            long-term memory (auto-injected, capped for injection)
    daily/YYYY-MM-DD.md  daily logs (compaction summaries land here too)
    scratchpad.md        project checklist
    notes/<name>.md      named notes

The project slug derives from the cwd (tail components + a short hash suffix
to avoid collisions). Every write is atomic (tmp file → fsync → rename) and
overwrites first copy the previous content to ``<file>.bak``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

#: Default injection cap for MEMORY.md (mirrors [memory] max_bytes).
DEFAULT_MAX_BYTES = 32768

#: Cap on search results.
MAX_SEARCH_HITS = 50

TRUNCATION_MARKER = "\n… [memory truncated] …\n"

#: Valid note names (also usable as slugs elsewhere).
NOTE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")

LONG_TERM_FILE = "MEMORY.md"
SCRATCHPAD_FILE = "scratchpad.md"


def project_slug(cwd: Path | str) -> str:
    """A stable, collision-safe slug for a project directory."""
    resolved = str(Path(cwd).resolve())
    digest = hashlib.sha1(resolved.encode()).hexdigest()[:8]
    parts = [p for p in re.split(r"[^a-zA-Z0-9]+", resolved) if p]
    tail = "-".join(parts[-2:]).lower()[:40].strip("-") or "root"
    return f"{tail}-{digest}"


def memory_root(cwd: Path | str, config_dir: Path | None = None) -> Path:
    """The store root for a project (``LECODE_CONFIG_DIR`` aware)."""
    if config_dir is None:
        from lecode.config.loader import config_dir as _config_dir

        config_dir = _config_dir()
    return Path(config_dir) / "memory" / project_slug(cwd)


def _cap_bytes(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore") + TRUNCATION_MARKER


@dataclass(frozen=True)
class SearchHit:
    """One regex match inside the memory store."""

    file: str  # path relative to the store root, e.g. "daily/2026-09-01.md"
    line_no: int  # 1-based
    line: str
    snippet: str  # short window around the first match on the line


class MemoryStore:
    """Read/write access to one project's memory files."""

    def __init__(self, root: Path | str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes

    # -- paths ---------------------------------------------------------------

    @property
    def long_term_path(self) -> Path:
        return self.root / LONG_TERM_FILE

    @property
    def scratchpad_path(self) -> Path:
        return self.root / SCRATCHPAD_FILE

    def daily_path(self, day: date | str | None = None) -> Path:
        if day is None:
            day = date.today()
        if isinstance(day, date):
            day = day.isoformat()
        return self.root / "daily" / f"{day}.md"

    def note_path(self, name: str) -> Path:
        if not NOTE_NAME.fullmatch(name):
            raise ValueError(f"invalid note name: {name!r} (want {NOTE_NAME.pattern})")
        return self.root / "notes" / f"{name}.md"

    # -- atomic writes ---------------------------------------------------------

    def _atomic_write(self, path: Path, content: str) -> None:
        """tmp file → fsync → rename; existing content is backed up to .bak."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    # -- long-term memory ---------------------------------------------------------

    def read_long_term(self, *, capped: bool = True) -> str:
        """MEMORY.md content; ``capped`` applies the injection byte cap."""
        text = self._read(self.long_term_path)
        return _cap_bytes(text, self.max_bytes) if capped else text

    def write_long_term(self, content: str) -> None:
        self._atomic_write(self.long_term_path, content.rstrip("\n") + "\n")

    def append_long_term(self, section: str) -> None:
        """Append under a dated ``## YYYY-MM-DD`` heading (created if absent)."""
        today = date.today().isoformat()
        current = self._read(self.long_term_path).rstrip("\n")
        heading = f"## {today}"
        if heading in current:
            new = f"{current}\n\n{section.strip()}\n"
        else:
            new = (
                f"{current}\n\n{heading}\n\n{section.strip()}\n"
                if current
                else (f"{heading}\n\n{section.strip()}\n")
            )
        self._atomic_write(self.long_term_path, new)

    def edit_long_term(self, old: str, new: str) -> None:
        """Exact-match replace; errors when ``old`` occurs 0 or 2+ times."""
        current = self._read(self.long_term_path)
        self._atomic_write(self.long_term_path, self._replace_unique(current, old, new))

    @staticmethod
    def _replace_unique(text: str, old: str, new: str) -> str:
        count = text.count(old)
        if count == 0:
            raise KeyError(f"text not found: {old[:80]!r}")
        if count > 1:
            raise ValueError(f"ambiguous: {count} occurrences of {old[:80]!r}")
        return text.replace(old, new, 1)

    # -- daily logs -----------------------------------------------------------------

    def append_daily(self, entry: str, day: date | str | None = None) -> None:
        """Append to the daily log under a timestamped ``## HH:MM`` heading."""
        path = self.daily_path(day)
        stamp = datetime.now().strftime("%H:%M")
        current = self._read(path).rstrip("\n")
        if not current:
            current = f"# Daily log {day if day is not None else date.today().isoformat()}"
        self._atomic_write(path, f"{current}\n\n## {stamp}\n\n{entry.strip()}\n")

    def read_daily(self, day: date | str | None = None) -> str:
        return self._read(self.daily_path(day))

    def write_daily(self, content: str, day: date | str | None = None) -> None:
        """Overwrite a daily log (today's by default)."""
        self._atomic_write(self.daily_path(day), content.rstrip("\n") + "\n")

    def flush_summary(self, summary: str) -> None:
        """Flush a compaction summary to today's daily log under Compaction."""
        path = self.daily_path()
        stamp = datetime.now().strftime("%H:%M")
        current = self._read(path).rstrip("\n")
        if not current:
            current = f"# Daily log {date.today().isoformat()}"
        self._atomic_write(path, f"{current}\n\n## Compaction — {stamp}\n\n{summary.strip()}\n")

    # -- scratchpad ----------------------------------------------------------------

    def read_scratchpad(self) -> str:
        return self._read(self.scratchpad_path)

    def write_scratchpad(self, content: str) -> None:
        self._atomic_write(self.scratchpad_path, content.rstrip("\n") + "\n")

    # -- notes ----------------------------------------------------------------------

    def write_note(self, name: str, content: str) -> None:
        self._atomic_write(self.note_path(name), content.rstrip("\n") + "\n")

    def read_note(self, name: str) -> str | None:
        """Note content, or ``None`` when the note does not exist."""
        path = self.note_path(name)
        return self._read(path) if path.is_file() else None

    def list_notes(self) -> list[str]:
        notes_dir = self.root / "notes"
        if not notes_dir.is_dir():
            return []
        return sorted(path.stem for path in notes_dir.glob("*.md"))

    def delete_note(self, name: str) -> bool:
        path = self.note_path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # -- search ---------------------------------------------------------------------

    def _search_files(self) -> list[Path]:
        """Files in ranking order: MEMORY, daily (recent first), notes, scratchpad."""
        files: list[Path] = []
        if self.long_term_path.is_file():
            files.append(self.long_term_path)
        daily_dir = self.root / "daily"
        if daily_dir.is_dir():
            files.extend(sorted(daily_dir.glob("*.md"), reverse=True))  # recent first
        notes_dir = self.root / "notes"
        if notes_dir.is_dir():
            files.extend(sorted(notes_dir.glob("*.md")))
        if self.scratchpad_path.is_file():
            files.append(self.scratchpad_path)
        return files

    def search(self, pattern: str, max_hits: int = MAX_SEARCH_HITS) -> list[SearchHit]:
        """Case-insensitive regex keyword search across all memory files.

        Hits come back in ranked file order (MEMORY.md first, then daily logs
        recent-first, then notes, then the scratchpad), capped at ``max_hits``.
        Raises :class:`ValueError` on an invalid regex.
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from e
        hits: list[SearchHit] = []
        for path in self._search_files():
            for line_no, line in enumerate(self._read(path).splitlines(), start=1):
                match = regex.search(line)
                if match is None:
                    continue
                start = max(0, match.start() - 40)
                end = min(len(line), match.end() + 40)
                hits.append(
                    SearchHit(
                        file=str(path.relative_to(self.root)),
                        line_no=line_no,
                        line=line.strip(),
                        snippet=line[start:end].strip(),
                    )
                )
                if len(hits) >= max_hits:
                    return hits
        return hits


def memory_injection(config, cwd: Path | str) -> str | None:
    """The rendered ``## Memory`` section for the system prompt, or ``None``.

    Cheap: no store reads when memory is disabled, no section when both
    MEMORY.md and the scratchpad are empty/absent.
    """
    if not config.memory.enabled:
        return None
    store = MemoryStore(memory_root(cwd), max_bytes=config.memory.max_bytes)
    long_term = store.read_long_term().strip()
    scratchpad = store.read_scratchpad().strip()
    if not long_term and not scratchpad:
        return None
    sections = []
    if long_term:
        sections.append(f"### Long-term memory\n\n{long_term}")
    if scratchpad:
        sections.append(f"### Scratchpad\n\n{scratchpad}")
    return "## Memory\n\n" + "\n\n".join(sections)
