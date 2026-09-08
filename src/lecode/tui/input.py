"""Input-editor extras: persisted history, kill ring, $EDITOR, path completion.

History is a prompt_toolkit ``History`` subclass backed by the **session
file itself**: submitted entries are appended as ``input`` events
(``{"text": ...}``; consecutive repeats deduped, capped in memory), and the
unsubmitted draft is persisted on exit as a ``draft`` event (last one wins)
so a restart offers it back via :meth:`SessionHistory.load_draft`. No
separate ``input_history.jsonl`` exists. In-session draft handling while
navigating (Up/Down) is prompt_toolkit's own buffer-history behavior.

The kill ring keeps Ctrl-K/Ctrl-U/Ctrl-W kills for Ctrl-Y yank; Ctrl-G opens
``$EDITOR`` on the current buffer via ``run_in_terminal`` (no alternate
screen to suspend); Tab on text completes path-ish tokens from a cached
``fd`` listing (``os.scandir`` fallback).
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import AsyncGenerator, Callable, Iterable
from pathlib import Path

from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History

from lecode.extras.proc import run_proc
from lecode.session.model import EventRecord

#: Maximum number of history entries offered to the editor.
HISTORY_CAP = 1000

#: Time-to-live for the cached fd file listing.
FILE_CACHE_TTL_S = 2.0

#: Timeout for the fd subprocess itself.
FD_TIMEOUT_S = 5.0

#: Word characters for Ctrl-W (kill word back).
_WORD_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


# -- history with drafts ------------------------------------------------------


class SessionHistory(History):
    """Per-session prompt_toolkit history with a draft slot, in the JSONL."""

    def __init__(self, store: object, session: object, cap: int = HISTORY_CAP) -> None:
        super().__init__()
        self._store = store
        self._session = session
        self._cap = cap
        self._strings: list[str] = []
        self._draft = ""
        self._load()

    def _load(self) -> None:
        strings: list[str] = []
        draft = ""
        for record in self._store.read_records(self._session):
            if not isinstance(record, EventRecord):
                continue
            value = record.data.get("text")
            if not isinstance(value, str):
                continue
            if record.kind == "input":
                if not strings or strings[-1] != value:
                    strings.append(value)
            elif record.kind == "draft":
                draft = value
        self._strings = strings[-self._cap :]
        self._draft = draft

    def rebind(self, session: object) -> None:
        """Point at another session (``/new``, ``/resume`` mid-TUI)."""
        self._session = session
        self._load()

    # -- prompt_toolkit History interface ------------------------------------

    def load_history_strings(self) -> Iterable[str]:
        return list(self._strings)

    def store_string(self, string: str) -> None:
        if not string.strip():
            return
        if self._strings and self._strings[-1] == string:
            return  # dedupe consecutive repeats
        self._strings.append(string)
        if len(self._strings) > self._cap:
            self._strings = self._strings[-self._cap :]
        self._store.append_event(self._session, "input", {"text": string})

    # -- drafts ----------------------------------------------------------------

    def save_draft(self, text: str) -> None:
        """Persist the unsubmitted buffer text (called on app exit)."""
        if not text.strip():
            return
        self._draft = text
        self._store.append_event(self._session, "draft", {"text": text})

    def load_draft(self) -> str:
        """Return the persisted draft (if any) and tombstone it."""
        draft = self._draft
        self._draft = ""
        if draft:
            # An empty draft event marks it consumed (append-only file).
            self._store.append_event(self._session, "draft", {"text": ""})
        return draft


# -- kill ring ------------------------------------------------------------------


class KillRing:
    """A minimal Emacs kill ring: kills push, yank pops the most recent."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def kill(self, text: str) -> None:
        if text:
            self._items.append(text)

    def yank(self) -> str | None:
        """The most recent kill, or ``None`` when the ring is empty."""
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)


def kill_to_end_of_line(text: str, cursor: int) -> str:
    """The text Ctrl-K would kill (to line end; the newline if at it)."""
    newline = text.find("\n", cursor)
    if newline == -1:
        return text[cursor:]
    if newline == cursor:
        return "\n"
    return text[cursor:newline]


def kill_to_start_of_line(text: str, cursor: int) -> str:
    """The text Ctrl-U would kill (to line start)."""
    start = text.rfind("\n", 0, cursor) + 1
    return text[start:cursor]


def kill_word_back(text: str, cursor: int) -> str:
    """The text Ctrl-W would kill (word back, then whitespace before it)."""
    i = cursor
    while i > 0 and text[i - 1] not in _WORD_CHARS:
        i -= 1
    while i > 0 and text[i - 1] in _WORD_CHARS:
        i -= 1
    return text[i:cursor]


# -- $EDITOR ---------------------------------------------------------------------


async def open_in_editor(current_text: str) -> str | None:
    """Open ``$EDITOR`` on the current input; return the edited text.

    ``None`` when ``$EDITOR`` is unset, the editor fails, or the file comes
    back unchanged/empty (an empty save is treated as "keep editing").
    """
    editor = os.environ.get("EDITOR")
    if not editor:
        return None
    fd, name = tempfile.mkstemp(prefix="lecode-input-", suffix=".md")
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(current_text)
        argv = [*shlex.split(editor), str(path)]

        def _run() -> None:
            # run_in_terminal needs no alt-screen suspend; run the editor
            # synchronously in the foreground terminal.
            subprocess.run(argv, check=False)

        await run_in_terminal(_run)
        edited = path.read_text(encoding="utf-8")
    except OSError:
        return None
    finally:
        path.unlink(missing_ok=True)
    edited = edited.rstrip("\n")
    if not edited or edited == current_text:
        return None
    return edited


# -- path completion ---------------------------------------------------------------


def _path_token_before_cursor(document: Document) -> str | None:
    """The path-ish token before the cursor, or ``None`` for plain words.

    Path-ish means: contains ``/``, or starts with ``~`` or ``.``.
    """
    before = document.text_before_cursor
    token = before.split(" ")[-1] if before else ""
    if not token:
        return None
    if token.startswith("/") and before == token:
        return None  # buffer-start slash-command trigger; the / picker owns it
    if "/" in token or token.startswith(("~", ".")):
        return token
    return None


class FileLister:
    """Cached ``fd`` file listing with an ``os.scandir`` fallback.

    Shared by the path completer and the ``@`` picker so one fd call feeds
    both. ``clock`` is injectable for TTL tests.
    """

    def __init__(self, cwd: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self._cwd = Path(cwd)
        self._clock = clock
        self._cache: list[str] | None = None
        self._cached_at = 0.0

    async def files(self) -> list[str]:
        now = self._clock()
        if self._cache is not None and now - self._cached_at < FILE_CACHE_TTL_S:
            return self._cache
        files = await self._fetch()
        self._cache = files
        self._cached_at = now
        return files

    async def _fetch(self) -> list[str]:
        try:
            result = await run_proc(
                ["fd", "--type", "f", "--type", "d", "--hidden", "--exclude", ".git"],
                cwd=self._cwd,
                timeout=FD_TIMEOUT_S,
            )
        except OSError:
            return self._scandir()
        if result.exit_code != 0 or result.timed_out:
            return self._scandir()
        return [line for line in result.stdout.splitlines() if line]

    def _scandir(self) -> list[str]:
        """Defensive one-level fallback when fd fails (fd is a hard dep)."""
        out: list[str] = []
        try:
            for entry in os.scandir(self._cwd):
                if entry.name == ".git":
                    continue
                out.append(entry.name + ("/" if entry.is_dir() else ""))
        except OSError:
            pass
        return out

    def refresh_now(self) -> None:
        """Drop the cache so the next completion re-lists."""
        self._cache = None

    def prefetch(self) -> None:
        """Kick off a background listing so the first completion doesn't wait."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.files())
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)


class PathCompleter(Completer):
    """Completes path-ish tokens from the cached project file listing."""

    def __init__(
        self,
        cwd: Path,
        clock: Callable[[], float] = time.monotonic,
        lister: FileLister | None = None,
    ) -> None:
        self._cwd = Path(cwd)
        self._lister = lister or FileLister(self._cwd, clock)

    async def get_completions_async(
        self, document: Document, complete_event: CompleteEvent
    ) -> AsyncGenerator[Completion, None]:
        token = _path_token_before_cursor(document)
        if token is None:
            return
        # fd lists paths relative to cwd; normalize the user's "./" away.
        prefix = token[2:] if token.startswith("./") else token
        for path in await self._lister.files():
            if path.startswith(prefix):
                yield Completion(path, start_position=-len(token))

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        # Unused: prompt_toolkit drives the async variant.
        return iter(())
