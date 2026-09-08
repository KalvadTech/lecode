"""Fuzzy trigger pickers rendered through the completion menu.

No fullscreen dialogs: ``@`` (files + agents), ``/`` (slash commands) and
``.`` (personas) are prompt_toolkit completers, so matches appear as a
completion menu below the input and Escape dismisses it (default behavior).
``/`` and ``.`` trigger only at buffer start; ``@`` triggers anywhere.

Insert-on-accept: an agent keeps the mention form (``@name `` — consumed by
:func:`lecode.context.agents.parse_mentions`), a file inserts its path, a
command inserts ``/name ``, a persona inserts ``.name ``.

Commands with argument rows also pick their arguments: after ``/model ``
the menu offers the catalog, after ``/resume `` the folder's sessions, and
so on (:class:`CommandArgCompleter` routes ahead of path completion).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import CompleteEvent, Completer, Completion, merge_completers
from prompt_toolkit.document import Document

from lecode.context.agents import AgentRegistry
from lecode.context.resources import list_available
from lecode.context.skills import SkillRegistry, skill_commands
from lecode.slash.catalog import BUILTIN_COMMANDS
from lecode.slash.registry import CommandRegistry, CompletionRow, SlashCommand
from lecode.tui.input import FileLister, PathCompleter

if TYPE_CHECKING:
    from lecode.tui.app import TuiApp

#: Max file completions offered by the ``@`` picker.
FILE_COMPLETION_LIMIT = 20

#: Word-boundary characters that earn a fuzzy bonus after them.
_BOUNDARY_CHARS = frozenset("-_/")


def fuzzy_score(query: str, candidate: str) -> int | None:
    """Score a subsequence match; ``None`` when ``query`` is not a subsequence.

    Case-insensitive. Higher is better: +1 per matched char, +8 when the
    match starts at the candidate's first char, +5 for each contiguous
    extension, +4 when a char lands right after ``-``, ``_`` or ``/``.
    The empty query matches everything with score 0.
    """
    if not query:
        return 0
    query = query.lower()
    candidate = candidate.lower()
    score = 0
    prev = -1
    start = 0
    for index, ch in enumerate(query):
        found = candidate.find(ch, start)
        if found == -1:
            return None
        score += 1
        if index == 0 and found == 0:
            score += 8
        elif found == prev + 1:
            score += 5
        elif candidate[found - 1] in _BOUNDARY_CHARS:
            score += 4
        prev = found
        start = found + 1
    return score


def _ranked[T](token: str, candidates: list[tuple[str, T]]) -> list[tuple[str, T]]:
    """Filter and sort ``(name, payload)`` candidates by fuzzy score."""
    scored = [
        (score, name, payload)
        for name, payload in candidates
        if (score := fuzzy_score(token, name)) is not None
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(name, payload) for _, name, payload in scored]


def prefix_matches[T](token: str, candidates: list[tuple[str, T]]) -> list[tuple[str, T]]:
    """Case-insensitive prefix matches, alphabetical (the ``/`` picker)."""
    prefix = token.lower()
    return sorted(
        (name, payload) for name, payload in candidates if name.lower().startswith(prefix)
    )


def command_candidates(skills: SkillRegistry) -> list[tuple[str, str]]:
    """Built-in commands plus skill-registered ones (first name wins)."""
    commands = list(BUILTIN_COMMANDS)
    known = {name for name, _ in commands}
    for info in skill_commands(skills).values():
        if info["name"] not in known:
            commands.append((info["name"], info["description"]))
    return commands


def trigger_token(document: Document) -> tuple[str, str] | None:
    """``(trigger, token)`` for ``@``/``/``/``.``, or ``None`` off-trigger."""
    before = document.text_before_cursor
    word = before.split(" ")[-1] if before else ""
    if not word:
        return None
    if word.startswith("@"):
        return "@", word[1:]
    if word[0] in ("/", ".") and before == word:
        return word[0], word[1:]  # only when first char of the buffer
    return None


def persona_names() -> list[str]:
    """Built-in (and overridden) persona names from the prompts resources."""
    return sorted(
        name.removeprefix("personas/").removesuffix(".md")
        for name in list_available("prompts")
        if name.startswith("personas/") and name.endswith(".md")
    )


class TriggerCompleter(Completer):
    """Completion-menu pickers for the ``@``, ``/`` and ``.`` triggers."""

    def __init__(
        self,
        lister: FileLister,
        agents: AgentRegistry,
        skills: SkillRegistry,
    ) -> None:
        self._lister = lister
        self._agents = agents
        self._skills = skills

    # -- completions ----------------------------------------------------------

    async def get_completions_async(
        self, document: Document, complete_event: CompleteEvent
    ) -> AsyncGenerator[Completion, None]:
        trigger = trigger_token(document)
        if trigger is None:
            return
        kind, token = trigger
        word_len = len(token) + 1
        if kind == "@":
            for name, agent in _ranked(token, [(a.name, a) for a in self._agents.visible()]):
                yield Completion(
                    f"@{name} ",
                    start_position=-word_len,
                    display=name,
                    display_meta=f"agent — {agent.description}",
                )
            files = _ranked(token, [(path, path) for path in await self._lister.files()])
            for path, _ in files[:FILE_COMPLETION_LIMIT]:
                yield Completion(
                    path,
                    start_position=-word_len,
                    display=path,
                    display_meta="file",
                )
        elif kind == "/":
            for name, description in prefix_matches(token, command_candidates(self._skills)):
                yield Completion(
                    f"/{name} ",
                    start_position=-word_len,
                    display=name,
                    display_meta=description,
                )
        else:  # "."
            for name, _ in _ranked(token, [(n, None) for n in persona_names()]):
                yield Completion(
                    f".{name} ",
                    start_position=-word_len,
                    display=name,
                    display_meta="persona",
                )

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        # Unused: prompt_toolkit drives the async variant.
        return iter(())


# -- command-argument pickers ------------------------------------------------------


def command_arg_context(
    document: Document, registry: CommandRegistry
) -> tuple[SlashCommand, list[str], str] | None:
    """``(command, args, partial)`` when the buffer offers argument rows.

    ``args`` are the tokens before the cursor's token, ``partial`` the token
    under the cursor. Resolution mirrors dispatch: exact name or unique
    prefix, case-sensitive — unknown, ambiguous and free-text commands get
    no picker. The command word itself belongs to the slash picker, so a
    space must have been typed (``/model`` → ``None``, ``/model `` → context).
    """
    text = document.text_before_cursor
    if not text.startswith("/") or "\n" in text:
        return None
    words = text[1:].split(" ")
    if len(words) < 2 or not words[0]:
        return None
    try:
        command = registry.match(words[0])
    except KeyError:  # unknown or ambiguous prefix — same as dispatch would say
        return None
    if command.arg_completions is None:
        return None
    # Empty tokens (double spaces) don't count as args — matches dispatch,
    # which splits on any whitespace run.
    return command, [w for w in words[1:-1] if w], words[-1]


def arg_ranked(partial: str, rows: list[CompletionRow]) -> list[CompletionRow]:
    """Argument rows for ``partial``: provider order for an empty query
    (catalog order, newest sessions first), otherwise fuzzy-filtered over
    the inserted value and the display label, best match first.
    """
    if not partial:
        return rows
    scored: list[tuple[int, CompletionRow]] = []
    for row in rows:
        best = None
        for field in (row[0], row[1]):
            score = fuzzy_score(partial, field)
            if score is not None and (best is None or score > best):
                best = score
        if best is not None:
            scored.append((best, row))
    scored.sort(key=lambda item: (-item[0], item[1][0]))
    return [row for _, row in scored]


class CommandArgCompleter(Completer):
    """Routes completion: command-argument pickers, else paths + triggers.

    While the buffer is a command offering argument rows, those win and
    path completion stays out — ``/model vendor/name`` is a model ref, not
    a file. Providers can hit the disk (sessions, messages), so rows flow
    through the app's single-slot cache (:meth:`TuiApp.arg_completion_rows`),
    shared with the panel.
    """

    def __init__(self, app: TuiApp, fallback: Completer) -> None:
        self._app = app
        self._fallback = fallback

    async def get_completions_async(
        self, document: Document, complete_event: CompleteEvent
    ) -> AsyncGenerator[Completion, None]:
        result = self._app.arg_completion_rows(document)
        if result is None:
            async for completion in self._fallback.get_completions_async(document, complete_event):
                yield completion
            return
        _, _, partial, rows = result
        for insert, display, meta in arg_ranked(partial, rows):
            yield Completion(
                f"{insert} ",  # trailing space: the next token starts fresh
                start_position=-len(partial),
                display=display,
                display_meta=meta,
            )

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        # Unused: prompt_toolkit drives the async variant.
        return iter(())


def build_completer(
    app: TuiApp, cwd: Path, lister: FileLister, agents: AgentRegistry, skills: SkillRegistry
) -> CommandArgCompleter:
    """The input completer: argument pickers first, then paths, then triggers."""
    return CommandArgCompleter(
        app,
        merge_completers(
            [PathCompleter(cwd, lister=lister), TriggerCompleter(lister, agents, skills)]
        ),
    )
