"""Fuzzy trigger pickers rendered through the completion menu.

No fullscreen dialogs: ``@`` (files + agents), ``/`` (slash commands) and
``.`` (personas) are prompt_toolkit completers, so matches appear as a
completion menu below the input and Escape dismisses it (default behavior).
``/`` and ``.`` trigger only at buffer start; ``@`` triggers anywhere.

Insert-on-accept: an agent keeps the mention form (``@name `` — consumed by
:func:`lecode.context.agents.parse_mentions`), a file inserts its path, a
command inserts ``/name ``, a persona inserts ``.name ``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from lecode.context.agents import AgentRegistry
from lecode.context.resources import list_available
from lecode.context.skills import SkillRegistry, skill_commands
from lecode.slash.catalog import BUILTIN_COMMANDS
from lecode.tui.input import FileLister

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
