"""The slash-command registry.

Built-in commands come from :mod:`lecode.slash.catalog` with handlers in
:mod:`lecode.slash.handlers`; skills with ``register_cmd: true`` merge on top
(a skill command injects the skill body as a user prompt when invoked).
Lookup is exact or unique-prefix, mirroring the model catalog and the
session store.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lecode.context.skills import SkillRegistry
    from lecode.tui.app import TuiApp

#: Handler signature: the app (feed + state seams) plus split arguments.
CommandHandler = Callable[["TuiApp", list[str]], Awaitable[None]]

#: One argument-picker row: ``(insert, display, meta)``.
CompletionRow = tuple[str, str, str]

#: Argument-picker provider: ``(app, args typed so far) -> rows``.
#: Providers gate on ``args`` — a consumed position returns no rows (the
#: picker closes), a nested one returns the next stage's rows.
ArgCompletions = Callable[["TuiApp", "list[str]"], "list[CompletionRow]"]


class UnknownCommandError(KeyError):
    """No command matched the query."""


class AmbiguousCommandError(KeyError):
    """A prefix query matched more than one command."""

    def __init__(self, query: str, matches: list[str]) -> None:
        self.query = query
        self.matches = matches
        super().__init__(f"ambiguous command '/{query}', matches: {', '.join(matches)}")


@dataclass(frozen=True)
class SlashCommand:
    """One invocable command."""

    name: str
    description: str
    handler: CommandHandler
    #: Usage hint shown by ``/help <name>`` (e.g. ``"<mode>"``).
    arg_hint: str | None = None
    #: Dropdown rows for the command's arguments (the shared picker panel);
    #: ``None`` = free-text arguments, no argument picker.
    arg_completions: ArgCompletions | None = None
    #: Inert-row text when a fresh top-level picker finds no rows (empty
    #: catalog, no sessions…). ``None`` = render nothing.
    arg_empty_hint: str | None = None


class CommandRegistry:
    """Register, look up, and prefix-match slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        """Exact lookup only."""
        return self._commands.get(name)

    def list(self) -> list[SlashCommand]:
        """All commands, sorted by name."""
        return [self._commands[name] for name in sorted(self._commands)]

    def match(self, query: str) -> SlashCommand:
        """Resolve ``query`` by exact name or unique prefix."""
        command = self.get(query)
        if command is not None:
            return command
        matches = sorted(name for name in self._commands if name.startswith(query))
        if len(matches) == 1:
            return self._commands[matches[0]]
        if len(matches) > 1:
            raise AmbiguousCommandError(query, matches)
        raise UnknownCommandError(query)

    def register_skills(self, skills: SkillRegistry) -> None:
        """Merge skill-registered commands (``register_cmd: true``) on top.

        A skill command injects the skill body (plus any arguments) as a user
        prompt. Built-ins win name collisions.
        """
        from lecode.context.skills import skill_commands

        for info in skill_commands(skills).values():
            if info["name"] in self._commands:
                continue
            body = info["body"]

            async def handler(app: TuiApp, args: list[str], _body: str = body) -> None:
                text = _body if not args else f"{_body}\n\n{' '.join(args)}"
                app.submit_prompt(text)

            self.register(SlashCommand(info["name"], info["description"], handler))
