"""Tests for the slash-command registry: lookup, prefix matching, skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from lecode.context.skills import Skill, SkillRegistry
from lecode.slash import (
    BUILTIN_COMMANDS,
    AmbiguousCommandError,
    CommandRegistry,
    SlashCommand,
    UnknownCommandError,
    build_registry,
)


async def _noop(app, args) -> None:
    pass


def _command(name: str, description: str = "desc") -> SlashCommand:
    return SlashCommand(name, description, _noop)


class FakeFeed:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.errors: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)


class FakeApp:
    """Minimal app double: handlers here only need feed + submit_prompt."""

    def __init__(self) -> None:
        self.feed = FakeFeed()
        self.submitted: list[str] = []

    def submit_prompt(self, text: str, *, echo: str | None = None) -> None:
        self.submitted.append(text)


# -- registration / lookup -------------------------------------------------------


def test_register_and_get():
    registry = CommandRegistry()
    registry.register(_command("alpha"))
    assert registry.get("alpha").description == "desc"
    assert registry.get("missing") is None


def test_register_replaces_same_name():
    registry = CommandRegistry()
    registry.register(_command("alpha", "first"))
    registry.register(_command("alpha", "second"))
    assert registry.get("alpha").description == "second"


def test_list_is_sorted():
    registry = CommandRegistry()
    for name in ("beta", "alpha", "gamma"):
        registry.register(_command(name))
    assert [c.name for c in registry.list()] == ["alpha", "beta", "gamma"]


# -- matching ---------------------------------------------------------------------


def test_match_exact():
    registry = CommandRegistry()
    registry.register(_command("alpha"))
    registry.register(_command("alpine"))
    assert registry.match("alpha").name == "alpha"


def test_match_unique_prefix():
    registry = CommandRegistry()
    registry.register(_command("alpha"))
    registry.register(_command("beta"))
    assert registry.match("bet").name == "beta"


def test_match_ambiguous_prefix_lists_matches():
    registry = CommandRegistry()
    registry.register(_command("alpha"))
    registry.register(_command("alpine"))
    with pytest.raises(AmbiguousCommandError) as exc_info:
        registry.match("alp")
    assert exc_info.value.matches == ["alpha", "alpine"]


def test_match_unknown():
    registry = CommandRegistry()
    registry.register(_command("alpha"))
    with pytest.raises(UnknownCommandError):
        registry.match("zzz")


# -- skill commands ------------------------------------------------------------------


def _skill(name="demo", register_cmd=True, cmd_info=None) -> Skill:
    return Skill(
        name=name,
        description="Demo skill",
        body="SKILL BODY",
        path=Path("/x/SKILL.md"),
        register_cmd=register_cmd,
        cmd_info=cmd_info,
    )


def test_skill_command_registered_with_cmd_info_description():
    registry = CommandRegistry()
    registry.register_skills(SkillRegistry({"demo": _skill(cmd_info="usage help")}))
    command = registry.get("demo")
    assert command is not None
    assert command.description == "usage help"


def test_skill_without_register_cmd_not_registered():
    registry = CommandRegistry()
    registry.register_skills(SkillRegistry({"demo": _skill(register_cmd=False)}))
    assert registry.get("demo") is None


async def test_skill_command_injects_body_as_user_prompt():
    registry = CommandRegistry()
    registry.register_skills(SkillRegistry({"demo": _skill()}))
    app = FakeApp()
    await registry.match("demo").handler(app, [])
    assert app.submitted == ["SKILL BODY"]


async def test_skill_command_appends_args_to_body():
    registry = CommandRegistry()
    registry.register_skills(SkillRegistry({"demo": _skill()}))
    app = FakeApp()
    await registry.match("demo").handler(app, ["some", "args"])
    assert app.submitted == ["SKILL BODY\n\nsome args"]


def test_skill_command_does_not_override_builtin():
    registry = build_registry(SkillRegistry({"help": _skill(name="help")}))
    assert registry.get("help").description == "Show help"


# -- build_registry --------------------------------------------------------------------


def test_build_registry_covers_the_catalog():
    registry = build_registry()
    assert {c.name for c in registry.list()} == {name for name, _ in BUILTIN_COMMANDS}


def test_build_registry_without_skills():
    registry = build_registry(None)
    assert registry.get("new") is not None


def test_arg_hints_registered():
    registry = build_registry()
    assert registry.get("rename").arg_hint == "<name>"
    assert registry.get("quit").arg_hint is None


def test_every_catalog_command_has_a_real_handler():
    from lecode.slash.handlers import _HANDLERS

    assert {name for name, _ in BUILTIN_COMMANDS} <= set(_HANDLERS)
