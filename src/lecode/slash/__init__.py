"""Slash-command infrastructure.

The static catalog (:mod:`lecode.slash.catalog`) feeds the pickers; the
registry (:mod:`lecode.slash.registry`) resolves exact or unique-prefix
invocations; handlers live in :mod:`lecode.slash.handlers` (skills with
``register_cmd: true`` merge on top).
"""

from __future__ import annotations

from lecode.slash.catalog import BUILTIN_COMMANDS
from lecode.slash.handlers import build_registry
from lecode.slash.registry import (
    AmbiguousCommandError,
    CommandRegistry,
    SlashCommand,
    UnknownCommandError,
)

__all__ = [
    "BUILTIN_COMMANDS",
    "AmbiguousCommandError",
    "CommandRegistry",
    "SlashCommand",
    "UnknownCommandError",
    "build_registry",
]
