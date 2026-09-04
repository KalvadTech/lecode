"""Lifecycle hooks: events, handler runner, verdict merge, tool decorator."""

from __future__ import annotations

from lecode.hooks.decorator import apply_hooks
from lecode.hooks.events import (
    EVENTS,
    POST_TOOL_USE,
    PRE_TOOL_USE,
    SESSION_END,
    SESSION_START,
    STOP,
    SUBAGENT_END,
    SUBAGENT_START,
    USER_PROMPT_SUBMIT,
    build_envelope,
)
from lecode.hooks.runner import (
    DEFAULT_HOOK_TIMEOUT_S,
    HookDispatcher,
    HookHandler,
    HookVerdict,
    MergedVerdict,
    dispatch_event,
    dispatcher_from_config,
    hooks_status,
    run_handler,
)

__all__ = [
    "DEFAULT_HOOK_TIMEOUT_S",
    "EVENTS",
    "POST_TOOL_USE",
    "PRE_TOOL_USE",
    "SESSION_END",
    "SESSION_START",
    "STOP",
    "SUBAGENT_END",
    "SUBAGENT_START",
    "USER_PROMPT_SUBMIT",
    "HookDispatcher",
    "HookHandler",
    "HookVerdict",
    "MergedVerdict",
    "apply_hooks",
    "build_envelope",
    "dispatch_event",
    "dispatcher_from_config",
    "hooks_status",
    "run_handler",
]
