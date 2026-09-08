"""Hook handler execution and verdict merging.

A handler is a shell command run via ``/bin/sh -c`` through the shared proc
wrapper; the event envelope goes in on stdin and the verdict comes back as a
JSON object on stdout::

    {"verdict": "allow"|"defer"|"ask"|"deny", "reason": "...", "rewritten_input": {...}}

A handler that exits 0 with empty stdout has **no opinion** — modelled as
``verdict="allow", matched=False`` (abstain), which never affects a merge.

Failure semantics are deny-safe: a crash, non-zero exit, timeout, or invalid
output becomes a **Deny** verdict for the enforced gates (``PreToolUse`` and
``UserPromptSubmit``, reason "hook failed: …") and a logged no-op (abstain,
``failed=True``) for every other event — a hook failure can never widen
access.

Merging is most-severe-wins: Deny > Ask > Defer > Allow/abstain. The first
non-empty reason wins; ``rewritten_input`` comes from the most severe handler
that provided one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lecode.extras.proc import run_proc
from lecode.hooks.events import (
    EVENTS,
    POST_TOOL_USE,
    PRE_TOOL_USE,
    USER_PROMPT_SUBMIT,
    build_envelope,
)

log = logging.getLogger(__name__)

#: Default per-handler timeout in seconds.
DEFAULT_HOOK_TIMEOUT_S = 10.0

#: Valid verdicts, lowest to highest severity.
SEVERITY = {"allow": 0, "defer": 1, "ask": 2, "deny": 3}

#: Enforced gates: a handler failure becomes Deny instead of abstain.
DENY_SAFE_EVENTS = frozenset({PRE_TOOL_USE, USER_PROMPT_SUBMIT})


@dataclass(frozen=True)
class HookHandler:
    """One configured hook: a shell command plus its timeout."""

    command: str
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S


@dataclass(frozen=True)
class HookVerdict:
    """One handler's answer to an event."""

    verdict: str  # "allow" | "defer" | "ask" | "deny"
    reason: str = ""
    rewritten_input: dict[str, Any] | None = None
    matched: bool = True  # False = abstain (no opinion)
    failed: bool = False  # handler crashed/timed out/produced garbage
    handler: str = ""  # the command that produced this verdict


@dataclass(frozen=True)
class MergedVerdict:
    """The merged outcome of all handlers for one event."""

    verdict: str = "allow"
    reason: str = ""
    rewritten_input: dict[str, Any] | None = None
    failed: bool = False  # any handler failed (for --hooks-test exit codes)


def _failure(handler: HookHandler, event: str, detail: str) -> HookVerdict:
    """A deny-safe failure verdict: Deny for enforced gates, abstain otherwise."""
    reason = f"hook failed: {detail}"
    if event in DENY_SAFE_EVENTS:
        return HookVerdict("deny", reason=reason, failed=True, handler=handler.command)
    log.warning("%s hook handler %r failed: %s", event, handler.command, detail)
    return HookVerdict("allow", reason=reason, matched=False, failed=True, handler=handler.command)


async def run_handler(handler: HookHandler, envelope: dict[str, Any]) -> HookVerdict:
    """Run one handler against one envelope; never raises."""
    event = str(envelope.get("event", ""))
    result = await run_proc(
        ["/bin/sh", "-c", handler.command],
        cwd=envelope.get("cwd") or None,
        input=json.dumps(envelope),
        timeout=handler.timeout_s,
    )
    if result.timed_out:
        return _failure(handler, event, f"timed out after {handler.timeout_s}s")
    if result.exit_code != 0:
        detail = f"exit code {result.exit_code}"
        if result.stderr.strip():
            detail += f": {result.stderr.strip()[:200]}"
        return _failure(handler, event, detail)
    stdout = result.stdout.strip()
    if not stdout:
        return HookVerdict("allow", matched=False, handler=handler.command)  # abstain
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return _failure(handler, event, "invalid JSON output")
    if not isinstance(data, dict) or str(data.get("verdict", "")).lower() not in SEVERITY:
        return _failure(handler, event, "missing or invalid 'verdict' in output")
    rewritten = data.get("rewritten_input")
    if rewritten is not None and not isinstance(rewritten, dict):
        rewritten = None  # lenient: a malformed rewrite is ignored
    return HookVerdict(
        str(data["verdict"]).lower(),
        reason=str(data.get("reason") or ""),
        rewritten_input=rewritten,
        handler=handler.command,
    )


async def dispatch_event(
    event: str, envelope: dict[str, Any], handlers: list[HookHandler]
) -> MergedVerdict:
    """Run all handlers for one event sequentially and merge most-severe-wins."""
    verdict = "allow"
    reason = ""
    rewritten: dict[str, Any] | None = None
    rewritten_severity = -1
    failed = False
    for handler in handlers:
        result = await run_handler(handler, envelope)
        failed = failed or result.failed
        if not result.matched:
            continue
        severity = SEVERITY[result.verdict]
        if severity > SEVERITY[verdict]:
            verdict = result.verdict
        if not reason and result.reason:
            reason = result.reason
        if result.rewritten_input is not None and severity > rewritten_severity:
            rewritten = result.rewritten_input
            rewritten_severity = severity
    return MergedVerdict(verdict=verdict, reason=reason, rewritten_input=rewritten, failed=failed)


@dataclass
class HookDispatcher:
    """Holds the configured handlers and builds envelopes for tool events."""

    handlers: dict[str, list[HookHandler]]
    cwd: Path
    session: Any | None = None

    async def pre_tool_use(self, tool_name: str, args: dict[str, Any]) -> MergedVerdict:
        envelope = build_envelope(
            PRE_TOOL_USE, self.cwd, session=self.session, tool_name=tool_name, tool_args=args
        )
        return await dispatch_event(PRE_TOOL_USE, envelope, self.handlers.get(PRE_TOOL_USE, []))

    async def post_tool_use(
        self, tool_name: str, args: dict[str, Any], content: str, is_error: bool
    ) -> MergedVerdict:
        envelope = build_envelope(
            POST_TOOL_USE,
            self.cwd,
            session=self.session,
            tool_name=tool_name,
            tool_args=args,
            result={"content": content, "is_error": is_error},
        )
        return await dispatch_event(POST_TOOL_USE, envelope, self.handlers.get(POST_TOOL_USE, []))

    async def fire(self, event: str, **payload: Any) -> MergedVerdict:
        """Dispatch any event with an envelope built from ``payload`` kwargs.

        The one entry point for observational fire sites (Stop, SessionStart,
        Notification, …): with no configured handlers it returns the default
        allow verdict without running anything.
        """
        envelope = build_envelope(event, self.cwd, session=self.session, **payload)
        return await dispatch_event(event, envelope, self.handlers.get(event, []))


def dispatcher_from_config(
    config: Any, cwd: Path | str, session: Any | None = None
) -> tuple[HookDispatcher | None, list[str]]:
    """Build a dispatcher from ``config.hooks``; unknown events warn + skip.

    Returns ``(None, warnings)`` when no usable hooks are configured.
    """
    warnings: list[str] = []
    handlers: dict[str, list[HookHandler]] = {}
    for event, commands in (config.hooks or {}).items():
        if event not in EVENTS:
            warnings.append(f"unknown hook event: {event}")
            continue
        handlers[event] = [
            c if isinstance(c, HookHandler) else HookHandler(str(c)) for c in commands
        ]
    if not handlers:
        return None, warnings
    return HookDispatcher(handlers, Path(cwd), session), warnings


def hooks_status(config: Any) -> list[dict[str, Any]]:
    """Configured handlers as rows for the ``/hooks`` status command (Phase 9)."""
    rows: list[dict[str, Any]] = []
    for event in EVENTS:
        for command in (config.hooks or {}).get(event, []):
            rows.append(
                {
                    "event": event,
                    "command": command if isinstance(command, str) else command.command,
                    "timeout_s": (
                        command.timeout_s
                        if isinstance(command, HookHandler)
                        else DEFAULT_HOOK_TIMEOUT_S
                    ),
                }
            )
    return rows
