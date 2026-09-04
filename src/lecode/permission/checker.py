"""The permission checker: pure, synchronous, no I/O.

Decision pipeline in :meth:`PermissionChecker.check`:

1. extract the match target (:func:`~lecode.permission.patterns.target_of`)
2. deny rules (global + overlay extras) are unbypassable → Deny
3. overlay ``denied_tools`` → Deny
4. overlay extra allow/ask rules, last match wins
5. global allow rules then ask rules, last match wins
6. session "allow always" grants → Allow
7. mode fallback (two modes: ``yolo`` allows everything, ``readonly`` allows
   read-class tools only; MCP read-equivalence makes Exa/context7/grep.app
   tools read-class in both modes)
8. doom-loop escalation wraps the result: 3rd identical consecutive call
   turns Allow into Ask with a coach reason; 4th+ is Deny

The seam for lifecycle hooks (Phase 7): hooks receive the returned
:class:`CheckResult` and may only narrow it (Allow → Ask/Deny, Ask → Deny),
never widen it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lecode.config.models import Config, PermissionMode, PermissionRule, PermissionRuleSet
from lecode.permission.patterns import (
    canonical_args,
    is_read_equiv_mcp,
    rule_matches,
    target_of,
)


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class CheckResult:
    decision: Decision
    reason: str
    matched_rule: PermissionRule | None = None


# -- interactive approval decisions (TUI ask prompt) ----------------------------


@dataclass(frozen=True)
class AllowOnce:
    """Approve this single call."""


@dataclass(frozen=True)
class AllowAlways:
    """Approve and grant ``pattern`` (glob over the target) for the session."""

    pattern: str


@dataclass(frozen=True)
class Deny:
    """Refuse the call."""


#: What an interactive approval callback returns.
ApprovalDecision = AllowOnce | AllowAlways | Deny


#: Tools that never modify anything.
READ_TOOLS = frozenset(
    {
        "read",
        "grep",
        "find_files",
        "list_dir",
        "lsp_diagnostics",
        "memory_read",
        "memory_search",
        "advisor",  # a model call, not a mutation — read-class in every mode
        "task",  # dispatches a subagent; its own calls are gated individually
    }
)

#: Doom-loop thresholds (consecutive identical calls).
DOOM_ASK_AT = 3
DOOM_DENY_AT = 4


@dataclass(frozen=True)
class AgentOverlay:
    """Per-agent permission narrowing layered over the global config.

    ``mode`` replaces the fallback mode; ``extra_rules`` are evaluated before
    global rules; ``denied_tools`` always Deny. Overlays can only narrow.
    """

    mode: PermissionMode | None = None
    extra_rules: PermissionRuleSet = field(default_factory=PermissionRuleSet)
    denied_tools: tuple[str, ...] = ()


class SessionPermissions:
    """Session-scoped "allow always" grants: (tool, glob-pattern) pairs."""

    def __init__(self, grants: list[tuple[str, str]] | None = None) -> None:
        self.grants: list[tuple[str, str]] = list(grants or [])

    def grant(self, tool: str, pattern: str) -> None:
        self.grants.append((tool, pattern))

    def matching_grant(self, tool: str, target: str) -> str | None:
        """The first grant pattern matching this call, if any."""
        for granted_tool, pattern in self.grants:
            if granted_tool == tool and fnmatch_glob(pattern, target):
                return pattern
        return None


def fnmatch_glob(pattern: str, target: str) -> bool:
    return rule_matches(PermissionRule(pattern=pattern, kind="glob"), target)


@dataclass
class _DoomTracker:
    last_call: tuple[str, str] | None = None
    count: int = 0

    def record(self, tool_name: str, args: dict[str, Any]) -> int:
        call = (tool_name, canonical_args(args))
        if call == self.last_call:
            self.count += 1
        else:
            self.last_call = call
            self.count = 1
        return self.count


class PermissionChecker:
    """Evaluates tool calls against rules, grants, and the mode fallback."""

    def __init__(
        self,
        config: Config,
        session_perms: SessionPermissions | None = None,
        mode: PermissionMode | None = None,
        cwd: Path | None = None,
        _overlay: AgentOverlay | None = None,
        _doom: _DoomTracker | None = None,
    ) -> None:
        self._config = config
        self._rules = config.permissions.rules
        self._mode: PermissionMode = mode or config.permissions.mode
        self._session = session_perms or SessionPermissions()
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._overlay = _overlay
        self._doom = _doom or _DoomTracker()

    @property
    def mode(self) -> PermissionMode:
        """The mode used by the fallback step."""
        return self._mode

    def set_mode(self, mode: PermissionMode) -> None:
        """Switch the fallback mode (``/permissions``); overlays still win."""
        self._mode = mode

    def for_agent(self, overlay: AgentOverlay) -> PermissionChecker:
        """A derived checker with the overlay applied (shares doom tracking)."""
        return PermissionChecker(
            self._config, self._session, self._mode, self._cwd, overlay, self._doom
        )

    def check(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent_overlay: AgentOverlay | None = None,
    ) -> CheckResult:
        """Decide Allow/Ask/Deny for one tool call. Pure; prompting is the TUI's job."""
        overlay = agent_overlay or self._overlay
        target = target_of(tool_name, args)
        result = self._base_decision(tool_name, target, overlay)
        return self._apply_doom_loop(tool_name, args, result)

    # -- pipeline steps ------------------------------------------------------

    def _base_decision(
        self, tool_name: str, target: str, overlay: AgentOverlay | None
    ) -> CheckResult:
        # Deny rules are unbypassable: global table + overlay extras.
        deny = self._last_match(self._rules.deny, tool_name, target)
        if overlay is not None:
            deny = self._last_match(overlay.extra_rules.deny, tool_name, target) or deny
        if deny is not None:
            return CheckResult(Decision.DENY, f"deny rule matched: {deny.pattern}", deny)

        # Overlay denied tools always deny.
        if overlay is not None and tool_name in overlay.denied_tools:
            return CheckResult(Decision.DENY, f"tool denied by agent overlay: {tool_name}")

        # Overlay extra allow/ask rules first (last match wins within them).
        if overlay is not None:
            extra = self._last_match_allow_ask(overlay.extra_rules, tool_name, target)
            if extra is not None:
                return extra

        # Global rules: allow table walked first, then ask table; last match wins.
        matched = self._last_match_allow_ask(self._rules, tool_name, target)
        if matched is not None:
            return matched

        # Session "allow always" grants (can never reach a deny: handled above).
        grant = self._session.matching_grant(tool_name, target)
        if grant is not None:
            return CheckResult(Decision.ALLOW, f"session grant: {grant}")

        # Mode fallback.
        mode = overlay.mode if overlay and overlay.mode else self._mode
        return self._mode_fallback(mode, tool_name, target)

    def _last_match(
        self, table: dict[str, list[PermissionRule]], tool_name: str, target: str
    ) -> PermissionRule | None:
        match = None
        for rule in table.get(tool_name, []):
            if rule_matches(rule, target):
                match = rule
        return match

    def _last_match_allow_ask(
        self, rules: PermissionRuleSet, tool_name: str, target: str
    ) -> CheckResult | None:
        result: CheckResult | None = None
        for rule in rules.allow.get(tool_name, []):
            if rule_matches(rule, target):
                result = CheckResult(Decision.ALLOW, f"allow rule matched: {rule.pattern}", rule)
        for rule in rules.ask.get(tool_name, []):
            if rule_matches(rule, target):
                result = CheckResult(Decision.ASK, f"ask rule matched: {rule.pattern}", rule)
        return result

    def _apply_doom_loop(
        self, tool_name: str, args: dict[str, Any], result: CheckResult
    ) -> CheckResult:
        count = self._doom.record(tool_name, args)
        if count >= DOOM_DENY_AT:
            return CheckResult(
                Decision.DENY,
                f"doom loop: same call made {count} times with identical arguments",
            )
        if count >= DOOM_ASK_AT and result.decision == Decision.ALLOW:
            return CheckResult(
                Decision.ASK,
                f"doom loop: same call made {count} times with identical arguments",
            )
        return result

    # -- mode fallback ---------------------------------------------------------

    def _is_read_class(self, tool_name: str) -> bool:
        return tool_name in READ_TOOLS or is_read_equiv_mcp(tool_name)

    def _mode_fallback(self, mode: PermissionMode, tool_name: str, target: str) -> CheckResult:
        reason = f"mode: {mode}"
        if mode == "yolo":
            return CheckResult(Decision.ALLOW, reason)
        # readonly: read-class tools are allowed, everything else is denied.
        decision = Decision.ALLOW if self._is_read_class(tool_name) else Decision.DENY
        return CheckResult(decision, reason)
