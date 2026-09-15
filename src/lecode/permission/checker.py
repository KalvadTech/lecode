"""The permission checker: pure, synchronous, no I/O.

Root policy evaluates deny rules, explicit read-only enforcement, allow/ask
rules, session grants, then mode fallback. Overlay rules and fallback are
intersected with that result and every ancestor policy: Deny > Ask > Allow.
Path rules match the original target and normalized absolute/relative paths
in the executing checker's cwd, including when evaluating ancestor policy.

Only the executing checker's doom tracker records the call: the 3rd identical
consecutive call turns Allow into Ask; the 4th+ is Deny. Agent views share a
tracker and grants; children have independent trackers and grant stores.

The seam for lifecycle hooks (Phase 7): hooks receive the returned
:class:`CheckResult` and may only narrow it (Allow → Ask/Deny, Ask → Deny),
never widen it.
"""

from __future__ import annotations

import os.path
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
        "ask_user",  # only asks; never touches anything itself
        "task",  # dispatches a subagent; its own calls are gated individually
        "tasks_list",
        "tasks_output",
        "tasks_wait",
    }
)

#: Doom-loop thresholds (consecutive identical calls).
DOOM_ASK_AT = 3
DOOM_DENY_AT = 4


@dataclass(frozen=True)
class AgentOverlay:
    """Per-agent permission narrowing layered over the global config.

    ``mode`` replaces the overlay layer's fallback; ``extra_rules`` precede
    global rules within that layer; ``denied_tools`` always Deny. The result
    is capped by the base and ancestor policies, so overlays can only narrow.
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
        *,
        read_only: bool = False,
    ) -> None:
        self._config = config
        self._rules = config.permissions.rules
        self._mode: PermissionMode = mode or config.permissions.mode
        self._session = session_perms or SessionPermissions()
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._overlay = _overlay
        self._doom = _doom or _DoomTracker()
        self._read_only = read_only
        self._parent: PermissionChecker | None = None

    @property
    def mode(self) -> PermissionMode:
        """The narrowest fallback mode in this policy's ancestry."""
        if (self._overlay is not None and self._overlay.mode == "readonly") or (
            self._parent is not None and self._parent.mode == "readonly"
        ):
            return "readonly"
        return self._mode

    @property
    def read_only(self) -> bool:
        """Whether policy guarantees denial of non read-class tools.

        A readonly fallback with writable rule/grant exceptions is not a safe
        shared-checkout policy. Conservatively treat Ask as writable too.
        """
        if self._read_only or (self._parent is not None and self._parent.read_only):
            return True
        mode = self._overlay.mode if self._overlay and self._overlay.mode else self._mode
        if mode != "readonly":
            return False
        rules = [self._rules]
        if self._overlay is not None:
            rules.append(self._overlay.extra_rules)
        return not (
            any(
                entries and not self._is_read_class(tool)
                for ruleset in rules
                for table in (ruleset.allow, ruleset.ask)
                for tool, entries in table.items()
            )
            or any(not self._is_read_class(tool) for tool, _ in self._session.grants)
        )

    def set_mode(self, mode: PermissionMode) -> None:
        """Switch the fallback mode (``/permissions``); overlays still win."""
        self._mode = mode

    def for_agent(self, overlay: AgentOverlay, *, read_only: bool = False) -> PermissionChecker:
        """Narrow this policy for an agent, sharing session grants and doom tracking."""
        derived = self.for_child(overlay, session_perms=self._session, read_only=read_only)
        derived._doom = self._doom
        return derived

    def for_child(
        self,
        overlay: AgentOverlay | None = None,
        *,
        cwd: Path | None = None,
        session_perms: SessionPermissions | None = None,
        read_only: bool = False,
    ) -> PermissionChecker:
        """Derive an isolated child capped by every ancestor's full policy.

        Uses the supplied child grants (fresh empty grants by default) and a
        fresh doom tracker. Ancestor evaluation never records a parent call.
        """
        derived = PermissionChecker(
            self._config,
            session_perms,
            self._mode,
            self._cwd if cwd is None else cwd,
            overlay,
            read_only=self._read_only or read_only,
        )
        derived._parent = self
        return derived

    def check(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent_overlay: AgentOverlay | None = None,
    ) -> CheckResult:
        """Decide Allow/Ask/Deny for one tool call. Pure; prompting is the TUI's job."""
        if agent_overlay is not None:
            return self.for_agent(agent_overlay).check(tool_name, args)
        target = target_of(tool_name, args)
        targets = (target,)
        if target and (
            tool_name in ("read", "write", "edit", "list_dir")
            or (tool_name in ("grep", "find_files") and args.get("path"))
        ):
            absolute = os.path.abspath(self._cwd / target)
            targets = (target, absolute, os.path.relpath(absolute, self._cwd))
        result = self._policy_decision(tool_name, targets)
        return self._apply_doom_loop(tool_name, args, result)

    # -- pipeline steps ------------------------------------------------------

    def _policy_decision(self, tool_name: str, targets: tuple[str, ...]) -> CheckResult:
        """Intersect every policy layer without recording an ancestor call."""
        results = [self._base_decision(tool_name, targets, None)]
        if self._overlay is not None:
            results.append(self._base_decision(tool_name, targets, self._overlay))
        if self._parent is not None:
            results.append(self._parent._policy_decision(tool_name, targets))
        priority = {Decision.ALLOW: 0, Decision.ASK: 1, Decision.DENY: 2}
        return max(results, key=lambda result: priority[result.decision])

    def _base_decision(
        self, tool_name: str, targets: tuple[str, ...], overlay: AgentOverlay | None
    ) -> CheckResult:
        # Deny rules are unbypassable: global table + overlay extras.
        deny = self._last_match(self._rules.deny, tool_name, targets)
        if overlay is not None:
            deny = self._last_match(overlay.extra_rules.deny, tool_name, targets) or deny
        if deny is not None:
            return CheckResult(Decision.DENY, f"deny rule matched: {deny.pattern}", deny)

        # Overlay denied tools always deny.
        if overlay is not None and tool_name in overlay.denied_tools:
            return CheckResult(Decision.DENY, f"tool denied by agent overlay: {tool_name}")

        # Read-only checkers can never be widened by overlay rules, global
        # rules, session grants, or the mode fallback.
        if self._read_only and not self._is_read_class(tool_name):
            return CheckResult(Decision.DENY, f"read-only: {tool_name} is not a read-class tool")

        # Overlay extra allow/ask rules first (last match wins within them).
        if overlay is not None:
            extra = self._last_match_allow_ask(overlay.extra_rules, tool_name, targets)
            if extra is not None:
                return extra

        # Global rules: allow table walked first, then ask table; last match wins.
        matched = self._last_match_allow_ask(self._rules, tool_name, targets)
        if matched is not None:
            return matched

        # Session "allow always" grants (can never reach a deny: handled above).
        for target in targets:
            grant = self._session.matching_grant(tool_name, target)
            if grant is not None:
                return CheckResult(Decision.ALLOW, f"session grant: {grant}")

        # Mode fallback.
        mode = overlay.mode if overlay and overlay.mode else self._mode
        return self._mode_fallback(mode, tool_name)

    def _last_match(
        self, table: dict[str, list[PermissionRule]], tool_name: str, targets: tuple[str, ...]
    ) -> PermissionRule | None:
        match = None
        for rule in table.get(tool_name, []):
            if any(rule_matches(rule, target) for target in targets):
                match = rule
        return match

    def _last_match_allow_ask(
        self, rules: PermissionRuleSet, tool_name: str, targets: tuple[str, ...]
    ) -> CheckResult | None:
        result: CheckResult | None = None
        for rule in rules.allow.get(tool_name, []):
            if any(rule_matches(rule, target) for target in targets):
                result = CheckResult(Decision.ALLOW, f"allow rule matched: {rule.pattern}", rule)
        for rule in rules.ask.get(tool_name, []):
            if any(rule_matches(rule, target) for target in targets):
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

    def _mode_fallback(self, mode: PermissionMode, tool_name: str) -> CheckResult:
        reason = f"mode: {mode}"
        if mode == "yolo":
            return CheckResult(Decision.ALLOW, reason)
        # readonly: read-class tools are allowed, everything else is denied.
        decision = Decision.ALLOW if self._is_read_class(tool_name) else Decision.DENY
        return CheckResult(decision, reason)
