"""The permission system: rules, six modes, overlays, doom-loop detection."""

from __future__ import annotations

from lecode.permission.builtin import BUILTIN_AGENT_OVERLAYS
from lecode.permission.checker import (
    AgentOverlay,
    AllowAlways,
    AllowOnce,
    ApprovalDecision,
    CheckResult,
    Decision,
    Deny,
    PermissionChecker,
    SessionPermissions,
)
from lecode.permission.patterns import target_of

__all__ = [
    "BUILTIN_AGENT_OVERLAYS",
    "AgentOverlay",
    "AllowAlways",
    "AllowOnce",
    "ApprovalDecision",
    "CheckResult",
    "Decision",
    "Deny",
    "PermissionChecker",
    "SessionPermissions",
    "target_of",
]
