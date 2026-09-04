"""Built-in agent permission overlays.

``build`` (the default agent) gets full access — no narrowing. ``plan`` is
readonly-equivalent: it may read freely but cannot write or run commands.
These map onto the permission modes; user-defined agents (Phase 6) layer
their own overlays the same way.
"""

from __future__ import annotations

from lecode.permission.checker import AgentOverlay

BUILTIN_AGENT_OVERLAYS: dict[str, AgentOverlay] = {
    "build": AgentOverlay(),
    "plan": AgentOverlay(mode="readonly"),
}
