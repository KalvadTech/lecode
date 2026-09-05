"""Custom user-defined agents (opencode lineage).

Agents are markdown files with YAML frontmatter in
``~/.config/lecode/agents/*.md`` (global, ``LECODE_CONFIG_DIR`` aware) and
``.lecode/agents/*.md`` (project, nearest from the cwd up to the git root).
The project layer wins on name collisions; user files may also override the
built-in agents (``build`` / ``plan`` / ``explore``) by name.

Frontmatter: ``description`` (required), ``mode: primary|subagent|all``
(default ``all``), ``model``, ``temperature``, ``permission`` (overlay mapping
onto :class:`~lecode.permission.checker.AgentOverlay`), ``hidden`` (excluded
from pickers), ``color``. The body is the agent's system prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from lecode.config.loader import config_dir
from lecode.config.models import PermissionMode, PermissionRuleSet
from lecode.context.agents_md import find_git_root
from lecode.context.frontmatter import split_frontmatter
from lecode.permission.builtin import BUILTIN_AGENT_OVERLAYS
from lecode.permission.checker import AgentOverlay

AgentMode = Literal["primary", "subagent", "all"]

AGENT_MODES: tuple[str, ...] = get_args(AgentMode)

#: Characters allowed inside an agent name (used by mention parsing).
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

_BUILD_BODY = ""
_PLAN_BODY = (
    "You are in planning mode: explore and design, but never modify files or run "
    "state-changing commands. Produce a concrete plan the user can approve."
)
_EXPLORE_BODY = (
    "You are a read-only exploration agent. Answer the question by searching and "
    "reading the codebase; be fast and factual. Never modify anything."
)


@dataclass(frozen=True)
class AgentDefinition:
    """One agent: built-in or loaded from a markdown file."""

    name: str
    description: str
    body: str  # the agent's system prompt
    mode: AgentMode = "all"
    model: str | None = None
    temperature: float | None = None
    overlay: AgentOverlay | None = None
    hidden: bool = False
    color: str | None = None
    builtin: bool = False


def _builtin_agents() -> dict[str, AgentDefinition]:
    return {
        "build": AgentDefinition(
            name="build",
            description="Full-access coding agent.",
            body=_BUILD_BODY,
            mode="primary",
            builtin=True,
        ),
        "plan": AgentDefinition(
            name="plan",
            description="Read-only planning agent.",
            body=_PLAN_BODY,
            mode="primary",
            overlay=BUILTIN_AGENT_OVERLAYS["plan"],
            builtin=True,
        ),
        "explore": AgentDefinition(
            name="explore",
            description="Fast read-only codebase explorer (subagent).",
            body=_EXPLORE_BODY,
            mode="subagent",
            overlay=AgentOverlay(mode="readonly"),
            builtin=True,
        ),
    }


def global_agents_dir() -> Path:
    """The global agents directory (``LECODE_CONFIG_DIR`` aware)."""
    return config_dir() / "agents"


def project_agents_dir(cwd: Path | None = None) -> Path | None:
    """Nearest ``.lecode/agents/`` from ``cwd`` up to the git root."""
    start = (cwd or Path.cwd()).resolve()
    stop = find_git_root(start) or start
    current = start
    while True:
        candidate = current / ".lecode" / "agents"
        if candidate.is_dir():
            return candidate
        if current == stop:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _normalize_rules(raw: Any) -> dict[str, Any]:
    """Accept rules as ``{pattern, kind}`` tables or bare glob strings."""
    if not isinstance(raw, dict):
        raise ValueError("'permission.rules' must be a mapping")
    normalized: dict[str, Any] = {}
    for table, tools in raw.items():
        if table not in ("allow", "ask", "deny"):
            raise ValueError(f"invalid permission rule table: {table!r}")
        if not isinstance(tools, dict):
            raise ValueError(f"'permission.rules.{table}' must be a mapping of tool → rules")
        normalized[table] = {
            str(tool): [
                {"pattern": rule} if isinstance(rule, str) else rule
                for rule in (rules if isinstance(rules, list) else [rules])
            ]
            for tool, rules in tools.items()
        }
    return normalized


def _parse_overlay(data: Any) -> AgentOverlay | None:
    """Map a frontmatter ``permission`` table onto an :class:`AgentOverlay`."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("'permission' must be a mapping")
    mode = data.get("mode")
    if mode is not None and mode not in get_args(PermissionMode):
        raise ValueError(f"invalid permission mode: {mode!r}")
    try:
        rules = PermissionRuleSet.model_validate(_normalize_rules(data.get("rules") or {}))
    except ValueError as e:
        raise ValueError(f"invalid permission rules: {e}") from e
    denied = data.get("denied_tools") or []
    if not isinstance(denied, list):
        raise ValueError("'permission.denied_tools' must be a list")
    return AgentOverlay(mode=mode, extra_rules=rules, denied_tools=tuple(str(t) for t in denied))


def _load_agent_file(path: Path, warnings: list[str]) -> AgentDefinition | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        warnings.append(f"{path}: unreadable: {e}")
        return None
    try:
        frontmatter, body = split_frontmatter(text)
        description = str(frontmatter.get("description") or "").strip()
        if not description:
            raise ValueError("missing required 'description' in frontmatter")
        mode = frontmatter.get("mode", "all")
        if mode not in AGENT_MODES:
            raise ValueError(f"invalid mode: {mode!r} (expected one of {AGENT_MODES})")
        temperature = frontmatter.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, int | float)
        ):
            raise ValueError(f"'temperature' must be a number, got {temperature!r}")
        overlay = _parse_overlay(frontmatter.get("permission"))
    except ValueError as e:
        warnings.append(f"{path}: {e}")
        return None
    color = frontmatter.get("color")
    model = frontmatter.get("model")
    return AgentDefinition(
        name=path.stem,
        description=description,
        body=body.strip(),
        mode=mode,
        model=str(model) if model else None,
        temperature=float(temperature) if temperature is not None else None,
        overlay=overlay,
        hidden=bool(frontmatter.get("hidden", False)),
        color=str(color) if color else None,
    )


class AgentRegistry:
    """All known agents (built-ins plus loaded files) plus load warnings."""

    def __init__(
        self,
        agents: dict[str, AgentDefinition] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self._agents = dict(agents or {})
        self.warnings = list(warnings or [])

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def names(self) -> list[str]:
        return sorted(self._agents)

    def visible(self) -> list[AgentDefinition]:
        """Agents shown in pickers/listings (hidden ones excluded)."""
        return [self._agents[name] for name in self.names() if not self._agents[name].hidden]

    def primaries(self) -> list[AgentDefinition]:
        """Tab-cyclable primaries: build, plan, then user primaries alphabetically."""
        eligible = {
            a.name: a
            for a in self._agents.values()
            if a.mode in ("primary", "all") and not a.hidden
        }
        ordered = [eligible.pop(name) for name in ("build", "plan") if name in eligible]
        ordered += [eligible[name] for name in sorted(eligible)]
        return ordered

    def subagents(self) -> list[AgentDefinition]:
        """Non-hidden agents invocable via the ``task`` tool / ``@mention``."""
        return [
            self._agents[name]
            for name in self.names()
            if self._agents[name].mode in ("subagent", "all") and not self._agents[name].hidden
        ]

    def overlay_for(self, name: str) -> AgentOverlay | None:
        agent = self._agents.get(name)
        return agent.overlay if agent is not None else None

    def cycle(self, current: str) -> str:
        """The next primary after ``current`` (wraps; unknown → first primary)."""
        names = [a.name for a in self.primaries()]
        if not names:
            return current
        if current not in names:
            return names[0]
        return names[(names.index(current) + 1) % len(names)]


def load_agents(cwd: Path | None = None, global_dir: Path | None = None) -> AgentRegistry:
    """Load agents: built-ins, then global files, then project files (later wins)."""
    warnings: list[str] = []
    agents = _builtin_agents()
    for root in (global_dir or global_agents_dir(), project_agents_dir(cwd)):
        if root is None or not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            definition = _load_agent_file(path, warnings)
            if definition is not None:
                agents[definition.name] = definition
    return AgentRegistry(agents, warnings)


def parse_mentions(text: str, registry: AgentRegistry) -> tuple[list[str], str]:
    """Extract ``@agent`` mentions; returns ``(agent_names, cleaned_text)``.

    A word starting with ``@`` matches when a known agent name is the word, or
    a prefix of it followed by a non-name character (``@plan, …`` works).
    Longest names are tried first; unknown ``@words`` are left untouched.
    """
    candidates = sorted(registry.names(), key=len, reverse=True)
    found: list[str] = []
    kept: list[str] = []
    # Split on whitespace but keep the separators: newlines and spacing in
    # the user's message are significant (multiline chatbox input). A removed
    # mention also drops one following space/tab run, so "@a, then @b x"
    # cleans to "then x" — but a following newline is kept.
    skip_space = False
    for word in re.split(r"(\s+)", text):
        if skip_space:
            skip_space = False
            if word.isspace() and "\n" not in word:
                continue
        matched = None
        if word.startswith("@"):
            token = word[1:]
            for name in candidates:
                if token == name or (
                    token.startswith(name) and token[len(name)] not in _NAME_CHARS
                ):
                    matched = name
                    break
        if matched is not None:
            found.append(matched)
            skip_space = True
        else:
            kept.append(word)
    return found, "".join(kept).strip(" \t")
