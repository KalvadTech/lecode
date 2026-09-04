"""Startup loading screen.

Printed to the normal scrollback right before the chat opens: one line per
subsystem explaining exactly what was loaded (config files, provider,
prompt, AGENTS.md context, skills, agents, memory, tools, permissions,
hooks, LSP, MCP). Headless mode never shows it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from lecode.context.agents_md import collect as collect_agents_md

if TYPE_CHECKING:
    from lecode.agent.builder import Runtime
    from lecode.config.loader import LoadedConfig
    from lecode.config.models import Config
    from lecode.extras.mcp_client import ServerStatus
    from lecode.providers import ProviderSpec
    from lecode.session.storage import Session, SessionStore
    from lecode.tui.themes import Theme

#: Step statuses.
OK = "ok"
WARN = "warn"
SKIP = "skip"

_MARKS = {OK: "✓", WARN: "!", SKIP: "–"}  # noqa: RUF001 — the en dash is the intended skip glyph

#: ASCII-art banner (figlet "standard") printed above the loading panel.
_BANNER = r""" _                    _
| | ___  ___ ___   __| | ___
| |/ _ \/ __/ _ \ / _` |/ _ \
| |  __/ (_| (_) | (_| |  __/
|_|\___|\___\___/ \__,_|\___|"""

_BYLINE = "by wowi42"


@dataclass(frozen=True)
class LoadStep:
    """One line of the loading screen: what loaded, with what detail."""

    label: str
    detail: str
    status: str = OK


def _rel(path: Path, cwd: Path) -> str:
    """Display ``path`` relative to ``cwd`` when possible."""
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _join_names(names: list[str], cap: int = 6) -> str:
    if len(names) <= cap:
        return ", ".join(names)
    return ", ".join(names[:cap]) + f" +{len(names) - cap} more"


def build_load_report(
    *,
    config: Config,
    loaded: LoadedConfig,
    runtime: Runtime,
    session: Session,
    store: SessionStore,
    cwd: Path,
    resumed: bool,
    provider_spec: ProviderSpec,
    key_source: str,
    read_only: bool = False,
    models_origin: str | None = None,
    models_count: int = 0,
    mcp_servers: list[ServerStatus] | None = None,
) -> list[LoadStep]:
    """Collect the per-subsystem lines describing what this session loaded."""
    from lecode.memory import memory_root  # deferred: pulls in the store layer

    steps: list[LoadStep] = []

    # session
    if resumed:
        from lecode.session.stats import session_stats

        stats = session_stats(store, session)
        detail = f"{session.meta.name} — resumed, {stats.message_count} messages"
    else:
        detail = f"{session.meta.name} — new session"
    steps.append(LoadStep("session", detail))

    # config files
    if loaded.sources:
        detail = " + ".join(_rel(Path(s), cwd) for s in loaded.sources)
    else:
        detail = "defaults (no config file)"
    status = WARN if loaded.warnings else OK
    if loaded.warnings:
        detail += f" — {len(loaded.warnings)} warning(s)"
    steps.append(LoadStep("config", detail, status))

    # provider / model / auth (never print the key itself)
    auth = {"none": "no API key", "cli": "key from --api-key"}.get(
        key_source, f"key from {key_source}"
    )
    steps.append(
        LoadStep(
            "provider",
            f"{provider_spec.name} · {provider_spec.base_url}\nmodel {config.llm.model} · {auth}",
        )
    )

    # model catalog provenance (live fetch, else empty)
    if models_origin == "live":
        steps.append(LoadStep("models", f"{models_count} fetched live from the provider"))
    elif models_origin is not None:
        steps.append(LoadStep("models", "unavailable (fetch failed)", WARN))

    # system prompt
    sp = config.llm.system_prompt
    if sp.custom:
        detail = "custom override (llm.system_prompt.custom)"
    elif sp.style == "rich":
        detail = "rich" + (f" + persona {sp.persona}" if sp.persona else "")
    else:
        detail = "minimal"
    steps.append(LoadStep("prompt", detail))

    # project context files
    entries = collect_agents_md(cwd)
    if entries:
        names = [_rel(p, cwd) for p, _ in entries]
        steps.append(LoadStep("context", _join_names(names, cap=4)))
    else:
        steps.append(LoadStep("context", "no AGENTS.md/CLAUDE.md found", SKIP))

    # skills
    skill_names = [s.name for s in runtime.skills.list()]
    if skill_names:
        steps.append(LoadStep("skills", _join_names(skill_names)))
    else:
        steps.append(LoadStep("skills", "none", SKIP))

    # agents (built-ins are always present; highlight user-defined ones)
    builtin = ("build", "plan", "explore")
    user_agents = [a.name for a in runtime.agents.visible() if a.name not in builtin]
    primaries = ", ".join(a.name for a in runtime.agents.primaries())
    detail = f"primaries: {primaries}"
    if user_agents:
        detail += f" · custom: {_join_names(user_agents)}"
    steps.append(LoadStep("agents", detail))

    # memory
    if config.memory.enabled:
        memory_file = memory_root(cwd) / "MEMORY.md"
        if memory_file.is_file():
            size = memory_file.stat().st_size
            steps.append(LoadStep("memory", f"long-term {size / 1024:.1f} KB injected"))
        else:
            steps.append(LoadStep("memory", "enabled, empty", SKIP))
    else:
        steps.append(LoadStep("memory", "disabled", SKIP))

    # tools
    names = runtime.registry.names()
    detail = f"{len(names)} tools"
    if runtime.ctx.auto_approve:
        detail += " · auto-approved"
    steps.append(LoadStep("tools", detail))

    # permissions
    mode = runtime.ctx.permission_checker.mode
    detail = f"mode {mode}" + (" · --safe" if read_only else "")
    rules = config.permissions.rules
    if rules.allow or rules.ask or rules.deny:
        detail += " · custom rules"
    steps.append(LoadStep("permissions", detail))

    # hooks
    if config.hooks:
        detail = ", ".join(f"{event}({len(cmds)})" for event, cmds in sorted(config.hooks.items()))
        steps.append(LoadStep("hooks", detail))
    else:
        steps.append(LoadStep("hooks", "none configured", SKIP))

    # pierre (post-task reviewer)
    if config.pierre.enabled:
        reviewer = config.pierre.model or config.llm.model
        steps.append(LoadStep("pierre", f"reviewing with {reviewer}"))
    else:
        steps.append(LoadStep("pierre", "off", SKIP))

    # lsp
    steps.append(
        LoadStep(
            "lsp",
            "enabled" if config.lsp.enabled else "disabled",
            OK if config.lsp.enabled else SKIP,
        )
    )

    # mcp — with live statuses when the caller connected the servers first
    if mcp_servers is not None:
        lines: list[str] = []
        for s in mcp_servers:
            if s.state == "connected":
                lines.append(f"{s.name}: connected · {s.tools} tools")
            elif s.state == "failed":
                lines.append(f"{s.name}: failed — {s.error or 'connect error'}")
            else:
                lines.append(f"{s.name}: disabled")
        exa_missing = (
            config.mcp.enable_exa
            and not os.environ.get("EXA_API_KEY")
            and not any(s.name == "exa" for s in mcp_servers)
        )
        if exa_missing:
            lines.append("exa: no EXA_API_KEY")
        if not lines:
            steps.append(LoadStep("mcp", "no servers", SKIP))
        else:
            degraded = exa_missing or any(s.state == "failed" for s in mcp_servers)
            steps.append(LoadStep("mcp", "\n".join(lines), WARN if degraded else OK))
        return steps

    # no live statuses (tests, headless): report the configuration only
    mcp_bits: list[str] = []
    mcp_status = OK
    if config.mcp.enable_exa:
        if os.environ.get("EXA_API_KEY"):
            mcp_bits.append("exa")
        else:
            mcp_bits.append("exa (no EXA_API_KEY)")
            mcp_status = WARN
    if config.mcp.enable_context7:
        mcp_bits.append("context7")
    configured = [n for n, s in config.mcp.servers.items() if s.enabled]
    if configured:
        mcp_bits.append(_join_names(sorted(configured), cap=3))
    if mcp_bits:
        steps.append(LoadStep("mcp", " · ".join(mcp_bits), mcp_status))
    else:
        steps.append(LoadStep("mcp", "no servers", SKIP))

    return steps


def render_loading_screen(
    console: Console,
    theme: Theme,
    *,
    session_name: str,
    steps: list[LoadStep],
    cwd: Path,
) -> None:
    """Render the banner and the loading panel: one status line per subsystem."""
    console.print(Text(_BANNER, style=theme.accent), justify="center")
    console.print(Text(_BYLINE, style=theme.muted), justify="center")
    console.print()
    body = Text()
    width = max(len(s.label) for s in steps)
    for i, step in enumerate(steps):
        if i:
            body.append("\n")
        mark_style = {
            OK: theme.success,
            WARN: theme.warning,
            SKIP: theme.muted,
        }[step.status]
        body.append(f" {_MARKS[step.status]} ", style=mark_style)
        body.append(step.label.ljust(width), style=theme.accent)
        body.append("  ")
        lines = step.detail.split("\n")
        body.append(lines[0], style=theme.muted if step.status == SKIP else theme.text)
        for extra in lines[1:]:
            body.append("\n" + " " * (width + 4), style=theme.text)
            body.append(extra, style=theme.text)
    panel = Panel(
        body,
        title=f"[{theme.accent}]lecode[/{theme.accent}] — {session_name}",
        subtitle=f"[{theme.muted}]{cwd}[/{theme.muted}]",
        border_style=theme.muted,
        padding=(0, 1),
    )
    console.print(panel)


def show_loading_screen(
    *,
    config: Config,
    loaded: LoadedConfig,
    runtime: Runtime,
    session: Session,
    store: SessionStore,
    cwd: Path,
    resumed: bool,
    provider_spec: ProviderSpec,
    key_source: str,
    console: Console,
    read_only: bool = False,
    models_origin: str | None = None,
    models_count: int = 0,
    mcp_servers: list[ServerStatus] | None = None,
) -> None:
    """Build the report and print it. Never raises into startup."""
    from lecode.tui.themes import THEME

    try:
        steps = build_load_report(
            config=config,
            loaded=loaded,
            runtime=runtime,
            session=session,
            store=store,
            cwd=cwd,
            resumed=resumed,
            provider_spec=provider_spec,
            key_source=key_source,
            read_only=read_only,
            models_origin=models_origin,
            models_count=models_count,
            mcp_servers=mcp_servers,
        )
        render_loading_screen(console, THEME, session_name=session.meta.name, steps=steps, cwd=cwd)
    except Exception:
        # The loading screen is informational; it must never break startup.
        import logging

        logging.getLogger(__name__).debug("loading screen failed", exc_info=True)
