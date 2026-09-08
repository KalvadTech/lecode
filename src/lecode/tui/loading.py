"""Startup loading screen.

Progressive: the ASCII banner prints the instant startup begins, then one
line per subsystem is appended as it finishes loading (config files,
session, provider, model catalog, prompt, AGENTS.md context, skills,
agents, memory, tools, permissions, hooks, pierre, LSP, MCP). Slow steps
(model fetch, MCP probe) get a ``…`` pending line first. Headless mode
never shows it. The panel renderer is kept for snapshot-style rendering.
"""

from __future__ import annotations

import logging
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
PENDING = "pending"

_MARKS = {OK: "✓", WARN: "!", SKIP: "–", PENDING: "…"}  # noqa: RUF001

#: Label column width for progressive lines (the longest label).
LABEL_WIDTH = len("permissions")

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


def session_step(session: Session, store: SessionStore, *, resumed: bool) -> LoadStep:
    """The session line: name, and for resumes the message count."""
    if resumed:
        from lecode.session.stats import session_stats

        stats = session_stats(store, session)
        return LoadStep("session", f"{session.meta.name} — resumed, {stats.message_count} messages")
    return LoadStep("session", f"{session.meta.name} — new session")


def config_step(loaded: LoadedConfig, cwd: Path) -> LoadStep:
    """Which config files were merged, plus a warning count."""
    if loaded.sources:
        detail = " + ".join(_rel(Path(s), cwd) for s in loaded.sources)
    else:
        detail = "defaults (no config file)"
    status = WARN if loaded.warnings else OK
    if loaded.warnings:
        detail += f" — {len(loaded.warnings)} warning(s)"
    return LoadStep("config", detail, status)


def provider_step(config: Config, provider_spec: ProviderSpec, key_source: str) -> LoadStep:
    """Provider, model, and where the key came from (never the key itself)."""
    auth = {"none": "no API key", "cli": "key from --api-key"}.get(
        key_source, f"key from {key_source}"
    )
    return LoadStep(
        "provider",
        f"{provider_spec.name} · {provider_spec.base_url}\nmodel {config.llm.model} · {auth}",
    )


def models_step(models_origin: str | None, models_count: int) -> LoadStep | None:
    """Model-catalog provenance; ``None`` when no fetch was attempted."""
    if models_origin == "live":
        return LoadStep("models", f"{models_count} fetched live from the provider")
    if models_origin is not None:
        return LoadStep("models", "unavailable (fetch failed)", WARN)
    return None


def mcp_step(config: Config, mcp_servers: list[ServerStatus] | None = None) -> LoadStep:
    """The MCP line: live per-server status when probed, else configuration."""
    # with live statuses (the caller connected the servers first)
    if mcp_servers is not None:
        lines: list[str] = []
        for s in mcp_servers:
            if s.state == "connected":
                lines.append(f"{s.name}: connected · {s.tools} tools")
            elif s.state == "failed":
                lines.append(f"{s.name}: failed — {s.error or 'connect error'}")
            elif s.state == "auth_required":
                lines.append(s.auth_hint)
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
            return LoadStep("mcp", "no servers", SKIP)
        degraded = exa_missing or any(s.state in ("failed", "auth_required") for s in mcp_servers)
        return LoadStep("mcp", "\n".join(lines), WARN if degraded else OK)

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
        return LoadStep("mcp", " · ".join(mcp_bits), mcp_status)
    return LoadStep("mcp", "no servers", SKIP)


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

    steps.append(session_step(session, store, resumed=resumed))
    steps.append(config_step(loaded, cwd))
    steps.append(provider_step(config, provider_spec, key_source))
    models = models_step(models_origin, models_count)
    if models is not None:
        steps.append(models)

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

    steps.append(mcp_step(config, mcp_servers))
    return steps


def print_banner(console: Console, theme: Theme) -> None:
    """The ASCII-art banner + byline — printed immediately at startup."""
    console.print(Text(_BANNER, style=theme.accent), justify="center")
    console.print(Text(_BYLINE, style=theme.muted), justify="center")
    console.print()


def format_step(step: LoadStep, theme: Theme, width: int = LABEL_WIDTH) -> Text:
    """One loading line: mark, padded label, detail (multi-line indented)."""
    mark_style = {
        OK: theme.success,
        WARN: theme.warning,
        SKIP: theme.muted,
        PENDING: theme.muted,
    }[step.status]
    body = Text()
    body.append(f" {_MARKS[step.status]} ", style=mark_style)
    body.append(step.label.ljust(width), style=theme.accent)
    body.append("  ")
    lines = step.detail.split("\n")
    body.append(lines[0], style=theme.muted if step.status in (SKIP, PENDING) else theme.text)
    for extra in lines[1:]:
        body.append("\n" + " " * (width + 4))
        body.append(extra, style=theme.text)
    return body


def print_step(console: Console, theme: Theme, step: LoadStep) -> None:
    """Print one progressive loading line."""
    console.print(format_step(step, theme))


class LoadingProgress:
    """Progressive startup screen: banner first, then a line per subsystem
    as it finishes loading. Informational only — nothing here may raise
    into startup."""

    def __init__(self, console: Console, theme: Theme | None = None) -> None:
        from lecode.tui.themes import THEME

        self._console = console
        self._theme = theme or THEME

    def banner(self) -> None:
        """Print the ASCII art immediately."""
        try:
            print_banner(self._console, self._theme)
        except Exception:
            logging.getLogger(__name__).debug("loading banner failed", exc_info=True)

    def step(self, step: LoadStep | None) -> None:
        """Print one finished step (``None`` = nothing to report)."""
        if step is None:
            return
        try:
            print_step(self._console, self._theme, step)
        except Exception:
            logging.getLogger(__name__).debug("loading step failed", exc_info=True)

    def pending(self, label: str, detail: str) -> None:
        """Print a ``…`` line for a slow step that just started."""
        self.step(LoadStep(label, detail, PENDING))


def render_loading_screen(
    console: Console,
    theme: Theme,
    *,
    session_name: str,
    steps: list[LoadStep],
    cwd: Path,
) -> None:
    """Render the banner and the loading panel: one status line per subsystem."""
    print_banner(console, theme)
    body = Text()
    width = max(len(s.label) for s in steps)
    for i, step in enumerate(steps):
        if i:
            body.append("\n")
        body.append_text(format_step(step, theme, width))
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
