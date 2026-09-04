"""Assembly of the agent runtime: permission checker, tool registry, prompt.

Pure wiring — no network here; the CLI builds the provider client itself.
Agents and skills (Phase 6) are discovered here too: the selected agent's
permission overlay is layered onto the checker and its prompt body is prepended
to the system prompt, ahead of the skills listing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lecode.agent.prompts import build_system_prompt
from lecode.agent.tools import ToolContext, ToolRegistry, core_tools
from lecode.config.models import Config, PermissionMode
from lecode.context.agents import AgentRegistry, load_agents
from lecode.context.skills import SkillRegistry, load_skills
from lecode.hooks import HookDispatcher, apply_hooks, dispatcher_from_config
from lecode.memory import MemoryStore, memory_injection, memory_root, memory_tools
from lecode.permission import PermissionChecker, SessionPermissions

if TYPE_CHECKING:
    from lecode.providers.catalog import Catalog
    from lecode.session.storage import Session, SessionStore


@dataclass
class Runtime:
    """Everything the agent loop needs, minus the provider."""

    registry: ToolRegistry
    ctx: ToolContext
    system_prompt: str
    agents: AgentRegistry = field(default_factory=AgentRegistry)
    skills: SkillRegistry = field(default_factory=SkillRegistry)
    hooks: HookDispatcher | None = None
    warnings: list[str] = field(default_factory=list)


def build_runtime(
    config: Config,
    cwd: Path,
    *,
    session: Session | None = None,
    store: SessionStore | None = None,
    auto_approve: bool = False,
    mode: PermissionMode | None = None,
    allowed_tools: list[str] | None = None,
    agent_name: str | None = "build",
    agent_registry: AgentRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
    catalog: Catalog | None = None,
) -> Runtime:
    """Build the permission checker, tool context, registry, and system prompt.

    ``mode`` overrides the configured permission mode (e.g. ``--safe``);
    ``allowed_tools`` filters the core tool registry (``--allowed-tools``).
    ``agent_name`` selects a custom agent: its overlay narrows the permission
    checker and its prompt body is prepended to the system prompt (before the
    skills listing). Unknown agent names are ignored (fail-open).
    ``catalog`` is the live model catalog (modality checks in tools); ``None``
    leaves tools with an empty, fail-open catalog.
    """
    grants = store.load_grants(session) if session is not None and store is not None else None
    session_perms = SessionPermissions(grants)
    checker = PermissionChecker(config, session_perms=session_perms, mode=mode, cwd=cwd)
    warnings: list[str] = []

    agents = agent_registry or load_agents(cwd)
    skills = skill_registry or load_skills(cwd)
    warnings.extend(agents.warnings)
    warnings.extend(skills.warnings)
    agent = agents.get(agent_name) if agent_name else None
    if agent is not None and agent.overlay is not None:
        checker = checker.for_agent(agent.overlay)

    ctx = ToolContext(
        cwd=Path(cwd),
        config=config,
        permission_checker=checker,
        session=session,
        session_store=store,
        session_perms=session_perms,
        auto_approve=auto_approve,
        catalog=catalog,
    )

    tools = core_tools()
    # The advisor and task tools are always registered; a disabled advisor
    # reports how to enable, and task needs no configuration at all.
    from lecode.agent.tools import advisor as advisor_tool
    from lecode.agent.tools import task as task_tool

    tools.append(advisor_tool.make_tool())
    tools.append(task_tool.make_tool())
    if config.memory.enabled:
        memory_store = MemoryStore(memory_root(cwd), max_bytes=config.memory.max_bytes)
        ctx.extras["memory"] = memory_store
        tools += memory_tools()
    if config.lsp.enabled:
        from lecode.lsp.manager import LspManager
        from lecode.lsp.tool import LSP_EXTRA
        from lecode.lsp.tool import make_tool as lsp_tool

        ctx.extras[LSP_EXTRA] = LspManager(config, Path(cwd))
        tools.append(lsp_tool())
    if allowed_tools is not None:
        keep = set(allowed_tools)
        tools = [tool for tool in tools if tool.name in keep]
    registry = ToolRegistry(tools)

    # Lifecycle hooks decorate every tool at build time (they can only narrow).
    hooks, hook_warnings = dispatcher_from_config(config, cwd, session=session)
    warnings.extend(hook_warnings)
    if hooks is not None:
        apply_hooks(registry, hooks)

    # Subagent seams for the task tool and direct @agent dispatch.
    ctx.extras["agents"] = agents
    ctx.extras["registry"] = registry
    ctx.extras["hooks"] = hooks

    extra_parts: list[str] = []
    if agent is not None and agent.body:
        extra_parts.append(agent.body)
    listing = skills.render_listing()
    if listing:
        extra_parts.append(listing)
    system_prompt = build_system_prompt(
        config,
        cwd,
        memory_text=memory_injection(config, cwd),
        extra="\n\n".join(extra_parts) or None,
    )

    return Runtime(
        registry=registry,
        ctx=ctx,
        system_prompt=system_prompt,
        agents=agents,
        skills=skills,
        hooks=hooks,
        warnings=warnings,
    )
