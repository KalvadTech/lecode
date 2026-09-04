"""System prompt assembly.

The base prompt is the embedded ``prompts/minimal.md`` (default) or
``prompts/rich.md`` (``llm.system_prompt.style = "rich"``), optionally layered
with a named persona snippet. ``llm.system_prompt.custom`` replaces the base
entirely. Project context (the AGENTS.md walk) is appended after the base,
then the memory and skills seams (Phases 7 and 6).
"""

from __future__ import annotations

from pathlib import Path

from lecode.config.models import Config
from lecode.context import agents_md, resources


def _base_prompt(config: Config, cwd: Path) -> str:
    prompt_cfg = config.llm.system_prompt
    if prompt_cfg.custom is not None:
        return prompt_cfg.custom
    base = resources.load_text("prompts", f"{prompt_cfg.style}.md", cwd)
    if prompt_cfg.style == "rich" and prompt_cfg.persona:
        persona = resources.load_text("prompts", f"personas/{prompt_cfg.persona}.md", cwd)
        base = base.rstrip() + "\n\n" + persona.strip()
    return base


def build_system_prompt(
    config: Config,
    cwd: Path,
    *,
    memory_text: str | None = None,
    extra: str | None = None,
) -> str:
    """Assemble the system prompt: base + AGENTS.md walk + memory + extras."""
    sections = [_base_prompt(config, cwd).strip()]

    context = agents_md.render(agents_md.collect(cwd))
    if context:
        sections.append(context)

    if memory_text:  # Phase 7 seam: persistent memory injection
        sections.append(memory_text.strip())
    if extra:  # Phase 6 seam: skills listing
        sections.append(extra.strip())

    return "\n\n".join(sections)
