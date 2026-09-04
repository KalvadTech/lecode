"""Skills system: discoverable ``SKILL.md`` packs (kon lineage).

Packs live in ``.agents/skills/<name>/SKILL.md`` (project, nearest from the
cwd up to the git root) and ``~/.agents/skills/<name>/SKILL.md`` (global;
overridable with ``LECODE_SKILLS_DIR`` for tests). Project wins on name
collisions. Frontmatter: ``name`` (defaults to the directory name),
``description`` (required — missing skips the pack with a warning),
``register_cmd`` (bool, expose as a slash command), ``cmd_info`` (usage help
for that command). Everything is fail-open: unreadable directories or invalid
packs collect warnings, never crash.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from lecode.context.agents_md import find_git_root
from lecode.context.frontmatter import split_frontmatter

SKILL_FILENAME = "SKILL.md"

#: Environment variable overriding the global skills directory.
GLOBAL_SKILLS_ENV_VAR = "LECODE_SKILLS_DIR"

#: What a skill name must look like to be registerable as a slash command.
COMMAND_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str  # the skill's instruction markdown
    path: Path
    register_cmd: bool = False
    cmd_info: str | None = None


def global_skills_dir() -> Path:
    """The global skills directory (``LECODE_SKILLS_DIR`` aware)."""
    env = os.environ.get(GLOBAL_SKILLS_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".agents" / "skills"


def project_skills_dir(cwd: Path | None = None) -> Path | None:
    """Nearest ``.agents/skills/`` from ``cwd`` up to the git root."""
    start = (cwd or Path.cwd()).resolve()
    stop = find_git_root(start) or start
    current = start
    while True:
        candidate = current / ".agents" / "skills"
        if candidate.is_dir():
            return candidate
        if current == stop:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _load_pack(skill_file: Path, warnings: list[str]) -> Skill | None:
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError as e:
        warnings.append(f"{skill_file}: unreadable: {e}")
        return None
    try:
        frontmatter, body = split_frontmatter(text)
    except ValueError as e:
        warnings.append(f"{skill_file}: {e}")
        return None
    description = str(frontmatter.get("description") or "").strip()
    if not description:
        warnings.append(f"{skill_file}: missing required 'description' in frontmatter")
        return None
    name = str(frontmatter.get("name") or skill_file.parent.name).strip()
    cmd_info = frontmatter.get("cmd_info")
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        path=skill_file,
        register_cmd=bool(frontmatter.get("register_cmd", False)),
        cmd_info=str(cmd_info) if cmd_info else None,
    )


class SkillRegistry:
    """The discovered skills plus load warnings."""

    def __init__(
        self, skills: dict[str, Skill] | None = None, warnings: list[str] | None = None
    ) -> None:
        self._skills = dict(skills or {})
        self.warnings = list(warnings or [])

    def __len__(self) -> int:
        return len(self._skills)

    def list(self) -> list[Skill]:
        """All skills, sorted by name."""
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def render_listing(self) -> str:
        """Compact markdown list for the system prompt's ``extra`` section."""
        if not self._skills:
            return ""
        lines = ["## Available skills", ""]
        for skill in self.list():
            lines.append(f"- `{skill.name}` — {skill.description}")
        return "\n".join(lines)

    def render_skill(self, name: str) -> str | None:
        """The full instruction body of one skill, for on-demand inclusion."""
        skill = self._skills.get(name)
        return skill.body if skill is not None else None


def load_skills(cwd: Path | None = None, global_dir: Path | None = None) -> SkillRegistry:
    """Discover skills in the global dir and the project dir (project wins)."""
    warnings: list[str] = []
    skills: dict[str, Skill] = {}
    for root in (global_dir or global_skills_dir(), project_skills_dir(cwd)):
        if root is None or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            skill_file = child / SKILL_FILENAME
            if not child.is_dir() or not skill_file.is_file():
                continue
            skill = _load_pack(skill_file, warnings)
            if skill is not None:
                skills[skill.name] = skill
    return SkillRegistry(skills, warnings)


def skill_commands(registry: SkillRegistry) -> dict[str, dict[str, str]]:
    """Slash-command mapping for ``register_cmd: true`` skills (Phase 9 seam).

    Keys are command names (validated slugs); values carry the command name,
    its help text (``cmd_info`` falling back to the description), and the
    skill body. Invalid slugs are skipped with a warning.
    """
    commands: dict[str, dict[str, str]] = {}
    for skill in registry.list():
        if not skill.register_cmd:
            continue
        if not COMMAND_SLUG.fullmatch(skill.name):
            registry.warnings.append(
                f"skill '{skill.name}': register_cmd ignored, not a valid command slug"
            )
            continue
        commands[skill.name] = {
            "name": skill.name,
            "description": skill.cmd_info or skill.description,
            "body": skill.body,
        }
    return commands
