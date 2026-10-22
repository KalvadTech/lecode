"""Configuration loading: global + project files, merging, migrations.

Global config lives in ``~/.config/lecode/`` (overridable with the
``LECODE_CONFIG_DIR`` environment variable). The only accepted format is
TOML (``config.toml``). A project-local ``.lecode/config.toml`` (nearest,
found by walking from the cwd up to the git root) is deep-merged over the
global config: dicts merge recursively, scalars and lists replace.

Setting ``LECODE_NO_PROJECT_CONFIG`` (or passing ``--no-project-config``)
disables every repository-supplied layer: the project config, project agents,
project skills, and project prompt overrides. Use it when lecode runs on a
repository it should not trust, such as a pull request under review or a CI
checkout.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lecode.config.migrations import migrate_config
from lecode.config.models import LEGACY_PERMISSION_MODES, Config
from lecode.context.agents_md import find_git_root

#: Environment variable overriding the global config directory.
CONFIG_ENV_VAR = "LECODE_CONFIG_DIR"

#: Environment variable that, when truthy, ignores every repository-supplied layer.
NO_PROJECT_CONFIG_ENV_VAR = "LECODE_NO_PROJECT_CONFIG"

_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})

#: The only accepted config file name (TOML only).
CONFIG_FILENAME = "config.toml"

_DEFAULT_CONFIG = """\
# lecode configuration — see `lecode --help` and the docs for all options.
schema_version = 1

# [llm]
# provider = "openrouter"
# model = "deepseek/deepseek-v4-flash"
# api_key = "sk-or-..."            # or use the OPENROUTER_API_KEY env var
# thinking = "medium"              # none | low | medium | high
# auth_policy = "auto"             # auto | required | none
"""


@dataclass
class LoadedConfig:
    """Result of :func:`load_config`.

    ``sources`` lists every file that contributed, lowest precedence first;
    CLI overrides (a later phase) apply on top of ``config``.
    """

    config: Config
    warnings: list[str] = field(default_factory=list)
    sources: list[Path] = field(default_factory=list)


def config_dir() -> Path:
    """Return the global config directory (``LECODE_CONFIG_DIR`` aware)."""
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "lecode"


def find_config_file(directory: Path) -> Path | None:
    """Return ``directory/config.toml`` when it exists."""
    candidate = directory / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def write_default_config(path: Path) -> None:
    """Write the commented default config, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_CONFIG, encoding="utf-8")


def _load_raw(path: Path) -> dict[str, Any]:
    """Parse a TOML config file into a plain dict."""
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping at top level: {path}")
    return data


def project_config_enabled() -> bool:
    """Whether repository-supplied config layers may load (``LECODE_NO_PROJECT_CONFIG``)."""
    return os.environ.get(NO_PROJECT_CONFIG_ENV_VAR, "").strip().lower() in _FALSE_VALUES


def lsp_off_reason(config: Config, *, read_only: bool) -> str | None:
    """Why no language server runs, or ``None`` when one may start.

    A language server can run repository code (build scripts, workspace
    interpreters), so ``--safe`` and ``--no-project-config`` keep it off.
    """
    if not config.lsp.enabled:
        return "disabled"
    if read_only:
        return "off under --safe"
    if not project_config_enabled():
        return "off under --no-project-config"
    return None


def find_project_path(
    start: Path, relative: str, accept: Callable[[Path], bool] = Path.is_dir
) -> Path | None:
    """Nearest ``relative`` path that ``accept`` admits, from ``start`` up to the git root.

    Always ``None`` while project config is disabled, so every repository
    layer goes through this one gate.
    """
    if not project_config_enabled():
        return None
    root = find_git_root(start)
    current = start.resolve()
    stop = root if root is not None else current
    while True:
        candidate = current / relative
        if accept(candidate):
            return candidate
        if current == stop:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def find_project_config(start: Path) -> Path | None:
    """Return the nearest ``.lecode/config.toml`` from ``start`` up to the git root."""
    return find_project_path(start, f".lecode/{CONFIG_FILENAME}", Path.is_file)


def deep_merge(base: Any, override: Any) -> Any:
    """Merge ``override`` over ``base``: dicts recursively, everything else replaces."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(base[key], value) if key in base else value
        return merged
    return override


def _collect_unknown_keys(
    raw: dict[str, Any], model_cls: type[BaseModel], prefix: str, warnings: list[str]
) -> None:
    """Append a warning for every raw key with no matching model field."""
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else key
        model_field = model_cls.model_fields.get(key)
        if model_field is None:
            warnings.append(f"unknown config key: {path}")
            continue
        annotation = model_field.annotation
        if (
            isinstance(value, dict)
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            _collect_unknown_keys(value, annotation, path, warnings)


def _dump_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_dump_toml_value(v) for v in value) + "]"
    raise ValueError(f"cannot serialize value to TOML: {value!r}")


def _dump_toml(data: dict[str, Any]) -> str:
    """Serialize a (possibly nested) config dict as TOML.

    Minimal writer covering the types a config file can contain: nested
    tables, scalars, and lists of scalars.
    """
    lines: list[str] = []

    def emit_table(table: dict[str, Any], prefix: str) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        sub_tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        for key, value in scalars.items():
            lines.append(f"{key} = {_dump_toml_value(value)}")
        for key, value in sub_tables.items():
            header = f"{prefix}.{key}" if prefix else key
            lines.append("")
            lines.append(f"[{header}]")
            emit_table(value, header)

    emit_table(data, "")
    return "\n".join(lines).strip() + "\n"


def _write_raw(path: Path, raw: dict[str, Any]) -> None:
    """Rewrite a config file (after migration) as TOML."""
    path.write_text(_dump_toml(raw), encoding="utf-8")


def load_config(cwd: Path | None = None) -> LoadedConfig:
    """Load, migrate, and merge the global and project configs.

    The global config file is auto-created on first run. Files with an older
    ``schema_version`` are migrated forward and rewritten in place.
    """
    start = (cwd or Path.cwd()).resolve()
    warnings: list[str] = []
    sources: list[Path] = []

    global_file = find_config_file(config_dir())
    if global_file is None:
        global_file = config_dir() / CONFIG_FILENAME
        write_default_config(global_file)
    sources.append(global_file)

    raw_global = _load_raw(global_file)
    raw_global, migration_warnings, changed = migrate_config(raw_global)
    warnings.extend(f"{global_file}: {w}" for w in migration_warnings)
    if changed:
        _write_raw(global_file, raw_global)

    merged = raw_global

    project_file = find_project_config(start)
    if project_file is not None:
        raw_project = _load_raw(project_file)
        raw_project, migration_warnings, changed = migrate_config(raw_project)
        warnings.extend(f"{project_file}: {w}" for w in migration_warnings)
        if changed:
            _write_raw(project_file, raw_project)
        merged = deep_merge(merged, raw_project)
        sources.append(project_file)

    _collect_unknown_keys(merged, Config, "", warnings)
    raw_perms = merged.get("permissions")
    legacy_mode = raw_perms.get("mode") if isinstance(raw_perms, dict) else None
    if legacy_mode in LEGACY_PERMISSION_MODES:
        warnings.append(f"permissions.mode: '{legacy_mode}' is deprecated; coerced to 'yolo'")
    config = Config.model_validate(merged)
    return LoadedConfig(config=config, warnings=warnings, sources=sources)
