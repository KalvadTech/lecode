"""Configuration loading: global + project files, merging, migrations.

Global config lives in ``~/.config/lecode/`` (overridable with the
``LECODE_CONFIG_DIR`` environment variable). TOML is preferred; YAML and
JSON are accepted when no TOML file exists. A project-local
``.lecode/config.toml`` (nearest, found by walking from the cwd up to the
git root) is deep-merged over the global config: dicts merge recursively,
scalars and lists replace.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from lecode.config.migrations import migrate_config
from lecode.config.models import LEGACY_PERMISSION_MODES, Config

#: Environment variable overriding the global config directory.
CONFIG_ENV_VAR = "LECODE_CONFIG_DIR"

#: Candidate config file names, in preference order.
CONFIG_BASENAMES = ("config.toml", "config.yaml", "config.yml", "config.json")

_DEFAULT_CONFIG = """\
# lecode configuration — see `lecode --help` and the docs for all options.
# TOML is preferred; config.yaml / config.json are also accepted.
schema_version = 1

# [llm]
# provider = "openrouter"
# model = "deepseek/deepseek-v4-flash"
# api_key = "sk-or-..."            # or use the OPENROUTER_API_KEY env var
# thinking = "medium"              # none | low | medium | high
# auth_policy = "auto"             # auto | required | none

# [ui]
# theme = "default"
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
    """Return the preferred config file in ``directory``, if any exists."""
    for name in CONFIG_BASENAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def write_default_config(path: Path) -> None:
    """Write the commented default config, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_CONFIG, encoding="utf-8")


def _load_raw(path: Path) -> dict[str, Any]:
    """Parse a config file into a plain dict, by file suffix."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        data = tomllib.loads(text) if text.strip() else {}
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text) if text.strip() else {}
    elif suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:  # pragma: no cover - unreachable via find_config_file
        raise ValueError(f"unsupported config file format: {path}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping at top level: {path}")
    return data


def _find_git_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the nearest directory containing ``.git``."""
    current = start.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def find_project_config(start: Path) -> Path | None:
    """Return the nearest ``.lecode/config.*`` from ``start`` up to the git root."""
    root = _find_git_root(start)
    current = start.resolve()
    stop = root if root is not None else current
    while True:
        found = find_config_file(current / ".lecode")
        if found is not None:
            return found
        if current == stop:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


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
    """Rewrite a config file (after migration) in its original format."""
    suffix = path.suffix.lower()
    if suffix == ".toml":
        text = _dump_toml(raw)
    elif suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(raw, sort_keys=False)
    else:
        text = json.dumps(raw, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


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
        global_file = config_dir() / "config.toml"
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
