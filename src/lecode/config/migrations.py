"""Schema-version migrations for config files.

``MIGRATIONS`` maps a ``schema_version`` to a function that transforms a raw
config mapping from that version to the next. Migrations are applied in
order until the mapping reaches :data:`CURRENT_SCHEMA_VERSION`, after which
the loader rewrites the file on disk.
"""

from __future__ import annotations

from collections.abc import Callable

from lecode.config.models import CURRENT_SCHEMA_VERSION

#: from_version -> migrator producing from_version + 1. Empty at v1.
MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def migrate_config(raw: dict, current: int | None = None) -> tuple[dict, list[str], bool]:
    """Migrate a raw config mapping forward to ``current``.

    Returns ``(migrated, warnings, changed)``. A file whose version is newer
    than ``current`` is returned unchanged with a warning.
    """
    if current is None:
        current = CURRENT_SCHEMA_VERSION
    version = raw.get("schema_version", 0)
    if not isinstance(version, int) or isinstance(version, bool):
        version = 0

    if version > current:
        warning = (
            f"config schema_version {version} is newer than the supported "
            f"version {current}; proceeding anyway"
        )
        return raw, [warning], False

    changed = version != current
    while version < current:
        migrator = MIGRATIONS.get(version)
        if migrator is not None:
            raw = migrator(raw)
        version += 1

    if changed:
        raw = {**raw, "schema_version": current}
    return raw, [], changed
