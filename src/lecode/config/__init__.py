"""Configuration system: pydantic models, loader, and schema migrations."""

from __future__ import annotations

from lecode.config.loader import LoadedConfig, config_dir, load_config
from lecode.config.models import Config

__all__ = ["Config", "LoadedConfig", "config_dir", "load_config"]
