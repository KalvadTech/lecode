"""Tests for config models, loading, merging, and migrations."""

from __future__ import annotations

import json

import pytest

from lecode.config.loader import deep_merge, load_config
from lecode.config.migrations import MIGRATIONS
from lecode.config.models import Config


@pytest.fixture
def global_dir(tmp_path, monkeypatch):
    """Redirect the global config dir into tmp_path."""
    d = tmp_path / "global"
    d.mkdir()
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(d))
    return d


@pytest.fixture
def project(tmp_path):
    """A git-rooted project dir with a subdirectory as the cwd."""
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    cwd = root / "sub"
    cwd.mkdir()
    return root, cwd


def test_defaults_validate_from_empty():
    config = Config.model_validate({})
    assert config.schema_version == 1
    assert config.llm.provider == "openrouter"
    assert config.llm.thinking == "medium"
    assert config.llm.auth_policy == "auto"
    assert config.llm.tls_verify is True
    assert config.llm.system_prompt.style == "minimal"
    assert config.compaction.enabled is True
    assert config.compaction.buffer_tokens == 20000
    assert config.agent.max_turns == 500
    assert config.tools.enabled == {}
    assert config.permissions.mode == "yolo"
    assert config.notifications.volume == 0.5
    assert config.mcp.enable_exa is True
    assert config.mcp.enable_context7 is False
    assert config.memory.max_bytes == 32768
    assert config.advisor.enabled is False
    assert config.advisor.mode == "model"


def test_first_run_creates_default_config(global_dir, tmp_path):
    result = load_config(cwd=tmp_path)
    created = global_dir / "config.toml"
    assert created.is_file()
    assert result.sources == [created]
    assert "schema_version" in created.read_text()
    assert result.config.llm.provider == "openrouter"


def test_load_toml(global_dir, tmp_path):
    (global_dir / "config.toml").write_text(
        'schema_version = 1\n[llm]\nmodel = "anthropic/claude-sonnet-4"\n'
    )
    result = load_config(cwd=tmp_path)
    assert result.config.llm.model == "anthropic/claude-sonnet-4"
    assert result.warnings == []


@pytest.mark.parametrize("legacy", ["standard", "restrictive", "planwrite", "guarded"])
def test_legacy_permission_mode_coerced_to_yolo(global_dir, tmp_path, legacy):
    (global_dir / "config.toml").write_text(
        f'schema_version = 1\n[permissions]\nmode = "{legacy}"\n'
    )
    result = load_config(cwd=tmp_path)
    assert result.config.permissions.mode == "yolo"
    assert any(f"'{legacy}' is deprecated" in w for w in result.warnings)


def test_load_yaml_when_no_toml(global_dir, tmp_path):
    (global_dir / "config.yaml").write_text("llm:\n  model: openai/gpt-4o\n")
    result = load_config(cwd=tmp_path)
    assert result.config.llm.model == "openai/gpt-4o"
    assert result.sources == [global_dir / "config.yaml"]


def test_load_json_when_no_toml_or_yaml(global_dir, tmp_path):
    (global_dir / "config.json").write_text(json.dumps({"ui": {"no_color": True}}))
    result = load_config(cwd=tmp_path)
    assert result.config.ui.no_color is True


def test_toml_preferred_over_yaml(global_dir, tmp_path):
    (global_dir / "config.toml").write_text('[llm]\nmodel = "a"\n')
    (global_dir / "config.yaml").write_text("llm:\n  model: b\n")
    result = load_config(cwd=tmp_path)
    assert result.config.llm.model == "a"


def test_project_config_merges_over_global(global_dir, project):
    _root, cwd = project
    (global_dir / "config.toml").write_text(
        'schema_version = 1\n[llm]\nmodel = "global-model"\nthinking = "low"\n'
        '[ui]\nhidden_models = ["a"]\n'
    )
    (cwd / ".lecode").mkdir()
    (cwd / ".lecode" / "config.toml").write_text(
        '[llm]\nmodel = "project-model"\n[ui]\nhidden_models = ["b"]\n'
    )
    result = load_config(cwd=cwd)
    # Scalar override wins, untouched sibling key survives the merge.
    assert result.config.llm.model == "project-model"
    assert result.config.llm.thinking == "low"
    # Lists replace, they do not concatenate.
    assert result.config.ui.hidden_models == ["b"]
    assert result.sources == [global_dir / "config.toml", cwd / ".lecode" / "config.toml"]


def test_nearest_project_config_wins(global_dir, project):
    root, cwd = project
    (root / ".lecode").mkdir()
    (root / ".lecode" / "config.toml").write_text('[llm]\nmodel = "root-model"\n')
    (cwd / ".lecode").mkdir()
    (cwd / ".lecode" / "config.toml").write_text('[llm]\nmodel = "near-model"\n')
    result = load_config(cwd=cwd)
    assert result.config.llm.model == "near-model"
    assert result.sources == [global_dir / "config.toml", cwd / ".lecode" / "config.toml"]


def test_project_config_outside_git_root_ignored(global_dir, tmp_path):
    """Only dirs from cwd up to the git root are considered."""
    cwd = tmp_path / "root" / "sub"
    (cwd / ".lecode").mkdir(parents=True)
    (cwd / ".lecode" / "config.toml").write_text('[llm]\nmodel = "near"\n')
    (tmp_path / ".lecode").mkdir()  # above the (absent) git root: never visited
    result = load_config(cwd=cwd)
    assert result.config.llm.model == "near"


def test_unknown_key_warnings(global_dir, tmp_path):
    (global_dir / "config.toml").write_text('bogus = 1\n[llm]\nbogus_nested = 2\nmodel = "x"\n')
    result = load_config(cwd=tmp_path)
    assert result.config.llm.model == "x"  # unknown keys are ignored, not fatal
    assert any("unknown config key: bogus" in w for w in result.warnings)
    assert any("unknown config key: llm.bogus_nested" in w for w in result.warnings)


def test_missing_schema_version_migrated_and_rewritten(global_dir, tmp_path):
    path = global_dir / "config.toml"
    path.write_text('[llm]\nmodel = "x"\n')
    result = load_config(cwd=tmp_path)
    assert result.config.schema_version == 1
    assert "schema_version = 1" in path.read_text()
    assert 'model = "x"' in path.read_text()


def test_migration_applied_from_registry(global_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("lecode.config.migrations.CURRENT_SCHEMA_VERSION", 2)

    def v1_to_v2(raw):
        raw = dict(raw)
        llm = dict(raw.get("llm", {}))
        if "model_name" in llm:
            llm["model"] = llm.pop("model_name")
        raw["llm"] = llm
        return raw

    monkeypatch.setitem(MIGRATIONS, 1, v1_to_v2)
    path = global_dir / "config.toml"
    path.write_text('schema_version = 1\n[llm]\nmodel_name = "legacy"\n')
    result = load_config(cwd=tmp_path)
    assert result.config.llm.model == "legacy"
    rewritten = path.read_text()
    assert "schema_version = 2" in rewritten
    assert 'model = "legacy"' in rewritten


def test_future_schema_version_warns_but_loads(global_dir, tmp_path):
    (global_dir / "config.toml").write_text('schema_version = 99\n[llm]\nmodel = "x"\n')
    result = load_config(cwd=tmp_path)
    assert result.config.llm.model == "x"
    assert any("newer" in w for w in result.warnings)


def test_deep_merge_semantics():
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2]}
    override = {"b": {"y": 3}, "c": [9], "d": True}
    assert deep_merge(base, override) == {
        "a": 1,
        "b": {"x": 1, "y": 3},
        "c": [9],
        "d": True,
    }
