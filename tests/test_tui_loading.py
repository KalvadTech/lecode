"""Tests for the startup loading screen (tui/loading.py)."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from lecode.agent.builder import build_runtime
from lecode.config.loader import LoadedConfig, config_dir, load_config
from lecode.config.models import Config
from lecode.providers import ProviderSpec
from lecode.session.storage import SessionStore
from lecode.tui.loading import (
    SKIP,
    WARN,
    build_load_report,
    render_loading_screen,
    show_loading_screen,
)
from lecode.tui.themes import load_theme


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    (tmp_path / ".git").mkdir()
    return tmp_path


def _make(env, config=None, *, name="demo", resumed=False):
    config = config or Config()
    store = SessionStore()
    session = store.create(name, env, model=config.llm.model)
    runtime = build_runtime(config, env, session=session, store=store)
    loaded = LoadedConfig(config=config, warnings=[], sources=[])
    spec = ProviderSpec(
        name="openrouter", base_url="https://openrouter.ai/api/v1", model=config.llm.model
    )
    steps = build_load_report(
        config=config,
        loaded=loaded,
        runtime=runtime,
        session=session,
        store=store,
        cwd=env,
        resumed=resumed,
        provider_spec=spec,
        key_source="none",
    )
    return steps, session, store, runtime, loaded, spec


def _labels(steps):
    return [s.label for s in steps]


def test_report_covers_all_subsystems(env):
    steps, *_ = _make(env)
    assert _labels(steps) == [
        "session",
        "config",
        "provider",
        "prompt",
        "context",
        "skills",
        "agents",
        "memory",
        "tools",
        "permissions",
        "hooks",
        "lsp",
        "mcp",
        "theme",
    ]


def test_session_step_new_and_resumed(env):
    steps, session, _store, runtime, loaded, spec = _make(env, name="fresh")
    assert steps[0].detail == "fresh — new session"
    # resumed shows the message count
    store2 = SessionStore()
    session2 = store2.open(session.meta.id)
    steps2 = build_load_report(
        config=Config(),
        loaded=loaded,
        runtime=runtime,
        session=session2,
        store=store2,
        cwd=env,
        resumed=True,
        provider_spec=spec,
        key_source="env",
    )
    assert "resumed" in steps2[0].detail


def test_config_step_sources_and_warnings(env, tmp_path):
    steps, _, _, _, _loaded, spec = _make(env)
    cfg = _step(steps, "config")
    assert cfg.detail == "defaults (no config file)"
    loaded_with = LoadedConfig(
        config=Config(),
        warnings=["unknown config key: bogus"],
        sources=[config_dir() / "config.toml", env / ".lecode" / "config.toml"],
    )
    steps2, _, _, _, _, _ = _make(env)
    steps2 = build_load_report(
        config=loaded_with.config,
        loaded=loaded_with,
        runtime=build_runtime(Config(), env),
        session=SessionStore().create("s2", env),
        store=SessionStore(),
        cwd=env,
        resumed=False,
        provider_spec=spec,
        key_source="env",
    )
    cfg2 = _step(steps2, "config")
    assert cfg2.status == WARN
    assert "1 warning(s)" in cfg2.detail
    assert ".lecode/config.toml" in cfg2.detail  # project path shown relative


def test_provider_step_shows_key_source_not_key(env, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-value")
    config = Config()
    store = SessionStore()
    session = store.create("demo", env)
    runtime = build_runtime(config, env, session=session, store=store)
    from lecode.auth import resolve_api_key

    spec = ProviderSpec(
        name="openrouter", base_url="https://openrouter.ai/api/v1", model=config.llm.model
    )
    key = resolve_api_key(spec.name, config)
    steps = build_load_report(
        config=config,
        loaded=LoadedConfig(config=config),
        runtime=runtime,
        session=session,
        store=store,
        cwd=env,
        resumed=False,
        provider_spec=spec,
        key_source=key.source,
    )
    provider = _step(steps, "provider")
    assert "key from env" in provider.detail
    assert "sk-secret" not in provider.detail


def test_context_step_lists_agents_md(env):
    (env / "AGENTS.md").write_text("root instructions\n")
    sub = env / "pkg"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("pkg instructions\n")
    steps, *_ = _make(env)
    ctx = _step(steps, "context")
    assert "AGENTS.md" in ctx.detail
    # a file *below* cwd is not on the git-root→cwd walk; from pkg it is
    steps, *_ = _make(sub)
    assert _step(steps, "context").status != SKIP
    # without any files it is a skip
    (env / "AGENTS.md").unlink()
    (sub / "AGENTS.md").unlink()
    steps, *_ = _make(env)
    assert _step(steps, "context").status == SKIP


def test_skills_and_agents_and_hooks_steps(env):
    skills_dir = env / ".agents" / "skills" / "lint-it"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\ndescription: Lint things\n---\nDo the linting.\n")
    agents_dir = env / ".lecode" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "researcher.md").write_text("---\ndescription: Research\n---\nResearch things.\n")
    config = Config(hooks={"PreToolUse": ["echo allow"], "Stop": ["echo ok", "echo done"]})
    steps, *_ = _make(env, config)
    assert _step(steps, "skills").detail == "lint-it"
    agents = _step(steps, "agents")
    assert "primaries: build, plan" in agents.detail
    assert "researcher" in agents.detail
    hooks = _step(steps, "hooks")
    assert "PreToolUse(1)" in hooks.detail and "Stop(2)" in hooks.detail


def test_memory_step_states(env):
    steps, *_ = _make(env)
    assert _step(steps, "memory").status == SKIP  # enabled but empty
    from lecode.memory import memory_root

    root = memory_root(env)
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text("# Facts\n- important\n")
    steps, *_ = _make(env)
    assert "KB injected" in _step(steps, "memory").detail
    steps, *_ = _make(env, Config(memory={"enabled": False}))
    assert _step(steps, "memory").detail == "disabled"


def test_mcp_step_states(env, monkeypatch):
    steps, *_ = _make(env)
    # exa on by default but no key
    assert _step(steps, "mcp").detail == "exa (no EXA_API_KEY)"
    monkeypatch.setenv("EXA_API_KEY", "k")
    config = Config(
        mcp={"enable_context7": True, "servers": {"mine": {"transport": "stdio", "command": "x"}}}
    )
    steps, *_ = _make(env, config)
    mcp = _step(steps, "mcp")
    assert "exa" in mcp.detail and "context7" in mcp.detail and "mine" in mcp.detail


def test_permissions_step(env):
    steps, *_ = _make(
        env,
        Config(permissions={"mode": "readonly"}),
    )
    assert "mode readonly" in _step(steps, "permissions").detail
    steps, *_ = _make(env)
    perm = _step(steps, "permissions")
    assert perm.detail == "mode yolo"


def test_render_outputs_panel_with_all_labels(env):
    steps, _session, *_ = _make(env)
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, no_color=True, width=100)
    theme = load_theme("default", Config(), env)
    render_loading_screen(console, theme, session_name="demo", steps=steps, cwd=env)
    text = out.getvalue()
    assert "lecode" in text and "demo" in text
    for label in ("config", "provider", "prompt", "tools", "theme"):
        assert label in text
    assert "✓" in text and "–" in text  # ok and skip marks  # noqa: RUF001


def test_show_loading_screen_never_raises(env, monkeypatch):
    # Break a dependency mid-report: memory_root blowing up must not propagate.
    import lecode.tui.loading as loading

    monkeypatch.setattr(
        loading, "build_load_report", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    console = Console(file=io.StringIO(), no_color=True)
    store = SessionStore()
    session = store.create("demo", env)
    show_loading_screen(
        config=Config(),
        loaded=LoadedConfig(config=Config()),
        runtime=build_runtime(Config(), env),
        session=session,
        store=store,
        cwd=env,
        resumed=False,
        provider_spec=ProviderSpec(name="x", base_url="http://x", model="m"),
        key_source="none",
        console=console,
    )  # no exception


def test_interactive_startup_prints_loading_screen(env, monkeypatch, capsys):
    """run_interactive shows the panel between the name prompt and the chat."""
    import lecode.cli as cli

    monkeypatch.setattr(cli, "prompt_session_name", _fake_name_prompt)
    monkeypatch.setattr(cli, "build_provider", lambda config, api_key=None: object())
    monkeypatch.setattr(cli, "TuiApp", _FakeTui)
    monkeypatch.setattr(cli, "_run_tui", _fake_run_tui)
    code = cli.run_interactive()
    assert code == 0
    out = capsys.readouterr().out
    assert "lecode" in out and "provider" in out and "theme" in out


async def _fake_name_prompt(store):
    return "loading-test"


class _FakeTui:
    def __init__(self, *args, **kwargs):
        pass


async def _fake_run_tui(tui, client):
    return 0


def _step(steps, label):
    return next(s for s in steps if s.label == label)


def test_load_config_used_by_report(env):
    """The report consumes a real LoadedConfig from the loader."""
    loaded = load_config()  # auto-creates the default file
    assert loaded.sources, "first run should create and list the default config"
