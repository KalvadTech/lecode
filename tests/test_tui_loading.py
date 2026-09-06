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
from lecode.tui.themes import THEME


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    (tmp_path / ".git").mkdir()
    return tmp_path


def _make(env, config=None, *, name="demo", resumed=False, **report_kwargs):
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
        **report_kwargs,
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
        "pierre",
        "lsp",
        "mcp",
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


def test_mcp_step_live_statuses(env, monkeypatch):
    from lecode.extras.mcp_client import ServerStatus
    from lecode.tui.loading import WARN

    monkeypatch.setenv("EXA_API_KEY", "k")
    servers = [
        ServerStatus("context7", "connected", tools=2),
        ServerStatus("exa", "connected", tools=3),
        ServerStatus("mine", "failed", error="TimeoutError: connect"),
        ServerStatus("off", "disabled"),
    ]
    steps, *_ = _make(env, mcp_servers=servers)
    mcp = _step(steps, "mcp")
    assert mcp.status == WARN  # one server failed
    assert "exa: connected · 3 tools" in mcp.detail
    assert "context7: connected · 2 tools" in mcp.detail
    assert "mine: failed — TimeoutError: connect" in mcp.detail
    assert "off: disabled" in mcp.detail


def test_mcp_step_live_statuses_all_connected(env, monkeypatch):
    from lecode.extras.mcp_client import ServerStatus
    from lecode.tui.loading import OK

    monkeypatch.setenv("EXA_API_KEY", "k")
    steps, *_ = _make(env, mcp_servers=[ServerStatus("exa", "connected", tools=3)])
    mcp = _step(steps, "mcp")
    assert mcp.status == OK
    assert mcp.detail == "exa: connected · 3 tools"


def test_mcp_step_live_statuses_exa_key_missing(env, monkeypatch):
    from lecode.extras.mcp_client import ServerStatus
    from lecode.tui.loading import WARN

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    steps, *_ = _make(env, mcp_servers=[ServerStatus("mine", "connected", tools=1)])
    mcp = _step(steps, "mcp")
    assert mcp.status == WARN
    assert "mine: connected · 1 tools" in mcp.detail
    assert "exa: no EXA_API_KEY" in mcp.detail


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
    render_loading_screen(console, THEME, session_name="demo", steps=steps, cwd=env)
    text = out.getvalue()
    assert "lecode" in text and "demo" in text
    for label in ("config", "provider", "prompt", "tools"):
        assert label in text
    assert "✓" in text and "–" in text  # ok and skip marks  # noqa: RUF001


def test_render_prints_ascii_banner_and_byline(env):
    steps, _session, *_ = _make(env)
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, no_color=True, width=100)
    render_loading_screen(console, THEME, session_name="demo", steps=steps, cwd=env)
    text = out.getvalue()
    # figlet "standard" banner for "lecode", byline below it
    assert "| | ___  ___ ___   __| | ___" in text
    assert "|_|\\___|\\___\\___/ \\__,_|\\___|" in text
    assert "by wowi42" in text
    assert text.index("by wowi42") < text.index("session")  # banner above the panel


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
    """run_interactive prints the banner and step lines before the chat."""
    import lecode.cli as cli

    monkeypatch.setattr(cli, "prompt_session_name", _fake_name_prompt)
    monkeypatch.setattr(cli, "build_provider", lambda config, api_key=None: object())
    monkeypatch.setattr(cli, "TuiApp", _FakeTui)
    monkeypatch.setattr(cli, "_run_tui", _fake_run_tui)
    code = cli.run_interactive()
    assert code == 0
    out = capsys.readouterr().out
    # ASCII banner up front, then progressive step lines
    assert "| | ___  ___ ___   __| | ___" in out
    assert "by wowi42" in out
    assert "config" in out and "provider" in out and "session" in out
    assert out.index("| | ___") < out.index("provider")  # banner before the steps


async def _fake_name_prompt(store):
    return "loading-test"


class _FakeTui:
    def __init__(self, *args, **kwargs):
        pass


async def _fake_run_tui(tui, client, background=None):
    return 0


def test_catalog_fetch_runs_in_background(env, monkeypatch):
    """The chat opens before the model-catalog fetch finishes; the fetched
    catalog is bound late via set_catalog."""
    import lecode.cli as cli
    from lecode.providers.catalog import Catalog
    from lecode.providers.live import LoadedCatalog

    events: list[str] = []
    monkeypatch.setattr(cli, "prompt_session_name", _fake_name_prompt)
    monkeypatch.setattr(cli, "build_provider", lambda config, api_key=None: object())

    class FakeTui:
        def __init__(self, *args, **kwargs):
            events.append("tui-created")
            self.catalog = kwargs.get("catalog")

        def set_catalog(self, catalog, *, origin, count):
            events.append(f"catalog-bound:{origin}:{count}")

    async def fake_run_tui(tui, client, background=None):
        events.append("chat-open")
        assert tui.catalog is not None and not tui.catalog._entries  # empty default
        await background()
        return 0

    def slow_fetch(config, api_key=None):
        events.append("fetch-done")
        return LoadedCatalog(Catalog.default(), "live", 427)

    monkeypatch.setattr(cli, "TuiApp", FakeTui)
    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)
    monkeypatch.setattr(cli, "fetch_catalog", slow_fetch)
    assert cli.run_interactive() == 0
    assert events == ["tui-created", "chat-open", "fetch-done", "catalog-bound:live:427"]


def _step(steps, label):
    return next(s for s in steps if s.label == label)


def test_load_config_used_by_report(env):
    """The report consumes a real LoadedConfig from the loader."""
    loaded = load_config()  # auto-creates the default file
    assert loaded.sources, "first run should create and list the default config"


# -- progressive rendering ------------------------------------------------------


def _record_console():
    out = io.StringIO()
    return Console(file=out, force_terminal=False, no_color=True, width=100), out


def test_progressive_banner_prints_immediately():
    from lecode.tui.loading import LoadingProgress

    console, out = _record_console()
    LoadingProgress(console).banner()
    text = out.getvalue()
    assert "| | ___  ___ ___   __| | ___" in text
    assert "by wowi42" in text


def test_progressive_steps_append_in_order():
    from lecode.tui.loading import LoadingProgress, LoadStep

    console, out = _record_console()
    progress = LoadingProgress(console)
    progress.banner()
    progress.step(LoadStep("config", "defaults (no config file)"))
    progress.pending("models", "fetching live from openrouter…")
    progress.step(LoadStep("models", "427 fetched live from the provider"))
    text = out.getvalue()
    assert "✓ config" in text
    assert "… models" in text  # pending line shown while the fetch runs
    assert text.index("… models") < text.index("427 fetched")
    # banner comes before any step
    assert text.index("| | ___") < text.index("✓ config")


def test_progressive_step_multiline_detail_indented():
    from lecode.tui.loading import LoadingProgress, LoadStep

    console, out = _record_console()
    LoadingProgress(console).step(LoadStep("provider", "openrouter · url\nmodel gpt · no key"))
    lines = out.getvalue().splitlines()
    assert lines[0].startswith(" ✓ provider")
    assert lines[1].startswith(" " * 4) and "model gpt" in lines[1]


def test_progressive_never_raises(monkeypatch):
    import lecode.tui.loading as loading
    from lecode.tui.loading import LoadingProgress, LoadStep

    console, _ = _record_console()
    progress = LoadingProgress(console)
    monkeypatch.setattr(
        loading, "print_banner", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    progress.banner()  # no exception
    monkeypatch.setattr(
        loading, "print_step", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    progress.step(LoadStep("config", "x"))  # no exception
    progress.step(None)  # nothing to report: fine
