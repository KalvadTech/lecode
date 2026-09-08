"""lecode command-line interface.

Scope: ``--version``, the startup dependency check (fd / rg / rtk), headless
mode (``-p/--prompt``: auto-approved tools, auto-named session, final
response on stdout, token/cost summary on stderr, exit codes 0 done /
1 error / 2 startup / 3 max turns), and the interactive TUI (default when
no ``-p`` is given): session-name prompt → session on disk → chat, with
``-r/--resume`` and ``-c/--continue`` reopening existing sessions.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any

import typer
from typer.core import TyperGroup, TyperOption

from lecode import __version__
from lecode.agent.builder import build_runtime
from lecode.agent.runner import AgentRunner, RunResult
from lecode.auth import AuthError, resolve_api_key
from lecode.config.loader import config_dir, find_config_file, load_config
from lecode.config.models import AuthPolicy, Config
from lecode.deps import find_missing_binaries, format_missing_error
from lecode.extras.background import BACKGROUND_EXTRA
from lecode.extras.chain import ChainResult, run_chain
from lecode.extras.loop_mode import (
    DEFAULT_MAX_ITERATIONS,
    LoopResult,
    loop_session_name,
    run_plan_loop,
)
from lecode.extras.status_signals import START, STOP, StatusEmitter
from lecode.extras.worktree import WorktreeError, WorktreeInfo, WorktreeManager
from lecode.hooks import (
    EVENTS,
    INTERRUPT,
    SESSION_END,
    SESSION_START,
    USER_PROMPT_SUBMIT,
    HookDispatcher,
    MergedVerdict,
    build_envelope,
    dispatch_event,
    dispatcher_from_config,
)
from lecode.providers import ProviderError, build_client, resolve_provider
from lecode.providers.catalog import Catalog
from lecode.providers.live import LoadedCatalog, load_catalog
from lecode.providers.openai_compat import ChatClient
from lecode.providers.types import ChatMessage
from lecode.session.naming import auto_name
from lecode.session.storage import (
    AmbiguousSessionError,
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
)
from lecode.setup_wizard import offer_first_run_setup, run_wizard
from lecode.telemetry import init_telemetry, shutdown_telemetry
from lecode.tui.app import TuiApp
from lecode.tui.name_prompt import prompt_session_name

#: Exit codes (headless mode uses the same taxonomy).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2
EXIT_MAX_TURNS = 3

#: Value produced when ``-p/--prompt`` is given without an argument: read stdin.
_STDIN_MARKER = ""

#: Value produced when ``-r/--resume`` is given without an argument: pick from
#: the current folder's sessions.
_PICK_MARKER = ""


class _OptionalValueOption(TyperOption):
    """Option whose value is optional: a bare flag yields ``self.marker``.

    The vendored click parser has no optional-value options, so the value
    lookup is made tolerant: when no argument (or another flag) follows, the
    option yields the marker instead of raising.
    """

    marker: str = ""

    def add_to_parser(self, parser: Any, ctx: Any) -> None:
        super().add_to_parser(parser, ctx)
        original = parser._get_value_from_state

        def tolerant_get_value(option_name: str, option: Any, state: Any) -> Any:
            if option.obj is self and (not state.rargs or state.rargs[0].startswith(("-",))):
                return self.marker
            return original(option_name, option, state)

        parser._get_value_from_state = tolerant_get_value


class _PromptOption(_OptionalValueOption):
    """``-p/--prompt``: a bare flag reads stdin."""

    marker = _STDIN_MARKER


class _ResumeOption(_OptionalValueOption):
    """``-r/--resume``: a bare flag opens the folder's session picker."""

    marker = _PICK_MARKER


class _LeCodeGroup(TyperGroup):
    """Swaps ``--prompt``/``--resume`` for the optional-value variants."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if param.name == "prompt":
                param.__class__ = _PromptOption
            elif param.name == "resume":
                param.__class__ = _ResumeOption


app = typer.Typer(
    name="lecode",
    help="A minimalist terminal AI coding agent.",
    add_completion=False,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"lecode {__version__}")
        raise typer.Exit(EXIT_OK)


def check_dependencies() -> None:
    """Abort startup (exit code 2) if fd/rg/rtk are missing from PATH."""
    missing = find_missing_binaries()
    if missing:
        typer.echo(format_missing_error(missing), err=True)
        raise typer.Exit(EXIT_STARTUP)


def build_provider(config: Config, api_key: str | None = None) -> ChatClient:
    """Resolve provider + API key from config, then build the streaming client.

    Kept as a separate function so tests can substitute a fake provider.
    """
    spec = resolve_provider(config)
    key = resolve_api_key(spec.name, config, cli_key=api_key)
    return build_client(spec, key)


def fetch_catalog(config: Config, api_key: str | None = None) -> LoadedCatalog:
    """Fetch the live model catalog (empty on failure); never raises.

    Uses a short-lived client on its own event loop: the session client's
    connection pool must stay on the loop that runs the session, or closing
    pooled connections later explodes with "Event loop is closed".
    """

    async def _fetch() -> LoadedCatalog:
        client = build_provider(config, api_key=api_key)
        try:
            return await load_catalog(client)
        finally:
            await _aclose(client)

    try:
        return asyncio.run(_fetch())
    except (AuthError, ValueError):  # same build errors the caller already handles
        return LoadedCatalog(Catalog.default(), "empty")


async def _run_headless(
    provider: Any, runner: AgentRunner, messages: list[ChatMessage]
) -> RunResult:
    try:
        return await runner.run(messages)
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()


async def _run_with_mcp(
    runtime: Any, provider: Any, runner: AgentRunner, messages: list[ChatMessage]
) -> RunResult:
    """Headless run with MCP servers attached (tools registered, shut down after)."""
    from lecode.extras.mcp_client import attach_mcp

    async def _notify(text: str) -> None:
        print(text, file=sys.stderr)

    manager = await attach_mcp(runtime.registry, runtime.ctx, notify=_notify)
    try:
        return await _run_headless(provider, runner, messages)
    finally:
        background = runtime.ctx.extras.get(BACKGROUND_EXTRA)
        if background is not None:
            await background.shutdown()
        await manager.shutdown()


async def _aclose(provider: Any) -> None:
    aclose = getattr(provider, "aclose", None)
    if aclose is not None:
        await aclose()


def _fire_cli_hook(
    dispatcher: HookDispatcher | None, event: str, **payload: Any
) -> MergedVerdict | None:
    """Fire a lifecycle hook from sync CLI code; ``None`` when nothing ran.

    Needs its own event loop (the CLI entry points are sync between
    ``asyncio.run`` calls). Only ``UserPromptSubmit``'s verdict is enforced;
    everything else is observational.
    """
    if dispatcher is None or not dispatcher.handlers.get(event):
        return None
    return asyncio.run(dispatcher.fire(event, **payload))


def _apply_cli_overrides(
    config: Config,
    *,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    auth_policy: AuthPolicy | None,
    tls_verify: bool,
    max_turns: int | None,
) -> None:
    """CLI flag overrides apply on top of the merged config."""
    if model:
        config.llm.model = model
    if provider:
        config.llm.provider = provider
    if base_url:
        config.llm.base_url = base_url
    if auth_policy:
        config.llm.auth_policy = auth_policy
    if not tls_verify:
        config.llm.tls_verify = False
    if max_turns is not None:
        config.agent.max_turns = max_turns


def _tool_filter(allowed_tools: str | None) -> list[str] | None:
    if not allowed_tools:
        return None
    return [name.strip() for name in allowed_tools.split(",") if name.strip()]


def _init_telemetry(config: Config) -> None:
    """Start Sentry/OTel if configured; problems degrade to warnings."""
    for warning in init_telemetry(config.telemetry, version=__version__):
        typer.echo(f"warning: {warning}", err=True)


async def _create_worktree(cwd: Path, name: str) -> tuple[WorktreeManager, WorktreeInfo]:
    """Create the ``--worktree`` isolation; the session then runs inside it."""
    manager = await WorktreeManager.discover(cwd)
    return manager, await manager.create(name)


def _worktree_exit_note(info: WorktreeInfo) -> str:
    return (
        f"worktree kept at {info.path} (branch {info.branch}) — "
        f"merge with /wt-merge or: git merge {info.branch}"
    )


def run_headless(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    auth_policy: AuthPolicy | None = None,
    tls_verify: bool = True,
    read_only: bool = False,
    allowed_tools: str | None = None,
    max_turns: int | None = None,
    worktree: str | None = None,
) -> int:
    """Run one prompt headlessly; returns the process exit code."""
    config = load_config().config
    _apply_cli_overrides(
        config,
        model=model,
        provider=provider,
        base_url=base_url,
        auth_policy=auth_policy,
        tls_verify=tls_verify,
        max_turns=max_turns,
    )
    _init_telemetry(config)

    try:
        client = build_provider(config, api_key=api_key)
    except (AuthError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP

    cwd = Path.cwd()
    wt_info: WorktreeInfo | None = None
    if worktree is not None:
        try:
            _, wt_info = asyncio.run(_create_worktree(cwd, worktree))
        except WorktreeError as e:
            typer.echo(f"error: {e}", err=True)
            return EXIT_STARTUP
        cwd = wt_info.path
    store = SessionStore()
    session = store.create(auto_name(store), cwd, model=config.llm.model)
    models = fetch_catalog(config, api_key)
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        auto_approve=True,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
        catalog=models.catalog,
    )
    runner = AgentRunner(
        client,
        runtime.registry,
        runtime.ctx,
        session=session,
        store=store,
        catalog=models.catalog,
    )
    signals = StatusEmitter(config.signals, session=session.name)

    _fire_cli_hook(runtime.hooks, SESSION_START)
    prompt_verdict = _fire_cli_hook(runtime.hooks, USER_PROMPT_SUBMIT, prompt=prompt)
    if prompt_verdict is not None and prompt_verdict.verdict == "deny":
        typer.echo(
            f"error: prompt blocked by hook: {prompt_verdict.reason or 'UserPromptSubmit hook'}",
            err=True,
        )
        _fire_cli_hook(runtime.hooks, SESSION_END)
        return EXIT_ERROR

    user_message: ChatMessage = {"role": "user", "content": prompt}
    store.append_message(session, user_message)
    messages: list[ChatMessage] = [
        {"role": "system", "content": runtime.system_prompt},
        user_message,
    ]
    signals.emit(START)
    try:
        result = asyncio.run(_run_with_mcp(runtime, client, runner, messages))
    except ProviderError as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_ERROR
    except KeyboardInterrupt:
        _fire_cli_hook(runtime.hooks, INTERRUPT)
        typer.echo("error: interrupted", err=True)
        return EXIT_ERROR
    finally:
        _fire_cli_hook(runtime.hooks, SESSION_END)
        signals.emit(STOP)
        shutdown_telemetry()

    typer.echo(result.final_text)
    if result.review:
        typer.echo(f"\npierre: {result.review}", err=True)
    totals = result.usage_totals
    typer.echo(
        f"tokens: {totals.input_tokens} in / {totals.output_tokens} out "
        f"· cost: ${totals.cost_usd:.4f}",
        err=True,
    )
    if wt_info is not None:
        typer.echo(_worktree_exit_note(wt_info), err=True)
    if result.stop_reason == "context_overflow":
        typer.echo("error: context full even after compaction — start a new session", err=True)
        return EXIT_MAX_TURNS
    if result.stop_reason == "max_turns":
        return EXIT_MAX_TURNS
    return EXIT_OK


def run_loop_mode(
    plan_file: str,
    *,
    loop_cmd: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    auth_policy: AuthPolicy | None = None,
    tls_verify: bool = True,
    read_only: bool = False,
    allowed_tools: str | None = None,
) -> int:
    """``--loop``: iterate the agent over a plan file until it is done.

    One auto-named ``loop-*`` session per run; every iteration is turns in
    that session. Progress lines go to stderr, each iteration's final text to
    stdout. Exit codes: 0 done / 1 error / 3 max iterations.
    """
    config = load_config().config
    _apply_cli_overrides(
        config,
        model=model,
        provider=provider,
        base_url=base_url,
        auth_policy=auth_policy,
        tls_verify=tls_verify,
        max_turns=None,
    )
    _init_telemetry(config)
    try:
        client = build_provider(config, api_key=api_key)
    except (AuthError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP

    cwd = Path.cwd()
    plan_path = Path(plan_file)
    if not plan_path.is_absolute():
        plan_path = cwd / plan_path
    store = SessionStore()
    session = store.create(loop_session_name(), cwd, model=config.llm.model)
    models = fetch_catalog(config, api_key)
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        auto_approve=True,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
        catalog=models.catalog,
    )
    runner = AgentRunner(
        client,
        runtime.registry,
        runtime.ctx,
        session=session,
        store=store,
        catalog=models.catalog,
    )
    signals = StatusEmitter(config.signals, session=session.name)

    async def run_iteration(prompt: str) -> str:
        store.append_message(session, {"role": "user", "content": prompt})
        messages: list[ChatMessage] = [
            {"role": "system", "content": runtime.system_prompt},
            *store.load_for_model(session),
        ]
        result = await runner.run(messages)
        return result.final_text

    async def _loop() -> LoopResult:
        try:
            return await run_plan_loop(
                run_iteration,
                plan_path,
                loop_cmd=loop_cmd,
                cwd=cwd,
                max_iterations=max_iterations,
                on_progress=lambda line: typer.echo(line, err=True),
                on_text=typer.echo,
            )
        finally:
            background = runtime.ctx.extras.get(BACKGROUND_EXTRA)
            if background is not None:
                await background.shutdown()
            await _aclose(client)

    signals.emit(START)
    _fire_cli_hook(runtime.hooks, SESSION_START)
    try:
        result = asyncio.run(_loop())
    except ProviderError as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_ERROR
    except KeyboardInterrupt:
        _fire_cli_hook(runtime.hooks, INTERRUPT)
        typer.echo("error: interrupted", err=True)
        return EXIT_ERROR
    finally:
        _fire_cli_hook(runtime.hooks, SESSION_END)
        signals.emit(STOP)
        shutdown_telemetry()
    if result.stop_reason == "error":
        typer.echo(f"error: {result.error}", err=True)
        return EXIT_ERROR
    if result.stop_reason == "max_iterations":
        typer.echo(
            f"max iterations ({result.iterations}) reached — {len(result.remaining)} item(s) left",
            err=True,
        )
        return EXIT_MAX_TURNS
    return EXIT_OK


def run_chain_mode(
    topic: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    auth_policy: AuthPolicy | None = None,
    tls_verify: bool = True,
    read_only: bool = False,
    allowed_tools: str | None = None,
) -> int:
    """``--chain``: brainstorm → plan → code → review over one topic.

    Each phase's output prints to stdout with a header as it completes.
    """
    config = load_config().config
    _apply_cli_overrides(
        config,
        model=model,
        provider=provider,
        base_url=base_url,
        auth_policy=auth_policy,
        tls_verify=tls_verify,
        max_turns=None,
    )
    _init_telemetry(config)
    try:
        client = build_provider(config, api_key=api_key)
    except (AuthError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP

    cwd = Path.cwd()
    store = SessionStore()
    session = store.create(auto_name(store), cwd, model=config.llm.model)
    chain_catalog = fetch_catalog(config, api_key).catalog
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        auto_approve=True,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
        catalog=chain_catalog,
    )
    signals = StatusEmitter(config.signals, session=session.name)

    def factory() -> AgentRunner:
        return AgentRunner(
            client,
            runtime.registry,
            runtime.ctx,
            session=session,
            store=store,
            catalog=chain_catalog,
        )

    def on_phase(phase: str, output: str) -> None:
        typer.echo(f"## {phase}\n\n{output}\n")

    async def _chain() -> ChainResult:
        try:
            return await run_chain(
                factory,
                topic,
                system_prompt=runtime.system_prompt,
                on_phase=on_phase,
            )
        finally:
            background = runtime.ctx.extras.get(BACKGROUND_EXTRA)
            if background is not None:
                await background.shutdown()
            await _aclose(client)

    signals.emit(START)
    _fire_cli_hook(runtime.hooks, SESSION_START)
    try:
        asyncio.run(_chain())
    except ProviderError as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_ERROR
    except KeyboardInterrupt:
        _fire_cli_hook(runtime.hooks, INTERRUPT)
        typer.echo("error: interrupted", err=True)
        return EXIT_ERROR
    finally:
        _fire_cli_hook(runtime.hooks, SESSION_END)
        signals.emit(STOP)
        shutdown_telemetry()
    return EXIT_OK


async def _run_tui(
    tui: TuiApp,
    provider: Any,
    background: Callable[[], Coroutine[Any, Any, None]] | None = None,
) -> int:
    """Run the TUI; ``background`` (e.g. the model-catalog fetch) starts with
    it and is cancelled if the user quits first."""
    bg = asyncio.ensure_future(background()) if background is not None else None
    try:
        return await tui.run()
    finally:
        if bg is not None:
            if not bg.done():
                bg.cancel()
            await asyncio.gather(bg, return_exceptions=True)
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()


def run_interactive(
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    auth_policy: AuthPolicy | None = None,
    tls_verify: bool = True,
    read_only: bool = False,
    allowed_tools: str | None = None,
    max_turns: int | None = None,
    resume: str | None = None,
    continue_last: bool = False,
    no_color: bool = False,
    worktree: str | None = None,
) -> int:
    """Interactive TUI path: banner → progressive load report → chat.

    Startup order (locked): dependency check (done by the caller) → ASCII
    banner (printed immediately) → first-run setup offer (no config + tty)
    → config load → session-name prompt. Ctrl-C/Ctrl-D at the prompt exits
    0 before any session file is created. Each subsystem prints its loading
    line as it finishes. ``-r/--resume <ref>`` / ``-c/--continue`` reopen an
    existing session and keep its name (no prompt).
    """
    from rich.console import Console

    from lecode.tui.loading import (
        LoadingProgress,
        build_load_report,
        config_step,
        provider_step,
        session_step,
    )

    # Banner first — before any slow work (setup wizard, network fetches).
    console = Console(no_color=no_color)
    progress = LoadingProgress(console)
    progress.banner()

    if (
        resume is None
        and not continue_last
        and find_config_file(config_dir()) is None
        and sys.stdin.isatty()
    ):
        asyncio.run(offer_first_run_setup())
    loaded = load_config()
    config = loaded.config
    _apply_cli_overrides(
        config,
        model=model,
        provider=provider,
        base_url=base_url,
        auth_policy=auth_policy,
        tls_verify=tls_verify,
        max_turns=max_turns,
    )
    if no_color:
        config.ui.no_color = True
    if config.ui.no_color:
        console.no_color = True
    _init_telemetry(config)

    cwd = Path.cwd()
    progress.step(config_step(loaded, cwd))
    wt_manager: WorktreeManager | None = None
    wt_info: WorktreeInfo | None = None
    original_cwd = cwd
    if worktree is not None:
        try:
            wt_manager, wt_info = asyncio.run(_create_worktree(cwd, worktree))
        except WorktreeError as e:
            typer.echo(f"error: {e}", err=True)
            return EXIT_STARTUP
        except KeyboardInterrupt:
            return EXIT_OK
        cwd = wt_info.path
        typer.echo(f"worktree: {wt_info.path} (branch {wt_info.branch})")
    store = SessionStore()
    if resume == _PICK_MARKER:
        # Bare -r: list this folder's sessions and let the user pick one.
        from lecode.tui.name_prompt import pick_session

        meta = asyncio.run(pick_session(store, cwd))
        if meta is None:
            return EXIT_OK
        session = store.open(meta.id)
    elif resume is not None or continue_last:
        try:
            meta = store.resolve(resume, cwd=cwd)
        except (SessionNotFoundError, AmbiguousSessionError) as e:
            typer.echo(f"error: {e}", err=True)
            return EXIT_STARTUP
        session = store.open(meta.id)
    else:
        try:
            name = asyncio.run(prompt_session_name(store))
        except KeyboardInterrupt:
            return EXIT_OK
        if name is None:
            return EXIT_OK
        session = store.create(name, cwd, model=config.llm.model)

    # One live lecode per session: refuse to attach when another process
    # holds the session lock (fail fast, before any network startup work).
    try:
        session_lock = store.acquire_lock(session)
    except SessionInUseError as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP

    resumed = resume is not None or continue_last
    progress.step(session_step(session, store, resumed=resumed))

    try:
        client = build_provider(config, api_key=api_key)
    except (AuthError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP
    spec = resolve_provider(config)
    key_source = resolve_api_key(spec.name, config, cli_key=api_key).source
    progress.step(provider_step(config, spec, key_source))

    # Build the runtime immediately — the two slow, network-bound steps (live
    # model catalog, MCP connect) are deferred to background tasks once the
    # chat is open. The catalog is bound late onto ctx: modality checks read
    # it at tool-call time.
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
        agent_name=session.meta.agent,
    )
    for warning in runtime.warnings:
        typer.echo(f"warning: {warning}", err=True)

    # The remaining local subsystems load with the runtime; print them now.
    try:
        already_shown = {"session", "config", "provider", "models", "mcp"}
        for step in build_load_report(
            config=config,
            loaded=loaded,
            runtime=runtime,
            session=session,
            store=store,
            cwd=cwd,
            resumed=resumed,
            provider_spec=spec,
            key_source=key_source,
            read_only=read_only,
        ):
            if step.label not in already_shown:
                progress.step(step)
    except Exception:
        # The loading screen is informational; it must never break startup.
        pass

    # The chat opens immediately: the model-catalog fetch runs as a
    # background task alongside the TUI (bound late via set_catalog), and
    # MCP servers attach in the background inside TuiApp.run — both report
    # into the feed when they land. Headless/loop paths still block on the
    # fetch, as they print no feed.
    progress.pending("models", f"fetching live from {spec.name} in the background…")
    if (
        config.mcp.enable_exa
        or config.mcp.enable_context7
        or any(s.enabled for s in config.mcp.servers.values())
    ):
        progress.pending("mcp", "connecting in the background…")
    console.print()

    tui = TuiApp(
        config,
        runtime,
        client,
        session,
        store,
        console=console,
        catalog=Catalog.default(),
        session_lock=session_lock,
    )
    if wt_info is not None and wt_manager is not None:
        tui.attach_worktree(wt_manager, wt_info, original_cwd)

    async def _background_models() -> None:
        # fetch_catalog manages its own event loop internally — run it in a
        # thread so it never blocks the TUI loop.
        models = await asyncio.to_thread(fetch_catalog, config, api_key)
        tui.set_catalog(models.catalog, origin=models.origin, count=models.remote_count)

    try:
        code = asyncio.run(_run_tui(tui, client, background=_background_models))
    except KeyboardInterrupt:
        shutdown_telemetry()
        return EXIT_OK
    if wt_info is not None:
        typer.echo(_worktree_exit_note(wt_info))
    shutdown_telemetry()
    return code


async def _hooks_test_async(dispatcher: HookDispatcher) -> bool:
    """Run every configured event's handlers against a synthetic envelope."""
    any_failed = False
    for event in EVENTS:
        handlers = dispatcher.handlers.get(event)
        if not handlers:
            continue
        envelope = build_envelope(
            event,
            dispatcher.cwd,
            session=dispatcher.session,
            tool_name="example_tool",
            tool_args={"example": True},
            result={"content": "example result", "is_error": False},
            prompt="example prompt",
            agent="example_agent",
        )
        merged = await dispatch_event(event, envelope, handlers)
        line = f"{event}: {merged.verdict}"
        if merged.reason:
            line += f" — {merged.reason}"
        if merged.failed:
            line += " (handler failed)"
        typer.echo(line)
        any_failed = any_failed or merged.failed
    return any_failed


def run_hooks_test() -> int:
    """Dry-run the configured hook pipeline without executing any tool."""
    config = load_config().config
    dispatcher, warnings = dispatcher_from_config(config, Path.cwd())
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    if dispatcher is None:
        typer.echo("no hooks configured")
        return EXIT_OK
    failed = asyncio.run(_hooks_test_async(dispatcher))
    return EXIT_ERROR if failed else EXIT_OK


def run_setup() -> int:
    """``--setup``: the onboarding wizard; standalone, exits after writing."""
    if not sys.stdin.isatty():
        typer.echo("error: --setup requires an interactive terminal", err=True)
        return EXIT_STARTUP
    try:
        path = asyncio.run(run_wizard())
    except (KeyboardInterrupt, EOFError):
        typer.echo("error: setup cancelled", err=True)
        return EXIT_ERROR
    typer.echo(f"setup complete — edit {path} for anything else")
    return EXIT_OK


@app.callback(cls=_LeCodeGroup, invoke_without_command=True)
def callback(
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = False,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Headless mode: run this prompt and exit. "
            "Given without a value, the prompt is read from stdin.",
        ),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model id.")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Provider name.")] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="OpenAI-compatible endpoint URL.")
    ] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", help="Provider API key.")] = None,
    auth_policy: Annotated[
        AuthPolicy | None, typer.Option("--auth-policy", help="auto | required | none")
    ] = None,
    no_tls_verify: Annotated[
        bool, typer.Option("--no-tls-verify", help="Disable TLS verification.")
    ] = False,
    read_only: Annotated[
        bool,
        typer.Option(
            "--safe",
            "--read-only",
            help="Read-only permission mode (default is yolo: everything allowed).",
        ),
    ] = False,
    allowed_tools: Annotated[
        str | None, typer.Option("--allowed-tools", help="Comma-separated tool allowlist.")
    ] = None,
    max_turns: Annotated[
        int | None, typer.Option("--max-turns", help="Maximum agent turns.")
    ] = None,
    hooks_test: Annotated[
        bool,
        typer.Option("--hooks-test", help="Dry-run the configured hook pipeline and exit."),
    ] = False,
    setup: Annotated[
        bool,
        typer.Option("--setup", help="Run the onboarding wizard and exit."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            "-r",
            help="Resume a session by id, id prefix, or name. "
            "Given without a value, pick from this folder's sessions.",
        ),
    ] = None,
    continue_last: Annotated[
        bool,
        typer.Option("--continue", "-c", help="Resume the most recent session."),
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color", help="Disable colored output.")] = False,
    worktree: Annotated[
        str | None,
        typer.Option("--worktree", help="Run inside an isolated git worktree+branch."),
    ] = None,
    loop: Annotated[
        str | None,
        typer.Option("--loop", help="Loop mode: iterate the agent over this plan file."),
    ] = None,
    loop_cmd: Annotated[
        str | None,
        typer.Option("--loop-cmd", help="Per-iteration verification command (--loop)."),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Loop iteration budget (--loop)."),
    ] = DEFAULT_MAX_ITERATIONS,
    chain: Annotated[
        str | None,
        typer.Option("--chain", help="Run a brainstorm→plan→code→review chain on TOPIC."),
    ] = None,
) -> None:
    """lecode — minimalist terminal AI coding agent."""
    if setup:
        raise typer.Exit(run_setup())
    if hooks_test:
        raise typer.Exit(run_hooks_test())
    check_dependencies()
    if sum(x is not None for x in (prompt, loop, chain)) > 1:
        typer.echo("error: --prompt, --loop and --chain are mutually exclusive", err=True)
        raise typer.Exit(EXIT_STARTUP)
    if loop is not None:
        raise typer.Exit(
            run_loop_mode(
                loop,
                loop_cmd=loop_cmd,
                max_iterations=max_iterations,
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                auth_policy=auth_policy,
                tls_verify=not no_tls_verify,
                read_only=read_only,
                allowed_tools=allowed_tools,
            )
        )
    if chain is not None:
        raise typer.Exit(
            run_chain_mode(
                chain,
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                auth_policy=auth_policy,
                tls_verify=not no_tls_verify,
                read_only=read_only,
                allowed_tools=allowed_tools,
            )
        )
    if prompt is None:
        raise typer.Exit(
            run_interactive(
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                auth_policy=auth_policy,
                tls_verify=not no_tls_verify,
                read_only=read_only,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
                resume=resume,
                continue_last=continue_last,
                no_color=no_color,
                worktree=worktree,
            )
        )
    if prompt == _STDIN_MARKER:
        if sys.stdin.isatty():
            typer.echo("error: -p without a value requires a piped prompt on stdin", err=True)
            raise typer.Exit(EXIT_STARTUP)
        prompt = sys.stdin.read().strip()
        if not prompt:
            typer.echo("error: empty prompt on stdin", err=True)
            raise typer.Exit(EXIT_ERROR)
    raise typer.Exit(
        run_headless(
            prompt,
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            auth_policy=auth_policy,
            tls_verify=not no_tls_verify,
            read_only=read_only,
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            worktree=worktree,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
