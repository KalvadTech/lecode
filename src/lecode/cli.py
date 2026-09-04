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
    HookDispatcher,
    build_envelope,
    dispatch_event,
    dispatcher_from_config,
)
from lecode.providers import ProviderError, build_client, resolve_provider
from lecode.providers.live import LoadedCatalog, load_catalog
from lecode.providers.openai_compat import ChatClient
from lecode.providers.types import ChatMessage
from lecode.session.naming import auto_name
from lecode.session.storage import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SessionStore,
)
from lecode.setup_wizard import offer_first_run_setup, run_wizard
from lecode.tui.app import TuiApp
from lecode.tui.name_prompt import prompt_session_name

#: Exit codes (headless mode uses the same taxonomy).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2
EXIT_MAX_TURNS = 3

#: Value produced when ``-p/--prompt`` is given without an argument: read stdin.
_STDIN_MARKER = ""


class _PromptOption(TyperOption):
    """``-p/--prompt`` whose value is optional: a bare flag reads stdin.

    The vendored click parser has no optional-value options, so the value
    lookup is made tolerant: when no argument (or another flag) follows, the
    option yields the stdin marker instead of raising.
    """

    def add_to_parser(self, parser: Any, ctx: Any) -> None:
        super().add_to_parser(parser, ctx)
        original = parser._get_value_from_state

        def tolerant_get_value(option_name: str, option: Any, state: Any) -> Any:
            if option.obj is self and (not state.rargs or state.rargs[0].startswith(("-",))):
                return _STDIN_MARKER
            return original(option_name, option, state)

        parser._get_value_from_state = tolerant_get_value


class _LeCodeGroup(TyperGroup):
    """Swaps the ``--prompt`` parameter for the optional-value variant."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for param in self.params:
            if param.name == "prompt":
                param.__class__ = _PromptOption


app = typer.Typer(
    name="lecode",
    help="A minimalist terminal AI coding agent.",
    add_completion=False,
    no_args_is_help=False,
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


def fetch_catalog(client: ChatClient) -> LoadedCatalog:
    """Fetch the live model catalog (cache/bundled fallback); never raises."""
    return asyncio.run(load_catalog(client, config_dir()))


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

    manager = await attach_mcp(runtime.registry, runtime.ctx)
    try:
        return await _run_headless(provider, runner, messages)
    finally:
        await manager.shutdown()


async def _aclose(provider: Any) -> None:
    aclose = getattr(provider, "aclose", None)
    if aclose is not None:
        await aclose()


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
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        auto_approve=True,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
    )
    runner = AgentRunner(
        client,
        runtime.registry,
        runtime.ctx,
        session=session,
        store=store,
        catalog=fetch_catalog(client).catalog,
    )
    signals = StatusEmitter(config.signals, session=session.name)

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
        typer.echo("error: interrupted", err=True)
        return EXIT_ERROR
    finally:
        signals.emit(STOP)

    typer.echo(result.final_text)
    totals = result.usage_totals
    typer.echo(
        f"tokens: {totals.input_tokens} in / {totals.output_tokens} out "
        f"· cost: ${totals.cost_usd:.4f}",
        err=True,
    )
    if wt_info is not None:
        typer.echo(_worktree_exit_note(wt_info), err=True)
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
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        auto_approve=True,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
    )
    runner = AgentRunner(
        client,
        runtime.registry,
        runtime.ctx,
        session=session,
        store=store,
        catalog=fetch_catalog(client).catalog,
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
            await _aclose(client)

    signals.emit(START)
    try:
        result = asyncio.run(_loop())
    except ProviderError as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_ERROR
    except KeyboardInterrupt:
        typer.echo("error: interrupted", err=True)
        return EXIT_ERROR
    finally:
        signals.emit(STOP)
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
    try:
        client = build_provider(config, api_key=api_key)
    except (AuthError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP

    cwd = Path.cwd()
    store = SessionStore()
    session = store.create(auto_name(store), cwd, model=config.llm.model)
    runtime = build_runtime(
        config,
        cwd,
        session=session,
        store=store,
        auto_approve=True,
        mode="readonly" if read_only else None,
        allowed_tools=_tool_filter(allowed_tools),
    )
    signals = StatusEmitter(config.signals, session=session.name)
    chain_catalog = fetch_catalog(client).catalog

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
            await _aclose(client)

    signals.emit(START)
    try:
        asyncio.run(_chain())
    except ProviderError as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_ERROR
    except KeyboardInterrupt:
        typer.echo("error: interrupted", err=True)
        return EXIT_ERROR
    finally:
        signals.emit(STOP)
    return EXIT_OK


async def _run_tui(tui: TuiApp, provider: Any) -> int:
    try:
        return await tui.run()
    finally:
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
    """Interactive TUI path: name prompt → session on disk → chat.

    Startup order (locked): dependency check (done by the caller) → first-run
    setup offer (no config + tty) → config load → session-name prompt. Ctrl-C/
    Ctrl-D at the prompt exits 0 before any session file is created.
    ``-r/--resume <ref>`` / ``-c/--continue`` reopen an existing session and
    keep its name (no prompt).
    """
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

    cwd = Path.cwd()
    wt_manager: WorktreeManager | None = None
    wt_info: WorktreeInfo | None = None
    original_cwd = cwd
    if worktree is not None:
        try:
            wt_manager, wt_info = asyncio.run(_create_worktree(cwd, worktree))
        except WorktreeError as e:
            typer.echo(f"error: {e}", err=True)
            return EXIT_STARTUP
        cwd = wt_info.path
        typer.echo(f"worktree: {wt_info.path} (branch {wt_info.branch})")
    store = SessionStore()
    if resume is not None or continue_last:
        try:
            meta = store.resolve(resume)
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

    try:
        client = build_provider(config, api_key=api_key)
    except (AuthError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return EXIT_STARTUP
    models = fetch_catalog(client)

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

    from rich.console import Console

    from lecode.tui.loading import show_loading_screen

    console = Console(no_color=config.ui.no_color)
    spec = resolve_provider(config)
    show_loading_screen(
        config=config,
        loaded=loaded,
        runtime=runtime,
        session=session,
        store=store,
        cwd=cwd,
        resumed=resume is not None or continue_last,
        provider_spec=spec,
        key_source=resolve_api_key(spec.name, config, cli_key=api_key).source,
        console=console,
        read_only=read_only,
        models_origin=models.origin,
        models_count=models.remote_count,
    )

    tui = TuiApp(config, runtime, client, session, store, console=console, catalog=models.catalog)
    if wt_info is not None and wt_manager is not None:
        tui.attach_worktree(wt_manager, wt_info, original_cwd)
    try:
        code = asyncio.run(_run_tui(tui, client))
    except KeyboardInterrupt:
        return EXIT_OK
    if wt_info is not None:
        typer.echo(_worktree_exit_note(wt_info))
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
        bool, typer.Option("--read-only", help="Read-only permission mode.")
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
        typer.Option("--resume", "-r", help="Resume a session by id, id prefix, or name."),
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
