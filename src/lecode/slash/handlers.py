"""Slash-command handlers.

Each handler is ``async (app, args) -> None``: it renders through
``app.feed`` and mutates app/session state via the app's public seams
(:class:`~lecode.tui.app.TuiApp`). Commands whose features land in later
phases (MCP, worktrees, loop, export, …) are registered as stubs that say
so. Pickers are inline numbered lists — no dialogs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

from lecode.config.models import PermissionMode, ThinkingLevel
from lecode.context.resources import load_text
from lecode.extras.export import ShareError, export_html, share_gist
from lecode.extras.loop_mode import DEFAULT_MAX_ITERATIONS
from lecode.extras.proc import run_proc
from lecode.extras.status_signals import GIT_CONFLICT
from lecode.extras.worktree import WorktreeError, WorktreeManager
from lecode.hooks import hooks_status
from lecode.memory import MemoryStore, memory_command, memory_root
from lecode.multimodal import describe_content, format_size, load_attachment
from lecode.providers import resolve_provider
from lecode.providers.catalog import (
    AmbiguousModelError,
    ModelNotFoundError,
)
from lecode.session.handoff import handoff as handoff_session
from lecode.session.naming import unique_name, validate_name
from lecode.session.stats import session_stats
from lecode.session.storage import AmbiguousSessionError, SessionNotFoundError
from lecode.slash.catalog import BUILTIN_COMMANDS
from lecode.slash.registry import (
    AmbiguousCommandError,
    CommandRegistry,
    SlashCommand,
    UnknownCommandError,
)
from lecode.tui import name_prompt
from lecode.tui.clipboard import osc8_link
from lecode.tui.input import open_in_editor
from lecode.tui.statusline import human_tokens

if TYPE_CHECKING:
    from lecode.context.skills import SkillRegistry
    from lecode.tui.app import TuiApp

#: Message seqs listed by ``/rewind`` without an argument.
REWIND_LIST_LIMIT = 10

#: Messages shown by ``/history``.
HISTORY_LIMIT = 30

#: Recent messages kept raw by ``/compact``; everything older is summarized.
COMPACT_KEEP_TAIL = 4

#: Chars of transcript sent to the summarizer by ``/compact``.
COMPACT_TRANSCRIPT_CAP = 100_000

COMPACT_PROMPT = (
    "Summarize this conversation for continuation by an AI coding agent. "
    "Capture the goal, decisions made, files touched, and the current state "
    "of the work. Be compact (a few hundred words at most); plain text."
)

STUB_MESSAGE = "/{name} is not yet available — it lands in a later release"

WELCOME_TEXT = """\
lecode — cheat sheet

  Enter            send · queue while the agent runs
  Shift-Enter      newline (Ctrl-J works everywhere)
  Alt-Enter        steer (priority queue)
  Ctrl-C           cancel turn / clear input / quit
  Ctrl-D           delete char / quit on empty input
  Tab              complete paths, or cycle agents on empty input
  Ctrl-G           open $EDITOR
  !cmd             run a shell command
  !!cmd            run a shell command and feed output to the model
  /command         slash commands (/help lists them)
  .persona text    one turn with a persona overlay
  @file / @agent   mention a file or agent

  /new /resume /session /undo /redo /rewind /retry /compact
  /model /thinking /permissions /memory /hooks /quit"""


# -- helpers ------------------------------------------------------------------


def _busy(app: TuiApp) -> bool:
    """Session-switching commands refuse while a turn is in flight."""
    if app.turn_busy():
        app.feed.error("finish or cancel the current turn first (Ctrl-C)")
        return True
    return False


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return " ".join(parts).strip()
    return str(content or "").strip()


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _ask_name(app: TuiApp) -> str | None:
    """The shared session-name prompt; ``None`` on abort."""
    name = await name_prompt.prompt_session_name(app.store)
    if name is None:
        app.feed.info("cancelled")
    return name


# -- sessions -----------------------------------------------------------------


async def cmd_new(app: TuiApp, args: list[str]) -> None:
    """``/new [name]``: start a fresh session and switch to it."""
    if _busy(app):
        return
    if args:
        wanted = " ".join(args)
        if (error := validate_name(wanted)) is not None:
            app.feed.error(error)
            return
        name = unique_name(wanted, app.store)
    else:
        name = await _ask_name(app)
        if name is None:
            return
    session = app.store.create(name, app.runtime.ctx.cwd, model=app.config.llm.model)
    if app.switch_session(session):
        app.feed.info(f"new session: {session.name}")


async def cmd_clear(app: TuiApp, args: list[str]) -> None:
    """``/clear``: drop the in-memory conversation (a marker event keeps the
    file intact; replay starts after it)."""
    app.store.append_event(app.session, "clear")
    app.reload_history()
    app.feed.info("conversation cleared (history kept on disk)")


async def cmd_resume(app: TuiApp, args: list[str]) -> None:
    """``/resume [ref]``: switch sessions; no arg lists them, ``--delete``
    removes one. Only sessions created in the current folder are considered.
    Resolution: id, unique id prefix, exact name, ``latest``."""
    cwd = app.runtime.ctx.cwd
    if _busy(app):
        return
    if args and args[0] == "--delete":
        _delete_session(app, args[1:])
        return
    if args:
        ref = " ".join(args)
        try:
            meta = app.store.resolve(ref, cwd=cwd)
        except (SessionNotFoundError, AmbiguousSessionError) as e:
            app.feed.error(str(e))
            return
        if meta.id == app.session.id:
            app.feed.info("already in this session")
            return
        session = app.store.open(meta.id)
        if app.switch_session(session):
            app.feed.info(f"resumed session: {session.name}")
        return
    sessions = app.store.list_sessions(cwd)
    if not sessions:
        app.feed.info("(no sessions in this folder)")
        return
    lines = []
    for index, meta in enumerate(sessions, 1):
        marker = " (current)" if meta.id == app.session.id else ""
        pid = app.store.lock_holder(meta.id)
        in_use = f" — in use (pid {pid})" if pid else ""
        lines.append(f"{index}. {meta.name}{marker} — {meta.id} — {meta.created_at}{in_use}")
    lines.append("")
    lines.append("switch: /resume <name-or-id> · delete: /resume --delete <name-or-id>")
    app.feed.info("\n".join(lines))


def _delete_session(app: TuiApp, args: list[str]) -> None:
    if not args:
        app.feed.error("usage: /resume --delete <name-or-id>")
        return
    ref = " ".join(args)
    try:
        meta = app.store.resolve(ref, cwd=app.runtime.ctx.cwd)
    except (SessionNotFoundError, AmbiguousSessionError) as e:
        app.feed.error(str(e))
        return
    if meta.id == app.session.id:
        app.feed.error("cannot delete the current session")
        return
    if (pid := app.store.lock_holder(meta.id)) is not None:
        app.feed.error(f"session '{meta.name}' is open in another lecode process (pid {pid})")
        return
    app.store.delete(meta.id)
    app.feed.info(f"deleted session: {meta.name}")


async def cmd_session(app: TuiApp, args: list[str]) -> None:
    """``/session``: metadata plus token/cost stats."""
    session = app.session
    stats = session_stats(app.store, session)
    roles = " · ".join(f"{role} x{count}" for role, count in sorted(stats.role_counts.items()))
    lines = [
        f"session: {session.name} ({session.id})",
        f"cwd: {session.meta.cwd}",
        f"created: {stats.created_at}"
        + (f" · last active: {stats.last_active}" if stats.last_active else ""),
        f"agent: {session.meta.agent} · model: {session.meta.model or app.config.llm.model}",
        f"messages: {stats.message_count} ({roles or 'none'})"
        f" · tombstones: {stats.tombstone_count}",
        f"tokens: {stats.input_tokens} in / {stats.output_tokens} out · cost ${stats.cost_usd:.4f}",
    ]
    app.feed.info("\n".join(lines))


async def cmd_undo(app: TuiApp, args: list[str]) -> None:
    """``/undo``: hide the last user turn (tombstone; redoable)."""
    if app.store.undo(app.session) is None:
        app.feed.info("nothing to undo")
        return
    app.reload_history()
    app.feed.info("undid the last user turn")


async def cmd_redo(app: TuiApp, args: list[str]) -> None:
    """``/redo``: cancel the most recent undo."""
    if not app.store.redo(app.session):
        app.feed.info("nothing to redo")
        return
    app.reload_history()
    app.feed.info("restored the undone turn")


async def cmd_rewind(app: TuiApp, args: list[str]) -> None:
    """``/rewind [seq]``: no arg lists recent user turns; with a seq, hides
    everything after it (a restore point is recorded first)."""
    if not args:
        turns = [m for m in app.store.load_messages(app.session) if m.role == "user"]
        if not turns:
            app.feed.info("(no turns yet)")
            return
        lines = ["recent turns:"]
        for record in turns[-REWIND_LIST_LIMIT:]:
            lines.append(f"  seq {record.seq}: {_clip(_text_of(record.message), 60)}")
        lines.append("rewind: /rewind <seq> — hides everything after that turn")
        app.feed.info("\n".join(lines))
        return
    try:
        seq = int(args[0])
    except ValueError:
        app.feed.error("usage: /rewind <seq>")
        return
    visible = app.store.load_messages(app.session)
    if not any(m.seq == seq for m in visible):
        app.feed.error(f"no visible message with seq {seq}")
        return
    hidden = sum(1 for m in visible if m.seq > seq)
    app.store.rewind_to(app.session, seq)
    app.reload_history()
    app.feed.info(f"rewound to seq {seq} ({hidden} message(s) hidden; restore point recorded)")


async def cmd_retry(app: TuiApp, args: list[str]) -> None:
    """``/retry``: undo the last turn and re-run its user prompt."""
    if _busy(app):
        return
    messages = app.store.load_messages(app.session)
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if last_user is None:
        app.feed.info("nothing to retry")
        return
    content = last_user.message.get("content")
    if not isinstance(content, str):
        app.feed.error("cannot retry a non-text message")
        return
    app.store.undo(app.session)
    app.reload_history()
    app.submit_prompt(content)


async def cmd_rename(app: TuiApp, args: list[str]) -> None:
    """``/rename <name>``: validate, deduplicate, append a rename event."""
    if not args:
        app.feed.error("usage: /rename <name>")
        return
    wanted = " ".join(args)
    if (error := validate_name(wanted)) is not None:
        app.feed.error(error)
        return
    if wanted.strip() == app.session.name:
        app.feed.info(f"already named: {app.session.name}")
        return
    name = unique_name(wanted, app.store)
    app.store.append_event(app.session, "rename", {"name": name})
    app.session.meta.name = name
    app.status.session_name = name
    app.refresh()
    app.feed.info(f"renamed to: {name}")


async def cmd_history(app: TuiApp, args: list[str]) -> None:
    """``/history``: the session's messages, one compact line each."""
    messages = app.store.load_messages(app.session)
    if not messages:
        app.feed.info("(empty session)")
        return
    lines = []
    for record in messages[-HISTORY_LIMIT:]:
        text = _clip(_text_of(record.message) or "(tool calls)", 72)
        lines.append(f"{record.seq:>4} {record.role:<9} {text}")
    app.feed.info("\n".join(lines))


async def cmd_handoff(app: TuiApp, args: list[str]) -> None:
    """``/handoff [name]``: seed a fresh session with a brief of this one."""
    if _busy(app):
        return
    if args:
        name = " ".join(args)
    else:
        name = await _ask_name(app)
        if name is None:
            return
    try:
        session = handoff_session(app.session, app.store, name)
    except ValueError as e:
        app.feed.error(str(e))
        return
    if app.switch_session(session):
        app.feed.info(f"handed off to: {session.name}")


async def cmd_compact(app: TuiApp, args: list[str]) -> None:
    """``/compact``: summarize all but the last few messages via the provider,
    then record a compaction event."""
    messages = app.store.load_messages(app.session)
    if len(messages) <= COMPACT_KEEP_TAIL:
        app.feed.info("not enough history to compact")
        return
    older, tail = messages[:-COMPACT_KEEP_TAIL], messages[-COMPACT_KEEP_TAIL:]
    transcript = "\n".join(f"{m.role}: {_text_of(m.message)}" for m in older)
    transcript = _clip(transcript, COMPACT_TRANSCRIPT_CAP)
    app.feed.info(f"compacting {len(older)} messages…")
    try:
        completed = await app.runner.provider.complete(
            [
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": transcript},
            ],
            model=app.config.llm.model,
        )
    except Exception as e:
        app.feed.error(f"compaction failed: {e}")
        return
    summary = (completed.content or "").strip()
    if not summary:
        app.feed.error("compaction failed: empty summary")
        return
    app.store.compact(app.session, summary, keep_from_seq=tail[0].seq)
    app.reload_history()
    app.feed.info(f"compacted {len(older)} messages into a {len(summary)}-char summary")


# -- model / provider / thinking -------------------------------------------------


async def cmd_model(app: TuiApp, args: list[str]) -> None:
    """``/model [ref]``: show or switch the model (id, unique prefix, name)."""
    if not args:
        current = app.config.llm.model
        try:
            info = app.catalog.get(current)
        except ModelNotFoundError:
            app.feed.info(f"model: {current} (not in catalog)")
            return
        app.feed.info(
            f"model: {info.id} ({info.name}) — ctx {human_tokens(info.context_window)} · "
            f"${info.pricing.prompt}/M in · ${info.pricing.completion}/M out"
        )
        return
    ref = " ".join(args)
    try:
        info = app.catalog.get(ref)
    except AmbiguousModelError as e:
        app.feed.error(str(e))
        return
    except ModelNotFoundError:
        app.feed.error(f"unknown model: {ref} (see /models, or /models-add {ref})")
        return
    app.config.llm.model = info.id
    app.runner.model = info.id
    app.status.model = info.id
    app.status.context_window = info.context_window
    app.refresh()
    app.feed.info(f"model: {info.id}")


def _is_hidden(model_id: str, hidden: list[str]) -> bool:
    return any(model_id == h or model_id.startswith(h) for h in hidden)


async def cmd_models(app: TuiApp, args: list[str]) -> None:
    """``/models``: the catalog, minus ``ui.hidden_models`` entries."""
    lines = []
    for entry in app.catalog.all():
        if _is_hidden(entry.id, app.config.ui.hidden_models):
            continue
        marker = " (current)" if entry.id == app.config.llm.model else ""
        lines.append(
            f"{entry.id}{marker} — ctx {human_tokens(entry.context_window)} · "
            f"${entry.pricing.prompt}/M in · ${entry.pricing.completion}/M out"
        )
    app.feed.info("\n".join(lines) or "(no models)")


async def cmd_model_subagent(app: TuiApp, args: list[str]) -> None:
    """``/model-subagent [model]``: show or set the subagent model override
    (session-scoped, in-memory); ``default`` resets to inheriting the main
    model. Agent frontmatter ``model:`` still wins over this override."""
    current = app.config.agent.subagent_model
    if not args:
        if current is None:
            app.feed.info(f"subagent model: (inherits main: {app.config.llm.model})")
        else:
            app.feed.info(f"subagent model: {current}")
        return
    ref = " ".join(args)
    if ref == "default":
        app.config.agent.subagent_model = None
        app.feed.info(f"subagent model: (inherits main: {app.config.llm.model})")
        return
    try:
        info = app.catalog.get(ref)
    except AmbiguousModelError as e:
        app.feed.error(str(e))
        return
    except ModelNotFoundError:
        app.feed.error(f"unknown model: {ref} (see /models-subagent)")
        return
    app.config.agent.subagent_model = info.id
    app.feed.info(f"subagent model: {info.id}")


async def cmd_models_subagent(app: TuiApp, args: list[str]) -> None:
    """``/models-subagent``: the catalog, marking the effective subagent model."""
    effective = app.config.agent.subagent_model or app.config.llm.model
    lines = []
    for entry in app.catalog.all():
        if _is_hidden(entry.id, app.config.ui.hidden_models):
            continue
        marker = " (subagent)" if entry.id == effective else ""
        lines.append(
            f"{entry.id}{marker} — ctx {human_tokens(entry.context_window)} · "
            f"${entry.pricing.prompt}/M in · ${entry.pricing.completion}/M out"
        )
    lines.append("")
    lines.append("set: /model-subagent <model> · reset: /model-subagent default")
    app.feed.info("\n".join(lines))


async def cmd_thinking(app: TuiApp, args: list[str]) -> None:
    """``/thinking [level]``: show/set the reasoning effort (next turns)."""
    levels = get_args(ThinkingLevel)
    if not args:
        app.feed.info(f"thinking: {app.config.llm.thinking} (one of: {', '.join(levels)})")
        return
    level = args[0].lower()
    if level not in levels:
        app.feed.error(f"unknown thinking level: {level} (one of: {', '.join(levels)})")
        return
    app.config.llm.thinking = level
    app.feed.info(f"thinking: {level} (applies from the next turn)")


# -- permissions ------------------------------------------------------------------

PERMISSION_MODES = get_args(PermissionMode)


async def cmd_permissions(app: TuiApp, args: list[str]) -> None:
    """``/permissions [mode]``: show or switch the permission mode."""
    if not args:
        checker = app.runtime.ctx.permission_checker
        rules = app.config.permissions.rules
        counts = " · ".join(
            f"{sum(len(v) for v in table.values())} {kind}"
            for kind, table in (("allow", rules.allow), ("ask", rules.ask), ("deny", rules.deny))
        )
        session_perms = app.runtime.ctx.session_perms
        grants = len(session_perms.grants) if session_perms is not None else 0
        app.feed.info(
            f"permission mode: {checker.mode} (modes: {', '.join(PERMISSION_MODES)})\n"
            f"rules: {counts} · session grants: {grants}"
        )
        return
    mode = args[0]
    if mode not in PERMISSION_MODES:
        app.feed.error(f"unknown mode: {mode} (modes: {', '.join(PERMISSION_MODES)})")
        return
    app.set_permission_mode(mode)
    app.feed.info(f"permission mode: {mode}")


async def cmd_toggle(app: TuiApp, args: list[str]) -> None:
    """``/toggle``: cycle the readonly ↔ yolo pair."""
    current = app.runtime.ctx.permission_checker.mode
    mode = "readonly" if current == "yolo" else "yolo"
    app.set_permission_mode(mode)
    app.feed.info(f"permission mode: {mode}")


# -- pierre ---------------------------------------------------------------------


async def cmd_pierre(app: TuiApp, args: list[str]) -> None:
    """``/pierre``: status, or tune the post-task reviewer (session-scoped)."""
    cfg = app.config.pierre
    if not args:
        app.feed.info(
            f"pierre: {'on' if cfg.enabled else 'off'} · model: {cfg.model or app.config.llm.model}"
        )
        return
    sub = args[0]
    if sub == "on":
        if not cfg.model:
            app.feed.error("pierre: no reviewer model set — pick one: /pierre model <id>")
            return
        if cfg.model == app.config.llm.model:
            app.feed.error(
                f"pierre: the reviewer must differ from the main model ({cfg.model})"
                " — pick another: /pierre model <id>"
            )
            return
        cfg.enabled = True
        app.feed.info(f"pierre: on — every finished task gets reviewed by {cfg.model}")
    elif sub == "off":
        cfg.enabled = False
        app.feed.info("pierre: off")
    elif sub == "model":
        if len(args) < 2:
            app.feed.error("usage: /pierre model <model-id>")
            return
        cfg.model = " ".join(args[1:])
        app.feed.info(f"pierre model: {cfg.model}")
    else:
        app.feed.error("usage: /pierre [on|off|model <id>]")


# -- interface ------------------------------------------------------------------


async def cmd_memory(app: TuiApp, args: list[str]) -> None:
    """``/memory <args>``: route to the persistent-memory commands."""
    if not app.config.memory.enabled:
        app.feed.info("memory is disabled ([memory] enabled = false)")
        return
    store = app.runtime.ctx.extras.get("memory")
    if store is None:
        store = MemoryStore(memory_root(app.runtime.ctx.cwd), max_bytes=app.config.memory.max_bytes)
    app.feed.info(memory_command(args, store))


async def cmd_hooks(app: TuiApp, args: list[str]) -> None:
    """``/hooks``: configured lifecycle-hook handlers."""
    rows = hooks_status(app.config)
    if not rows:
        app.feed.info("no hooks configured")
        return
    lines = [f"{row['event']}: {row['command']} (timeout {row['timeout_s']}s)" for row in rows]
    app.feed.info("\n".join(lines))


async def cmd_doctor(app: TuiApp, args: list[str]) -> None:
    """``/doctor``: health check — binaries, config, provider, connectivity,
    and every subsystem, with a pass/warn/fail mark per line."""
    import asyncio
    import shutil

    from lecode.auth import AuthError, resolve_api_key
    from lecode.config.loader import load_config
    from lecode.deps import REQUIRED_BINARIES

    config = app.config
    lines: list[str] = []
    issues = 0

    def report(status: str, text: str) -> None:
        nonlocal issues
        mark = {"ok": "✓", "warn": "!", "fail": "✗", "skip": "–"}[status]  # noqa: RUF001
        if status in ("warn", "fail"):
            issues += 1
        lines.append(f"{mark} {text}")

    # external binaries (hard dependencies)
    for name, hint in REQUIRED_BINARIES.items():
        path = shutil.which(name)
        if path:
            report("ok", f"{name}: {path}")
        else:
            report("fail", f"{name}: MISSING — install: {hint}")

    # config files (fresh load: picks up edits since startup)
    try:
        loaded = load_config(app.runtime.ctx.cwd)
    except Exception as e:
        report("fail", f"config: failed to load — {e}")
    else:
        sources = ", ".join(str(s) for s in loaded.sources) or "defaults (no config file)"
        report("ok", f"config: {sources}")
        for warning in loaded.warnings:
            report("warn", f"config warning: {warning}")

    # provider + API key (never prints the key)
    try:
        spec = resolve_provider(config)
        key = resolve_api_key(spec.name, config)
        auth = "no API key" if key.source == "none" else f"key from {key.source}"
        status = "warn" if key.source == "none" and config.llm.auth_policy != "none" else "ok"
        report(status, f"provider: {spec.name} · {spec.base_url} · {auth}")
    except (AuthError, ValueError) as e:
        report("fail", f"provider: {e}")

    # model + catalog
    model = config.llm.model
    try:
        info = app.catalog.get(model)
    except (ModelNotFoundError, AmbiguousModelError):
        report("warn", f"model: {model} — not in the catalog (pricing/ctx unknown)")
    else:
        report(
            "ok",
            f"model: {info.id} — ctx {human_tokens(info.context_window)}"
            f" · ${info.pricing.prompt}/M in · ${info.pricing.completion}/M out",
        )

    # connectivity: one live /models fetch (also validates the key)
    app.feed.info("doctor: probing the provider…")
    from lecode.cli import fetch_catalog  # deferred: cli imports the TUI

    fetched = await asyncio.to_thread(fetch_catalog, config)
    if fetched.origin == "live":
        report("ok", f"connectivity: reachable — {fetched.remote_count} models listed")
    else:
        report("fail", "connectivity: catalog fetch failed (network/auth/base URL?)")

    # MCP servers
    manager = app.runtime.ctx.extras.get("mcp")
    statuses = manager.status() if manager is not None else []
    if statuses:
        for s in statuses:
            if s.state == "connected":
                report("ok", f"mcp {s.name}: connected · {s.tools} tools")
            elif s.state == "failed":
                report("warn", f"mcp {s.name}: failed — {s.error or 'connect error'}")
            elif s.state == "auth_required":
                report("warn", f"mcp {s.auth_hint}")
            else:
                report("skip", f"mcp {s.name}: disabled")
    else:
        report("skip", "mcp: no servers configured")

    # session file
    stats = session_stats(app.store, app.session)
    size = app.session.path.stat().st_size if app.session.path.exists() else 0
    report(
        "ok",
        f"session: {app.session.path} — {format_size(size)}, {stats.message_count} messages",
    )

    # persistent memory
    if config.memory.enabled:
        root = memory_root(app.runtime.ctx.cwd)
        long_term = root / "MEMORY.md"
        detail = f"{root}"
        if long_term.is_file():
            detail += f" · MEMORY.md {format_size(long_term.stat().st_size)}"
        else:
            detail += " · MEMORY.md absent"
        report("ok", f"memory: {detail}")
    else:
        report("skip", "memory: disabled")

    # hooks, lsp, telemetry
    rows = hooks_status(config)
    report("ok", f"hooks: {len(rows)} handler(s)") if rows else report("skip", "hooks: none")
    report("ok", "lsp: enabled") if config.lsp.enabled else report("skip", "lsp: disabled")
    telemetry = config.telemetry
    if telemetry.enabled:
        bits = []
        bits.append("sentry" if telemetry.sentry_dsn else "no sentry DSN")
        bits.append(
            f"otlp {telemetry.otlp_endpoint}" if telemetry.otlp_endpoint else "no OTLP endpoint"
        )
        status = "ok" if telemetry.sentry_dsn or telemetry.otlp_endpoint else "warn"
        report(status, f"telemetry: enabled · {' · '.join(bits)}")
    else:
        report("skip", "telemetry: disabled")

    # permissions + tools
    checker = app.runtime.ctx.permission_checker
    tools = len(app.runtime.registry.names())
    report("ok", f"permissions: {checker.mode} · tools: {tools}")

    lines.append("")
    lines.append("doctor: all good" if issues == 0 else f"doctor: {issues} issue(s) — see above")
    app.feed.info("\n".join(lines))


async def cmd_agents(app: TuiApp, args: list[str]) -> None:
    """``/agents``: primaries and subagents, hidden ones marked."""
    agents = app.runtime.agents
    current = app.status.agent
    lines = ["primaries:"]
    for agent in agents.primaries():
        marker = " (current)" if agent.name == current else ""
        lines.append(f"  {agent.name}{marker} — {agent.description}")
    lines.append("subagents:")
    for agent in agents.subagents():
        marker = " (current)" if agent.name == current else ""
        lines.append(f"  {agent.name}{marker} — {agent.description}")
    hidden = [agents.get(name) for name in agents.names() if agents.get(name).hidden]
    if hidden:
        lines.append("hidden:")
        for agent in hidden:
            lines.append(f"  {agent.name} (hidden) — {agent.description}")
    app.feed.info("\n".join(lines))


async def cmd_queue(app: TuiApp, args: list[str]) -> None:
    """``/queue``: pending steered and queued prompts."""
    steered, queued = app.queued_prompts()
    if not steered and not queued:
        app.feed.info("(queues empty)")
        return
    lines = []
    if steered:
        lines.append("steered:")
        lines += [f"  {_clip(describe_content(text), 72)}" for text in steered]
    if queued:
        lines.append("queued:")
        lines += [f"  {_clip(describe_content(text), 72)}" for text in queued]
    app.feed.info("\n".join(lines))


# -- multimodal attachments ---------------------------------------------------------


async def cmd_add(app: TuiApp, args: list[str]) -> None:
    """``/add <path…>``: attach image/audio/PDF files to the next submission."""
    if not args:
        app.feed.error("usage: /add <path>…")
        return
    for raw in args:
        path = Path(raw)
        if not path.is_absolute():
            path = app.runtime.ctx.cwd / path
        try:
            attachment = load_attachment(path)
        except (OSError, ValueError) as e:
            app.feed.error(str(e))
            continue
        app.attachments.add(attachment)
        app.feed.info(
            f"📎 {attachment.path.name} ({attachment.media_kind}, "
            f"{format_size(attachment.size_bytes)})"
        )


async def cmd_drop(app: TuiApp, args: list[str]) -> None:
    """``/drop <n|name>``: drop a pending attachment; no arg lists them."""
    if not args:
        attachments = app.attachments.list()
        if not attachments:
            app.feed.info("(no pending attachments)")
            return
        lines = [
            f"{i}. {a.path.name} ({a.media_kind}, {format_size(a.size_bytes)})"
            for i, a in enumerate(attachments, 1)
        ]
        app.feed.info("\n".join(lines))
        return
    dropped = app.attachments.drop(args[0])
    if dropped is None:
        app.feed.error(f"no such attachment: {args[0]} (see /drop)")
        return
    app.feed.info(f"dropped {dropped.path.name}")


async def cmd_drop_all(app: TuiApp, args: list[str]) -> None:
    """``/drop-all``: drop every pending attachment."""
    count = app.attachments.clear()
    app.feed.info(f"dropped {count} attachment(s)" if count else "(no pending attachments)")


async def cmd_btw(app: TuiApp, args: list[str]) -> None:
    """``/btw <text>``: a side note prepended to the next submission."""
    if not args:
        app.feed.error("usage: /btw <note>")
        return
    app.add_note(" ".join(args))
    app.feed.info("noted — included with your next message")


async def cmd_copy(app: TuiApp, args: list[str]) -> None:
    """``/copy``: last assistant response to the clipboard."""
    await app.copy_last_response()


# -- export / import / share -----------------------------------------------------


async def cmd_export(app: TuiApp, args: list[str]) -> None:
    """``/export [path]``: standalone HTML of the session (logical view)."""
    out = Path(" ".join(args)) if args else None
    if out is not None and not out.is_absolute():
        out = app.runtime.ctx.cwd / out
    try:
        path = export_html(app.session, app.store, out)
    except OSError as e:
        app.feed.error(f"export failed: {e}")
        return
    app.feed.info(f"exported: {path}")


async def cmd_import(app: TuiApp, args: list[str]) -> None:
    """``/import <path>``: validate and import a session JSONL file."""
    if not args:
        app.feed.error("usage: /import <path>")
        return
    path = Path(args[0])
    if not path.is_absolute():
        path = app.runtime.ctx.cwd / path
    try:
        session = app.store.import_session(path)
    except (OSError, ValueError) as e:
        app.feed.error(f"import failed: {e}")
        return
    app.feed.info(f"imported: {session.name} ({session.id}) — resume with /resume {session.id}")


async def cmd_share(app: TuiApp, args: list[str]) -> None:
    """``/share``: create a secret gist with the session JSONL + summary."""
    try:
        url = await share_gist(app.session, app.store)
    except ShareError as e:
        app.feed.error(str(e))
        return
    app.feed.info(f"shared: {osc8_link(url, url, no_color=app.config.ui.no_color)}")


# -- git worktrees ------------------------------------------------------------------


async def cmd_worktree(app: TuiApp, args: list[str]) -> None:
    """``/worktree <name>``: create an isolated worktree+branch and switch in."""
    if _busy(app):
        return
    if app._worktree is not None:
        app.feed.error(f"already in worktree '{app._worktree.name}' (/wt-exit first)")
        return
    if not args:
        app.feed.error("usage: /worktree <name>")
        return
    try:
        manager = await WorktreeManager.discover(app.runtime.ctx.cwd)
        info = await manager.create(args[0])
    except WorktreeError as e:
        app.feed.error(str(e))
        return
    app._enter_worktree(manager, info)


async def cmd_wt_merge(app: TuiApp, args: list[str]) -> None:
    """``/wt-merge``: merge the worktree branch back; conflicts are reported,
    never auto-resolved (the merge stays in progress for manual resolution)."""
    if _busy(app):
        return
    if app._worktree is None or app._worktree_manager is None:
        app.feed.error("not in a worktree (see /worktree)")
        return
    try:
        result = await app._worktree_manager.merge_back(app._worktree.name)
    except WorktreeError as e:
        app.feed.error(str(e))
        return
    if result.merged:
        app.feed.info(result.message)
    elif result.conflicts:
        app._signals.emit(GIT_CONFLICT, files=result.conflicts)
        listing = "\n".join(f"  {path}" for path in result.conflicts)
        app.feed.error(f"{result.message}\nconflicts:\n{listing}")
    else:
        app.feed.error(result.message)


async def cmd_wt_exit(app: TuiApp, args: list[str]) -> None:
    """``/wt-exit [--delete] [--force]``: leave the worktree and return to the
    original cwd; dirty worktrees refuse without ``--force``."""
    if _busy(app):
        return
    if app._worktree is None or app._worktree_manager is None:
        app.feed.error("not in a worktree (see /worktree)")
        return
    delete = "--delete" in args
    force = "--force" in args
    name = app._worktree.name
    try:
        await app._worktree_manager.exit_worktree(name, delete_branch=delete, force=force)
    except WorktreeError as e:
        app.feed.error(str(e))
        return
    app._exit_worktree()
    app.feed.info(f"left worktree '{name}'" + (" (branch deleted)" if delete else ""))


# -- power features: plan loop / prompt chain -------------------------------------


async def cmd_loop(app: TuiApp, args: list[str]) -> None:
    """``/loop <plan-file> [max]``: iterate the agent over a markdown checklist
    plan in the background; ``/loop stop`` cancels."""
    if args and args[0] == "stop":
        if not app.loop_running():
            app.feed.info("no loop running")
            return
        assert app._loop_task is not None
        app._loop_task.cancel()
        app.feed.info("stopping loop…")
        return
    if app.loop_running():
        app.feed.error("a loop is already running (/loop stop first)")
        return
    if _busy(app):
        return
    if not args:
        app.feed.error("usage: /loop <plan-file> [max-iterations]")
        return
    plan_path = Path(args[0])
    if not plan_path.is_absolute():
        plan_path = app.runtime.ctx.cwd / plan_path
    if not plan_path.is_file():
        app.feed.error(f"no such plan file: {plan_path}")
        return
    max_iterations = DEFAULT_MAX_ITERATIONS
    if len(args) > 1:
        if not args[1].isdigit() or int(args[1]) < 1:
            app.feed.error("usage: /loop <plan-file> [max-iterations] (positive integer)")
            return
        max_iterations = int(args[1])
    app.start_loop(plan_path, max_iterations)
    app.feed.info(
        f"loop started: {plan_path.name} (max {max_iterations} iterations) — /loop stop cancels"
    )


async def cmd_chain(app: TuiApp, args: list[str]) -> None:
    """``/chain <topic>``: brainstorm → plan → code → review as one turn."""
    if _busy(app):
        return
    topic = " ".join(args).strip()
    if not topic:
        app.feed.error("usage: /chain <topic>")
        return
    app.start_chain(topic)


async def cmd_mcp(app: TuiApp, args: list[str]) -> None:
    """``/mcp`` — states; ``tools|reconnect|auth|logout <name>``."""
    manager = app.runtime.ctx.extras.get("mcp")
    if manager is None or not manager.status():
        app.feed.info("no MCP servers configured")
        return
    if args and args[0] == "reconnect":
        if len(args) < 2:
            app.feed.error("usage: /mcp reconnect <name>")
            return
        status = await manager.reconnect(args[1])
        if status is None:
            app.feed.error(f"unknown MCP server: {args[1]}")
            return
        if status.state == "connected":
            app.feed.info(f"mcp: {status.name} reconnected ({status.tools} tools)")
        else:
            app.feed.error(f"mcp: {status.name} reconnect failed: {status.error}")
        return
    if args and args[0] == "auth":
        if len(args) < 2:
            app.feed.error("usage: /mcp auth <name>")
            return
        name = args[1]
        if manager.server_tools(name) is None:
            app.feed.error(f"unknown MCP server: {name}")
            return
        app.feed.info(f"mcp: {name}: opening your browser for OAuth login — approve there…")
        status = await manager.authenticate(
            name,
            announce=lambda url: app.feed.info(
                f"mcp: {name}: authorization URL (a different browser works too): {url}"
            ),
        )
        if status.error and status.state == "connected":
            # e.g. /mcp auth on a server without auth = "oauth"
            app.feed.error(f"mcp: {name}: {status.error}")
        elif status.state == "connected":
            app.feed.info(f"mcp: {name} authenticated ({status.tools} tools)")
        elif status.state == "auth_required":
            app.feed.error(
                f"mcp: {name} authentication failed: {status.error} — try /mcp auth {name}"
            )
        else:
            app.feed.error(f"mcp: {name} authentication failed: {status.error}")
        return
    if args and args[0] == "logout":
        if len(args) < 2:
            app.feed.error("usage: /mcp logout <name>")
            return
        status = await manager.logout(args[1])
        if status is None:
            app.feed.error(f"unknown MCP server: {args[1]}")
            return
        if status.error:
            app.feed.error(f"mcp: {status.name}: {status.error}")
        else:
            app.feed.info(f"mcp: {status.name} logged out — /mcp auth {status.name} to reconnect")
        return
    if args and args[0] == "tools":
        if len(args) < 2:
            app.feed.error("usage: /mcp tools <name>")
            return
        tools = manager.server_tools(args[1])
        if tools is None:
            app.feed.error(f"unknown MCP server: {args[1]}")
            return
        if not tools:
            app.feed.info(f"mcp: {args[1]} exposes no tools (not connected)")
            return
        lines = [f"{tool.name} — {(tool.description or '').strip()}" for tool in tools]
        app.feed.info(f"mcp: {args[1]} tools:\n" + "\n".join(lines))
        return
    lines = []
    for status in manager.status():
        if status.state == "connected":
            lines.append(f"{status.name}: connected ({status.tools} tools)")
        elif status.state == "disabled":
            lines.append(f"{status.name}: disabled")
        elif status.state == "auth_required":
            lines.append(status.auth_hint)
        else:
            lines.append(f"{status.name}: failed — {status.error}")
    app.feed.info("\n".join(lines))


# -- onboarding & docs-ish commands -----------------------------------------------


AGENTS_MD_SKELETON = """\
# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

<!-- One paragraph: what this is, the stack, the rough layout. -->

## Build & test

- install: `…`
- test: `…`
- lint: `…`

## Conventions

<!-- Code style, naming, where things live, what to avoid. -->

## Guardrails

- Never commit or push without being asked.
- Keep changes minimal and covered by tests.
"""


async def cmd_init(app: TuiApp, args: list[str]) -> None:
    """``/init``: write a starter AGENTS.md in the cwd (never overwrites)."""
    path = app.runtime.ctx.cwd / "AGENTS.md"
    if path.exists():
        app.feed.error(f"AGENTS.md already exists: {path}")
        return
    path.write_text(AGENTS_MD_SKELETON, encoding="utf-8")
    app.feed.info(f"wrote {path} — fill in the placeholders for this project")


#: ``/tutor`` static topic map — built-in help text, no LLM call.
TUTOR_TOPICS: dict[str, str] = {
    "sessions": (
        "Sessions persist to disk as JSONL. /new starts one, /resume switches "
        "(or --delete), /rename renames, /undo and /rewind walk history back, "
        "/compact summarizes old context, /handoff seeds a fresh session."
    ),
    "permissions": (
        "Two modes: yolo (default, everything allowed) and readonly (read-class "
        "tools only); switch with /mode or /toggle. Rules in [permissions.rules] "
        "allow/ask/deny are glob or regex, last match wins, deny is "
        "unbypassable. Repeat identical calls escalate (doom-loop guard)."
    ),
    "worktrees": (
        "/worktree <name> moves the session into an isolated git worktree+branch; "
        "/wt-merge merges it back (conflicts listed), /wt-exit returns without "
        "merging. --worktree <name> does this from the CLI."
    ),
    "loop": (
        "/loop <plan.md> [max] iterates the agent over a markdown checklist until "
        "every box is ticked; /loop stop cancels. CLI: --loop plan.md with "
        "--loop-cmd '<verify command>' feeding output into the next iteration."
    ),
    "chain": "/chain <topic> runs brainstorm → plan → code → review as one turn.",
    "mcp": (
        "MCP servers are configured under [mcp.servers] (stdio or http); Exa web "
        "search is auto-configured when EXA_API_KEY is set, context7 with "
        "enable_context7 = true. /mcp shows state; /mcp tools|reconnect|auth|logout "
        '<name>. HTTP servers with auth = "oauth" log in via /mcp auth '
        "(opens your browser once; credentials are reused afterwards). Tools "
        "appear as mcp:<server>:<tool>."
    ),
    "memory": (
        "Persistent markdown memory: MEMORY.md (auto-injected), daily logs, "
        "scratchpad, named notes. /memory inspects it; the agent uses the "
        "memory_* tools. Disable with [memory] enabled = false."
    ),
    "hooks": (
        "Lifecycle hooks ([hooks] event = commands) wrap tool calls and events; "
        "they can only narrow permission verdicts. /hooks lists them, "
        "--hooks-test dry-runs the pipeline. See docs/hooks.md."
    ),
    "agents": (
        "Custom agents are markdown files (~/.config/lecode/agents, .lecode/agents) "
        "with model/prompt/permission overlays. Tab cycles build/plan; @mention or "
        "the task tool invokes subagents. /agents lists them."
    ),
}


async def cmd_tutor(app: TuiApp, args: list[str]) -> None:
    """``/tutor <topic>``: built-in explainer for one lecode feature."""
    if not args:
        app.feed.info("usage: /tutor <topic> — topics: " + ", ".join(sorted(TUTOR_TOPICS)))
        return
    text = TUTOR_TOPICS.get(args[0])
    if text is None:
        app.feed.error(f"unknown topic: {args[0]} (topics: {', '.join(sorted(TUTOR_TOPICS))})")
        return
    app.feed.info(text)


#: Cap on the diff/file content ``/review`` sends.
REVIEW_CONTENT_CAP = 50_000


async def cmd_review(app: TuiApp, args: list[str]) -> None:
    """``/review [file…]``: review the working-tree diff, or the listed files."""
    if _busy(app):
        return
    cwd = app.runtime.ctx.cwd
    if args:
        parts = []
        for name in args:
            path = Path(name)
            if not path.is_absolute():
                path = cwd / path
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                app.feed.error(f"cannot read {name}: {e}")
                return
            parts.append(f"### {name}\n\n```\n{_clip(text, REVIEW_CONTENT_CAP)}\n```")
        subject = "these files:\n\n" + "\n\n".join(parts)
    else:
        result = await run_proc(["git", "diff", "HEAD"], cwd=cwd, timeout=10)
        if result.exit_code != 0:
            result = await run_proc(["git", "diff"], cwd=cwd, timeout=10)
        if result.exit_code != 0:
            app.feed.error("not a git repository — or pass files: /review <file>…")
            return
        diff = result.stdout.strip()
        if not diff:
            app.feed.info("nothing to review (working tree matches HEAD)")
            return
        subject = f"this git diff:\n\n```diff\n{_clip(diff, REVIEW_CONTENT_CAP)}\n```"
    persona = load_text("prompts", "personas/reviewer.md", cwd).strip()
    app.submit_prompt(f"{persona}\n\nReview {subject}")


async def cmd_notifications(app: TuiApp, args: list[str]) -> None:
    """``/notifications [on|off]``: show or toggle audio notifications."""
    cfg = app.config.notifications
    if args:
        if args[0] not in ("on", "off"):
            app.feed.error("usage: /notifications [on|off]")
            return
        cfg.enabled = args[0] == "on"
        app.feed.info(f"notifications: {args[0]} (session only — edit config to persist)")
        return
    app.feed.info(
        f"notifications: {'on' if cfg.enabled else 'off'} · volume {cfg.volume} · "
        f"finish {'on' if cfg.on_finish else 'off'}, "
        f"error {'on' if cfg.on_error else 'off'}, "
        f"approval {'on' if cfg.on_approval else 'off'}"
    )


async def cmd_prompt(app: TuiApp, args: list[str]) -> None:
    """``/prompt``: print the assembled system prompt for this session."""
    app.feed.info(app.runtime.system_prompt)


async def cmd_editsys(app: TuiApp, args: list[str]) -> None:
    """``/editsys``: edit the effective system prompt in $EDITOR; the save
    becomes this session's ``llm.system_prompt.custom`` override."""
    edited = await open_in_editor(app.runtime.system_prompt)
    if edited is None:
        app.feed.info("unchanged (editor closed without edits, or $EDITOR unset)")
        return
    app.config.llm.system_prompt.custom = edited
    app.runtime.system_prompt = edited
    app.feed.info("system prompt overridden for this session")


# -- help / welcome / quit ---------------------------------------------------------


async def cmd_help(app: TuiApp, args: list[str]) -> None:
    """``/help [cmd]``: one command's usage, or the catalog grouped."""
    if args:
        try:
            command = app.commands.match(args[0])
        except UnknownCommandError:
            app.feed.error(f"unknown command: /{args[0]}")
            return
        except AmbiguousCommandError as e:
            app.feed.error(str(e))
            return
        hint = f" {command.arg_hint}" if command.arg_hint else ""
        app.feed.info(f"/{command.name}{hint} — {command.description}")
        return
    known: set[str] = set()
    lines = []
    for label, names in CATEGORIES:
        lines.append(f"{label}:")
        for name in names:
            command = app.commands.get(name)
            if command is None:
                continue
            known.add(name)
            lines.append(f"  /{name} — {command.description}")
    extra = [c for c in app.commands.list() if c.name not in known]
    if extra:
        lines.append("Skills:")
        lines += [f"  /{c.name} — {c.description}" for c in extra]
    lines.append("details: /help <command>")
    app.feed.info("\n".join(lines))


async def cmd_welcome(app: TuiApp, args: list[str]) -> None:
    """``/welcome``: the shortcuts cheat-sheet."""
    app.feed.info(WELCOME_TEXT)


async def cmd_quit(app: TuiApp, args: list[str]) -> None:
    """``/quit``/``/exit``: leave the chat."""
    app.request_quit()


# -- registry assembly -----------------------------------------------------------

#: ``/help`` grouping (display order).
CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "Sessions",
        [
            "new",
            "clear",
            "resume",
            "session",
            "undo",
            "redo",
            "rewind",
            "retry",
            "rename",
            "history",
            "handoff",
            "compact",
            "export",
            "import",
            "share",
        ],
    ),
    (
        "Model & provider",
        [
            "model",
            "models",
            "model-subagent",
            "models-subagent",
            "thinking",
            "reasoning",
        ],
    ),
    ("Permissions", ["permissions", "mode", "toggle"]),
    ("Worktrees", ["worktree", "wt-merge", "wt-exit"]),
    ("Power features", ["loop", "chain", "mcp", "review"]),
    (
        "Interface",
        [
            "notifications",
            "copy",
            "queue",
            "btw",
            "help",
            "welcome",
            "tutor",
            "quit",
            "exit",
        ],
    ),
    (
        "Context & memory",
        [
            "memory",
            "hooks",
            "agents",
            "add",
            "drop",
            "drop-all",
            "prompt",
            "compress",
            "editsys",
            "init",
        ],
    ),
]

_HANDLERS = {
    "new": cmd_new,
    "clear": cmd_clear,
    "resume": cmd_resume,
    "session": cmd_session,
    "undo": cmd_undo,
    "redo": cmd_redo,
    "rewind": cmd_rewind,
    "retry": cmd_retry,
    "rename": cmd_rename,
    "history": cmd_history,
    "quit": cmd_quit,
    "exit": cmd_quit,
    "handoff": cmd_handoff,
    "compact": cmd_compact,
    "model": cmd_model,
    "models": cmd_models,
    "model-subagent": cmd_model_subagent,
    "models-subagent": cmd_models_subagent,
    "thinking": cmd_thinking,
    "reasoning": cmd_thinking,
    "permissions": cmd_permissions,
    "mode": cmd_permissions,
    "toggle": cmd_toggle,
    "memory": cmd_memory,
    "hooks": cmd_hooks,
    "agents": cmd_agents,
    "queue": cmd_queue,
    "btw": cmd_btw,
    "copy": cmd_copy,
    "help": cmd_help,
    "welcome": cmd_welcome,
    "pierre": cmd_pierre,
    "add": cmd_add,
    "drop": cmd_drop,
    "drop-all": cmd_drop_all,
    "export": cmd_export,
    "import": cmd_import,
    "share": cmd_share,
    "worktree": cmd_worktree,
    "wt-merge": cmd_wt_merge,
    "wt-exit": cmd_wt_exit,
    "loop": cmd_loop,
    "chain": cmd_chain,
    "mcp": cmd_mcp,
    "init": cmd_init,
    "tutor": cmd_tutor,
    "review": cmd_review,
    "notifications": cmd_notifications,
    "prompt": cmd_prompt,
    "compress": cmd_compact,
    "editsys": cmd_editsys,
    "doctor": cmd_doctor,
}

#: Usage hints shown by ``/help <name>``.
ARG_HINTS = {
    "new": "[name]",
    "resume": "[name-or-id] | --delete <name-or-id>",
    "rewind": "[seq]",
    "rename": "<name>",
    "handoff": "[name]",
    "model": "[model]",
    "model-subagent": "[model]",
    "thinking": "[none|low|medium|high]",
    "reasoning": "[none|low|medium|high]",
    "permissions": "[mode]",
    "mode": "[mode]",
    "memory": "[show|edit|search|log|notes]",
    "btw": "<text>",
    "help": "[command]",
    "add": "<path>…",
    "drop": "[n|name]",
    "export": "[path]",
    "import": "<path>",
    "worktree": "<name>",
    "wt-exit": "[--delete] [--force]",
    "loop": "<plan-file> [max-iterations]",
    "chain": "<topic>",
    "mcp": "[tools|reconnect|auth|logout <name>]",
    "tutor": "<topic>",
    "review": "[file…]",
    "notifications": "[on|off]",
}


def _make_stub(name: str):
    """A 'not yet available' handler for later-phase commands."""

    async def handler(app: TuiApp, args: list[str], _name: str = name) -> None:
        app.feed.info(STUB_MESSAGE.format(name=_name))

    return handler


def build_registry(skills: SkillRegistry | None = None) -> CommandRegistry:
    """The default registry: catalog commands + skill-registered commands.

    Commands without a Phase 9 handler are registered as stubs.
    """
    registry = CommandRegistry()
    for name, description in BUILTIN_COMMANDS:
        handler = _HANDLERS.get(name) or _make_stub(name)
        registry.register(SlashCommand(name, description, handler, arg_hint=ARG_HINTS.get(name)))
    if skills is not None:
        registry.register_skills(skills)
    return registry
