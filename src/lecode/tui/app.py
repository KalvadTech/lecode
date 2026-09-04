"""The interactive TUI application.

prompt_toolkit owns a bottom-pinned multiline input with the fixed
statusline as bottom toolbar (re-rendered on state change and on a 0.3s
spinner refresh while a turn runs). The transcript is the append-only Rich
:class:`~lecode.tui.feed.Feed` — normal scrollback, no alternate screen, no
mouse. One asyncio loop: submissions either start a turn task on the
:class:`~lecode.agent.runner.AgentRunner` or queue (Enter) / steer
(Alt-Enter) while a turn is running. The TUI never calls providers
directly; everything flows through the runner's event taxonomy.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import Application
from prompt_toolkit.completion import merge_completers
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from lecode.agent.runner import (
    AgentRunner,
    Done,
    Error,
    Reasoning,
    Retrying,
    Token,
    ToolCall,
    ToolResult,
)
from lecode.config.models import PermissionMode
from lecode.context.agents import parse_mentions
from lecode.context.resources import load_text
from lecode.extras.chain import run_chain
from lecode.extras.loop_mode import run_plan_loop
from lecode.extras.mcp_client import MCP_EXTRA, attach_mcp
from lecode.extras.proc import run_proc
from lecode.extras.status_signals import START, STOP, StatusEmitter
from lecode.extras.subagents import SubagentError, SubagentOutcome, run_subagent
from lecode.multimodal import (
    AttachmentStore,
    MessageContent,
    check_modalities,
    describe_content,
    extract_attachment_refs,
    format_size,
    to_content_parts,
)
from lecode.permission import (
    AllowAlways,
    AllowOnce,
    ApprovalDecision,
    Deny,
    PermissionChecker,
    target_of,
)
from lecode.providers.openai_compat import ProviderError
from lecode.providers.types import ContentPart
from lecode.session.stats import session_stats
from lecode.slash.handlers import build_registry
from lecode.slash.registry import AmbiguousCommandError, CommandRegistry, UnknownCommandError
from lecode.tui.clipboard import copy_to_clipboard
from lecode.tui.feed import Feed
from lecode.tui.input import (
    FileLister,
    JsonlHistory,
    KillRing,
    PathCompleter,
    history_path,
    kill_to_end_of_line,
    kill_to_start_of_line,
    kill_word_back,
    open_in_editor,
)
from lecode.tui.notify import Notifier
from lecode.tui.permission import ApprovalPrompt, approval_prompt_text
from lecode.tui.pickers import TriggerCompleter, persona_names
from lecode.tui.statusline import (
    CachedBranch,
    StatusLineState,
    StatusState,
    render_statusline,
)
from lecode.tui.themes import load_theme

if TYPE_CHECKING:
    from lecode.agent.builder import Runtime
    from lecode.config.models import Config
    from lecode.session.storage import Session, SessionStore

#: Max pending messages in each of the input and steer queues.
QUEUE_LIMIT = 5

#: Spinner/statusline refresh period while the app runs.
SPINNER_INTERVAL_S = 0.3

#: Timeout for ``!cmd`` shell-outs.
SHELL_TIMEOUT_S = 120.0

#: Exit code returned by :meth:`TuiApp.run` (interactive exits cleanly).
EXIT_OK = 0


class TuiApp:
    """Interactive chat: prompt_toolkit input + statusline over a Rich feed."""

    def __init__(
        self,
        config: Config,
        runtime: Runtime,
        provider: Any,
        session: Session,
        store: SessionStore,
        console: Console | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._store = store
        self._session = session
        self._theme = load_theme(config.ui.theme, config)
        self._console = console or Console(no_color=config.ui.no_color)
        self._feed = Feed(self._console, self._theme, collapse_thinking=config.ui.collapse_thinking)

        self._cwd = Path(runtime.ctx.cwd)
        self._agent_name = session.meta.agent or "build"
        #: Checker without an agent overlay; Tab-cycling re-derives from it.
        self._base_checker = runtime.ctx.permission_checker

        # Inline permission prompt: Ask verdicts route through the app, but
        # only while the interactive loop runs (:meth:`run` installs the
        # callback); headless turns keep the default "requires approval".
        self._approval = ApprovalPrompt()
        self._notifier = Notifier(config.notifications)
        self._last_response = ""
        #: Pending advisor-handoff answer (the next Enter resolves it).
        self._handoff_future: asyncio.Future[str | None] | None = None
        #: Active worktree isolation (``/worktree``, ``--worktree``).
        self._worktree: Any | None = None  # WorktreeInfo
        self._worktree_manager: Any | None = None  # WorktreeManager
        self._original_cwd: Path | None = None
        #: Background plan-loop task (``/loop``); while it runs, prompts refuse.
        self._loop_task: asyncio.Task[None] | None = None
        #: Unix-socket status signals (``[signals]``); inert when disabled.
        self._signals = StatusEmitter(config.signals, session=session.name)

        self._input_queue: asyncio.Queue[MessageContent] = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self._steer_queue: asyncio.Queue[MessageContent] = asyncio.Queue(maxsize=QUEUE_LIMIT)
        #: ``/btw`` side notes prepended to the next user submission.
        self._pending_notes: list[str] = []
        #: Pending attachments for the next user submission (``/add``, ``@path``).
        self._attachments = AttachmentStore()
        #: Model catalog for ``/model``/``/models`` (lazy; ``/models-add`` merges in).
        self._catalog: Any | None = None
        self._commands = build_registry(runtime.skills)
        self._runner = AgentRunner(
            provider,
            runtime.registry,
            runtime.ctx,
            session=session,
            store=store,
            steer_queue=self._steer_queue,
            input_queue=self._input_queue,
        )
        # Subagent progress (the task tool and direct @agent turns) renders
        # inline through the feed; installed here so tests driving _submit
        # directly get it too.
        self._runtime.ctx.extras["subagent_events"] = self._on_child_event

        self._status = StatusState(
            session_name=session.name,
            agent=self._agent_name,
            model=config.llm.model,
            cwd=self._cwd,
            context_window=config.agent.context_window,
        )
        self._branch = CachedBranch()
        self._history: list[dict[str, Any]] = [{"role": "system", "content": runtime.system_prompt}]
        self._history += store.load_for_model(session)

        self._turn_task: asyncio.Task[None] | None = None
        self._pending: set[asyncio.Task[None]] = set()  # in-flight submit tasks
        self._quit = False
        self._app: Application[None] | None = None
        self._spinner_task: asyncio.Task[None] | None = None

        # Input editor extras (history persisted under the config dir).
        self._input_history = JsonlHistory(history_path())
        self._kill_ring = KillRing()
        # One shared fd-backed file list feeds the path completer and pickers.
        self._file_lister = FileLister(self._cwd)
        self._completer = merge_completers(
            [
                PathCompleter(self._cwd, lister=self._file_lister),
                TriggerCompleter(self._file_lister, runtime.agents, runtime.skills),
            ]
        )
        self._input_area: TextArea | None = None

    # -- public seams for slash-command handlers -------------------------------

    @property
    def feed(self) -> Feed:
        return self._feed

    @property
    def config(self) -> Config:
        return self._config

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def session(self) -> Session:
        return self._session

    @property
    def attachments(self) -> AttachmentStore:
        """Pending attachments for the next submission (``/add``, ``@path``)."""
        return self._attachments

    @property
    def runner(self) -> AgentRunner:
        return self._runner

    @property
    def status(self) -> StatusState:
        """The statusline state (mutate fields, then :meth:`refresh`)."""
        return self._status

    @property
    def commands(self) -> CommandRegistry:
        return self._commands

    def refresh(self) -> None:
        """Re-render the statusline after state changes."""
        self._invalidate()

    @property
    def catalog(self) -> Any:
        """The model catalog (lazy default; ``/models-add`` merges into it)."""
        if self._catalog is None:
            from lecode.providers.catalog import Catalog

            self._catalog = Catalog.default()
        return self._catalog

    def submit_prompt(self, text: str, *, echo: str | None = None) -> None:
        """Render and queue/start a user prompt (skill commands, ``/retry``)."""
        self._feed.user_message(text if echo is None else echo)
        self._enqueue_or_start(text)

    def request_quit(self) -> None:
        """``/quit``/``/exit``: cancel any turn and leave the loop."""
        self._quit = True
        self.cancel_turn()
        if self._app is not None:
            self._app.exit()

    async def copy_last_response(self) -> None:
        """``/copy``: last assistant response to the clipboard."""
        if not self._last_response:
            self._feed.info("nothing to copy")
            return
        if await copy_to_clipboard(self._last_response):
            self._feed.info(f"copied {len(self._last_response)} chars")
        else:
            self._feed.error("clipboard unavailable")

    def set_theme(self, name: str) -> None:
        """``/theme``: swap the theme and re-render the statusline colors."""
        self._theme = load_theme(name, self._config)
        self._config.ui.theme = name
        self._feed.set_theme(self._theme)
        self._invalidate()

    def switch_session(self, session: Session) -> None:
        """Point the app (runner, checker, history, statusline) at ``session``."""
        self._session = session
        self._runner.session = session
        self._runtime.ctx.session = session
        self._signals.session = session.name
        if self._runtime.hooks is not None:
            self._runtime.hooks.session = session
        self._agent_name = session.meta.agent or "build"
        checker = self._base_checker
        agent = self._runtime.agents.get(self._agent_name)
        if agent is not None and agent.overlay is not None:
            checker = checker.for_agent(agent.overlay)
        self._runtime.ctx.permission_checker = checker
        self._last_response = ""
        # Per-session advisor budget resets with the session.
        advisor = self._runtime.registry.get("advisor")
        if advisor is not None and hasattr(advisor, "reset_uses"):
            advisor.reset_uses()
        self._reload_history()
        self._status.session_name = session.name
        self._status.agent = self._agent_name
        stats = session_stats(self._store, session)
        self._status.input_tokens = stats.input_tokens
        self._status.output_tokens = stats.output_tokens
        self._status.cost_usd = stats.cost_usd
        self._status.context_used = 0
        self._invalidate()

    def _reload_history(self) -> None:
        """Rebuild the in-memory history from the session file."""
        self._history = [{"role": "system", "content": self._runtime.system_prompt}]
        self._history += self._store.load_for_model(self._session)

    def reload_history(self) -> None:
        """Public wrapper used by undo/redo/rewind/clear/compact handlers."""
        self._reload_history()

    def turn_busy(self) -> bool:
        """Whether a turn or plan loop is in flight (switching commands refuse)."""
        return self._turn_running() or self.loop_running()

    def loop_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """``/permissions``: switch the fallback mode on both checkers."""
        self._config.permissions.mode = mode
        self._base_checker.set_mode(mode)
        self._runtime.ctx.permission_checker.set_mode(mode)

    def add_note(self, note: str) -> None:
        """``/btw``: stash a side note prepended to the next submission."""
        self._pending_notes.append(note)

    def add_catalog_model(self, entry: Any) -> None:
        """``/models-add``: merge a custom entry into the in-memory catalog."""
        self._catalog = self.catalog.merge([entry])

    def set_cwd(self, path: Path) -> None:
        """Repoint ctx, permission checker, completers and statusline at ``path``.

        Used by worktree enter/exit; session perms and the permission mode
        carry over (a fresh checker, doom tracking resets).
        """
        path = Path(path)
        self._cwd = path
        ctx = self._runtime.ctx
        ctx.cwd = path
        base = PermissionChecker(
            self._config,
            session_perms=ctx.session_perms,
            mode=self._base_checker.mode,
            cwd=path,
        )
        self._base_checker = base
        agent = self._runtime.agents.get(self._agent_name)
        ctx.permission_checker = (
            base.for_agent(agent.overlay)
            if agent is not None and agent.overlay is not None
            else base
        )
        if self._runtime.hooks is not None:
            self._runtime.hooks.cwd = path
        self._file_lister = FileLister(path)
        self._completer = merge_completers(
            [
                PathCompleter(path, lister=self._file_lister),
                TriggerCompleter(self._file_lister, self._runtime.agents, self._runtime.skills),
            ]
        )
        if self._input_area is not None:
            self._input_area.completer = self._completer
        self._status.cwd = path
        self._invalidate()

    def attach_worktree(self, manager: Any, info: Any, original_cwd: Path) -> None:
        """Record an already-entered worktree (the CLI ``--worktree`` path)."""
        self._worktree_manager = manager
        self._worktree = info
        self._original_cwd = Path(original_cwd)

    def _enter_worktree(self, manager: Any, info: Any) -> None:
        """``/worktree``: switch the session cwd into a fresh worktree."""
        self.attach_worktree(manager, info, self._cwd)
        self.set_cwd(info.path)
        self._feed.info(f"worktree: {info.path} (branch {info.branch})")

    def _exit_worktree(self) -> None:
        """``/wt-exit``: return to the pre-worktree cwd and clear the state."""
        if self._original_cwd is not None:
            self.set_cwd(self._original_cwd)
        self._worktree = None
        self._worktree_manager = None
        self._original_cwd = None

    def queued_prompts(self) -> tuple[list[MessageContent], list[MessageContent]]:
        """(steered, queued) pending prompt snapshots for ``/queue``."""
        return (list(self._steer_queue._queue), list(self._input_queue._queue))

    # -- application construction -------------------------------------------

    def _toolbar(self) -> ANSI:
        """The statusline as prompt_toolkit formatted text (ANSI via Rich)."""
        text = render_statusline(self._status, self._theme, width=self._console.width)
        with self._console.capture() as capture:
            self._console.print(text, end="")
        return ANSI(capture.get())

    def _spawn(self, coro: Any) -> None:
        """Track a fire-and-forget submit task, surfacing any exception."""
        task: asyncio.Task[None] = asyncio.ensure_future(coro)
        self._pending.add(task)

        def _done(done: asyncio.Task[None]) -> None:
            self._pending.discard(done)
            if not done.cancelled() and (exc := done.exception()) is not None:
                self._feed.error(f"internal error: {exc}")

        task.add_done_callback(_done)

    def _build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()
        approval_pending = Condition(lambda: self._approval.is_pending)

        @kb.add("y", filter=approval_pending)
        def _approve_once(event: Any) -> None:
            self._approval.resolve(AllowOnce())

        @kb.add("a", filter=approval_pending)
        def _approve_always(event: Any) -> None:
            pending = self._approval.pending
            pattern = pending.target if pending is not None else "*"
            self._approval.resolve(AllowAlways(pattern=pattern))

        @kb.add("n", filter=approval_pending)
        def _deny(event: Any) -> None:
            self._approval.resolve(Deny())

        @kb.add("escape", filter=approval_pending)
        def _deny_escape(event: Any) -> None:
            self._approval.resolve(Deny())

        @kb.add("enter")
        def _enter(event: Any) -> None:
            if self._approval.is_pending:
                return  # y/a/n/ESC only while an approval is pending
            if self._handoff_future is not None and not self._handoff_future.done():
                text = event.current_buffer.text.strip()
                if text:
                    event.current_buffer.append_to_history()
                event.current_buffer.reset()
                self._handoff_future.set_result(text or None)
                return
            text = event.current_buffer.text
            if text.strip():
                event.current_buffer.append_to_history()
            event.current_buffer.reset()
            self._spawn(self._submit(text))

        @kb.add("escape", "enter")
        def _alt_enter(event: Any) -> None:
            if self._approval.is_pending:
                return
            text = event.current_buffer.text
            if text.strip():
                event.current_buffer.append_to_history()
            event.current_buffer.reset()
            self._spawn(self._submit(text, steer=True))

        @kb.add("c-c")
        def _ctrl_c(event: Any) -> None:
            if self._approval.is_pending:
                self._approval.resolve(Deny())
                return
            if self._handoff_future is not None and not self._handoff_future.done():
                self._handoff_future.set_result(None)  # decline the handoff
                return
            if self.cancel_turn():
                return
            if event.current_buffer.text:
                event.current_buffer.reset()
                return
            self._quit = True
            event.app.exit()

        @kb.add("c-d")
        def _ctrl_d(event: Any) -> None:
            if event.current_buffer.text:
                event.current_buffer.delete()
            else:
                self._quit = True
                event.app.exit()

        @kb.add("tab")
        def _tab(event: Any) -> None:
            if event.current_buffer.text:
                event.current_buffer.start_completion()
            else:
                self.cycle_agent()

        @kb.add("c-k")
        def _ctrl_k(event: Any) -> None:
            buf = event.current_buffer
            killed = kill_to_end_of_line(buf.text, buf.cursor_position)
            self._kill_ring.kill(killed)
            pos = buf.cursor_position
            buf.text = buf.text[:pos] + buf.text[pos + len(killed) :]
            buf.cursor_position = pos

        @kb.add("c-u")
        def _ctrl_u(event: Any) -> None:
            buf = event.current_buffer
            killed = kill_to_start_of_line(buf.text, buf.cursor_position)
            pos = buf.cursor_position
            buf.text = buf.text[: pos - len(killed)] + buf.text[pos:]
            buf.cursor_position = pos - len(killed)

        @kb.add("c-w")
        def _ctrl_w(event: Any) -> None:
            buf = event.current_buffer
            killed = kill_word_back(buf.text, buf.cursor_position)
            pos = buf.cursor_position
            buf.text = buf.text[: pos - len(killed)] + buf.text[pos:]
            buf.cursor_position = pos - len(killed)

        @kb.add("c-y")
        def _ctrl_y(event: Any) -> None:
            yanked = self._kill_ring.yank()
            if yanked:
                event.current_buffer.insert_text(yanked)

        @kb.add("c-g")
        def _ctrl_g(event: Any) -> None:
            buf = event.current_buffer

            async def _edit() -> None:
                edited = await open_in_editor(buf.text)
                if edited is None:
                    if not os.environ.get("EDITOR"):
                        self._feed.info("$EDITOR is not set")
                    return
                buf.text = edited
                buf.cursor_position = len(edited)

            self._spawn(_edit())

        return kb

    def _build_app(self, input: Input | None = None, output: Output | None = None) -> Application:
        draft = self._input_history.load_draft()
        self._input_area = TextArea(
            prompt="> ",
            text=draft,
            multiline=True,
            focusable=True,
            history=self._input_history,
            completer=self._completer,
        )
        toolbar = Window(
            content=FormattedTextControl(self._toolbar),
            height=1,
            dont_extend_height=True,
        )
        return Application(
            layout=Layout(HSplit([self._input_area, toolbar])),
            key_bindings=self._build_keybindings(),
            full_screen=False,
            mouse_support=False,
            paste_mode=True,  # bracketed paste: multiline pastes never submit
            input=input,
            output=output,
        )

    # -- the driver -----------------------------------------------------------

    async def run(self, *, input: Input | None = None, output: Output | None = None) -> int:
        """Run the interactive loop until quit; returns the exit code."""
        self._status.git_branch = await self._branch.get(self._cwd)
        self._app = self._build_app(input=input, output=output)
        self._runtime.ctx.approval_callback = self._request_approval
        self._runtime.ctx.extras["advisor_handoff"] = self._request_advisor_handoff
        await attach_mcp(self._runtime.registry, self._runtime.ctx)
        self._file_lister.prefetch()
        self._spinner_task = asyncio.ensure_future(self._spinner_loop())
        try:
            with patch_stdout():
                try:
                    await self._app.run_async()
                except (EOFError, KeyboardInterrupt):
                    self._quit = True  # e.g. stdin EOF on a non-tty
        finally:
            self._spinner_task.cancel()
            self._approval.cancel()
            self._runtime.ctx.approval_callback = None
            self._runtime.ctx.extras.pop("advisor_handoff", None)
            if self._handoff_future is not None and not self._handoff_future.done():
                self._handoff_future.set_result(None)
            self.cancel_turn()
            if self._turn_task is not None:
                await asyncio.gather(self._turn_task, return_exceptions=True)
            lsp = self._runtime.ctx.extras.get("lsp")
            if lsp is not None:
                await lsp.shutdown()
            mcp = self._runtime.ctx.extras.get(MCP_EXTRA)
            if mcp is not None:
                await mcp.shutdown()
            if self._input_area is not None and self._input_area.text.strip():
                self._input_history.save_draft(self._input_area.text)
            self._app = None
        self.print_totals()
        return EXIT_OK

    async def _spinner_loop(self) -> None:
        """Advance the spinner and refresh branch/statusline periodically."""
        while True:
            await asyncio.sleep(SPINNER_INTERVAL_S)
            if self._status.state is StatusLineState.RUNNING:
                self._status.spinner_frame += 1
            self._status.git_branch = await self._branch.get(self._cwd)
            if self._app is not None:
                self._app.invalidate()

    # -- submissions ----------------------------------------------------------

    async def _submit(self, text: str, *, steer: bool = False) -> None:
        """Route one submitted line: shell-outs, slash commands, or LLM input."""
        text = text.rstrip("\n")
        if not text.strip():
            return
        if text.startswith("!!"):
            await self._run_shell(text[2:].strip(), share_with_llm=True, steer=steer)
        elif text.startswith("!"):
            await self._run_shell(text[1:].strip(), share_with_llm=False, steer=steer)
        elif text.startswith("/"):
            await self.handle_command(text)
        elif text.startswith(".") and self._submit_persona(text, steer=steer):
            pass  # .persona <text>: handled (persona system-prompt overlay)
        else:
            if self.loop_running():
                self._feed.info("a plan loop is running — /loop stop first")
                return
            mentions, cleaned = parse_mentions(text, self._runtime.agents)
            invocable = {a.name for a in self._runtime.agents.subagents()}
            targets = [name for name in mentions if name in invocable]
            if targets and not self._turn_running():
                # Direct @agent dispatch: a side query run by the subagent.
                # While a turn runs, mentions keep the note behavior in
                # _prepare_message (the message queues as normal input).
                self._feed.user_message(text)
                self._turn_task = asyncio.ensure_future(
                    self._run_subagent_turn(targets[0], cleaned or text)
                )
                return
            prepared = self._prepare_message(text)
            echo = self._attachment_echo()
            content = self._with_attachments(prepared)
            if content is None:
                return  # modality error already rendered; attachments kept
            self._feed.user_message(text + echo)
            self._enqueue_or_start(content, steer=steer)

    def _with_attachments(self, text: str) -> MessageContent | None:
        """Attach pending attachments to ``text``; ``None`` = blocked (modality)."""
        attachments = self._attachments.list()
        if not attachments:
            return text
        error = check_modalities(attachments, self._runner.model, self.catalog)
        if error is not None:
            self._feed.error(error)
            return None
        parts: list[ContentPart] = [{"type": "text", "text": text}]
        parts += to_content_parts(attachments)
        self._attachments.clear()
        return parts

    def _attachment_echo(self) -> str:
        """``  📎 a.png, b.pdf`` suffix for the user-message echo."""
        attachments = self._attachments.list()
        if not attachments:
            return ""
        return "  📎 " + ", ".join(a.path.name for a in attachments)

    def _prepare_message(self, text: str) -> str:
        """Pull ``@path`` media refs into attachments, then apply ``@agent``
        routing notes and ``/btw`` pending notes."""
        before = len(self._attachments)
        text = extract_attachment_refs(text, self._cwd, self._attachments)
        for attachment in self._attachments.list()[before:]:
            self._feed.info(
                f"📎 {attachment.path.name} ({attachment.media_kind}, "
                f"{format_size(attachment.size_bytes)})"
            )
        mentions, cleaned = parse_mentions(text, self._runtime.agents)
        if mentions:
            names = ", ".join(f"@{name}" for name in mentions)
            text = (
                f"(The user mentioned agent(s) {names}; no subagent was "
                f"dispatched — treat the mention as context.)\n{cleaned}"
            )
        if self._pending_notes:
            notes = "\n".join(self._pending_notes)
            self._pending_notes = []
            text = f"{notes}\n\n{text}"
        return text

    def _submit_persona(self, text: str, *, steer: bool) -> bool:
        """``.persona <text>``: submit with a persona system-prompt overlay.

        Returns ``True`` when the input was consumed (known persona).
        """
        name, _, rest = text[1:].partition(" ")
        if name not in persona_names():
            return False
        try:
            body = load_text("prompts", f"personas/{name}.md", cwd=self._cwd)
        except FileNotFoundError:
            self._feed.error(f"persona not found: {name}")
            return True
        if not rest.strip():
            self._feed.error(f"usage: .{name} <text>")
            return True
        self._feed.user_message(text)
        self._enqueue_or_start(rest.strip(), steer=steer, overlay=body.strip())
        return True

    def _enqueue_or_start(
        self, text: MessageContent, *, steer: bool = False, overlay: str | None = None
    ) -> None:
        """Queue while a turn runs (5+5 limits), else start the turn now.

        ``overlay`` is an extra system message for this turn only (persona
        prefix). Queued while a turn runs, it degrades to an inline note (and
        is dropped when the message carries attachment parts).
        """
        if self._turn_running():
            if overlay is not None and isinstance(text, str):
                text = f"{overlay}\n\n{text}"
            queue = self._steer_queue if steer else self._input_queue
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull:
                which = "steer" if steer else "input"
                self._feed.info(f"{which} queue is full ({QUEUE_LIMIT})")
            self._sync_queue_status()
            return
        self._turn_task = asyncio.ensure_future(self._run_turn(text, overlay=overlay))

    async def _run_shell(self, cmd: str, *, share_with_llm: bool, steer: bool) -> None:
        """``!cmd`` shows output locally; ``!!cmd`` also feeds it to the LLM."""
        if not cmd:
            return
        self._feed.user_message(("!!" if share_with_llm else "!") + cmd)
        result = await run_proc(["bash", "-c", cmd], cwd=self._cwd, timeout=SHELL_TIMEOUT_S)
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        if result.timed_out:
            output += "\n(timed out)"
        is_error = result.exit_code != 0 or result.timed_out
        self._feed.tool_result("shell", output or "(no output)", is_error=is_error)
        if share_with_llm:
            self._enqueue_or_start(f"!{cmd}\n\n{output}", steer=steer)

    async def handle_command(self, text: str) -> None:
        """Dispatch a ``/command`` line through the slash registry."""
        parts = text[1:].split()
        query = parts[0] if parts else ""
        args = parts[1:]
        try:
            command = self._commands.match(query)
        except UnknownCommandError:
            self._feed.info(f"unknown command: /{query}")
            return
        except AmbiguousCommandError as e:
            matches = ", ".join(f"/{name}" for name in e.matches)
            self._feed.info(f"ambiguous command: /{query} ({matches})")
            return
        await command.handler(self, args)

    # -- inline permission prompt ----------------------------------------------

    def _invalidate(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    async def _request_approval(
        self, tool_name: str, args: dict[str, Any], reason: str
    ) -> ApprovalDecision:
        """``ctx.approval_callback``: inline y/a/n/ESC ask during a turn."""
        target = target_of(tool_name, args)
        if reason:
            self._feed.info(reason)  # doom-loop coach reasons land here too
        self._feed.permission(approval_prompt_text(tool_name, target))
        future = self._approval.request(tool_name, target, reason)
        self._status.state = StatusLineState.AWAITING_APPROVAL
        self._invalidate()
        self._spawn(self._notifier.approval_needed())
        try:
            return await future
        finally:
            self._approval.cancel()
            self._status.state = StatusLineState.RUNNING
            self._invalidate()

    # -- inline advisor handoff ----------------------------------------------

    async def _request_advisor_handoff(self, question: str, focus: str | None) -> str | None:
        """``ctx.extras['advisor_handoff']``: show the advisor's question and
        await the next Enter as the human's guidance (Ctrl-C declines)."""
        line = f"advisor asks: {question}"
        if focus:
            line += f" (focus: {focus})"
        self._feed.permission(line)
        self._feed.info("type your guidance, Enter to send — Ctrl-C declines")
        self._handoff_future = asyncio.get_running_loop().create_future()
        self._status.state = StatusLineState.AWAITING_APPROVAL
        self._invalidate()
        try:
            return await self._handoff_future
        finally:
            self._handoff_future = None
            if self._status.state is StatusLineState.AWAITING_APPROVAL:
                self._status.state = StatusLineState.RUNNING
            self._invalidate()

    # -- turns ------------------------------------------------------------------

    def _turn_running(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    def cancel_turn(self) -> bool:
        """Cancel the in-flight turn (Ctrl-C); ``True`` if one was cancelled."""
        if self._turn_running():
            self._turn_task.cancel()
            return True
        return False

    def _next_queued(self) -> MessageContent | None:
        """Pop the next pending message, steer queue (priority) first."""
        for queue in (self._steer_queue, self._input_queue):
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        return None

    def _sync_queue_status(self) -> None:
        self._status.queued = self._input_queue.qsize()
        self._status.steered = self._steer_queue.qsize()

    async def _run_turn(self, text: MessageContent, *, overlay: str | None = None) -> None:
        message: dict[str, Any] = {"role": "user", "content": text}
        self._history.append(message)
        self._store.append_message(self._session, message)
        run_history = self._history
        if overlay is not None:
            # Persona overlay: an extra system message for this turn only —
            # appended after the base system prompt, never persisted.
            run_history = [self._history[0], {"role": "system", "content": overlay}]
            run_history += self._history[1:]
        self._status.state = StatusLineState.RUNNING
        self._feed.stream_start()
        self._signals.emit(START)
        result = None
        cancelled = False
        try:
            result = await self._runner.run(run_history, on_event=self._on_event)
        except asyncio.CancelledError:
            cancelled = True
            self._feed.info("turn cancelled")
        except ProviderError:
            pass  # already rendered via the Error event
        finally:
            self._feed.stream_end()
            self._signals.emit(STOP)
            self._status.state = StatusLineState.IDLE
        if result is not None:
            if result.final_text:
                self._last_response = result.final_text
            totals = result.usage_totals
            self._status.input_tokens += totals.input_tokens
            self._status.output_tokens += totals.output_tokens
            self._status.cost_usd += totals.cost_usd
            # Rough proxy for the meter: tokens sent to the model this run.
            self._status.context_used = totals.input_tokens
        # Queued messages were persisted by the runner; rebuild from disk so
        # the next turn sees the same history the model saw.
        self._history = [{"role": "system", "content": self._runtime.system_prompt}]
        self._history += self._store.load_for_model(self._session)
        follow_up = None if cancelled or self._quit else self._next_queued()
        self._sync_queue_status()
        if follow_up is not None:
            self._feed.user_message(describe_content(follow_up))
            self._turn_task = asyncio.ensure_future(self._run_turn(follow_up))

    async def _run_subagent_turn(self, name: str, prompt: str) -> None:
        """A direct ``@agent`` submission: the subagent answers as a side
        query — the exchange is not persisted to the session."""
        self._status.state = StatusLineState.RUNNING
        outcome: SubagentOutcome | None = None
        try:
            outcome = await run_subagent(
                self._runtime.ctx,
                self._runtime.registry,
                self._runtime.agents,
                name=name,
                prompt=prompt,
                on_event=self._on_child_event,
            )
        except SubagentError as e:
            self._feed.error(str(e))
        except asyncio.CancelledError:
            self._feed.info("turn cancelled")
        finally:
            self._status.state = StatusLineState.IDLE
        if outcome is not None:
            self._feed.assistant_text(outcome.text)
            self._last_response = outcome.text
            self._status.input_tokens += outcome.input_tokens
            self._status.output_tokens += outcome.output_tokens
            self._status.cost_usd += outcome.cost_usd

    # -- plan loop / prompt chain ---------------------------------------------

    def start_loop(self, plan_path: Path, max_iterations: int) -> None:
        """``/loop``: run a plan loop over this session in the background."""

        async def run_iteration(prompt: str) -> str:
            message: dict[str, Any] = {"role": "user", "content": prompt}
            self._history.append(message)
            self._store.append_message(self._session, message)
            self._feed.stream_start()
            result = await self._runner.run(list(self._history), on_event=self._on_event)
            # The runner persisted everything; rebuild from disk like _run_turn.
            self._history = [{"role": "system", "content": self._runtime.system_prompt}]
            self._history += self._store.load_for_model(self._session)
            return result.final_text

        async def _loop() -> None:
            self._status.state = StatusLineState.RUNNING
            try:
                result = await run_plan_loop(
                    run_iteration,
                    plan_path,
                    cwd=self._cwd,
                    max_iterations=max_iterations,
                    on_progress=self._feed.info,
                )
            except asyncio.CancelledError:
                self._feed.info("loop stopped")
                return
            except ProviderError as e:
                self._feed.error(f"loop failed: {e}")
                return
            finally:
                self._feed.stream_end()
                self._status.state = StatusLineState.IDLE
            if result.stop_reason == "done":
                self._feed.info(f"loop done: plan complete after {result.iterations} iteration(s)")
            elif result.stop_reason == "max_iterations":
                self._feed.info(
                    f"loop stopped: max iterations ({result.iterations}) — "
                    f"remaining: {', '.join(result.remaining)}"
                )
            else:
                self._feed.error(f"loop failed: {result.error}")

        self._loop_task = asyncio.ensure_future(_loop())

    def start_chain(self, topic: str) -> None:
        """``/chain``: brainstorm → plan → code → review as the turn task."""
        self._turn_task = asyncio.ensure_future(self._run_chain(topic))

    async def _run_chain(self, topic: str) -> None:
        """Run the chain; each phase renders as it completes (not persisted)."""

        def factory() -> AgentRunner:
            # Fresh runner per phase; no session binding → chain is a side
            # computation rendered to the feed, not written to the session.
            return AgentRunner(self._runner.provider, self._runtime.registry, self._runtime.ctx)

        def on_phase(phase: str, output: str) -> None:
            self._feed.assistant_text(f"## {phase}\n\n{output}")

        self._status.state = StatusLineState.RUNNING
        try:
            result = await run_chain(
                factory,
                topic,
                system_prompt=self._runtime.system_prompt,
                cwd=self._cwd,
                on_phase=on_phase,
            )
        except asyncio.CancelledError:
            self._feed.info("turn cancelled")
            return
        except ProviderError as e:
            self._feed.error(f"chain failed: {e}")
            return
        finally:
            self._status.state = StatusLineState.IDLE
        if result.final:
            self._last_response = result.final

    def _on_event(self, event: Any) -> None:
        """Runner event → feed rendering (+ statusline state)."""
        if isinstance(event, Token):
            self._feed.stream_token(event.text)
        elif isinstance(event, Reasoning):
            self._feed.stream_token(event.text, thinking=True)
        elif isinstance(event, ToolCall):
            self._feed.tool_call(event.name, " ".join(event.arguments.split()))
        elif isinstance(event, ToolResult):
            self._feed.tool_result(event.name, event.content, event.is_error)
        elif isinstance(event, Error):
            self._feed.error(event.message)
            self._spawn(self._notifier.error())
        elif isinstance(event, Retrying):
            self._feed.retrying(event.attempt, event.delay)
        elif isinstance(event, Done):
            self._feed.stream_end()
            self._spawn(self._notifier.task_finish())

    def _on_child_event(self, agent: str, event: Any) -> None:
        """Subagent runner event → feed rendering, prefixed with the agent.

        Token/Reasoning are skipped: they would interleave with the parent's
        own stream.
        """
        if isinstance(event, ToolCall):
            self._feed.tool_call(f"{agent}/{event.name}", " ".join(event.arguments.split()))
        elif isinstance(event, ToolResult):
            self._feed.tool_result(f"{agent}/{event.name}", event.content, event.is_error)
        elif isinstance(event, Error):
            self._feed.error(f"{agent}: {event.message}")
        elif isinstance(event, Retrying):
            self._feed.retrying(event.attempt, event.delay)
        elif isinstance(event, Done):
            self._feed.info(f"{agent} finished ({event.stop_reason}, {event.turns} turn(s))")

    # -- agents / totals ----------------------------------------------------------

    def cycle_agent(self) -> str:
        """Tab on empty input: switch to the next primary agent."""
        self._agent_name = self._runtime.agents.cycle(self._agent_name)
        agent = self._runtime.agents.get(self._agent_name)
        checker = self._base_checker
        if agent is not None and agent.overlay is not None:
            checker = checker.for_agent(agent.overlay)
        self._runtime.ctx.permission_checker = checker
        self._status.agent = self._agent_name
        self._feed.info(f"agent: {self._agent_name}")
        return self._agent_name

    def print_totals(self) -> None:
        """Cost-reporting requirement: full-session totals on exit."""
        stats = session_stats(self._store, self._session)
        self._feed.info(
            f"Session {self._session.name}: tokens {stats.input_tokens} in / "
            f"{stats.output_tokens} out · cost ${stats.cost_usd:.4f}"
        )
