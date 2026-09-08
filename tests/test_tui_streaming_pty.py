"""Regression tests: the live feed must survive redraws on a real tty.

Runs the full TuiApp against a pseudo-terminal with a live pyte terminal
emulator on the other end (answering cursor-position requests with the true
cursor position, like a real terminal). The fake provider streams slowly so
spinner/statusline redraws interleave with the streamed tokens — the exact
conditions under which prompt_toolkit's patch_stdout erase/redraw cycle used
to erase the in-progress answer line.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import struct
import sys
import termios
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

pyte = pytest.importorskip("pyte")

from prompt_toolkit.data_structures import Size  # noqa: E402
from prompt_toolkit.input.vt100 import Vt100Input  # noqa: E402
from prompt_toolkit.output import ColorDepth  # noqa: E402
from prompt_toolkit.output.vt100 import Vt100_Output  # noqa: E402
from rich.console import Console  # noqa: E402
from tests.fakes import FakeProvider, sample_catalog  # noqa: E402

from lecode.agent.builder import build_runtime  # noqa: E402
from lecode.config.models import Config, PermissionRule  # noqa: E402
from lecode.session.storage import SessionStore  # noqa: E402
from lecode.tui.app import TuiApp  # noqa: E402

ROWS, COLS = 50, 100

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "openpty"), reason="needs a pty"
)


class SlowProvider(FakeProvider):
    """FakeProvider that streams slowly, like a real model."""

    def __init__(self, script: list[dict[str, Any]], *, delay: float = 0.3) -> None:
        super().__init__(script)
        self._delay = delay

    async def _stream(self, entry):  # type: ignore[override]
        for event in [e async for e in super()._stream(entry)]:
            await asyncio.sleep(self._delay)
            yield event


def _screen_lines(screen: pyte.HistoryScreen) -> list[str]:
    # History entries are pyte line buffers (position → Char), not strings.
    def render(line) -> str:
        if isinstance(line, str):
            return line.rstrip()
        if not line:
            return ""
        return "".join(line[i].data for i in range(max(line) + 1)).rstrip()

    return [render(ln) for ln in screen.history.top] + [
        screen.display[i].rstrip() for i in range(ROWS)
    ]


async def _wait_for(lines: Callable[[], list[str]], needle: str, timeout: float = 15) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if any(needle in ln for ln in lines()):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {needle!r}:\n" + "\n".join(lines()))


Driver = Callable[[int, "pyte.HistoryScreen"], Coroutine[Any, Any, list[str]]]


def _prompt_then_quit(prompt: str) -> Driver:
    """Driver: send ``prompt``, wait for the turn stats line, then /quit."""

    async def drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
        lines = lambda: _screen_lines(screen)  # noqa: E731
        await asyncio.sleep(0.8)
        os.write(master, prompt.encode() + b"\r")
        await _wait_for(lines, "answer:")
        os.write(master, b"/quit\r")
        return lines()

    return drive


async def _run_pty_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: list[dict[str, Any]],
    driver: Driver,
    *,
    configure: Callable[[Config], None] | None = None,
    delay: float = 0.3,
    color_depth: ColorDepth | None = None,
) -> list[str]:
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("TERM", "xterm-256color")

    saved_stdout = os.dup(1)
    os.dup2(slave, 1)
    real_stdout = os.fdopen(os.dup(1), "w")
    monkeypatch.setattr("sys.stdout", real_stdout)

    screen = pyte.HistoryScreen(COLS, ROWS, history=500)
    stream = pyte.Stream(screen)
    stop_reading = False

    def reader_thread() -> None:
        while not stop_reading:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                return
            if not chunk:
                return
            stream.feed(chunk.decode("utf-8", errors="replace"))
            # Answer cursor-position requests with the real cursor position.
            for _ in range(chunk.count(b"\x1b[6n")):
                os.write(
                    master,
                    f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}R".encode(),
                )

    thread = threading.Thread(target=reader_thread, daemon=True)
    thread.start()
    try:
        config = Config()
        config.notifications.enabled = False
        if configure is not None:
            configure(config)
        store = SessionStore()
        session = store.create("pty-test", tmp_path, model=config.llm.model)
        runtime = build_runtime(config, tmp_path, session=session, store=store)
        provider = SlowProvider(script, delay=delay)
        console = Console(force_terminal=True, width=COLS)
        app = TuiApp(
            config,
            runtime,
            provider,
            session,
            store,
            console=console,
            catalog=sample_catalog(),
        )
        inp = Vt100Input(os.fdopen(os.dup(slave), "r"))
        out = Vt100_Output(
            real_stdout, lambda: Size(rows=ROWS, columns=COLS), default_color_depth=color_depth
        )

        # patch_stdout's proxy binds the current AppSession's output at
        # creation, and its flush thread always resolves the default session.
        # Earlier tests may have cached a pytest-capture-wrapped output on the
        # default session — point it at our pty output so the flush path is
        # the production one (erase → print → redraw through run_in_terminal).
        from prompt_toolkit.application.current import get_app_session

        session_pt = get_app_session()
        saved_output = session_pt._output
        session_pt._output = out
        try:
            task = asyncio.ensure_future(app.run(input=inp, output=out))
            # Gate keystrokes on the first real render: prompt_toolkit only
            # enters raw mode (and flushes pre-start input) as the app takes
            # over the terminal — on a slow runner a byte written after a
            # fixed sleep can land before that and be echoed by the line
            # discipline instead of delivered to the input.
            await _wait_for(lambda: _screen_lines(screen), "dir:")
            driven = asyncio.ensure_future(driver(master, screen))
            await asyncio.wait_for(task, timeout=30)
            return await driven
        finally:
            session_pt._output = saved_output
    finally:
        stop_reading = True
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


async def test_streamed_answer_survives_redraws(tmp_path, monkeypatch):
    """The answer must be on screen/scrollback intact after the turn ends."""
    script = [
        {
            "reasoning": ["thinking..."],
            "text": ["Par", "is."],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    ]
    lines = await _run_pty_app(
        tmp_path, monkeypatch, script, _prompt_then_quit("what is the capital of France?")
    )
    visible = [ln for ln in lines if ln.strip()]
    dump = "\n".join(visible)
    assert any(ln == "Paris." for ln in visible), (
        "streamed answer missing from the terminal:\n" + dump
    )
    arrow = [i for i, ln in enumerate(visible) if "→" in ln and "round 1" in ln]
    assert arrow, "no LLM-call line found:\n" + dump
    assert visible[arrow[0] + 1] == "Paris."  # no eaten-line gap after the call line


async def test_long_answer_survives_intact(tmp_path, monkeypatch):
    """A multi-paragraph streamed answer: every line lands, none eaten/duplicated."""
    markers = [f"MARKER{i:02d}" for i in range(30)]
    chunks = [m + "\n\n" for m in markers]  # blank line: separate paragraphs
    script = [{"text": chunks, "usage": {"input_tokens": 10, "output_tokens": 60}}]
    lines = await _run_pty_app(
        tmp_path, monkeypatch, script, _prompt_then_quit("long answer please"), delay=0.02
    )
    dump = "\n".join(lines)
    for marker in markers:
        assert dump.count(marker) == 1, f"{marker} missing or duplicated:\n{dump}"


async def test_tool_round_then_answer(tmp_path, monkeypatch):
    """A tool call round followed by a final answer: all of it on screen."""
    script = [
        {
            "tool_calls": [{"name": "bash", "arguments": '{"command": "echo pty-tool-out"}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {"text": ["all done here"], "usage": {"input_tokens": 20, "output_tokens": 3}},
    ]
    lines = await _run_pty_app(
        tmp_path, monkeypatch, script, _prompt_then_quit("run something"), delay=0.05
    )
    dump = "\n".join(lines)
    assert "bash" in dump and "pty-tool-out" in dump, "tool call/result missing:\n" + dump
    assert "all done here" in dump, "final answer missing:\n" + dump


async def test_slash_menu_renders_and_no_match_row(tmp_path, monkeypatch):
    """Typing '/mod' shows the dropdown on the real terminal (several rows at
    once, not clipped); an unknown prefix shows the inert 'No matching
    commands' row and Enter still submits."""
    script = [{"text": ["ok"], "usage": {"input_tokens": 10, "output_tokens": 2}}]

    async def drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
        lines = lambda: _screen_lines(screen)  # noqa: E731
        await asyncio.sleep(0.8)
        os.write(master, b"/mod")
        await _wait_for(lines, "Switch permission mode")  # menu row 1 of /mod
        menu = "\n".join(lines())
        assert "List available models" in menu, menu  # row 4 visible, not clipped
        os.write(master, b"\x15/zzzz")  # clear line, unknown command prefix
        await _wait_for(lines, "No matching commands")
        snapshot = [ln for ln in lines() if "No matching commands" in ln]
        os.write(master, b"\r")
        await _wait_for(lines, "unknown command: /zzzz")
        os.write(master, b"\x15/quit\r")
        return [*lines(), "<captured>", *snapshot]

    lines = await _run_pty_app(tmp_path, monkeypatch, script, drive)
    dump = "\n".join(lines)
    assert "No matching commands" in dump, "no-match row never rendered:\n" + dump
    assert "unknown command: /zzzz" in dump, "enter did not submit normally:\n" + dump


async def test_at_files_use_the_themed_panel(tmp_path, monkeypatch):
    """@ file completions render in the same themed panel as /commands."""
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")

    async def drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
        visible = lambda: screen.display  # noqa: E731
        try:
            await asyncio.sleep(0.8)
            os.write(master, b"@not")
            await _wait_for(visible, "context  1 match")
            heading = next(i for i, line in enumerate(screen.display) if "context" in line)
            row = heading + 1
            assert "notes.md" in screen.display[row]
            x = screen.display[row].index("notes.md")
            meta_x = screen.display[row].index("file")
            assert screen.buffer[row][x].fg == "ece7f7"  # theme.text
            assert screen.buffer[row][meta_x].fg == "8a80a3"  # theme.muted
            assert screen.buffer[row][x].bg == "1c162b"  # panel background
            assert screen.display[heading - 1].startswith("┌")
            os.write(master, b"\x15/quit\r")
            return screen.display[:]
        finally:
            os.write(master, b"\x03\x04")

    await _run_pty_app(tmp_path, monkeypatch, [], drive, color_depth=ColorDepth.DEPTH_24_BIT)


async def test_slash_panel_is_anchored_compact_and_styled(tmp_path, monkeypatch):
    async def drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
        visible = lambda: screen.display  # noqa: E731
        try:
            await asyncio.sleep(0.8)
            os.write(master, b"/mod")
            await _wait_for(visible, "commands  5 matches")
            heading = next(i for i, line in enumerate(screen.display) if "commands" in line)
            footer = next(i for i, line in enumerate(screen.display) if "Enter/Tab" in line)
            assert footer == heading + 6, "exactly five command rows, no padding"
            assert "↑↓ navigate  Enter/Tab select  Esc close" in screen.display[footer]
            assert screen.display[heading - 1].startswith("┌")
            assert screen.display[footer + 1].startswith("└")
            assert screen.display[footer + 2].strip(), "statusline follows without a blank gap"
            row = heading + 1
            command_x = screen.display[row].index("mode")
            meta_x = screen.display[row].index("Switch permission mode")
            assert screen.buffer[row][command_x].fg == "ece7f7"  # theme.text
            assert screen.buffer[row][meta_x].fg == "8a80a3"
            assert screen.buffer[row][command_x].bg == "1c162b"
            assert screen.buffer[heading - 1][0].fg == "8a80a3"

            os.write(master, b"\x1b[B\x1b[B")
            await _wait_for(visible, "> model")
            selected = next(i for i, line in enumerate(screen.display) if "> model" in line)
            x = screen.display[selected].index("model")
            assert screen.buffer[selected][x].bold
            assert screen.buffer[selected][x].fg == "a78bfa"
            assert screen.buffer[selected][x].bg == "35264f"
            meta_x = screen.display[selected].index("Switch the model")
            assert screen.buffer[selected][meta_x].fg == "8a80a3"
            assert not screen.buffer[selected][meta_x].bold
            assert screen.buffer[selected][COLS - 2].bg == "35264f"
            assert screen.display[heading - 1].startswith("┌"), "panel must not follow cursor"

            os.write(master, b"\x15/cop")
            await _wait_for(visible, "commands  1 match")
            heading = next(i for i, line in enumerate(screen.display) if "commands" in line)
            assert "copy" in screen.display[heading + 1]
            assert "Enter/Tab" in screen.display[heading + 2]
            assert screen.display[heading + 4].strip(), "one match leaves no eight-row gap"
            return screen.display[:]
        finally:
            os.write(master, b"\x03\x04")

    await _run_pty_app(tmp_path, monkeypatch, [], drive, color_depth=ColorDepth.DEPTH_24_BIT)


@pytest.mark.parametrize("columns", [60, 100])
async def test_slash_panel_caps_rows_scrolls_and_collapses(tmp_path, monkeypatch, columns):
    monkeypatch.setattr(sys.modules[__name__], "COLS", columns)

    async def drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
        visible = lambda: screen.display  # noqa: E731
        try:
            await asyncio.sleep(0.8)
            os.write(master, b"/")
            await _wait_for(visible, "commands")
            heading = next(i for i, line in enumerate(screen.display) if "commands" in line)
            footer = next(i for i, line in enumerate(screen.display) if "Enter/Tab" in line)
            assert footer == heading + 9, "only eight matches visible"
            assert "Esc close" in screen.display[footer], "footer fits narrow terminals"
            os.write(master, b"\x1b[A")  # Up from no selection wraps to the last match.
            await _wait_for(visible, "> wt-merge")
            assert "Merge the worktree back" in "\n".join(screen.display)
            assert "commands" in screen.display[heading]
            assert "Enter/Tab" in screen.display[footer]

            os.write(master, b"\x1b")
            await _wait_for(visible, "│> / ")  # wait for Escape's Alt-sequence timeout
            assert not any("commands" in line or "Enter/Tab" in line for line in screen.display)
            prompt = next(i for i, line in enumerate(screen.display) if "> /" in line)
            assert "dir:" in screen.display[prompt + 2], "closed panel reserves no space"

            os.write(master, b"\x15/zzzz")
            await _wait_for(visible, "commands  0 matches")
            heading = next(i for i, line in enumerate(screen.display) if "commands  0" in line)
            assert "No matching commands" in screen.display[heading + 1]
            assert "Enter/Tab" in screen.display[heading + 2]
            assert "dir:" in screen.display[heading + 4], "empty state has only one row"
            return screen.display[:]
        finally:
            os.write(master, b"\x03\x04")

    await _run_pty_app(tmp_path, monkeypatch, [], drive)


async def test_permission_prompt_approves_tool(tmp_path, monkeypatch):
    """An ask rule prompts inline; pressing y runs the tool and shows output."""

    def configure(config: Config) -> None:
        config.permissions.rules.ask["bash"] = [PermissionRule(pattern="*")]

    script = [
        {
            "tool_calls": [{"name": "bash", "arguments": '{"command": "echo pty-approved-out"}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {"text": ["approved done"], "usage": {"input_tokens": 20, "output_tokens": 3}},
    ]

    async def drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
        lines = lambda: _screen_lines(screen)  # noqa: E731
        await asyncio.sleep(0.8)
        os.write(master, b"run it\r")
        await _wait_for(lines, "allow bash")
        os.write(master, b"y")
        await _wait_for(lines, "answer:")
        os.write(master, b"/quit\r")
        return lines()

    lines = await _run_pty_app(
        tmp_path, monkeypatch, script, drive, configure=configure, delay=0.05
    )
    dump = "\n".join(lines)
    assert "allow bash" in dump, "approval prompt never shown:\n" + dump
    assert "pty-approved-out" in dump, "approved tool output missing:\n" + dump
    assert "approved done" in dump, "final answer missing:\n" + dump
