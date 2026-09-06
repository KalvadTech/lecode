"""Regression test: streamed answers must survive app redraws on a real tty.

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
from pathlib import Path

import pytest

pyte = pytest.importorskip("pyte")

from prompt_toolkit.data_structures import Size  # noqa: E402
from prompt_toolkit.input.vt100 import Vt100Input  # noqa: E402
from prompt_toolkit.output.vt100 import Vt100_Output  # noqa: E402
from rich.console import Console  # noqa: E402
from tests.fakes import FakeProvider, sample_catalog  # noqa: E402

from lecode.agent.builder import build_runtime  # noqa: E402
from lecode.config.models import Config  # noqa: E402
from lecode.session.storage import SessionStore  # noqa: E402
from lecode.tui.app import TuiApp  # noqa: E402

ROWS, COLS = 50, 100

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(os, "openpty"), reason="needs a pty"
)


class SlowProvider(FakeProvider):
    """FakeProvider that streams slowly, like a real model."""

    async def _stream(self, entry):  # type: ignore[override]
        for event in [e async for e in super()._stream(entry)]:
            await asyncio.sleep(0.3)
            yield event


async def _drive(master: int, screen: pyte.HistoryScreen) -> list[str]:
    """Send a prompt, wait for the answer round to finish, then quit."""

    def lines() -> list[str]:
        return [ln.rstrip() for ln in screen.history.top] + [
            screen.display[i].rstrip() for i in range(ROWS)
        ]

    await asyncio.sleep(0.8)
    os.write(master, b"what is the capital of France?\r")
    deadline = asyncio.get_running_loop().time() + 15
    while asyncio.get_running_loop().time() < deadline:
        if any("answer:" in ln for ln in lines()):
            break
        await asyncio.sleep(0.1)
    os.write(master, b"/quit\r")
    return lines()


async def _run_pty_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
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
        store = SessionStore()
        session = store.create("pty-test", tmp_path, model=config.llm.model)
        runtime = build_runtime(config, tmp_path, session=session, store=store)
        provider = SlowProvider(
            [
                {
                    "reasoning": ["thinking..."],
                    "text": ["Par", "is."],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ]
        )
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
        out = Vt100_Output(real_stdout, lambda: Size(rows=ROWS, columns=COLS))

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
            driver = asyncio.ensure_future(_drive(master, screen))
            await asyncio.wait_for(task, timeout=20)
            return await driver
        finally:
            session_pt._output = saved_output
    finally:
        stop_reading = True
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


async def test_streamed_answer_survives_redraws(tmp_path, monkeypatch):
    """The answer must be on screen/scrollback intact after the turn ends."""
    lines = await _run_pty_app(tmp_path, monkeypatch)
    visible = [ln for ln in lines if ln.strip()]
    dump = "\n".join(visible)
    assert any(ln == "Paris." for ln in visible), (
        "streamed answer missing from the terminal:\n" + dump
    )
    arrow = [i for i, ln in enumerate(visible) if "→" in ln and "round 1" in ln]
    assert arrow, "no LLM-call line found:\n" + dump
    assert visible[arrow[0] + 1] == "Paris."  # no eaten-line gap after the call line
