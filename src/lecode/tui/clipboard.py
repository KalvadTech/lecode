"""Clipboard integration (OSC 52 → pbcopy → xclip) and OSC 8 hyperlinks.

Copy order: the OSC 52 escape when stdout is a terminal (works over SSH and
in tmux; silently unsupported terminals just ignore it), then ``pbcopy``
(macOS), then ``xclip`` (Linux). Everything is best-effort and never raises.
"""

from __future__ import annotations

import base64
import sys

from lecode.extras.proc import run_proc

_CLIPBOARD_TIMEOUT_S = 5.0


def osc52_sequence(text: str) -> str:
    """The OSC 52 escape writing ``text`` to the terminal clipboard."""
    payload = base64.b64encode(text.encode()).decode()
    return f"\x1b]52;c;{payload}\x1b\\"


def osc8_link(url: str, text: str, *, no_color: bool = False) -> str:
    """An OSC 8 hyperlink; plain ``text`` when colors are disabled."""
    if no_color:
        return text
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


async def copy_to_clipboard(text: str, *, tty: bool | None = None) -> bool:
    """Copy ``text`` to the system clipboard; ``True`` on the first success."""
    if tty is None:
        tty = sys.stdout.isatty()
    if tty:
        try:
            sys.stdout.write(osc52_sequence(text))
            sys.stdout.flush()
            return True
        except OSError:
            pass  # fall through to the external tools
    for argv in (["pbcopy"], ["xclip", "-selection", "clipboard"]):
        try:
            result = await run_proc(argv, input=text, timeout=_CLIPBOARD_TIMEOUT_S)
        except OSError:
            continue
        if result.exit_code == 0:
            return True
    return False
