"""rtk output compaction for the bash tool.

``rtk rewrite`` compacts noisy command output (stdin → stdout). Every failure
mode — binary missing, non-zero exit, timeout — is fail-open: the original
text is returned unchanged.
"""

from __future__ import annotations

import shutil

from lecode.extras.proc import run_proc

RTK_TIMEOUT_S = 5.0

#: Resolved once at import: the rtk binary path, or None.
RTK_PATH = shutil.which("rtk")


async def compact_output(text: str, *, rtk_path: str | None = None) -> str:
    """Run ``rtk rewrite`` on ``text``; fail-open to the original on any error."""
    binary = rtk_path if rtk_path is not None else RTK_PATH
    if binary is None or not text:
        return text
    try:
        result = await run_proc([binary, "rewrite"], input=text, timeout=RTK_TIMEOUT_S)
    except OSError:
        return text
    if result.timed_out or result.exit_code != 0 or not result.stdout.strip():
        return text
    return result.stdout
