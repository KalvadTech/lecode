"""rtk command rewriting for the bash tool.

``rtk rewrite <command>`` maps a shell command to its token-optimized rtk
proxy (``ls -la`` → ``rtk ls -la``, ``git status`` → ``rtk git status``,
``pytest`` → ``rtk pytest`` …). The rewritten string is printed on stdout —
without trailing newline — when an equivalent exists; stdout stays empty
otherwise. The exit code is not meaningful across rtk versions (0.46 exits
3 on a rewrite, 1 on no-equivalent), so success is "non-empty stdout".

Every failure mode — binary missing, empty stdout, timeout — is fail-open:
the original command is returned unchanged.
"""

from __future__ import annotations

import shutil

from lecode.extras.proc import run_proc

RTK_TIMEOUT_S = 5.0


def default_path() -> str | None:
    """The rtk binary path, resolved per call (PATH may change after import)."""
    return shutil.which("rtk")


async def rewrite_command(command: str, *, rtk_path: str | None = None) -> str:
    """Rewrite ``command`` to its rtk-proxy equivalent; fail-open to the original."""
    binary = rtk_path if rtk_path is not None else default_path()
    if binary is None or not command.strip():
        return command
    try:
        result = await run_proc([binary, "rewrite", command], timeout=RTK_TIMEOUT_S)
    except OSError:
        return command
    if result.timed_out:
        return command
    rewritten = result.stdout.strip()
    return rewritten or command
