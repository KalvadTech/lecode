"""A tiny MCP server (stdio) for tests: echo / boom / crash / flaky tools.

- ``echo``  — returns ``echo: <text>``.
- ``boom``  — raises inside the tool (isError result, connection stays up).
- ``crash`` — kills the server process mid-call (connection-level failure).
- ``flaky`` — kills the process on its first-ever call (state file in
  ``MOCK_MCP_STATE`` survives the restart), succeeds after.
"""

from __future__ import annotations

import os
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("mock")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def boom() -> str:
    """Always fails at the tool level."""
    raise RuntimeError("boom")


@server.tool()
def crash() -> str:
    """Kill the server process mid-call."""
    os._exit(1)


@server.tool()
def flaky() -> str:
    """Crash on the first call across restarts; succeed afterwards."""
    state_path = os.environ["MOCK_MCP_STATE"]
    try:
        with open(state_path, encoding="utf-8") as f:
            calls = int(f.read() or "0")
    except OSError:
        calls = 0
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(str(calls + 1))
    if calls == 0:
        os._exit(1)
    return f"ok after {calls + 1} calls"


if __name__ == "__main__":
    sys.exit(server.run("stdio") or 0)
