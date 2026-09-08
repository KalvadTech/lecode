# MCP servers

lecode connects to [MCP](https://modelcontextprotocol.io) servers with the
official Python SDK, over **stdio**, **streamable HTTP**, and **SSE** (the
legacy HTTP transport). Their tools show up as first-class lecode tools named
`mcp:<server>:<tool>` and go through the permission system like everything
else.

## Auto-configured servers

| server | when | transport |
|---|---|---|
| **Exa** (web search) | on by default; needs `EXA_API_KEY` in the environment | http `https://mcp.exa.ai/mcp` |
| **context7** (docs lookup) | off by default; set `enable_context7 = true` | http `https://mcp.context7.com/mcp` |

Exa authentication: the key is sent as the `?exaApiKey=` query parameter and
as an `Authorization: Bearer` header. Without `EXA_API_KEY`, Exa is skipped
silently.

## Configured servers

```toml
[mcp.servers.filesystem]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { }                # merged over a safe default environment
timeout_s = 30           # per-call timeout (default 30)
enabled = true

[mcp.servers.internal]
transport = "http"       # or "sse" for the legacy transport
url = "https://mcp.internal.example/mcp"
headers = { Authorization = "Bearer …" }

[mcp.servers.remote]
transport = "sse"
url = "https://mcp.remote.example/sse"
oauth = true             # interactive OAuth authorization-code flow
```

A `[mcp.servers]` entry named `exa` or `context7` replaces the auto-configured
definition.

## OAuth

Remote servers (`http` / `sse`) can set `oauth = true` instead of a static
`Authorization` header. On the first connect the SDK runs the
authorization-code flow with PKCE: your browser opens at the server's
authorization page (the URL is also printed in the feed / on stderr, so it
works over SSH), and an ephemeral localhost listener captures the redirect.

Tokens and the dynamic client registration persist in
`<config_dir>/mcp_auth/<server>.json` (mode `0600`), so headless runs and
restarts need no interaction — expired tokens are refreshed automatically.
A revoked or expired grant that fails with 401 is dropped and re-authorized
on the next call.

If a server with `oauth = true` connects where no browser flow can complete,
that server fails with an actionable error while everything else starts
normally — authorize it once interactively with `lecode` + `/mcp login <name>`.

## Permissions

Tools from the read-only-ish servers **exa**, **context7**, and **grep-app**
are read-equivalent: allowed in both `yolo` and `readonly` modes. Every
other MCP tool follows the mode fallback (allowed in `yolo`, denied in
`readonly`) and can be matched by rules:

```toml
# rule keys are exact tool names; the pattern matches the call target
# (for MCP tools the target is the canonical mcp:<server>:<tool> name)
[[permissions.rules.allow."mcp:internal:query"]]
pattern = "*"
```

Rule targets for MCP tools are the canonical `mcp:<server>:<tool>` name.

## Failure behavior

- Connecting happens at session start with a ~10s per-server budget; a broken
  or slow server never blocks the others or startup.
- A failed tool call gets exactly **one** reconnect attempt (fresh session),
  then an error result.
- Everything is fail-open: MCP trouble produces error tool results, never
  agent crashes.

## `/mcp`

```
/mcp                  per-server state: connected (n tools) / failed / disabled
/mcp tools <name>     list one server's tools
/mcp reconnect <name> drop and re-establish a server session
/mcp login <name>     delete stored OAuth tokens, re-run the browser flow
/mcp logout <name>    delete stored OAuth tokens, reconnect without them
```
