# MCP servers

lecode connects to [MCP](https://modelcontextprotocol.io) servers with the
official Python SDK, over **stdio** and **streamable HTTP**. Their tools show
up as first-class lecode tools named `mcp:<server>:<tool>` and go through the
permission system like everything else.

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
transport = "http"
url = "https://mcp.internal.example/mcp"
headers = { Authorization = "Bearer …" }
```

A `[mcp.servers]` entry named `exa` or `context7` replaces the auto-configured
definition.

OAuth for HTTP servers is intentionally not wired up (it needs an interactive
browser flow); pass bearer tokens via `headers` instead.

## Permissions

Tools from the read-only-ish servers **exa**, **context7**, and **grep-app**
are read-equivalent: auto-allowed in `standard` and `readonly` modes. Every
other MCP tool follows the mode fallback (`ask` in standard, denied in
readonly) and can be matched by rules:

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
```
