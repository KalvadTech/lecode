# MCP servers

lecode connects to [MCP](https://modelcontextprotocol.io) servers with the
official Python SDK, over **stdio**, **streamable HTTP**, and **SSE** (the
legacy HTTP transport). Their tools show up as first-class lecode tools named
`mcp:<server>:<tool>` and go through the permission system like everything
else.

There are no built-in servers; everything comes from `[mcp.servers]`. To add
a hosted server like Exa or context7, configure it yourself:

```toml
[mcp.servers.exa]
transport = "http"
url = "https://mcp.exa.ai/mcp?exaApiKey=<key>"
headers = { Authorization = "Bearer <key>" }

[mcp.servers.context7]
transport = "http"
url = "https://mcp.context7.com/mcp"
```

## Web research with Parallel

To look up current library documentation or read web pages while coding, add
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to your existing `~/.config/lecode/config.toml` (or the project's
`.lecode/config.toml`). For a new file, put `schema_version = 2` on the first
line, before any tables.

```toml
[mcp.servers.parallel]
transport = "http"
url = "https://search.parallel.ai/mcp"
headers = { "User-Agent" = "lecode (https://github.com/KalvadTech/lecode)" }
```

No Parallel account, API key, or OAuth setup is needed. The anonymous tier
provides free web search using Fast mode and page fetching for light use,
with rate limits.

Restart lecode after saving the config. Run `/mcp` to check that `parallel`
is connected, then `/mcp tools parallel` to see `web_search` and `web_fetch`.
lecode exposes them to the model as `mcp:parallel:web_search` and
`mcp:parallel:web_fetch`. Try a prompt like:

```text
Use Parallel to find the current Python asyncio TaskGroup documentation.
Fetch the official documentation page, summarize its cancellation behavior,
and include the source URLs. Do not edit files.
```

The `parallel` server follows the normal MCP permission fallback: allowed
in `yolo`, denied in `readonly`. To opt in to just these two tools in the
`readonly` permission mode, add these rules to the same config:

```toml
[[permissions.rules.allow."mcp:parallel:web_search"]]
pattern = "*"

[[permissions.rules.allow."mcp:parallel:web_fetch"]]
pattern = "*"
```

Existing deny rules and stricter agent restrictions still apply. Other
servers and your model configuration are unchanged.

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
auth = "oauth"           # interactive OAuth 2.1 authorization-code flow
```

## OAuth 2.1

Remote servers (`http` / `sse`) that advertise OAuth (like GlitchTip)
authenticate interactively — set `auth = "oauth"` instead of a static
`Authorization` header:

```toml
[mcp.servers.glitchtip]
transport = "http"
url = "https://your-glitchtip.example.com/mcp"
auth = "oauth"
```

- On startup lecode connects with **cached credentials only** — an expired
  access token is refreshed silently from its refresh token; a missing or
  rejected refresh shows `authentication required`. Startup never blocks on
  a browser.
- `/mcp auth glitchtip` runs the interactive login: it opens your browser
  and prints the authorization URL in the feed (paste it into a different
  browser if you prefer — e.g. over SSH); the redirect lands back on a
  loopback port lecode serves (`http://127.0.0.1:<port>/callback`).
- `/mcp login glitchtip` is the same but drops the cached credentials first,
  forcing a fresh browser flow (e.g. to switch accounts).
- `/mcp logout glitchtip` drops the session and the persisted credentials.
- Credentials (access + refresh tokens, client registration) are stored per
  endpoint under `~/.config/lecode/mcp-auth/` (0600 files, 0700 directory,
  plaintext JSON). `LECODE_CONFIG_DIR` moves them.
- A revoked or expired grant that fails a tool call with 401 is dropped on
  the spot; the server then shows `authentication required` until you
  re-authorize with `/mcp auth`.

The protocol itself — resource/server metadata discovery, dynamic client
registration, PKCE, token exchange and refresh — is handled by the SDK's
OAuth client; lecode supplies storage, the browser step, and the loopback
callback. `auth = "oauth"` conflicts with a static `Authorization` header
(that header path is the alternative for servers without OAuth).

## Permissions

Tools from servers **named** `exa`, `context7`, or `grep-app` (read-only-ish
services) are read-equivalent: allowed in both `yolo` and `readonly` modes.
Every other MCP tool follows the mode fallback (allowed in `yolo`, denied in
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
                      / authentication required
/mcp enable <name>    connect a server and add its tools to the model's context
/mcp disable <name>   disconnect a server and drop its tools from the context
/mcp tools <name>     list one server's tools
/mcp reconnect <name> drop and re-establish a server session
/mcp auth <name>      interactive OAuth login (opens the browser; reuses
                      still-valid cached credentials)
/mcp login <name>     like auth, but drops cached credentials first
/mcp logout <name>    drop a server session and its stored credentials
```

`enable`/`disable` are session-scoped: they flip the in-memory flag so a
noisy or rarely used server's tool specs stop riding every request, without
editing the config. Set `enabled = false` under `[mcp.servers.<name>]` to
disable a server permanently.
