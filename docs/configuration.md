# Configuration reference

lecode reads TOML config only (`config.toml`):

- **Global**: `~/.config/lecode/config.toml` (the directory is overridable
  with the `LECODE_CONFIG_DIR` env var). Auto-created with commented defaults
  on first run.
- **Project**: `.lecode/config.toml`, found by walking from the cwd up to
  the git root. Deep-merged over the global config: dicts merge recursively,
  scalars and lists replace.
- **CLI flags** apply on top of both (`--model`, `--provider`, `--base-url`,
  `--api-key`, …).

Unknown keys produce startup warnings. `schema_version = 1` is the current
on-disk schema; older files are migrated forward automatically.

The API key resolution chain is: `--api-key` > provider env var
(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `[custom_providers.*].api_key_env`)
> `[llm].api_key` in the config file. If you store the key in the file, keep
it owner-only (`chmod 600`); `lecode --setup` does that for you.

## `[llm]`

| field | default | meaning |
|---|---|---|
| `provider` | `"openrouter"` | `openrouter` or a `[custom_providers]` name |
| `model` | `"deepseek/deepseek-v4-flash"` | default model id |
| `api_key` | unset | provider key (env vars preferred) |
| `base_url` | unset | override the provider's endpoint |
| `thinking` | `"medium"` | `none` \| `low` \| `medium` \| `high` |
| `connect_timeout_s` | `10.0` | HTTP connect timeout |
| `read_timeout_s` | `300.0` | HTTP read (streaming) timeout |
| `auth_policy` | `"auto"` | `auto` \| `required` \| `none` — fail when no key found |
| `tls_verify` | `true` | set `false` for self-signed endpoints |

The model catalog is fetched live from the provider's `/models` endpoint at
startup (context windows, pricing, modalities). When the fetch fails the
catalog is empty — models lose pricing/modality annotations and costs report
as unknown until the provider reports usage — and nothing is cached on disk. Plain
OpenAI-shaped `/models` responses (id only) get a 128k
default context window and zeroed pricing.

### `[llm.system_prompt]`

| field | default | meaning |
|---|---|---|
| `style` | `"minimal"` | `minimal` \| `rich` base prompt |
| `custom` | unset | replaces the base prompt entirely |
| `persona` | unset | persona name (`prompts/personas/<name>.md`) appended in rich style |

## `[compaction]`

| field | default | meaning |
|---|---|---|
| `enabled` | `true` | auto-compact when approaching the context window |
| `buffer_tokens` | `20000` | headroom kept below the window |
| `on_overflow` | `"continue"` | `continue` \| `pause` when even compaction can't fit |
| `mid_turn_threshold` | unset | absolute token count that triggers compaction on tool-loop rounds after the first (instead of window − buffer) |

## `[agent]`

| field | default | meaning |
|---|---|---|
| `max_turns` | `500` | hard turn budget per prompt |
| `context_window` | `200000` | assumed model window for compaction |
| `turn_cooldown_ms` | `0` | sleep between turns |
| `tool_idle_timeout_s` | `300` | abort a turn after this much provider silence |
| `subagent_model` | unset | model for subagents (`/model-subagent`); inherits main |

## `[tools]`

| field | default | meaning |
|---|---|---|
| `enabled` | `{}` | per-tool on/off, e.g. `enabled = { bash = false }` |
| `allowlist` | `[]` | when non-empty, only these tools are registered |

## `[ui]`

| field | default | meaning |
|---|---|---|
| `collapse_thinking` | `true` | collapse reasoning blocks in the feed |
| `show_welcome` | `true` | show the welcome cheat-sheet on startup |
| `hidden_models` | `[]` | model ids hidden from `/models` |
| `no_color` | `false` | disable colored output |

The theme is fixed: one dark, Kalvad-purple palette ("kalvad"). There is no
theme selection, no theme files, and no color overrides.

## `[permissions]`

`mode` is one of `yolo` (default; everything allowed) or `readonly` (read-class
tools allowed, everything else denied). Legacy mode values (`standard`,
`restrictive`, `planwrite`, `guarded`) still load but are coerced to `yolo`
with a deprecation warning. Rules live under `[permissions.rules]` as
`allow` / `ask` / `deny` tables mapping tool names to pattern lists:

```toml
[permissions]
mode = "yolo"

[[permissions.rules.allow.bash]]
pattern = "git status"

[[permissions.rules.ask.write]]
pattern = "*.py"

[[permissions.rules.deny.bash]]
pattern = "rm -rf*"
kind = "glob"   # "glob" (default) or "regex"
```

Last match wins within a table; deny rules are unbypassable (even in `yolo`),
and `ask` rules still prompt. Read-class tools (`read`, `grep`, `find_files`,
`list_dir`, `lsp_diagnostics`, `memory_read`, `memory_search`,
`task`, and Exa/context7/grep.app MCP tools) are the only tools allowed in
`readonly`. A 3rd identical consecutive call escalates Allow → Ask, the 4th
is denied (doom-loop guard).

## `[notifications]`

Sound notifications (afplay/paplay/aplay, terminal bell fallback) and desktop
notifications (osascript on macOS, notify-send on Linux).

| field | default | meaning |
|---|---|---|
| `enabled` | `true` | master switch (`/notifications on\|off` toggles per session) |
| `volume` | `0.5` | 0.0–1.0 (afplay only) |
| `sound` | `true` | sound channel (player or bell) |
| `desktop` | `true` | desktop-notification channel |
| `on_finish` / `on_error` / `on_approval` | `true` | per-event toggles (both channels) |

## `[signals]`

Lifecycle status events as JSON datagrams over a Unix socket — for status
bars and tmux integration.

| field | default | meaning |
|---|---|---|
| `enabled` | `false` | emit `start` / `stop` / `git-conflict` events |
| `socket_path` | `<config_dir>/lecode.sock` | datagram destination |

Payload: `{"event", "session", "ts", …}`; failures are dropped silently.

## `[mcp]`

| field | default | meaning |
|---|---|---|
| `enable_exa` | `true` | auto-configure Exa web search when `EXA_API_KEY` is set |
| `enable_context7` | `false` | auto-configure the context7 docs server |

`[mcp.servers.<name>]` entries:

| field | default | meaning |
|---|---|---|
| `transport` | `"stdio"` | `stdio` \| `http` (streamable HTTP) \| `sse` (legacy) |
| `command` / `args` / `env` | — | stdio: argv and extra environment |
| `url` / `headers` | — | http/sse: endpoint and extra headers (static bearer tokens go here) |
| `auth` | — | http/sse: `"oauth"` enables the interactive OAuth 2.1 flow; credentials in `<config_dir>/mcp-auth/` (0600) |
| `timeout_s` | `30.0` | per-call timeout |
| `enabled` | `true` | disabled servers are skipped |

See [mcp.md](mcp.md).

## `[lsp]`

| field | default | meaning |
|---|---|---|
| `enabled` | `true` | spawn language servers, append diagnostics to write/edit |

`[lsp.servers.<lang>]` overrides the built-in registry (`command`,
`file_patterns`) or defines a new language. Diagnostics are fail-open: LSP
trouble never blocks the agent.

## `[memory]`

| field | default | meaning |
|---|---|---|
| `enabled` | `true` | persistent markdown memory + the four `memory_*` tools |
| `max_bytes` | `32768` | injection cap for `MEMORY.md` |

See [memory.md](memory.md).

## `[pierre]`

Post-task reviewer: after every completed task, a second model compares the
request with the result and gives feedback in the feed.

| field | default | meaning |
|---|---|---|
| `enabled` | `false` | review every finished task + `/pierre` |
| `model` | unset | reviewer model id; `/pierre on` requires one, different from the main model |

## `[hooks]`

Event name → list of shell commands:

```toml
[hooks]
PreToolUse = ["./ci/check-tool.sh"]
Stop = ["say done"]
```

Events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`,
`PermissionResult`, `UserPromptSubmit`, `Stop`, `SessionStart`, `SessionEnd`,
`SubagentStart`, `SubagentEnd`, `PreCompact`, `PostCompact`, `Interrupt`,
`Notification`. Hooks can only narrow permission verdicts; only `PreToolUse`
and `UserPromptSubmit` deny-verdicts are enforced. See [hooks.md](hooks.md).

## `[custom_providers]`

- `[custom_providers.<name>]` — extra OpenRouter-compatible providers:

```toml
[custom_providers.local]
base_url = "http://localhost:11434/v1"
api_key_env = "LOCAL_API_KEY"   # optional
auth_policy = "none"            # auto | required | none
# headers = { X-Team = "infra" }
```

## `[telemetry]`

Opt-in Sentry error reporting + OpenTelemetry metrics. Requires the
`telemetry` extra (`uv tool install 'lecode[telemetry]'`); every failure mode
is fail-open (telemetry never breaks the agent).

| field | default | meaning |
|---|---|---|
| `enabled` | `false` | master switch |
| `sentry_dsn` | unset | Sentry (or GlitchTip) DSN for error reports |
| `otlp_endpoint` | unset | OTLP/HTTP base URL for metrics (e.g. `http://localhost:4318`) |
| `export_interval_s` | `60.0` | metric export interval |
| `service_name` | `"lecode"` | OTel `service.name` resource |
| `environment` | `"dev"` | Sentry/OTel environment tag |

Metrics: `lecode.turns`, `lecode.turn.duration_s`, `lecode.tokens.input`,
`lecode.tokens.output`, `lecode.cost_usd`, `lecode.tool_calls`,
`lecode.tool.duration_s` (all tagged by model / tool / stop reason). Sentry
receives provider and tool errors with `send_default_pii = false`.
