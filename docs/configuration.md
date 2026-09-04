# Configuration reference

lecode reads TOML (preferred), YAML, or JSON config:

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
| `provider` | `"openrouter"` | `openrouter`, `openai`, or a `[custom_providers]` name |
| `model` | `"openai/gpt-5-mini"` | default model id |
| `api_key` | unset | provider key (env vars preferred) |
| `base_url` | unset | override the provider's endpoint |
| `thinking` | `"medium"` | `none` \| `low` \| `medium` \| `high` |
| `connect_timeout_s` | `10.0` | HTTP connect timeout |
| `read_timeout_s` | `300.0` | HTTP read (streaming) timeout |
| `auth_policy` | `"auto"` | `auto` \| `required` \| `none` — fail when no key found |
| `tls_verify` | `true` | set `false` for self-signed endpoints |

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
| `mid_turn_threshold` | unset | token count that triggers mid-turn compaction |

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
| `theme` | `"default"` | theme name (see `/themes`) |
| `collapse_thinking` | `true` | collapse reasoning blocks in the feed |
| `show_welcome` | `true` | show the welcome cheat-sheet on startup |
| `hidden_models` | `[]` | model ids hidden from `/models` |
| `no_color` | `false` | disable colored output |

## `[permissions]`

`mode` is one of `standard` (default), `restrictive`, `readonly`, `planwrite`,
`guarded`, `yolo`. Rules live under `[permissions.rules]` as
`allow` / `ask` / `deny` tables mapping tool names to pattern lists:

```toml
[permissions]
mode = "standard"

[[permissions.rules.allow.bash]]
pattern = "git status"

[[permissions.rules.ask.write]]
pattern = "*.py"

[[permissions.rules.deny.bash]]
pattern = "rm -rf*"
kind = "glob"   # "glob" (default) or "regex"
```

Last match wins within a table; deny rules are unbypassable. Read-class tools
(`read`, `grep`, `find_files`, `list_dir`, `lsp_diagnostics`, `memory_read`,
`memory_search`, `advisor`, `task`, and Exa/context7/grep.app MCP tools) are
auto-allowed in `standard` and `readonly`. A 3rd identical consecutive call
escalates Allow → Ask, the 4th is denied (doom-loop guard).

## `[notifications]`

Audio notifications (afplay/paplay/aplay, terminal bell fallback).

| field | default | meaning |
|---|---|---|
| `enabled` | `true` | master switch (`/notifications on\|off` toggles per session) |
| `volume` | `0.5` | 0.0–1.0 (afplay only) |
| `on_finish` / `on_error` / `on_approval` | `true` | per-event toggles |

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
| `transport` | `"stdio"` | `stdio` \| `http` (streamable HTTP) |
| `command` / `args` / `env` | — | stdio: argv and extra environment |
| `url` / `headers` | — | http: endpoint and extra headers (bearer tokens go here) |
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

## `[advisor]`

| field | default | meaning |
|---|---|---|
| `enabled` | `false` | the advisor tool + `/advisor` |
| `model` | unset | advisor model id (defaults to the main model) |
| `max_uses` | `5` | per-session call budget |
| `context_limit_kb` | `32` | conversation context sent along |
| `mode` | `"model"` | `model` \| `handoff` (ask the human inline) |

## `[hooks]`

Event name → list of shell commands:

```toml
[hooks]
PreToolUse = ["./ci/check-tool.sh"]
Stop = ["say done"]
```

Events: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`,
`SessionStart`, `SessionEnd`, `SubagentStart`, `SubagentEnd`. Hooks can only
narrow permission verdicts. See [hooks.md](hooks.md).

## `[colors]`, `[model_presets]`, `[custom_providers]`

- `[colors]` — theme color overrides (`name = "#hex"`).
- `[model_presets]` — `alias = "model-id"` shortcuts added to `/models`
  (`/models-add` writes here).
- `[custom_providers.<name>]` — extra OpenAI-compatible providers:

```toml
[custom_providers.local]
base_url = "http://localhost:11434/v1"
api_key_env = "LOCAL_API_KEY"   # optional
auth_policy = "none"            # auto | required | none
# headers = { X-Team = "infra" }
```
