# lecode — build plan (from scratch)

## Product definition

**lecode** is a minimalist terminal AI coding agent, written in Python. Installable as a
CLI via `uv tool install`, runs in the terminal's normal scrollback (no alternate
screen), talks to **OpenRouter** and any **generic OpenAI-compatible API** (local
servers included), executes tools under a rich permission system, persists sessions, and
supports power features: subagents, MCP servers (with Exa and context7 auto-configured),
git worktrees, skills, custom user-defined agents, LSP diagnostics, persistent memory,
a post-task reviewer model, multimodal input, lifecycle hooks, session export, loop
mode, prompt chaining.

**Mandatory session naming**: every interactive start asks for a session name up front,
before the chat opens — the prompt cannot be skipped or left empty (Ctrl-C/Ctrl-D
quits instead). The goal is enforced order: every session is named, browsable, and
greppable from the moment it exists. Headless mode (`-p`, `--loop`) skips the prompt and
auto-generates a name; `--continue`/`--resume` keep the existing session's name.

Requires three external binaries on PATH: **`fd`, `ripgrep` (rg), and `rtk`** — checked
at startup with a clear error and install hint if missing.

Design stance (from kon): the default harness stays tiny — **default system prompt under
~300 tokens**, project context loaded externally via AGENTS.md files and skills. A rich,
detailed prompt (personas, mode prompts) is available as a config option.

(Release-time check: if `lecode` is taken on PyPI, publish as `lecode-agent` while
keeping the `lecode` command name.)

## Feature lineage

### From kon (github.com/0xku/kon)

- Tiny default system prompt; externalized project context (AGENTS.md from git root down
  to cwd + global file)
- Skills system: `SKILL.md` packs in `.agents/skills/` (project) and `~/.agents/skills/`
  (global), YAML frontmatter (`name`, `description`, `register_cmd`, `cmd_info`),
  discoverable by the model and optionally registered as slash commands
- UX extras: steer queue (Alt+Enter queues a priority message processed before normal
  queued prompts; 5+5 queue limits), `!cmd` (run shell, show result) and `!!cmd`
  (also feed output to the LLM), Tab path completion, `/copy` (last response to
  clipboard), `/handoff` (synthesize a focused prompt from the current session into a
  new one), session picker with delete, `/session` stats
- Audio notifications on task finish / error / awaiting approval (volume config)
- Append-only **JSONL** session files (inspectable, robust; full history survives
  compaction)
- Config schema versioning with automatic forward migration
- Thinking levels (`/thinking`), collapsible thinking display, turn cooldown and
  tool-call idle timeout knobs
- Headless `-p` mode with meaningful exit codes (0 done / 1 error / 2 startup / 3 max turns)
- `--base-url` + auth-policy flags for local/custom OpenAI-compatible endpoints
- **`fd` and `ripgrep` as hard dependencies** (kon-style), plus `rtk` — verified at
  startup

Not taken from kon: OAuth provider logins (auth stays API-key-only), auto-downloading
binaries, two-mode permissions (too coarse), dedicated per-vendor provider plugins,
DuckDuckGo web tools (Exa MCP instead), anonymous sessions.

### From zerostack (github.com/gi-dellav/zerostack)

- Permission *system*: per-tool glob + regex rule layers, last-match-wins, unbypassable
  denies, six modes (standard / restrictive / readonly / planwrite / guarded / yolo),
  session "allow always" allowlist, doom-loop detection; Exa / context7 / grep.app MCP
  tools treated as read-equivalent
- Session undo / redo / rewind-to-turn
- Subagents (`task` tool), MCP client (stdio + HTTP + OAuth for MCP servers) with
  **Exa auto-configured by default and context7 one flag away**, git worktrees, loop
  mode, prompt chaining, JSONL import + HTML export + gist sharing, themes,
  slash-command catalog, prompt caching (via OpenRouter), model catalog with pricing
- **LSP integration**: diagnostics appended to edits
- **`rtk` output compaction** for bash
- **Persistent memory**: Markdown store — long-term `MEMORY.md` (auto-injected, 32 KB
  cap), daily logs, project scratchpad checklists, named notes; four tools
  (`memory_write/edit/read/search`, regex keyword search); compaction summaries flushed
  to the daily log; atomic writes + `.bak` backups; `/memory` commands
- **Pierre mode**: a post-task reviewer — a second model compares the user's request
  with the finished result and reports whether it delivered; `/pierre on|off|model`
- **Multimodal input**: image/audio/PDF attachments via `/add` and the `@` picker
  (20 MB cap), sent as OpenAI-compatible content parts where the selected model
  supports them
- **Lifecycle hooks**: user-defined subprocess hooks — `PreToolUse`, `PostToolUse`,
  `Stop`, `UserPromptSubmit`, session and subagent start/end; handlers receive a JSON
  envelope on stdin and return verdicts (`Allow` / `Defer` / `Ask` / `Deny`,
  most-severe-wins merge) plus optional input rewrites; hooks can only narrow
  permissions, never grant; `/hooks` status command and `--hooks-test` dry-run
- Setup wizard (`--setup`)

### From opencode (opencode.ai)

- **Custom user-defined agents**: markdown files in `~/.config/lecode/agents/` (global)
  and `.lecode/agents/` (project) with frontmatter (`description`, `mode:
  primary|subagent|all`, `model`, `temperature`, `permission` overlay, `hidden`,
  `color`). Primary agents are cycled with **Tab** (built-ins: `build` = full access,
  `plan` = read-only/ask — maps onto the permission modes); subagents are invoked
  automatically by the model via `task` or manually by `@mention`. Agent permissions
  layer over the global permission config.

Considered and deferred (not v1): formatter-runners after edits, customizable keybinds,
parent↔child session navigation, JS-style plugins, desktop app, `/connect` billing
portal.

## Locked technical decisions

- **Language/runtime**: Python 3.12+, single `asyncio` event loop.
- **TUI**: `prompt_toolkit` + `Rich`. Append-only feed, pinned input box. No alternate
  screen, no mouse capture, no full-screen dialogs.
- **Session naming**: interactive startup always begins with a name prompt (a single
  inline prompt_toolkit line, validated non-empty, duplicates suffixed `-2`, `-3`, …);
  skipped in `-p`/`--loop` (auto-name from timestamp) and on `-c`/`-r` resume.
- **Statusline**: one fixed, well-designed statusline (session name · agent · model ·
  cwd+git branch · context meter · tokens · cost · spinner/queue state) — **not
  user-configurable** beyond theme colors. No segment-layout engine.
- **Providers**: one thin hand-written async streaming client on `httpx` implementing
  the OpenAI Chat Completions protocol (SSE parsed directly). OpenRouter is a preset
  over the same client (base URL, app-identity headers, model listing + pricing,
  provider routing hints, prompt caching pass-through). No LiteLLM, no vendor SDKs.
- **Auth**: API keys only. Priority chain: CLI flag > env var (`OPENROUTER_API_KEY`,
  `OPENAI_API_KEY`, or per-custom-provider env) > config file. Custom/local endpoints
  may run keyless (`auth = "auto|required|none"` policy per provider).
- **External binaries (mandatory)**: `fd` powers `find_files`, `rg` powers `grep`,
  `rtk` compacts `bash` output (`rtk rewrite`, fail-open with 5s timeout on individual
  rewrites). Startup verifies all three exist on PATH and exits with install
  instructions otherwise (exit code 2). No auto-download, no fallback code paths.
- **Packaging**: PyPI package, `uv tool install`.
- **Tooling**: `uv`, `pytest` + `pytest-asyncio` + `respx`, `ruff`, `pydantic` v2,
  `typer`.

## Feature specification

### Core
- **CLI** (typer): model/provider/api-key/base-url flags, `-p` headless mode,
  `-c`/`--continue`, `-r`/`--resume <id|prefix|name>`, `--extra-tools`, auth-policy and
  TLS-skip flags for local endpoints, tool allowlisting, read-only mode, `--setup`,
  `--hooks-test`. Startup dependency check for `fd`/`rg`/`rtk`; interactive startup
  then runs the mandatory session-name prompt before entering the chat.
- **Config**: `~/.config/lecode/config.toml` (TOML preferred; YAML/JSON accepted),
  auto-created on first run, **schema version + automatic migration**, project-local
  `.lecode/config.toml` deep-merged over global, unknown-key warnings, CLI overrides.
  Sections: `[llm]` (provider/model/thinking level/timeouts/base URL/auth policy/TLS),
  `[llm.system_prompt]` (minimal default; `style = "rich"` opts into the detailed
  prompt + personas), `[compaction]`, `[agent]` (max turns, context window, cooldown),
  `[tools]`, `[ui]` (theme, thinking collapse, welcome shortcuts, hidden models),
  `[permissions]`, `[notifications]`, `[mcp]` (`enable_exa` default true,
  `enable_context7` default false, server table), `[lsp]` (server overrides),
  `[memory]`, `[pierre]`, `[hooks]` (event → handler commands), plus permission rule
  tables, colors, custom provider definitions (name → base_url +
  api_key_env + headers).
- **Providers**: OpenRouter (first-class preset: model catalog refresh, pricing,
  app-identity headers, caching) + generic OpenAI-compatible endpoints (Ollama, LM
  Studio, llama.cpp, corporate gateways, DeepSeek/xAI/ZhiPu-style hosts — all via
  `[custom_providers]` config or `--base-url`). Streaming everywhere, tool calling,
  retry with exponential backoff + jitter and per-error retryability, `/models`
  listing, static bundled model catalog (context size + pricing).
- **Agent loop**: multi-turn streaming runner — stream a turn, collect tool calls,
  execute in parallel (results paired by call id), append, re-invoke; empty-response
  guard (max 3); "Please continue." injection when a turn ends without a final answer;
  instant cancellation of stream + in-flight tools; per-turn token/cost accounting;
  configurable max turns (default ~500) and turn cooldown.
- **Tools**: core 6 — `read` (pagination, repeat-read guard, image rendering for
  multimodal models), `write` (atomic), `edit` (two engines: fuzzy
  whitespace-normalized search/replace and CRC-anchored line addressing), `bash`
  (timeout, truncation, idle timeout, `rtk` output compaction), `grep` (regex + glob +
  context; implemented over `rg`), `find_files` (glob; implemented over `fd`); plus
  `list_dir`, `todo_write`, `lsp_diagnostics`, the four `memory_*` tools;
  MCP-provided web search/fetch via Exa.
- **Custom agents**: built-in primaries `build` (full access) and `plan` (read-only +
  ask), cycled with Tab and shown in the statusline; user agents from markdown files
  (global + project) with per-agent model/temperature/prompt/permission overlay;
  subagents invocable via `task` or `@mention`; hidden agents excluded from pickers.
- **LSP integration**: spawn language servers for files the agent writes/edits;
  diagnostics appended to `write`/`edit` results (fail-open — LSP problems never block
  the agent); built-in server registry (per-language command + file patterns) with
  `[lsp]` config overrides; `lsp_diagnostics` tool for on-demand queries; raw async
  JSON-RPC client over stdio.
- **Permissions**: per-tool glob and regex rule layers, last-match-wins, deny rules
  unbypassable, six modes with per-mode fallbacks, per-agent permission overlays,
  session-scoped "allow always" allowlist, doom-loop detection (≥3 identical
  consecutive calls → coach/ask/deny), interactive ask with AllowOnce /
  AllowAlways(pattern) / Deny. Exa web search, context7, and grep.app MCP tools
  classified read-equivalent → auto-allowed like read tools. Hooks sit strictly
  outside the checker: they can only narrow a decision, never grant one.
  `/permissions` switches modes.
- **Lifecycle hooks**: subprocess handlers per event (`PreToolUse`, `PostToolUse`,
  `Stop`, `UserPromptSubmit`, session start/end, subagent start/end); JSON envelope on
  stdin, verdict + optional rewritten input on stdout; `PreToolUse` verdicts merge
  most-severe-wins (Deny > Ask > Defer > Allow); tools are decorated at build time;
  `--hooks-test` dry-runs the hook pipeline without executing tools.
- **Memory**: project-scoped Markdown store (project slug from cwd); `MEMORY.md`
  long-term (auto-injected into the system context, 32 KB cap), daily log files,
  scratchpad checklist, named notes; tools `memory_write` / `memory_edit` /
  `memory_read` / `memory_search` (regex keyword search); compaction summaries appended
  to the daily log; atomic writes with `.bak` backups; `/memory show|edit|search|log`
  commands.
- **Pierre mode**: after every completed task, a second model compares the request
  with the result and gives feedback in the feed; `/pierre on|off|model`.
- **Multimodal input**: `/add` and `@` accept image/audio/PDF files (20 MB cap);
  attachments sent as OpenAI-compatible `image_url` / document content parts; a clear
  error when the selected model lacks the modality; attachment list shown in the feed.
- **Sessions**: append-only **JSONL** per session in the config dir, **always named**
  (mandatory prompt at interactive startup; `/new` asks for a name too); resume by id /
  unique prefix / recency / name; session picker (sorted, searchable, with delete);
  `/session` metadata + token stats; `/rename`; undo / redo / rewind-to-turn with
  restore point; `/handoff` (asks for the new session's name); per-session permission
  allowlist.
- **Cost reporting**: money is a first-class metric — every time a chat finishes
  (interactive run end, `/quit`, and headless completion), display the session totals:
  money spent (USD, from catalog pricing), input tokens, output tokens. Running
  totals also live in the statusline (tokens · cost).
- **Context management**: token estimation calibrated by provider usage; manual
  (`/compact`) and automatic compaction near the window (summarize old messages, keep
  recent tail; `on_overflow = continue|pause`, buffer tokens); optional mid-turn
  threshold; long tool results truncated head/tail with overflow to files; full history
  always retained in the JSONL file.
- **Context loading**: `AGENTS.md` / `CLAUDE.md` from global dir and ancestors (git root
  → cwd); **skills** from `.agents/skills/` and `~/.agents/skills/` (SKILL.md +
  frontmatter, model-discovered, `register_cmd: true` exposes them as slash commands);
  16 built-in named prompt personas (rich prompt mode and `.prompt` prefix),
  user-overridable; memory injection as above.
- **TUI**: startup session-name prompt (validated, unskippable); multiline input, Emacs
  bindings + kill ring, history with drafts (persisted), bracketed paste, Ctrl+G opens
  `$EDITOR`, Tab path completion (and Tab agent-cycle when the input is empty), `@`
  fuzzy file/agent picker, `/` command picker, `.` prompt picker, argument pickers for
  models (with `hidden_models` trimming) / themes / providers, two-stage rewind picker;
  queued prompts (Enter while running) + steer queue (Alt+Enter, priority, 5+5 limits);
  streaming markdown feed with syntax highlighting; collapsible thinking blocks;
  **fixed statusline** (session name · agent · model · git branch · context meter ·
  tokens · cost · state); braille spinner; inline permission prompt (y/a/n/ESC);
  17+ JSON themes (`/theme`); OSC 8 hyperlinks;
  clipboard (OSC 52 / pbcopy / xclip), `/copy`; audio notifications; `--no-color`.
- **Slash commands** (~50): `/new /clear /resume /session /undo /redo /rewind /retry
  /rename /history /quit /exit /handoff /compact /compress /model /models
  /thinking /reasoning /permissions /mode /toggle /theme /themes /prompt
  /editsys /add /drop /drop-all /init /help /welcome
  /tutor /review /btw /queue /copy /export /import /share /loop /worktree /wt-exit
  /wt-merge /mcp /model-subagent /models-subagent /notifications /memory /pierre
  /doctor /hooks /agents` + skill-registered commands. Prefixes: `!cmd`, `!!cmd`,
  `.prompt`, `@file`, `@agent`.
- **Headless**: `-p [prompt]` (inline or stdin), tools auto-approved, final response to
  stdout, exit codes 0/1/2/3, auto-generated session name; `--loop` iterative mode
  against a plan file with optional per-iteration command and max iterations.

### Power features
- **Subagents**: `task` tool spawning parallel child agents (per-prompt, timeout +
  response cap) — built-in read-only `explore` plus user-defined agents; own model
  (`/model-subagent`), tool calls visible in feed; subagent lifecycle hooks fire.
- **MCP client**: stdio + streamable-HTTP servers via the official `mcp` Python SDK
  (OAuth here is for MCP servers only, unrelated to LLM auth), per-server
  timeouts/reconnect, tools under the permission system, `/mcp` management.
  **Auto-configured servers**: Exa web search (default on, needs `EXA_API_KEY` or
  dashboard key flow) and context7 docs lookup (default off, `enable_context7 = true`).
- **Git worktrees**: `--worktree <name>` isolated worktree+branch, merge-back on exit
  (`/wt-merge /wt-exit`), conflict detection.
- **Export**: standalone HTML (`/export`), re-importable JSONL (`/import`), secret-gist
  sharing (`/share`).
- **Prompt chaining**: brainstorm → plan → code → review phases with auto-transitions.
- **Status signals**: start/stop/git-conflict events over a Unix socket.

### Explicit non-goals (v1)
- No alternate screen, mouse capture, full-screen dialogs.
- **No OAuth for LLM providers; no dedicated Anthropic/Gemini/Copilot/Codex/xAI/Azure
  plugins** — API keys + OpenAI-compatible protocol only.
- No anonymous sessions in interactive mode (name is mandatory).
- No configurable statusline layout (one fixed design).
- No editor (ACP) integration, no output-compaction beyond `rtk`.
- No formatter-runners, customizable keybinds, or child-session navigation (deferred
  opencode ideas).
- No OS sandboxing of shell commands (config stub for later).
- No auto-downloading of binaries; `fd`/`rg`/`rtk` must be installed by the user.
- Windows support is best-effort (targets macOS/Linux).

## Project layout

```
lecode/
├── pyproject.toml            # console_script: lecode = lecode.cli:main
├── src/lecode/
│   ├── cli.py                # typer CLI + startup checks (fd/rg/rtk) + name prompt hook
│   ├── config/               # pydantic models, loader, schema migrations, project merge
│   ├── auth.py               # API-key priority chain (CLI > env > config), auth policy
│   ├── providers/            # openai_compat.py (the one client), openrouter.py (preset:
│   │                         # headers, catalog, pricing), retry.py, catalog.py
│   ├── agent/
│   │   ├── runner.py         # asyncio multi-turn streaming loop
│   │   ├── builder.py        # prompt assembly (minimal|rich), tool/skill/agent wiring
│   │   └── tools/            # core tools (rg/fd/rtk backed) + task +
│   │                         # memory_* + MCP bridge
│   ├── lsp/                  # async JSON-RPC client, server registry, manager,
│   │                         # lsp_diagnostics tool
│   ├── session/              # JSONL model/storage, naming, compaction, undo/rewind,
│   │                         # handoff, input history
│   ├── permission/           # patterns, checker, six modes, agent overlays, doom-loop,
│   │                         # MCP read-equiv
│   ├── hooks/                # event dispatcher, subprocess runner, verdict merge,
│   │                         # tool decorator, --hooks-test
│   ├── memory/               # markdown store, 4 tools, injection, /memory handlers
│   ├── multimodal.py         # attachment detection/encoding, modality capability check
│   ├── context/              # AGENTS.md walk, skills loader, agents loader,
│   │                         # prompt/theme resources
│   ├── tui/
│   │   ├── app.py            # prompt_toolkit application + asyncio wiring
│   │   ├── feed.py           # append-only Rich feed (markdown, thinking collapse)
│   │   ├── input.py          # editor, bindings, history, triggers, queues, Tab complete
│   │   ├── pickers.py        # fuzzy pickers (files/agents/commands/models/sessions…)
│   │   ├── statusline.py     # one fixed, polished statusline (name · agent · model …)
│   │   ├── notify.py         # audio notifications
│   │   └── themes.py
│   ├── slash/                # command registry + handlers (skills register here too)
│   ├── extras/               # subagents, mcp_client (exa/context7 presets), worktree,
│   │                         # loop_mode, export, chain, status_signals, rtk.py
│   ├── setup_wizard.py       # linear onboarding dialogs
│   └── data/                 # prompts/, themes/, models.json, export template
├── tests/
└── docs/
```

## Architecture notes

- One `asyncio` loop; components communicate over `asyncio.Queue`s with a small event
  taxonomy (`Token / Reasoning / ToolCall / ToolResult / Error / Retrying / Done`, user
  input, permission ask/reply). TUI never calls providers;
  runner never renders.
- Startup sequence (interactive): dependency check → config load → `--setup` if
  unconfigured → **session-name prompt** (loop until non-empty; Ctrl-C/Ctrl-D exits
  cleanly with code 0 before any session file is created) → session created on disk
  under that name → chat opens. `/new` and `/handoff` reuse the same prompt component.
- Cancellation is `asyncio.Task.cancel()`; subagents in a task group cancelled together;
  steer queue is a second priority input queue drained first between turns.
- One streaming client implementation (OpenAI Chat Completions with SSE via
  `httpx.AsyncClient.stream()` + line iteration); OpenRouter behavior is a thin preset
  layer over it (headers, catalog/pricing fetch, routing params). Subagents and the
  pierre reviewer reuse the same client with different model/prompt params.
- External binaries: one shared async subprocess wrapper
  (`asyncio.create_subprocess_exec`, timeout, output caps) used by `grep` (rg),
  `find_files` (fd), `bash` (rtk), and the hook handlers. Presence verified once at
  startup; individual `rtk rewrite` failures at runtime stay fail-open (original output
  used). Hook subprocess timeouts deny-safe (hook failure never widens access).
- Hooks are applied as a decorator around every tool at build time; the permission
  checker runs first and hooks can only narrow its verdict (Allow → Ask/Deny, never
  the reverse).
- LSP manager: lazily spawns one server per language root on first write/edit to a
  matching file; async JSON-RPC over stdio; diagnostics pulled via
  `textDocument/diagnostic` (or `publishDiagnostics` notifications) and appended to the
  tool result; every failure mode is swallowed (fail-open) with a debug log.
- Multimodal: attachment files are sniffed by extension + magic bytes, base64-encoded,
  and attached to the user message as OpenAI content parts; the model catalog carries
  per-model modality flags to fail fast with a clear message.
- Static data (prompts, themes, model catalog, HTML template) ships as package data via
  `importlib.resources`; override precedence embedded < global dir < project dir.
- External programs (`$EDITOR`, `git`) via prompt_toolkit's `run_in_terminal` — trivial
  with no alternate screen to suspend.

## Testing strategy

- `pytest` + `pytest-asyncio` (auto mode) — async tests must run under
  `uv run python -m pytest`; `respx` for HTTP-level mocking. CI runners install
  fd/ripgrep/rtk (or a shim for rtk) before the suite runs.
- **Fake provider**: scripted streaming responder (text chunks, tool calls, errors) with
  request/history capture — headless end-to-end tests of the full agent loop. Pierre
  and subagents run against the same fake.
- **Provider contract tests**: SSE decode, tool-call round-trips, multimodal payload
  shaping, retry classification, OpenRouter headers/catalog handling, error mapping,
  keyless local endpoints.
- **TUI tests**: prompt_toolkit `create_pipe_input` drives the app headlessly; feed and
  statusline asserted with Rich `Console(record=True)` snapshots; the startup name
  prompt is covered by pipe-input tests (empty input rejected, duplicate suffixed,
  Ctrl-C aborts before session creation).
- **Tool tests**: grep/find_files run against real `rg`/`fd` (skipped with a clear
  marker if absent); startup dependency check tested with a doctored PATH; rtk
  compaction tested with a fake `rtk` shim, including runtime fail-open on rewrite
  errors; LSP tested against a mock JSON-RPC server subprocess (spawn, initialize,
  diagnostics, fail-open on crash).
- **Hooks tests**: corpus of fake handler scripts (each verdict, rewrites, timeouts,
  crashes) asserting merge order and narrow-only semantics; `--hooks-test` dry-run
  coverage.
- **Memory tests**: store CRUD, injection budget caps, `.bak` behavior, search ranking.
- Coverage emphasis (safety-critical first): permission checker + modes + agent
  overlays + MCP read-equivalence, hook verdict merging, edit engines, session JSONL
  storage / naming / compaction / undo / handoff, config merge + migrations, skills and
  agents loaders (frontmatter validation), runner retry/cancellation, queue/steer
  ordering. Target ~550–700 tests at v1.
- CI: `ruff check`, `ruff format --check`, `pytest` on Python 3.12/3.13, Linux + macOS.

## Execution phases (each ends green: pytest + ruff)

1. **Bootstrap** — uv project, pyproject (deps: prompt_toolkit, rich, httpx, pydantic,
   PyYAML, mcp, typer; dev: pytest, pytest-asyncio, respx, ruff), CI (installing
   fd/rg), startup dependency check, `lecode --version`.
2. **Foundations** — config system (models, loader, migrations, project merge, default
   writer), API-key auth chain, model catalog (incl. modality flags), AGENTS.md walk,
   prompt/theme resources.
3. **Provider layer** — the OpenAI-compatible streaming client (SSE, tool calling,
   multimodal content parts, retry) + OpenRouter preset (headers, catalog refresh,
   pricing) + custom-provider resolution. respx contract tests.
4. **Session + permissions** — JSONL sessions, **naming (mandatory, duplicate-safe)**,
   compaction, undo/redo/rewind, handoff; permission checker (both rule layers, six
   modes, doom-loop, MCP read-equivalence). Heaviest test phase.
5. **Tools + agent loop** — core tools with permission gating (rg/fd/rtk backed),
   prompt assembly (minimal default + rich option), multi-turn runner, cancellation,
   queues. `-p` headless mode works against the fake provider + one real provider.
6. **Skills + custom agents** — skills loader and `register_cmd` slash registration;
   agents loader (markdown + frontmatter), build/plan primaries, Tab cycling,
   `@mention` subagent invocation, per-agent permission overlays.
7. **Hooks + memory** — hook events, subprocess runner, verdict merge, tool decorator,
   `--hooks-test`, `/hooks`; memory store, four tools, injection, `/memory`.
8. **TUI** — startup session-name prompt, input editor (Emacs, history, Tab completion,
   triggers, steer queue, agent cycling), streaming markdown feed with thinking
   collapse, pickers (incl. session picker with delete), the fixed statusline, spinner,
   themes, inline permission prompt, clipboard, `/copy`, notifications, attachment
   display.
9. **Slash commands + pierre + multimodal** — full command registry; pierre post-task
   review (`/pierre`); multimodal `/add` + `@` attachments with capability
   checks.
10. **Power features** — in order: subagents, prompt chaining, export/import/share,
    git worktrees, loop mode, status signals, LSP integration (registry, manager,
    diagnostics-in-results, `lsp_diagnostics`), then MCP client with Exa/context7
    auto-config (largest, own sub-phase).
11. **Onboarding + release** — `--setup` wizard, first-run defaults, `/init`, `/welcome`,
    docs (including fd/rg/rtk install instructions); PyPI publish workflow (verify the
    `lecode` name, else `lecode-agent`); clean-machine `uv tool install` → setup →
    real coding task; full CI green.

## Sizing

Roughly 19–26k LOC of Python (memory ~1k, hooks ~1.2k, pierre ~0.2k, multimodal ~0.3k,
custom agents ~0.8k, LSP ~0.8k on top of the ~14–19k base). Phases 2–5 are the critical
path to a usable agent; phase 8 is the largest single chunk; MCP + LSP (phase 10) are
the biggest extras.

## Verification gates

- Every phase: `uv run python -m pytest` green, `uv run ruff check` clean.
- End of phase 5: scripted multi-turn tool-use transcripts pass against the fake
  provider; live smoke test against OpenRouter and one local OpenAI-compatible server;
  `-p` exit codes verified; startup check rejects a PATH missing fd/rg/rtk.
- End of phase 7: hook corpus passes (verdict merge, narrow-only, timeout deny-safe);
  memory injection respects caps.
- End of phase 8: interactive start requires a session name (empty rejected, duplicate
  suffixed, abort leaves no session file); `-p` skips the prompt.
- End of phase 11: clean-machine `uv tool install` → `--setup` → one real coding task
  completed (including one hook firing, one pierre review, one image attachment); suite
  green in CI.
