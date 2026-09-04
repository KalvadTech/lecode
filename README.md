# lecode

**A minimalist terminal AI coding agent.** No IDE, no electron, no noise —
just your terminal, an OpenAI-compatible model, and a sharp set of tools.

```text
╭────────────────────────── lecode — fix-auth ───────────────────────────╮
│  ✓ session      fix-auth — new session                                 │
│  ✓ config       ~/.config/lecode/config.toml + .lecode/config.toml     │
│  ✓ provider     openrouter · https://openrouter.ai/api/v1              │
│                 model deepseek/deepseek-v4-flash · key from env        │
│  ✓ models       327 fetched live from the provider                     │
│  ✓ prompt       minimal                                                │
│  ✓ context      AGENTS.md · docs/AGENTS.md                             │
│  – skills       none                                                   │
│  ✓ agents       primaries: build, plan · custom: researcher            │
│  ✓ memory       long-term 2.4 KB injected                              │
│  ✓ tools        15 tools                                               │
│  ✓ permissions  mode yolo · custom rules                               │
│  – hooks        none configured                                        │
│  ✓ lsp          enabled                                                │
│  ! mcp          exa (no EXA_API_KEY)                                   │
╰─────────────────── ~/github.com/you/your-project ─────────────────────╯
```

Every interactive start shows you exactly what was loaded — config files,
provider, prompt, project context, skills, agents, memory, permissions,
hooks, LSP, MCP — before the chat opens.

## Why lecode

- **Tiny harness, big ecosystem.** The default system prompt is under 300
  tokens; project context comes from your `AGENTS.md` files, skills, and
  memory — not from a bloated prompt.
- **OpenRouter, or anything OpenRouter-compatible.** OpenRouter is the
  first-class preset (live catalog, pricing, caching); other gateways —
  Ollama, LM Studio, llama.cpp, corporate proxies — plug in with one
  `--base-url` flag or a `[custom_providers]` entry. Keyless local endpoints
  supported. The model catalog is fetched live from the provider at startup
  (cached on disk, bundled snapshot as offline fallback).
- **Real permissions, not vibes.** Two modes (`yolo` by default, `readonly`
  when you want a look-but-don't-touch agent), glob + regex rules,
  last-match-wins, unbypassable denies, doom-loop detection, and lifecycle
  hooks that can narrow — never widen — any decision.
- **Money is a metric.** Live token + cost totals in the statusline, and a
  `tokens in/out · cost` summary every time a chat ends.
- **Inspectable by design.** Sessions are append-only JSONL you can grep;
  sessions are always named; every session is resumable, undoable,
  rewindable, exportable.

## Requirements

- Python 3.12+
- Three external binaries on `PATH` (verified at startup; no auto-download,
  no fallback):

  | binary | powers | install |
  |--------|--------|---------|
  | `fd`   | `find_files` | `brew install fd` · `apt install fd-find` · `cargo install fd-find` |
  | `rg`   | `grep` | `brew install ripgrep` · `apt install ripgrep` · `cargo install ripgrep` |
  | `rtk`  | bash output compaction | see <https://github.com/rtk-ai/rtk> |

Targets macOS and Linux; Windows is best-effort.

## Install

```sh
uv tool install lecode
```

(If the `lecode` name is ever taken on PyPI, the package is also published
as `lecode-agent` — the command stays `lecode`.)

## Quickstart

```sh
lecode --setup                       # import from pi/opencode or answer 4 questions
export OPENROUTER_API_KEY=sk-or-...  # or keep the key in config.toml
cd your-project
lecode                               # name the session, then ask for something
```

```text
fix-auth · build · deepseek/deepseek-v4-flash · lecode:main · ctx ▓▓░░░ 18% · ↑4.1k ↓0.9k · $0.0062 · ⠼
```

One fixed statusline: session · agent · model · cwd:branch · context meter ·
tokens · cost · state. No configuration needed.

Useful things to type:

```text
/help                     all slash commands, grouped
/welcome                  key bindings cheat-sheet
/tutor permissions        explain one feature
!make test                run a shell command, see the output
!!pytest -x               run it AND feed the output to the model
@src/auth.py              attach a file (images/PDF/audio too)
.plan refactor this       run with a persona
Tab                       cycle agents: build ⇄ plan
Alt-Enter                 steer the agent mid-turn
```

## Headless

```sh
lecode -p "write a haiku about this repo"     # one prompt, then exit
git diff | lecode -p "review this diff"       # a bare -p reads stdin
lecode --loop plan.md --loop-cmd "make test"  # iterate until the plan is done
lecode --chain "redesign the parser"          # brainstorm→plan→code→review
```

The final answer goes to stdout; a `tokens: <in> in / <out> out · cost:
$X.XXXX` summary goes to stderr, so scripts can pipe the answer cleanly.

Exit codes: `0` done · `1` error · `2` startup (missing deps, bad flags,
non-tty `--setup`) · `3` max turns / max loop iterations.

## A tour of the power features

- **Permissions** — two modes (`yolo` allows everything, `readonly` allows
  read-class tools only; default `yolo`), per-tool glob/regex rules, session
  allow-always grants, per-agent overlays. `/permissions`, `/mode`, `/toggle`.
- **Sessions** — always named, append-only JSONL, `/new` `/resume`
  `/undo` `/redo` `/rewind` `/retry` `/compact` `/handoff` `/rename`,
  searchable picker with delete, HTML export (`/export`), secret-gist
  sharing (`/share`), re-import (`/import`).
- **Custom agents** — markdown files in `~/.config/lecode/agents/` or
  `.lecode/agents/` with their own model, temperature, prompt and permission
  overlay. Tab cycles primaries; `@agent` or the `task` tool runs subagents.
- **Skills** — `SKILL.md` packs in `.agents/skills/` (project) or
  `~/.agents/skills/` (global), discoverable by the model, optionally
  registered as slash commands.
- **Memory** — persistent per-project markdown: long-term `MEMORY.md`
  (auto-injected, 32 KB cap), daily logs, scratchpad, named notes.
  `/memory` to inspect; the agent has `memory_*` tools.
- **Hooks** — shell commands on lifecycle events (`PreToolUse`,
  `PostToolUse`, `Stop`, …) that return verdicts; they can only narrow
  permissions. `--hooks-test` dry-runs the pipeline.
- **Advisor** — a second, stronger model the agent consults mid-task for
  strategy, with a per-session budget — or routed to *you* in handoff mode.
- **MCP** — stdio + streamable-HTTP servers. Exa web search is
  preconfigured (needs `EXA_API_KEY`); context7 is one flag away.
- **LSP** — diagnostics from real language servers appended to `write`/`edit`
  results; fail-open, never blocks.
- **Worktrees** — `--worktree <name>` or `/worktree` for isolated branches,
  `/wt-merge` / `/wt-exit` with conflict detection.
- **Multimodal** — `/add image.png` or `@file.pdf`; capability-checked
  against the model.
- **Telemetry (opt-in)** — Sentry/GlitchTip error reports and OpenTelemetry
  metrics (turns, tokens, cost, tool calls) via `[telemetry]`; needs the
  `telemetry` extra, fail-open by design.

## Configuration

Global config: `~/.config/lecode/config.toml` (override the directory with
`LECODE_CONFIG_DIR`); a project-local `.lecode/config.toml` deep-merges over
it. YAML and JSON config files are accepted too. The setup wizard writes the
file with `0600` permissions because it can hold an API key — prefer the
`OPENROUTER_API_KEY` / `OPENAI_API_KEY` env vars to keep secrets out of it.

Full reference: [docs/configuration.md](docs/configuration.md).

## Docs

- [docs/configuration.md](docs/configuration.md) — every config section and default
- [docs/hooks.md](docs/hooks.md) — hook events, envelope, verdict protocol
- [docs/memory.md](docs/memory.md) — the memory store
- [docs/agents-and-skills.md](docs/agents-and-skills.md) — custom agents and skills
- [docs/mcp.md](docs/mcp.md) — MCP servers, Exa, context7
- [docs/build-plan.md](docs/build-plan.md) — the full product definition

## Development

```sh
uv sync
uv run python -m pytest        # 1000+ tests
uv run ruff check && uv run ruff format --check
prek install                   # git hooks: ruff on commit, pytest on push
```

## License

MIT
