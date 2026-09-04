# lecode

A minimalist terminal AI coding agent. Talks to OpenRouter and any generic
OpenAI-compatible API (local servers included), executes tools under a rich
permission system, persists sessions to disk, and ships power features:
subagents, MCP servers (Exa auto-configured, context7 one flag away), git
worktrees, LSP diagnostics, persistent memory, an advisor model, loop mode,
prompt chaining, lifecycle hooks, and session export.

## Requirements

- Python 3.12+
- External binaries on `PATH` (no auto-download, no fallback — startup
  verifies them):

  | binary | install |
  |--------|---------|
  | `fd`   | `brew install fd` · `apt install fd-find` · `cargo install fd-find` |
  | `rg`   | `brew install ripgrep` · `apt install ripgrep` · `cargo install ripgrep` |
  | `rtk`  | see <https://github.com/rtk-ai/rtk> |

## Install

```sh
uv tool install lecode
```

(If the `lecode` package name is ever taken/conflicting on PyPI, the project
is also published as `lecode-agent` — the command stays `lecode`.)

For development:

```sh
uv sync
uv run lecode --version
```

## Quickstart

```sh
lecode --setup        # provider, API key, model, theme — 30 seconds
export OPENROUTER_API_KEY=sk-or-...   # or keep the key in config.toml
lecode                # name the session, then ask for something
```

The interactive UI opens with a session-name prompt; from there just type.
`/help` lists the slash commands, `/welcome` shows the key bindings,
`/tutor <topic>` explains a feature. Ctrl-C cancels a turn; `/quit` exits.

## Headless usage

```sh
lecode -p "write a haiku about this repo"     # one prompt, then exit
git diff | lecode -p "review this diff"       # a bare -p reads stdin
lecode --loop plan.md --loop-cmd "make test"  # iterate until the plan is done
lecode --chain "redesign the parser"          # brainstorm→plan→code→review
```

Exit codes: `0` done · `1` error · `2` startup (missing deps, bad flags,
non-tty `--setup`) · `3` max turns / max loop iterations.

## Configuration

Global config lives in `~/.config/lecode/config.toml` (env var
`LECODE_CONFIG_DIR` overrides the directory); a project-local
`.lecode/config.toml` deep-merges over it. The setup wizard writes the file
with `0600` permissions because it can hold an API key — prefer the
`OPENROUTER_API_KEY` / `OPENAI_API_KEY` env vars if you'd rather keep secrets
out of the file.

See [docs/configuration.md](docs/configuration.md) for the full reference.

## Feature highlights

- **Permissions**: six modes, glob/regex rules, doom-loop guard, per-agent
  overlays — `/permissions` and `/mode`.
- **Sessions**: JSONL on disk, `/new` `/resume` `/undo` `/rewind` `/compact`
  `/handoff`, HTML export and gist sharing.
- **Custom agents & skills**: markdown-defined agents with permission
  overlays, `SKILL.md` packs, `@mentions`, subagents via the `task` tool.
- **MCP**: stdio + streamable-HTTP servers; Exa web search works out of the
  box with `EXA_API_KEY` — see [docs/mcp.md](docs/mcp.md).
- **LSP**: diagnostics appended to `write`/`edit` results, fail-open.
- **Memory**: persistent markdown memory across sessions — `/memory`, and
  [docs/memory.md](docs/memory.md).
- **Hooks**: shell commands on lifecycle events that can narrow permissions —
  [docs/hooks.md](docs/hooks.md).
- **Worktrees**: `/worktree`, `/wt-merge`, `--worktree` for isolated changes.

More docs: [docs/agents-and-skills.md](docs/agents-and-skills.md),
[docs/build-plan.md](docs/build-plan.md) (the full product definition).

## Test

```sh
uv run python -m pytest
uv run ruff check
uv run ruff format --check
```
