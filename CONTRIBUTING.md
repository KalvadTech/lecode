# Contributing

lecode is a small, dependency-shy codebase. Contributions are welcome; the
bar is that the checks below pass and the change fits the project's
minimalist bent.

## Setup

```sh
git clone git@github.com:KalvadTech/lecode.git
cd lecode
uv sync --extra telemetry   # telemetry extra is exercised in CI
prek install                # git hooks: ruff on commit, pytest on push
```

Python 3.15+ is required. CLI startup needs `fd`, `rg`, and `rtk` on `PATH`
(search tests skip without their binaries; CI uses an rtk pass-through shim).

## Checks — run in CI order

```sh
uv run ruff check
uv run ruff format --check
uv run python -m pytest
```

Focused test: `uv run python -m pytest tests/test_agent_builder.py`; append
`::test_name` for a single case.

## Conventions

- **Read [AGENTS.md](AGENTS.md) first.** It documents the runtime boundaries
  (builder wiring stays network-free, tool dispatch goes through the
  permission pipeline, TUI preserves scrollback, sessions are append-only
  JSONL) and the test-isolation rules.
- Tests isolate `LECODE_CONFIG_DIR`, `LECODE_SKILLS_DIR`, and the project cwd
  under `tmp_path`; reuse `tests/fakes.py` for providers and catalogs.
- Commits follow conventional commits: `fix:` → patch, `feat:` → minor,
  `feat!:` (or any `!`) → major. Releases are manual; `pyproject.toml`
  carries a static version.
- Docs live in `docs/`. If you change behavior the README or a doc file
  describes, update it in the same commit.

## Pull requests

Keep them focused — one change per PR. A maintainer will review; CI must be
green on Linux and macOS.
