# Repository guidance

## Development and verification

- Python minimum is 3.12; CI tests 3.13 and 3.14 on Linux/macOS. Use `uv`;
  `uv sync --extra telemetry` includes the optional dependencies exercised in CI.
- Run checks from the repo root, in CI order:
  ```sh
  uv run ruff check
  uv run ruff format --check
  uv run python -m pytest
  ```
- Focused test: `uv run python -m pytest tests/test_agent_builder.py`; append
  `::test_read_only_mode_denies_writes` to run one case.
- CLI startup requires `fd`, `rg`, and `rtk` on PATH. Debian's `fd-find` installs
  `fdfind`, so expose it as `fd`. Search tests skip without their binaries;
  CI supplies an RTK pass-through shim, not real compaction coverage.
- `.pre-commit-config.yaml` runs Ruff checks and formatting on commit, the full
  pytest suite on pre-push. The formatting hook modifies files.

## Runtime boundaries

- `src/lecode/cli.py` dispatches interactive, headless, loop, and chain modes.
  They share `agent/builder.py:build_runtime()` for wiring and
  `agent/runner.py:AgentRunner` for the model/tool loop. Keep builder wiring
  network-free; interactive catalog/MCP startup is deferred until chat opens.
- Route tool execution through `agent/tools/base.py:ToolRegistry.dispatch_result()`:
  it validates arguments and enforces permissions/approval. Auto-approval never
  overrides Deny; hooks and agent overlays can only narrow permissions.
- TUI changes must preserve normal terminal scrollback, not an alternate-screen UI.
  `tui/feed.py` keeps live tokens in the layout and flushes completed text to
  scrollback; partial-line printing through `patch_stdout` can erase streaming output.
  Check `tests/test_tui_feed.py`, `tests/test_tui_app.py`, and
  `tests/test_tui_streaming_pty.py` for rendering changes.
- `session/storage.py` stores append-only JSONL. Replay through `load_for_model()`
  so compaction, clear events, and undo tombstones apply. Keep lock sidecars on
  release: unlinking them allows competing processes to lock different inodes.

## Isolation and sources of truth

- `config/loader.py:load_config()` creates defaults and rewrites migrated config.
  Tests constructing runtimes/config/session stores should isolate
  `LECODE_CONFIG_DIR`, `LECODE_SKILLS_DIR`, and project cwd under `tmp_path`.
  The shared `tool_ctx` fixture is not autouse and isolates only the config directory.
- Reuse `tests/fakes.py` for providers/catalogs. MCP tests launch mock subprocesses
  and localhost servers; disable auto servers as in `tests/test_mcp.py:mcp_config`
  to avoid ambient credentials activating Exa. Keep the SSE shutdown reset in
  `tests/conftest.py`; it prevents order-dependent failures across server tests.
- `docs/build-plan.md` is historical, not the current spec: permissions now have
  two modes, config is TOML-only, and the catalog is live with an empty fallback.
  For config or hook changes, consult `docs/configuration.md` or `docs/hooks.md`
  and reconcile with the implementation.
- Releases use conventional commits and release-please, with wheel/sdist attached
  to GitHub Releases, not PyPI. `.github/workflows/release.yml` separately re-locks
  `uv.lock` after release-please bumps `pyproject.toml`.
