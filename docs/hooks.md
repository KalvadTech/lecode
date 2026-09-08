# Lifecycle hooks

Hooks are shell commands fired on lifecycle events. They wrap tool calls and
session events, and they can **only narrow** permission verdicts (Allow →
Ask/Deny) — never widen them. Deny rules stay unbypassable.

## Configuration

```toml
[hooks]
PreToolUse = ["./ci/check-tool.sh", "python3 ./ci/audit.py"]
PostToolUse = ["./ci/log-result.sh"]
Stop = ["say done"]
```

Each event maps to a list of commands. Commands run via `/bin/sh -c` in the
project cwd, with a default 10s timeout each.

## Events

| event | when | extra envelope fields |
|---|---|---|
| `PreToolUse` | before a tool runs (after the permission check) — **enforced** | `tool` |
| `PostToolUse` | after a tool runs successfully | `tool`, `result` |
| `PostToolUseFailure` | after a tool returns an error result or raises — fires **instead of** `PostToolUse` | `tool`, `result` or `reason` |
| `PermissionRequest` | an Ask verdict goes to the interactive approval prompt | `tool` |
| `PermissionResult` | a permission Ask resolves: `decision` is `allow_once` / `allow_always` / `deny` / `auto` (auto-approved) | `tool`, `decision` |
| `UserPromptSubmit` | a prompt is submitted (TUI and `-p`) — **enforced**: deny blocks the prompt | `prompt` |
| `Stop` | an agent run finishes | `reason` (stop reason: `done`, `empty`, `max_turns`, `context_overflow`) |
| `SessionStart` / `SessionEnd` | session lifecycle (chat open/quit, `-p`/`--loop`/`--chain` runs, session switch) | — |
| `SubagentStart` / `SubagentEnd` | subagent (task/@mention) lifecycle | `agent` |
| `PreCompact` / `PostCompact` | around context compaction (`/compact` and automatic) | — |
| `Interrupt` | Ctrl-C cancels a running turn (TUI) or the headless run | — |
| `Notification` | a desktop/sound notification is sent | `kind` (`approval` / `error` / `finish`), `tool` or `reason` for approval/error |

Only `PreToolUse` and `UserPromptSubmit` are enforced (a deny blocks the
tool call / the prompt). Every other event is observational: verdicts are
ignored.

## The envelope

Handlers receive one JSON object on **stdin**:

```json
{
  "event": "PreToolUse",
  "ts": "2026-09-01T12:00:00+00:00",
  "session": {"id": "…", "name": "…"},
  "cwd": "/path/to/project",
  "tool": {"name": "bash", "args": {"command": "ls"}}
}
```

`session` is `null` outside a session. `result` (PostToolUse/PostToolUseFailure)
is `{"content": "…", "is_error": false}`.

## The verdict protocol

Handlers answer with one JSON object on **stdout**:

```json
{"verdict": "allow", "reason": "…", "rewritten_input": {"command": "ls -la"}}
```

- `verdict`: `allow` \| `defer` \| `ask` \| `deny` (severity order:
  deny > ask > defer > allow).
- `reason`: shown to the user/agent when the verdict narrows access.
- `rewritten_input` (optional): replacement tool arguments, applied only for
  `PreToolUse`; taken from the most severe handler that provided one.
- **Abstain**: exit 0 with empty stdout — the handler has no opinion.

When several handlers are configured for one event, the most severe verdict
wins; the first non-empty reason is reported.

## Failure semantics (deny-safe)

A crash, non-zero exit, timeout, or invalid JSON becomes:

- **Deny** for the enforced events (`PreToolUse`, `UserPromptSubmit`; reason
  `hook failed: …`) — a broken guard blocks the tool/prompt rather than
  letting it through;
- a logged **abstain** for every other event.

## Inspecting and testing

- `/hooks` lists the configured handlers.
- `lecode --hooks-test` dry-runs every configured event against a synthetic
  envelope and prints each merged verdict (exit 1 when any handler failed);
  no tool is executed.
