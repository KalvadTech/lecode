# Agents and skills

Two markdown-driven extension points: **agents** (personas with their own
model/prompt/permissions) and **skills** (domain knowledge packs the model
can load on demand).

## Agents

Agents are markdown files with YAML frontmatter:

- global: `~/.config/lecode/agents/*.md` (`LECODE_CONFIG_DIR`-aware)
- project: `.lecode/agents/*.md` (nearest from the cwd up to the git root)

The project layer wins on name collisions, and user files may override the
built-ins (`build`, `plan`, `explore`, `general`) by name. `/agents` lists them; Tab
cycles the primary agents in the TUI.

```markdown
---
description: Reviews code changes for correctness and style
mode: all              # primary | subagent | all (default: all)
model: openai/gpt-5    # optional model override
temperature: 0.2       # optional
hidden: false          # hidden agents stay out of pickers
color: "#ffaa00"       # optional statusline color
permission:            # optional overlay — can only narrow
  mode: readonly       #   replaces the fallback permission mode
  denied_tools: [bash] #   always denied for this agent
  rules:               #   evaluated before the global rules
    allow:
      bash: [{ pattern: "git diff*" }]
---

You are a careful code reviewer. …
```

- **Primary** agents (`mode: primary|all`) are selectable as the session's
  main agent.
- **Subagents** (`mode: subagent|all`) are invocable mid-turn via the `task`
  tool or an `@mention` in the prompt.
- The `permission` overlay narrows only: a `deny` is unbypassable,
  `denied_tools` always deny, and the overlay's rules/mode can never grant
  more than the global config.

## Persistent workers: review, validate, integrate

`build` is the primary coding agent and `plan` is the read-only primary.
`explore` is a read-only subagent; `general` is a general-purpose coding subagent
that can write, subject to all inherited permissions. It does not grant access
denied by the parent, including a read-only parent. Global/project agent files
still override built-ins by name, and hidden/primary-only agents are not eligible
for delegation. The system prompt lists eligible subagents at runtime creation.

Create a coding worker with
`task(agent='general', prompt='Implement and verify ...', run_in_background=True)`.
The result returns the actual `worker_id`. Then use
`workers(action='send', id=<returned worker_id>, text='Follow-up ...')` to send
feedback, or `workers(action='list')` to discover existing IDs. `workers` controls
existing workers only: sending to an invented ID or an agent name does not spawn
one. Omit `run_in_background` to wait for the task's answer.

Writable workers require a Git repository with a committed HEAD and an attached
branch. If the session started outside such a repository, restart it from the
repository: a shell `cd` does not change the session's runtime cwd.

The `task` tool starts persistent workers. Read-only workers share their parent's
cwd. Write workers get an isolated branch and checkout based on their immediate
parent's committed HEAD. Their sidecar pins the parent checkout, branch, and
base commit, including for nested workers. Dirty parent changes require human
confirmation because they are not copied into the child.

The supervising model can carry out the following workflow without asking for
routine review/integration approval, subject to the normal tool permissions:

1. Let the worker finish, or stop it and wait for it and its descendants to become
   idle. The worker may checkpoint changes with `bash` on **its own branch**.
2. Call `workers` with `{"action":"inspect","id":"WORKER_ID"}` (`review` is
   an alias). This returns the assignment, pinned parent/base, current parent and
   worker HEADs, and the actual committed diff. Review that exact diff against
   the assignment. A worker's completion message alone is not a review.
3. If changes are needed, use `send` to request them and review again. Inspection
   of dirty workers also returns their tracked uncommitted diff and untracked
   paths. Commit/checkpoint in the worker, resolve any merge, and inspect again
   before integration. These controls never commit the user's root checkout.
4. After accepting the diff, the **immediate supervisor** explicitly calls
   `{"action":"integrate","id":"WORKER_ID","reviewed_head":"WORKER_HASH","reviewed_parent_head":"PARENT_HASH"}`.
   Both must be the exact full hashes returned by the accepted review, not fresh
   lookups at integration time.
   General controls can address descendants, but a grandparent model cannot
   integrate a grandchild directly. Integrate the grandchild into its parent,
   then review and integrate that parent separately.
5. Configured validation commands run in the child's cwd through the registered
   `bash` permission gate. Fresh worker-bound permission and hook contexts remain
   constrained by the supervisor. A denial, hook rewrite, execution error,
   missing process exit result, timeout, or failed check blocks integration.
   The integration primitive rechecks both checkouts before fast-forwarding the
   pinned parent. Any changed worker or parent HEAD, including a parent rewind,
   demands another review; hashes are never automatically refreshed. It never
   blindly merges after a validation failure.
6. Once integrated, call `{"action":"cleanup","id":"WORKER_ID"}` if the
   workspace is no longer needed. Cleanup refuses uncommitted or unmerged work.
   Clean up nested child workspaces before their parents. Worker records,
   transcripts, and sidecars remain available; there is no automatic discard
   or push.

Configure project validation in TOML (see [configuration](configuration.md)):

```toml
[worktree]
validation = ["uv run ruff check", "uv run ruff format --check", "uv run python -m pytest"]
```

Only this configured command list is accepted. Models cannot supply replacement
checks or an `allow_unvalidated` flag. An empty list requires an explicit human
confirmation even with tool auto-approval enabled. Without a human callback,
unvalidated integration is unavailable. Confirmations identify the worker,
agent, and cwd.

### Human controls

`/agent` lists workers; use either a worker ID or its roster number:

| Command | Effect |
| --- | --- |
| `/agent ID` | Open the retained transcript/detail view |
| `/agent ID inspect` | Show assignment, pinned destination, both HEADs, and diff |
| `/agent ID integrate WORKER_HASH PARENT_HASH` | Validate and integrate using both full reviewed hashes |
| `/agent ID cleanup` | Remove a clean, integrated workspace; retain its transcript |
| `/agent ID recover` | Ask before recreating a missing checkout from committed history |
| `/agent ID send TEXT` | Queue feedback or a follow-up |
| `/agent ID stop [tree]` | Stop the worker, optionally its descendants |
| `/agent ID resume [TEXT]` | Resume a retained worker |
| `/agent ID submit` | Submit a human-started worker's result to its parent |
| `/agent ID focus` | Focus the composer on that worker |

Mutating slash controls use the same `workers` dispatch permissions as model
calls. Nested integration first passes the current root's permission gate, then
runs in its immediate supervisor's context with the complete ancestor policy.
Validation also respects current root restrictions after a root agent switch;
neither human controls nor child grants widen ancestor Deny or Ask decisions.
Cleanup has no force/discard argument. Recovery requires human confirmation and
warns that old uncommitted data is unrecoverable.

Before a retained write worker follows up or resumes, its workspace guard
reconciles committed parent progress into a clean checkout. Dirty work is
preserved; conflicts remain in the worker for resolution. Missing checkouts
fail until a human confirms recovery. Workspace maintenance reserves the idle
worker against concurrent follow-ups, resumes, and nested worker creation.

## Skills

Skills are `SKILL.md` packs:

- project: `.agents/skills/<name>/SKILL.md`
- global: `~/.agents/skills/<name>/SKILL.md` (`LECODE_SKILLS_DIR` overrides)

```markdown
---
name: pdf              # defaults to the directory name
description: Work with PDF files — extract, merge, inspect   # required
register_cmd: true     # also expose as /<name>
cmd_info: "usage help for the slash command"
---

# Working with PDFs

Use `pdftotext` for extraction …
```

The name + description listing goes into the system prompt, so the model
knows the pack exists and can read it when relevant. With
`register_cmd: true`, `/pdf` injects the skill body as a user prompt (any
arguments are appended). Project packs win on name collisions. Everything is
fail-open: unreadable or invalid packs produce warnings, never crashes.

## Related

- AGENTS.md in the project root is injected into the system prompt
  automatically (walk up to the git root); `/init` writes a starter file.
- Personas (`prompts/personas/*.md`) are one-turn prompt overlays:
  `.persona <text>` in the input, or `[llm.system_prompt] persona = "name"`.
