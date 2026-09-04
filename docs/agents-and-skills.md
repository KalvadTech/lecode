# Agents and skills

Two markdown-driven extension points: **agents** (personas with their own
model/prompt/permissions) and **skills** (domain knowledge packs the model
can load on demand).

## Agents

Agents are markdown files with YAML frontmatter:

- global: `~/.config/lecode/agents/*.md` (`LECODE_CONFIG_DIR`-aware)
- project: `.lecode/agents/*.md` (nearest from the cwd up to the git root)

The project layer wins on name collisions, and user files may override the
built-ins (`build`, `plan`, `explore`) by name. `/agents` lists them; Tab
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
