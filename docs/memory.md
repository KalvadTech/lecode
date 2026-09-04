# Persistent memory

lecode keeps a project-scoped markdown memory store across sessions. The
agent reads and writes it through tools; you inspect it with `/memory`.

## Layout

Under `<config_dir>/memory/<project-slug>/` (the slug derives from the
project path plus a short hash):

```
MEMORY.md            long-term memory — auto-injected into the system prompt
daily/YYYY-MM-DD.md  daily logs (compaction summaries land here too)
scratchpad.md        project checklist
notes/<name>.md      named notes
```

Every write is atomic (tmp file → fsync → rename) and overwrites first copy
the previous content to `<file>.bak`.

## What the agent sees

`MEMORY.md` is injected into the system prompt on every turn, capped at
`[memory] max_bytes` (default 32768). The agent manages memory with four
tools:

| tool | purpose |
|---|---|
| `memory_read` | read MEMORY.md / a daily log / scratchpad / a note |
| `memory_write` | write (append or replace) one of the memory files |
| `memory_edit` | search/replace inside a memory file |
| `memory_search` | regex search across the store (≤ 50 hits) |

`memory_read` and `memory_search` are read-class (auto-allowed in standard
and readonly modes); `memory_write` and `memory_edit` are write-class.

## `/memory`

```
/memory show              print MEMORY.md
/memory edit              print the uncapped MEMORY.md for manual editing
/memory search <pattern>  search the store
/memory log [date]        print a daily log (today by default)
/memory notes             list named notes
```

## Configuration

```toml
[memory]
enabled = true      # false removes the tools and the injection
max_bytes = 32768   # MEMORY.md injection cap
```
