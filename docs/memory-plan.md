# Persistent memory upgrade plan

Status: approved for implementation (2026-09-15). Design rationale and external
evidence live in [memory-comparison.md](memory-comparison.md).

## Architecture

Keep the existing storage and add one store:

```
Markdown  -> human-managed notes                (project scope)
JSONL     -> raw transcript + working summary   (session scope)
SQLite    -> auto facts, provenance, revisions, exclusions (project scope)
```

Borrow Pi's incremental extraction and Mastra's source-linked recall without
adopting either framework. No embeddings, no background worker pool.

Delivery order: fix correctness, then storage/scope, source recall, incremental
working memory, forgetting safeguards, and finally enable automatic learning.
Automatic durable learning stays disabled until phase 5 is green.

## Phase 1: Fix compaction and injection correctness

- Prefix coverage in `session/compaction.py`: the summarized range must equal
  the range removed from replay; `keep_from_seq` is the first visible message
  not covered. Never start the kept tail on a `role: "tool"` message. Return
  `None` when nothing is safely coverable.
- Compact events gain the covered range and boundary hashes
  (`session/storage.py`).
- `load_for_model` replays a summary only when its range intersects no active
  tombstone and no later `clear`.
- Shared `refresh_system_prompt(runtime)`; call at run start and session
  switch; replace `history[0]` rather than appending system messages.
- Fix the `docs/memory.md` daily-flush claim.

## Phase 2: Storage and project scope

- New `src/lecode/memory/facts.py`: `FactStore` on stdlib `sqlite3` with WAL,
  `busy_timeout`, `BEGIN IMMEDIATE` writes, hash-derived ids, and tables for
  facts, revisions, provenance, and exclusions.
- `resolve_project_root(cwd)`: git ancestry with linked-worktree resolution to
  the common dir; resolved cwd fallback outside git.
- Wire `ToolContext.project_root` / `scope`, builder effective root, CLI
  worktree pass-through, and keep `set_cwd` from rebinding the durable root.
- Subagents keep read access but lose durable writers.
- `memory_recall` is read-class.

## Phase 3: Source-linked recall

- Source-ref validation: seqs present, boundary hashes match, no corrupt-skip,
  not hidden or excluded. Statuses: `valid`, `missing`, `stale`, `hidden`.
- `memory_recall` returns facts plus bounded exact source text and refuses
  hidden or excluded ranges. Session lookup by id across the project.

## Phase 4: Incremental working memory

- Chain summaries: previous valid summary plus only newly covered messages.
- Bounded output; persist compaction usage in the totals.
- Working summary joins the refreshed prompt.
- New `[memory]` config keys; `auto_learn` defaults to false.

## Phase 5: Forgetting and recovery safeguards

- Forget flow: exclusion row first, then purge managed facts and revisions,
  then a `forget` event under the session lock (deferred if another process
  holds it). No physical JSONL rewrite.
- One filtering predicate covers replay, recall, summarization, and remember;
  intersecting summaries go inert and rebuild lazily at the next compaction.
- Import aliases only when unique and referenced; session deletion orphans
  facts.

## Phase 6: Enable automatic learning and evaluate

- Extraction at safe compaction boundaries with the current model; candidates
  promoted conservatively; contradictions surfaced; facts stored as untrusted
  evidence.
- `auto_learn=false` until this phase passes acceptance.
- Evaluation protocol: identical sessions, model, and budget; measure durable
  recall, >5-compaction continuity, exact tool-output recovery, interruption
  recovery, and total tokens/cost/latency.

## Non-goals

No embeddings or vector database, no background worker pool, no framework
adoption, no cross-project or user-level memory, no committing memory to the
repository, no raw-transcript erasure, no forensic-deletion guarantees, and no
silent legacy-note merging.

## Confirmed defaults

1. Legacy Markdown migration: one-time non-destructive copy when only a
   cwd-scoped store exists; if both exist, keep both and use the project dir.
   Never delete or merge silently.
2. Undo/redo: facts sourced from undone turns are suppressed while the
   tombstone is active and restored on redo.
3. Forget is logical invalidation plus purge of managed rows; raw JSONL stays
   until explicit session deletion.
4. Session deletion orphans facts; forget is the erasure command.
5. Auto-learning reads user and assistant text; tool output only corroborates.
   Facts are injected as data, never instructions.

## Validation

Every phase: `uv run python -m pytest` on the touched files plus
`uv run ruff check`. Full suite once at the end.