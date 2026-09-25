# Persistent memory upgrade plan

Status: implemented. This document records lecode's memory architecture, delivery
phases, and validation. See [memory.md](memory.md) for current usage and behavior.

## Architecture

Keep the existing storage and add one store:

```
Markdown  -> human-managed notes                (project scope)
JSONL     -> raw transcript + working summary   (session scope)
SQLite    -> auto facts, provenance, revisions, exclusions (project scope)
```

Use incremental extraction and source-linked recall within lecode's existing
runtime. No embeddings, no background worker pool.

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

Implementation status (2026-09-15): implemented with offline scripted-provider
acceptance coverage. **Auto-learning stays off by default.** A subsequently reported
live rejection and successful natural-language smoke test are described in
[memory.md](memory.md); there is no measured recall-quality or cost/latency result.

- Both compaction callers pass live parent context. After a successful summary,
  one bounded call uses the same provider/current model and newly covered raw
  text, with strict JSON and captured exact source ranges. Read-only, disabled,
  child and nonpersistent contexts skip extraction.
- Natural-language update (2026-09-16): the existing model classifies durable user
  preferences within longer messages, with no fixed prefix. Promotion requires
  `text=quote`, an exact substring of one original user record, and keeps the
  temporary-marker check scoped to that quote. The prompt excludes tasks, negative
  or quoted/hypothetical examples and embedded instructions, retaining contextual
  qualifications. These are model judgments, not deterministic intent or injection
  proofs. Fabricated/paraphrased evidence is rejected. Literal local
  `read`-corroborated file observations keep their existing contract;
  conflicts/corrections become inspectable proposals, not automatic revisions.
  Comparison is bounded and model-assisted, not semantic contradiction proof.
- Phase 5 transactions now support automatic remember's generation/source-version
  recheck and normalized exact-text deduplication against all revisions. Source
  snapshots precede the await; stale candidates are discarded. Failed learning
  preserves a successful summary and records call usage separately once.
- Runtime prompt refresh includes only valid durable evidence, whole facts and
  revision/source references, within `facts_max_bytes` and the total `max_bytes`
  budget including scratchpad. `/memory facts` and read-class `memory_list` expose
  IDs/status and the latest learning proposals. Base prompt edits remain separate
  from managed injection; runtime/session-store close APIs release fact connections.
- Scripted public-path checks cover opt-in/off, cross-session injection, six
  compactions, exact source recall, unsupported/tool-instruction rejection,
  duplicates/conflicts, await races/failures, source invalidation, corrections,
  clear/undo/redo, byte bounds and combined usage. Individual files and focused
  subsets were used during each phase. Final validation below includes the full
  suite; no live-provider benchmark was run.
- Actual configuration, model-classification and exact-source checks, bounded-context and
  proposal-inspection limitations, and the paired baseline/hybrid evaluation
  checklist are in [memory.md](memory.md).

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

Natural-language update (2026-09-16): the exact multiline regression failed before
the substring change, then passed; three prefix-free variants failed before the
grammar removal, then passed. A sanitized-marker regression failed before checking
the original captured record, then passed. Final focused validation:

```bash
uv run --no-sync python -m pytest tests/test_memory_learning.py tests/test_memory_recall.py tests/test_compaction.py -q
uv run --no-sync ruff check src/lecode/memory/learning.py tests/test_memory_learning.py
uv run --no-sync ruff format --check src/lecode/memory/learning.py tests/test_memory_learning.py
git diff --check
```

Result: **148 focused tests passed**, including conflicts/corrections and
stale-generation races. The subsequent full suite passed **1,522 tests**; Ruff lint
and formatting across `src` and `tests`, and whitespace checks passed. The user
also confirmed a live multiline preference was learned as a valid source-linked
fact. This smoke test is not a systematic classification evaluation.
Fake-provider responses test the grounding/recall contract, not semantic
resistance to quoted prompt injection. No standalone typecheck was run.

Individual touched test files and focused subsets were run during each phase.
Historical phase-6 validation, before the natural-language update: after the final
two-axis review and its fixes, the orchestrator ran:

```bash
uv sync --locked --extra telemetry
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
git diff --check
```

Result: **1,484 tests passed**; lint, formatting and whitespace checks passed.
The telemetry extra was already declared and locked; no dependency files changed.
No standalone typechecker is configured or installed, so no typecheck result is
claimed. No live-provider recall, cost or latency benchmark was performed.

Final review: no remaining hard Standards findings or Spec findings. A small
duplicate recall-context construction remains a nonblocking maintainability note.
