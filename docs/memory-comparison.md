# Memory comparison: lecode and observational memory

Date: 2026-09-15

## Recommendation

**Keep lecode's Markdown memory, explicit tools, and JSONL sessions. Fix context
refresh and compaction correctness first. Then, if session evaluations justify
it, add incremental, source-linked working memory and on-demand transcript recall.**

Pi's design is conceptually better suited to automatic long-session continuity,
but it is not proven better overall or cheaper, and its current implementation
is not recommended as an as-is replacement. Mastra OM with retrieval enabled is
the stronger prebuilt candidate to evaluate, not a demonstrated benchmark winner
or Python drop-in. Letta MemFS supplies useful techniques, not a migration
recommendation. These are fit assessments based on the evidence below.

## Scope and evidence

The original name “laroute” referred to **lecode in this workspace**:
[KalvadTech/lecode](https://github.com/KalvadTech/lecode), inspected at
`0fb73ad25d8af0eddc55547ba2cff7730706f158`. It is Python package version 0.2.0
([pyproject.toml](../pyproject.toml), lines 1-6). Local links below refer to that
baseline; line numbers describe the inspected revision.

[pi-observational-memory][pi-root] was inspected at
`78a1efcfdd46332253fb289724f05b26dfc7769e`. Mastra source checks used
`274c51875045d0d247e963ddd0226b6051a1cd20`; Mastra and Letta documentation was
consulted on the date above, including Context7 queries after resolving library
IDs. External risks are source analysis or upstream reports, not reproduced
failures. No real-world memory-fidelity or cost benchmark was conducted.

## Compact comparison

| Dimension | lecode today | Pi observational memory |
|---|---|---|
| Scope | Notes shared across sessions with the same resolved cwd | Session-isolated notes; forks seed from parent |
| Automatic extraction | Model chooses explicit memory writes; separate compaction | Parallel Observers extract facts; one Consolidator maintains topics |
| Retrieval/provenance | Regex note search; source JSONL exists, no dedicated transcript-recall tool | Topic map plus file reads/search; observations have chunk coverage, not individual source pointers |
| Compaction | Re-summarizes older logical history; keeps four messages | Deterministic observation rendering plus journey/map and boundary-aligned raw tail |
| Cost | Memory reads/writes need no separate memory model; compaction calls current model | Additional Observer/Consolidator calls; worker-cost telemetry |
| Integration | Existing Python implementation | Pi-specific TypeScript extension and Pi subprocesses |
| Reliability | Atomic note writes and one backup; context/compaction gaps | Atomic files and observer coordination; coverage/promotion gaps |

Sources: [lecode store][local-store], lines 41-287; [tools][local-tools], lines
51-234; [compaction][local-compact], lines 45-84; [session replay][local-session],
lines 378-453; [Pi README][pi-readme], [Observer][pi-observer],
[compaction hook][pi-compact], [Consolidator][pi-consolidator], and
[cost tracking][pi-cost].

## What lecode already provides, and what needs fixing

### Durable project memory is a useful foundation

The store lives under the config root, keyed by resolved cwd plus a hash, with
`MEMORY.md`, daily logs, named notes, and scratchpad. This is cwd-scoped, not a
repository-wide identity shared automatically by different checkout paths.
Writes use temporary files, `fsync`, and replacement, retaining one `.bak`;
there is no project-level read-modify-write lock
([store.py][local-store], lines 41-117).

The model explicitly writes/edits durable facts. Reads access current disk
contents and are uncapped unless pagination is requested. Search is a linear,
case-insensitive regex scan, limited to 50 hits in fixed order: long-term,
daily newest-first, notes, scratchpad. This is simple recall, not semantic
ranking ([tools.py][local-tools], lines 51-229;
[store.py][local-store], lines 222-266).

### Injection is a snapshot, not refreshed memory

Runtime construction renders memory into the system prompt; the TUI reuses that
prompt across turns. Tool writes do not refresh this injected snapshot, although
tool responses and explicit reads remain available
([builder.py][local-builder], lines 125-136;
[app.py][local-app], lines 1462-1481 and 1517-1520).

The default long-term cap is **32,768 bytes**, preserving the prefix while newer
appends go at the end. Recent corrections can therefore fall outside injection.
Scratchpad is injected without that cap; daily logs and notes are not injected.
The cap does not bound the entire prompt. Subagents share memory tools but do
not receive automatic memory-text injection
([store.py][local-store], lines 26-27, 59-63, 125-146, 269-287;
[subagents.py][local-subagents], lines 127-159).

### Compaction currently has correctness gaps

The shared compactor loads original logical messages, keeps the last **four
messages, not four turns**, serializes older role/content text, and truncates
that input to approximately 100,000 characters. It calls the current model and
records `summary` plus `keep_from_seq`. Repeated compaction does not combine the
previous summary with only new messages. There is no enforced output-token cap
or tool-call/result boundary protection
([compaction.py][local-compact], lines 20-84).

Consequently, later portions of the older history can be omitted from the
summarizer input while their messages are removed from replay. Splitting a tool
exchange can also produce an unsuitable tail. These are **active-context
omissions**, not deletion of the original JSONL: replay selects the latest
summary and tail while stored messages remain
([storage.py][local-session], lines 378-453).

Automatic budgeting is incomplete too. Usage resets to zero per runner call,
and compaction considers the preceding response's input usage before the next
iteration. It does not preflight the first request of the next user turn;
missing usage and fresh large tool results can escape the trigger. Defaults are
a 200k fallback window, 20k buffer, and continuing on overflow, not a hard
end-to-end budget ([runner.py][local-runner], lines 272-341 and 605-652;
[config/models.py][local-config], lines 52-70).

Finally, [memory.md](memory.md), line 13, says compaction summaries reach daily
logs. The compactor does not call `flush_summary`; that helper has test-only
callers. Treat this as a documentation/implementation mismatch, not an existing
memory pipeline ([compaction.py][local-compact], lines 45-84;
[store.py][local-store], lines 180-187;
[store tests](../tests/test_memory_store.py)).

## What Pi improves, and its limits

Pi extracts timestamped observations from independent transcript chunks, commits
them to a branch-local ledger, and renders them deterministically at compaction.
There is **no separate Reflector worker**: a Consolidator rewrites older facts
into session topic files and a descriptive journey. The compaction hook waits
for relevant observers and preserves tool-safe chunk boundaries. This separates
incremental extraction from prompt assembly
([Observer][pi-observer], [compaction][pi-compact],
[Consolidator][pi-consolidator]).

Actual defaults are 10k-token chunks, four observers, a 15k pool trigger with a
10k target, 150k context trigger, 20k tail target, and 1k journey target. Workers
default to OpenRouter `z-ai/glm-5.3`; the README example instead shows Sonnet and
different thresholds. Budgets are estimates/targets, not strict caps on the
complete prompt; topic-map growth is uncapped
([configuration][pi-config], [memory-map rendering][pi-map]).

Important limitations:

- **Scope:** fresh sessions do not share notes. Forks copy parent memory once;
  topic files and journey do not roll back with `/tree`
  ([session persistence][pi-session]).
- **Coverage risk:** dispatch advances a watermark before success, and later
  completed chunks can conceal an earlier failed slice. The source does not
  enforce contiguous successful coverage before compaction
  ([Observer][pi-observer], [coverage logic][pi-progress]).
- **Unverified promotion:** a clean Consolidator exit tombstones the supplied,
  still-active batch without verifying saved facts. Tombstones filter the active
  pool; original observation records remain. Failed runs retain observations
  but may leave partial topic rewrites. Topic files have no built-in version
  history, so exact prior contents are not guaranteed recoverable
  ([Consolidator][pi-consolidator], [ledger fold][pi-fold],
  [file tools][pi-tools]).
- **Operational maturity:** version 0.1.0 has 14 test files, but its spawn smoke
  tests cover arguments/IPC rather than real model workers. Unmerged upstream
  reports address NUL-containing prompts and Linux `E2BIG` from oversized argv
  ([manifest][pi-package], [tests][pi-tests], [PR #1][pi-pr1], [PR #3][pi-pr3]).

“Deterministic” does not mean lossless: extraction and topic rewriting remain
model-dependent. Source sessions provide recovery evidence, not guaranteed
automatic recall. Cost tracking sums Pi usage data reported by workers at
`agent_end`; it does not demonstrate net savings and can miss interrupted runs
without a final handoff ([worker accounting][pi-cost], [README][pi-readme]).

## Relevant alternatives

**Mastra OM with retrieval enabled** combines Observer/Reflector compression
with observation-group source ranges and a `recall` tool. Basic transcript
browsing needs no vector database; semantic search is optional. Thread scope
is default; shared resource scope remains experimental and disables async
buffering ([official guide][mastra-guide]).

Its standalone `ObservationalMemory` engine is documented, but requires Mastra
`MemoryStorage`; the illustrated integration uses `Memory`, a processor, and
Mastra `Agent`. Direct use is positioned for experimentation or processor
ordering. No official Python-native OM library was documented in the sources
checked. This is a stronger prebuilt evaluation candidate, not a Python drop-in
or framework-neutral service ([standalone reference][mastra-ref],
[TypeScript framework documentation](https://mastra.ai/docs)).

**Letta MemFS** uses git-versioned Markdown, always-visible `system/` files,
on-demand reference files, and worktrees for background memory updates. Its
provided implementation belongs to the Letta runtime; cloud adoption is
optional because local execution is supported. Borrowing those techniques fits
lecode better than migrating solely for memory
([MemFS][letta-memfs], [self-hosting][letta-local]).

## Third option: improve the existing architecture

### Prerequisites: repair current behavior

1. Refresh the injected memory view at a defined request boundary and budget
   the whole outgoing context, including scratchpad and new tool results.
2. Make compaction cover exactly what replay removes, preserve tool exchanges,
   and retain the previous usable context if generation or persistence fails.
3. Resolve the daily-summary documentation mismatch and explicitly define
   correction/recency behavior for capped project memory.

### Optional capability: source-linked working memory

Use three complementary layers: **bounded curated project memory**, an
**incremental session working summary**, and a **recent tool-safe raw tail**.
Keep durable preferences separate from temporary task state.

When justified, update the working summary from its previous version plus the
newly covered message range. Attach session ID and sequence-range references;
provide bounded, on-demand raw transcript recall for exact errors, identifiers,
and tool output. Existing session IDs, message sequences, and append-only events
are reusable building blocks ([session/model.py](../src/lecode/session/model.py),
lines 18-69; [storage.py][local-session], lines 418-453).

Persist and validate the new summary and its coverage together **before**
advancing the replay cutoff. A failed or empty result must not advance coverage.
Initially reuse the compaction path synchronously. Add background workers only
if measured latency warrants their cancellation, retry, and stale-result
complexity; add vector search only if existing search and source recall fail
the evaluation. These are proposed changes, not implemented capabilities.

## Validation and decision gate

The primary agent ran this unchanged-baseline check:

```bash
uv run --no-sync python -m pytest tests/test_memory_store.py tests/test_memory_tools.py tests/test_memory_commands.py tests/test_compaction.py tests/test_agent_runner.py tests/test_session_storage.py -q
```

Result: **113 passed in 0.54s**. An earlier direct `pytest` invocation failed
collection on `tests.fakes`; module invocation succeeded. These are unit and
integration-wiring checks, not LLM memory-fidelity benchmarks. External tests
were not run.

Compare the current system, corrected baseline, and hybrid on identical
sessions, main model, tool outputs, and context budgets. Record any separate
memory-model choices. Test:

- Cross-session preferences and later corrections, including capped memory.
- Continuity through more than five compactions without redoing completed work.
- Exact recovery of source tool output and identifiers from compressed history.
- Interruptions, failed writes/extraction, resume, and retries without skipped
  coverage or duplicate promotion.
- Total agent **plus memory-worker** input/output tokens, available cached-token
  usage, cost, and latency, alongside task success and supported factual recall.

Adopt incremental extraction only if it improves those outcomes enough to
justify its cost and complexity. External compression/accuracy claims alone
cannot establish that result for lecode.

[local-store]: ../src/lecode/memory/store.py
[local-tools]: ../src/lecode/memory/tools.py
[local-compact]: ../src/lecode/session/compaction.py
[local-session]: ../src/lecode/session/storage.py
[local-builder]: ../src/lecode/agent/builder.py
[local-app]: ../src/lecode/tui/app.py
[local-subagents]: ../src/lecode/extras/subagents.py
[local-runner]: ../src/lecode/agent/runner.py
[local-config]: ../src/lecode/config/models.py
[pi-root]: https://github.com/amosblomqvist/pi-observational-memory/tree/78a1efcfdd46332253fb289724f05b26dfc7769e
[pi-readme]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/README.md
[pi-observer]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/hooks/observer-trigger.ts
[pi-compact]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/hooks/compaction-hook.ts
[pi-consolidator]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/hooks/consolidator-trigger.ts
[pi-cost]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/agent/cost.ts
[pi-config]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/config.ts
[pi-map]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/memory/index-render.ts
[pi-session]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/memory/session.ts
[pi-progress]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/ledger/progress.ts
[pi-fold]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/src/ledger/fold.ts
[pi-tools]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/agent/consolidator/tools.ts
[pi-package]: https://github.com/amosblomqvist/pi-observational-memory/blob/78a1efcfdd46332253fb289724f05b26dfc7769e/package.json
[pi-tests]: https://github.com/amosblomqvist/pi-observational-memory/tree/78a1efcfdd46332253fb289724f05b26dfc7769e/tests
[pi-pr1]: https://github.com/amosblomqvist/pi-observational-memory/pull/1
[pi-pr3]: https://github.com/amosblomqvist/pi-observational-memory/pull/3
[mastra-guide]: https://mastra.ai/docs/memory/observational-memory
[mastra-ref]: https://mastra.ai/reference/memory/observational-memory#standalone-usage
[letta-memfs]: https://docs.letta.com/concepts/memfs.md
[letta-local]: https://docs.letta.com/self-hosting.md
