# Persistent memory

lecode keeps project-scoped Markdown notes, session JSONL transcripts and working
summaries, and source-linked SQLite facts. Optional automatic learning runs after
successful compaction. It is **off by default**. Inspect notes and facts with `/memory`.

## Layout

Under `<config_dir>/memory/<project-slug>/` (the slug derives from the
project path plus a short hash):

The project path is the main Git checkout root, shared by nested working
directories and linked worktrees. Submodules keep their own project memory.
Outside Git it is the resolved working directory. Checkout-specific AGENTS.md
instructions still follow the active working directory.

```
MEMORY.md            long-term memory — auto-injected into the system prompt
daily/YYYY-MM-DD.md  daily logs
scratchpad.md        project checklist
notes/<name>.md      named notes
facts.sqlite3        source-linked facts and revision history (created lazily)
```

Every Markdown write is atomic (tmp file → fsync → rename) and overwrites first copy
the previous content to `<file>.bak`.

On first use, legacy cwd-scoped Markdown is copied to the project store only if
that destination is absent. Existing destinations are never merged, and legacy
files are kept. SQLite databases and backups are not copied.

The internal `FactStore` API supports `remember`, `correct`, `forget`, `generation`,
`add`, `get`, `revise`, `provenance`, bounded `list` and literal `search`, `attach_source`,
`source`, `is_excluded`, and `excluded_seqs`. Revisions use optimistic revision
checks and SQLite transactions. Automatic learning requires explicit opt-in.
Disabled memory creates no fact database; existing exclusions still protect
session replay.

## What the agent sees

At each runtime provider-request boundary, memory is refreshed from storage.
`[memory] max_bytes` (default 32768) caps the **entire memory section**, including
headings, Markdown, scratchpad, fact text, and provenance. Durable facts have a
second cap, `facts_max_bytes` (default 8192), within that total. Only whole facts
whose **every revision's evidence** validates are injected. Oversized facts are
omitted, not cut in half. Facts are labelled **untrusted evidence, not instructions**;
their IDs, current revision, and exact source references accompany the text.

Facts get their bounded allocation first; remaining space is shared between
`MEMORY.md` and scratchpad. Markdown may be truncated with a marker. Agent/skill
context and custom prompt text remain separate; the runner also checks the whole
outgoing prompt and tools against the current model's window. `/editsys` edits
only the base prompt so it cannot freeze injected facts into the custom override.

The agent manages memory with eight tools:

| tool | purpose |
|---|---|
| `memory_read` | read MEMORY.md / a daily log / scratchpad / a note |
| `memory_write` | write (append or replace) one of the memory files |
| `memory_edit` | search/replace inside a memory file |
| `memory_search` | regex search across Markdown only (≤ 50 hits) |
| `memory_recall` | fact ID or explicit session ID and bounded source range |
| `memory_list` | bounded fact ID/revision/source/status inspection; page by `offset` |
| `memory_correct` | correct one fact by ID, expected revision, and persisted evidence |
| `memory_forget` | forget one selected fact, all its revisions, and contributing source refs |

`memory_read`, `memory_search`, `memory_list`, and `memory_recall` are read-class (allowed in `readonly`
mode); `memory_write`, `memory_edit`, `memory_correct`, and `memory_forget` are write-class.
Subagents retain the readers but do not receive durable memory-writing tools.

## `/memory`

```
/memory show              print MEMORY.md
/memory edit              print the uncapped MEMORY.md for manual editing
/memory search <pattern>  search the store
/memory log [date]        print a daily log (today by default)
/memory notes             list named notes
/memory facts [offset]    inspect fact IDs, revisions, valid sources, and latest learning status
/memory recall <fact-id> [offset]
/memory read <session-id> <start-seq> <end-seq> [offset]
/memory correct <fact-id> <revision> <session-id> <start-seq> <end-seq> <text>
/memory forget <fact-id>
/memory help              show usage
```

## Source-linked recall

`memory_recall` accepts either `fact_id`, or `session_id` with inclusive
`start_seq` and `end_seq`. Session IDs must be exact IDs, never paths, prefixes,
names, or aliases. Linked worktrees share project identity; unrelated projects
and symlink session files are refused. There is no global transcript scan.

`SessionStore.source_snapshot` captures actual visible messages in at most 200
sequence positions. Events can occupy positions between messages. Its immutable
`SourceRef(session_id, seqs, digest)` records the exact message sequences and a
SHA-256 digest of **all complete referenced JSON records**, canonically encoded
with sorted keys. Tool names, arguments, outputs, usage, and other metadata are
covered, including middle records. `validate_source` rechecks project identity,
IDs, digests, corruption, and session recall visibility on every use.

`FactStore.attach_source` validates storage and existing exclusions before
attaching a reference once to its matching revision. Opening an existing database
adds a `source_refs` table keyed by `(fact_id, revision)` with `session_id`, JSON
`seqs`, and `digest`; existing facts, revisions, provenance, and exclusions remain
intact. Legacy provenance is not automatically trusted or backfilled.

Recall returns JSON with IDs and `valid`, `missing`, `stale`, `hidden`, or
`unverified` status. Only `valid` results contain fact text and source evidence;
these remain untrusted data, not instructions or proof that the claim is true.
Tombstones suppress evidence; redo restores tombstoned evidence. `/clear` resets
working context and direct session recall, but does not revoke existing durable
facts. The shared source validator accepts `purpose="working"` (default, including
new extraction/correction) or `purpose="durable"` (existing fact recall). Both
purposes reject missing, tampered, undone, and excluded sources.
Deleting a source or importing
it under a collision-renamed ID does not remap old provenance. Original IDs are
used only when their exact records still validate; missing identity stays missing.

Output is capped at **16,384 UTF-8 bytes**, including JSON metadata. `source_text`
is a character page of a JSON array containing sequence IDs and original message
structures. Concatenate pages before parsing that array. Pass `next_offset` back
as `offset`; `null` means finished. `limit` defaults to 4096 characters and is
clamped to 8192, with smaller pages when JSON escaping requires it. Fact text is
capped at 512 characters with `fact_text_truncated`. Binary attachment payloads
are replaced with omission markers, while their complete records remain hashed.
Each page is validated anew; compare the returned source digest before combining
pages from direct session recall.

Children receive a read-only `RecallContext` backed by the parent's source store.
Child session persistence remains disabled, and recall does not modify parent
extras or grant children durable-writing tools. Working history remains session
scoped; facts remain in the project store.

## Incremental working memory

`/compact` and automatic compaction use the previous **validated** working
summary plus only newly covered complete exchanges. Tool names, call IDs,
arguments, and results remain in the summarizer input. Binary payloads use
omission markers; the transcript and prior summary are labelled inert data.
An incomplete/cancelled exchange blocks further coverage, and at least four
recent messages remain raw, with a complete tool-safe exchange boundary.

The input is bounded before assembly (at most 100,000 UTF-8 bytes, further
limited by the current model window). Summary output is explicitly capped at
2,048 tokens or the smaller current-model/headroom limit; oversized, empty,
truncated, or tool-calling responses cannot advance the cutoff.

Each durable JSONL compact event contains the summary, exact cumulative
`SourceRef` lineage in chunks of at most 200 sequence positions, the prior
compact event's sequence/digest identity, model, and reported usage. Sources
are fsynced before derivation. Session identity, source bytes, visibility, and
exclusions are rechecked before the fsynced append, without a new lock or SQLite
transaction across the model call. Clear, undo, changed source records, or changed
prior revisions invalidate affected summaries. Legacy summaries without validated
lineage are inert; raw history is replayed and rebuilt lazily.

The runner preflights every request, including the refreshed system prompt,
tools, working summary, and new results. It uses a conservative UTF-8-size
estimate with framing/media allowance and per-model calibration from reported
usage, **not an exact tokenizer**. Catalog limits take precedence; unknown models
use `[agent] context_window`. `[compaction] buffer_tokens` reserves output
headroom, and `mid_turn_threshold` can trigger earlier compaction. If compaction
fails but the request fits, the run continues. An unsafe request pauses even
with `on_overflow="continue"`; chain phases stop on this pause too. Manual
`/compact` refuses an active turn.

Reported compaction usage contributes once to session/run totals, including
rejected responses. Rejected attempts use a separate `memory_usage` event without
changing the summary cutoff. Missing usage is stored as null and counted in
`unknown_usage_calls`; numeric totals are known subtotals, not a zero-cost claim.

`SessionStore.exclusion_reader(session_id)` resolves the source session's durable
project FactStore, with `bind_facts(project_root, facts)` for runtime wiring. Shared
session stores retain separate project bindings; changing a checkout cwd does not
rebind other sessions. Replay, source validation, and compaction use the same
visibility predicate, including the project exclusion generation. Compaction
never writes summaries to Markdown/daily logs. With `auto_learn=true`, a successful
compaction can then make one separate, bounded extraction call described below.

Limits: lineage grows with covered history, and validation currently rereads
JSONL for each bounded snapshot. Large incomplete exchanges can prevent further
compaction; no source content is silently discarded to make them fit.

## Correction, forgetting, and recovery (Phase 5)

`FactStore.remember(text, ref, sessions=..., project_root=...)` atomically creates
a fact with a validated `SourceRef`. `correct(fact_id, text, expected_revision=...,
ref=..., sessions=..., project_root=..., expected_generation=None)` atomically
appends a revision with new evidence. Correction requires persisted user or
assistant text; tool output alone cannot justify it. Evidence identity is checked,
not the semantic truth of the claim. The lower-level `add`/`revise` APIs remain
available for unverified storage, and also enforce exclusions inside their write
transaction. `get`/`search` are administrative, unverified reads: model-facing
consumers must use source-validated recall.

`memory_correct` takes exactly `fact_id`, `expected_revision`, `text`, and
`source_snapshot: {session_id, seqs, digest}`. Obtain the snapshot from a successful
`memory_recall` result's `session_id` and `source` fields. The slash command captures
the specified persisted range before dispatching the same tool. Neither path
guesses evidence from the session's next sequence number. Both mutation commands
use the normal permissions and lifecycle hooks. Children receive recall access,
but no FactStore handle, session writer, or durable-writing tools.

`forget(fact_id)` uses **one `BEGIN IMMEDIATE` transaction** to:

1. Install content-free exclusions for every source position in every revision's
   provenance and attached source ref, including superseded revisions.
2. Record a content-free fact-ID retry marker and advance a monotonic generation.
3. Delete the selected fact, revisions, provenance, and attached refs.

It returns `True` on the first successful forget and `False` on retry. Unknown IDs
fail clearly. Overlapping facts are retained administratively but suppressed from
recall, and excluded evidence cannot be added, revised, or attached again. No
pattern-based bulk deletion, source-session scan, or source-session deletion is
performed. Exclusions are authoritative: this implementation needs no JSONL forget
marker or pending journal. A crash immediately after SQLite commit cannot make
the evidence visible again. SQLite transactions do not include JSONL writes.

Forget makes intersecting summaries logically inert; raw JSONL is unchanged.
Compaction attempts a lazy rebuild from filtered context. If no safe complete
prefix remains and context exceeds the request budget, the runner pauses.

To prevent recalled text from being rephrased into fresh provenance, managed
generated messages carry the exclusion generation captured before their producing
request. **Any project forget invalidates older generated messages project-wide**,
including legacy assistant/tool messages without an epoch, recalled tool results,
handoff seeds, and background output. This deliberately suppresses some unrelated
generated content, without deleting independently authored user messages or
Markdown. Precise transitive lineage could narrow that cutoff in a later phase.
Chain phases and child requests retain their input epoch; background-task reads
and notifications also reject older epochs.

The live runner rebuilds cached history when the epoch changes, checks again
before provider calls and before accepting responses, and stamps persisted output
with the original request epoch. A forget during a provider await discards the
stale response instead of promoting it. Corrections derived from old requests
also fail their epoch check inside the SQLite transaction.

**Limits:** a request already sent to a provider cannot be recalled, and tokens
already streamed to the terminal cannot be unsent. Raw history and exports remain;
this is logical managed-memory invalidation, not forensic erasure. Existing
Markdown, manual external copies, and ordinary shell/raw-file reads are outside
this guarantee; there is no OS sandbox claim. Deleting a source session orphans
facts instead of purging them, and imports never fabricate source aliases.
Session deletion respects attach locks and retains the lock sidecar inode.

## Optional automatic learning (Phase 6)

Both `/compact` and automatic runner compaction use
`compact_session(..., config=..., catalog=..., ctx=..., on_usage=...)`. The live
parent `ToolContext` is required for learning. The current model ID, configured
thinking level, and the same provider instance are used; there is no provider
switch, background worker, embedding service, or new dependency.

Learning runs only **after the working summary was successfully persisted** and
only when `memory.enabled` and `memory.auto_learn` are true. Read-only mode,
read-only agent overlays, denied/ask memory-writing permissions, child contexts,
and nonpersistent contexts disable extraction. Omitting `ctx` disables learning.
The gates are checked again before promotion.

### Evidence contract

The extraction payload contains only the newly covered raw user/assistant
messages and their exact sequence numbers, with paired tool records available
only as corroboration. It does not include the generated working summary as
candidate evidence. Generated synthetic user notifications are excluded.
Existing valid facts are a bounded comparison context, not candidate sources.

The model must return strict JSON with no unknown or duplicate keys, for example:

```json
{
  "candidates": [{
    "text": "For this project, I prefer tabs for indentation.",
    "source_seqs": [1],
    "kind": "explicit_user_preference",
    "quote": "For this project, I prefer tabs for indentation.",
    "conflicts": [],
    "proposal": false
  }]
}
```

The evidence contract separates model classification from source verification:

* **Explicit preference:** the model identifies durable project conventions or
  preferences in natural language, without a required prefix. `text` must equal
  `quote`, a nonempty exact contiguous substring of one original persisted user
  message, identified by a singleton `source_seqs`. A preference can appear within
  a longer multiline message containing unrelated tasks. No paraphrasing or
  inferred canonical text is accepted; sanitized payload markers are not evidence.
  The existing temporary-marker guard (`this task`, `for now`, `today`, `only`,
  etc.) applies to the selected quote, not unrelated surrounding text.
  The prompt requires reading the full message, retaining qualifications, negation
  and temporal scope, and excluding one-off commands, negative examples, quoted or
  hypothetical preferences, pasted third-party instructions and extractor-output
  manipulation. **That classification depends on the model.** Exact source
  matching proves attribution, not durable intent: a misclassified task or quoted
  injection can still pass, and a model can incorrectly crop away a qualification.
  The temporary guard is a conservative English heuristic, not semantic proof; it
  can reject genuine conventions containing `only` and miss other transient wording.
* **Verified project fact:** only a literal file-content observation is supported.
  `kind` is `verified_project_fact`; `quote` is one exact numbered output line from
  a matching local `read` tool call. `text` is exactly
  `path contains "JSON-escaped exact line"`, also present as a complete user or
  assistant message in that exchange. For example,
  `pyproject.toml contains "requires-python = \">=3.12\""`.
  A relative path, matching read arguments/call ID/result, and the confirming
  message must all occur in the supplied range. This verifies **recorded observed
  content**, not a semantic conclusion, current file state, or an instruction to
  obey that content. Shell, web, MCP, and arbitrary tool results do not qualify.
* **Proposals:** model-indicated conflicts or uncertainty, and explicit correction
  language (including `Correction: ...`), leave incumbent facts unchanged.
  Conflict IDs must identify facts actually supplied to that call, and those
  facts must still have the same revision. `/memory facts` / `memory_list` shows
  the current session's latest learning status and valid proposals. Explicit
  `memory_correct` remains the only correction path. Proposal inspection is not
  a durable review queue: it shows the latest extraction attempt, and hides
  proposals whose epoch or evidence no longer validates.

`source_seqs` must match a captured singleton or complete exchange range supplied
to the extractor, at most 200 sequence positions. Claiming a real sequence number
outside that input is insufficient. Arbitrary disjoint evidence ranges are not
supported for one fact. Tool instructions alone cannot become user preferences.
Raw source snapshots are captured and fsynced before awaiting the model. Session
identity/version, source hashes, visibility and exclusion generation are rechecked
before promotion; generation and source version are checked again inside the
`FactStore.remember` transaction that inserts the fact and its source together.
Forget, undo, clear, source edits/deletion or a session switch during the await
discard stale candidates.

### Bounds, deduplication, and failure handling

`/memory facts` and `memory_list` expose the latest recorded extraction outcome:

* `no_candidates`: the model returned a valid, empty candidates array.
* `rejected`: candidates were returned, but none passed validation.
* `learned` / `proposed`: validated evidence was remembered (possibly deduplicated)
  or left for explicit review. Other candidates in the same response may be rejected.
* `failed`: the response could not be processed or extraction failed.
* `stale`: context or evidence changed before promotion.

New learning events include `reason_counts`, a bounded map of fixed codes to counts.
Each rejected candidate contributes its **first failing check**, not every possible
reason. Response-level failures contribute one count. Codes are:

| Codes | Meaning |
|---|---|
| `response_schema`, `invalid_json` | Invalid completion/envelope, candidate limit, or JSON (including duplicate keys) |
| `output_too_large`, `unexpected_tool_calls`, `incomplete_response` | Output byte limit, tool calls, or unsupported finish reason |
| `candidate_schema`, `invalid_text` | Candidate fields/kind/proposal flag, or text/quote type, encoding, or bounds |
| `invalid_source`, `invalid_preference_source` | Unavailable/noncaptured sequence range, or preference not backed by one user message |
| `temporary_preference`, `exact_text_mismatch` | Selected quote has a temporary marker, or text differs from quote / quote is absent from the original user text |
| `invalid_preference` | Legacy fixed-prefix rejection; retained for reading older diagnostics |
| `invalid_conflicts`, `unverified_project_evidence` | Unknown conflict references or missing exact read corroboration |
| `stale_context`, `extraction_error` | Context changed or an otherwise unclassified extraction error |

Diagnostics contain no rejected text, quotes, raw provider output, or exception
messages. Existing validated proposals retain their source-checked inspection.
Old events without counts remain readable; their `rejected` status cannot
retrospectively distinguish empty output from invalid candidates. Cancellation
still propagates with accounting in `finally`; preparation skips before a call
do not create an extraction event. These diagnostics describe checks, not why a
provider chose its response, and do not guarantee learning.

* One extraction call per successful boundary; payload at most **16,000 UTF-8
  bytes**, further limited by the current model window and reserved headroom.
  The fixed extraction prompt is included in the model-window check. Only the
  fitting portion of a large newly covered prefix is considered; omitted learning
  input is not retried later, but raw recall and summary coverage remain intact.
* Output at most **2,048 tokens**, or smaller catalog/headroom limit, and at most
  three UTF-8 bytes per allowed output token. At most **four candidates**, with
  each text/quote at most **512 UTF-8 bytes**. Empty, malformed, truncated,
  tool-calling or unsupported completions cannot promote unsupported facts.
* Existing-fact comparison inspects at most 50 ID-ordered facts and supplies at
  most 4,096 serialized bytes. Normal injection also inspects at most 50 facts;
  `memory_list` pages 20 at a time with a 16,384-byte output cap. Use recall/list
  for facts that do not fit automatic context.
* Automatic `remember(..., deduplicate=True)` compares **case-folded,
  whitespace-collapsed exact text** against all stored revisions, including
  corrected-away text. An equivalent new source returns the incumbent without
  reattaching or reverting it. Canonical text-plus-source IDs still prevent
  retries. There is no inferred semantic key, fuzzy deduplication, or algorithmic
  contradiction proof; the model may miss conflicts outside its bounded context.
* Each learning call records its usage exactly once in a separate
  `memory_usage` event with `purpose="learning"`, including rejected output.
  Summarization usage remains on the successful compact event. Both count in run
  and session totals; absent usage counts as unknown, not zero. If the source
  session was deleted, no usage event can be appended there; the run callback
  still accounts for the call.
* Extraction failure does not rewind a successful summary or delete raw text.
  Cancellation can propagate, but the already-written summary survives. Existing
  forget/undo invalidation still applies independently; in particular Phase 5's
  conservative generation cutoff can invalidate unrelated older generated text.

`memory_write` and `memory_edit` still operate on Markdown only. Automatic learning
does not modify those files. `/clear` preserves durable facts, undo suppresses
their evidence and redo restores it, and missing/corrupt source sessions orphan
facts out of normal injection. For embedded use, stop in-flight work and call
`Runtime.close()` (or `SessionStore.close()` / `FactStore.close()` for standalone
stores). CLI and TUI shutdown paths close memory connections.

## Configuration

```toml
[memory]
enabled = true      # false removes the tools and the injection
max_bytes = 32768   # total memory section, including scratchpad, labels and facts
auto_learn = false  # explicitly set true to opt in at successful compaction boundaries
facts_max_bytes = 8192 # validated 0..65536; whole valid facts within the total cap
```

Example: enable `auto_learn = true`, send
`We use tabs for indentation in this repo.` (on its own or within a longer
message), then continue through enough
complete exchanges to compact that message. Run `/compact` or let normal
compaction trigger, and inspect `/memory facts`. A new independent session in
the same project can receive the validated fact. **Learning is not guaranteed**:
the model must emit a qualifying candidate and the evidence and budgets must
validate. Use the source ID from inspection with `/memory recall <fact-id>`.

## Reproducible evaluation protocol

The earlier reported live opt-in `/compact` check returned `rejected` with no
proposals or facts; raw extraction output was unavailable. The later multiline
release-notes case exposed the old whole-message `exact_text_mismatch` restriction.
After the natural-language change, the user confirmed a successful live smoke
test with `z-ai/glm-5.3-flash`: a release-note preference embedded in a multiline
message became a valid source-linked fact after compaction, with status `learned`
and no rejection reasons. This is one observed success, not a quality benchmark.
The change also has offline scripted-provider coverage. These tests verify exact quote acceptance,
source recall and validation failures; they do not measure the model's ability to
distinguish real preferences from tasks, negative examples or quoted injections.
The scripted provider tests
are acceptance checks, not recall-quality, cost-saving or latency benchmarks.

Offline reproduction using already-installed development dependencies:

```sh
uv run --no-sync python -m pytest tests/test_memory_learning.py -q
uv run --no-sync python -m pytest tests/test_memory_facts.py -q
uv run --no-sync python -m pytest tests/test_compaction.py -q
uv run --no-sync python -m pytest tests/test_memory_recall.py -q
uv run --no-sync python -m pytest tests/test_agent_runner.py -q -k 'memory or compact or context or refresh or generation'
```

For a future live comparison, use two isolated disposable config directories and
identical project fixtures, initial Markdown, scripted sessions, model/provider,
sampling settings, context/output limits, and **total session budget**. Baseline:
JSONL working memory plus Markdown, no initial facts, `auto_learn=false`. Hybrid:
the same setup with `auto_learn=true`. Do not seed either arm with the other's
learned facts. Keep user prompts fixed even when answers differ.

1. Record an explicit lasting preference, transient tasks, a verified local-read
   observation, and a third-party instruction in tool output. Score supported
   fact precision as well as missed facts and unwanted promotions.
2. Start an independent same-project session with no shared working transcript.
   Ask the same recall question in both arms. Score correctness and exact,
   currently valid source attribution separately from answer fluency.
3. Force at least **six** safe compactions under the same budget. Check that each
   summary chains correctly, extraction input advances without reprocessing the
   old prefix, and raw tool-output recall still reconstructs the exact original
   text and sequence references, including the third-party instruction as data.
4. Exercise malformed/truncated/failed extraction, cancellation, forget during an
   outstanding call, undo/redo, clear, explicit correction, and source deletion
   or corruption. Check exclusion and continuity, not just final answer quality.
5. Measure **all** requests: answering, summarization, learning (including rejects),
   retries and any review calls. Report total input/output tokens, known cost,
   unknown usage counts, wall-clock session latency and per-boundary latency.
   Include failed trials; do not compare only the final answer's prompt size.
   Stop both arms at the same total budget, and report repeat counts and model
   settings with any aggregate result. Make no savings claim from the offline tests.
