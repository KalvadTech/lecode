# CLI reference

Headless, loop, and chain modes: exit codes, error classification, retries,
spend and time limits, and machine-readable output.

```sh
lecode -p "write a haiku about this repo"     # one prompt, then exit
lecode -p "review this repo" --max-cost 1 --timeout 120
git diff | lecode -p "review this diff"       # a bare -p reads stdin
lecode --loop plan.md --loop-cmd "make test"  # iterate until the plan is done
lecode --chain "redesign the parser"          # brainstorm→plan→code→review
lecode -p "review this repo" --output-format json
```

The final answer goes to stdout; a `tokens: <in> in / <out> out · cost:
$X.XXXX` summary goes to stderr, so scripts can pipe the answer cleanly.
Incomplete accounting uses `cost: known $X.XXXX`; this is only the known
portion of spending, not a complete total.

## Exit codes

Headless, loop, and chain modes use these exit codes:

| Code | Meaning |
|------|---------|
| `0` | Done |
| `1` | Generic error, including other HTTP failures |
| `2` | Startup error: missing dependencies or required credentials, bad flags, non-tty `--setup` |
| `3` | Max turns or max loop iterations |
| `4` | Provider authentication failure |
| `5` | Provider budget or credit exhaustion |
| `6` | Model/resource not found, including HTTP 404 (which can also mean an incorrect endpoint) |
| `7` | Provider rate limit, after retries are exhausted |
| `8` | Upstream, timeout, conflict, or transport failure |
| `9` | Unknown or malformed in-stream error |
| `10` | Context overflow |
| `11` | Cost limit or unknown spend (single prompt only) |
| `12` | Execution timeout (single prompt only) |

## Error classification and retries

HTTP and in-stream errors share classification. Recognized
`error.metadata.error_type` values take precedence over symbolic `error.code`
and `error.type`, followed by HTTP status (or a numeric in-stream code).
In-stream codes can be integers or three-digit ASCII strings. Recognized
symbolic codes are `authentication`, `invalid_api_key`, `payment_required`,
`insufficient_quota`, `credit_balance_exhausted`, `model_not_found`, `not_found`,
`rate_limit_exceeded`, `provider_overloaded`, `provider_unavailable`, `server`,
`server_error`, and `timeout`. Classification never guesses from message text.

Budget failures are not retried, including quota errors reported as HTTP 429.
Other retries use the numeric HTTP/in-stream status when supplied, retaining
the existing set: 408, 409, 429, 500, 502, 503, 504 and transport failures.
Symbolic-only in-stream errors use their corresponding status.
Unknown or malformed in-stream errors are not
retried. Interactive mode continues to display provider failures and stay open.

## `--max-cost`

For single-prompt headless runs, `--max-cost` accepts a positive USD threshold.
All model calls share it, including workers, subagents, compaction, learning,
reviews, and retries. Once accounted spending reaches it, no further model
requests start. Requests already in flight can exceed it. Unknown or invalid
model prices reject the request before execution, and missing or invalid usage
stops further requests. Explicit zero prices are valid for free models.

## `--timeout`

`--timeout` accepts positive seconds and covers active execution from MCP
connection through model calls, tools, and review. Synchronous startup and
resource cleanup are outside this deadline, so process exit can occur later.
Neither flag applies to interactive, loop, or chain mode.

## `--output-format json`

`--output-format json` works with `--prompt`, `--loop`, and `--chain`.
It prints one JSON object on stdout with `final_text`, `stop_reason`, `turns`,
`input_tokens`, `output_tokens`, `cost_usd`, `model`, and `usage_incomplete`.
Progress and diagnostics go to stderr. Text output remains the default.

For loops and chains, `final_text` is the last iteration or phase's answer,
and `turns` counts completed main model calls across the run. Token and cost
totals include recorded worker and subagent usage, compaction, memory learning,
and review. `model` identifies the main configured model. Cost is in USD,
using provider-reported cost when available and catalog pricing otherwise.

Failures also emit JSON. Known partial usage is retained as a lower bound
with `usage_incomplete: true`; unavailable metrics are `null`. Stop reasons
include `startup_error`, `error`, `interrupted`, `blocked`, and the runner or
loop's existing reasons (`done`, `empty`, `max_turns`, `context_overflow`,
`max_iterations`, `cost_limit`, `timeout`). Help, version, and argument-parsing errors use normal CLI
output; JSON is unavailable for interactive, setup, and hooks-test modes.
