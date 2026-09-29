"""End-to-end tests for headless mode (``lecode -p``) with a fake provider."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import respx
from tests.fakes import FakeProvider, sample_catalog
from typer.testing import CliRunner

from lecode.cli import (
    EXIT_CONTEXT_OVERFLOW,
    EXIT_COST_LIMIT,
    EXIT_MAX_TURNS,
    EXIT_OK,
    EXIT_STARTUP,
    EXIT_TIMEOUT,
    _run_with_mcp,
    app,
)
from lecode.providers.catalog import Catalog
from lecode.providers.live import LoadedCatalog
from lecode.providers.openai_compat import ChatClient, ProviderError

runner = CliRunner()


def _headless_records(tmp_path):
    for path in (tmp_path / "cfg" / "sessions").glob("*.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if records[0]["name"].startswith("session-"):
            return records
    raise AssertionError("headless session missing")


@pytest.fixture
def headless(tmp_path, monkeypatch):
    """Isolate config/session dirs, stub the dep check, return a script setter."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.toml").write_text("[mcp]\nenable_exa = false\n")
    monkeypatch.setattr("lecode.cli.find_missing_binaries", lambda: [])

    async def no_sleep(delay):
        pass

    monkeypatch.setattr("lecode.providers.retry.sleep", no_sleep)

    def use_script(script: list[dict]) -> FakeProvider:
        provider = FakeProvider(script)
        monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: provider)
        return provider

    return tmp_path, use_script


@pytest.fixture(params=["prompt", "loop", "chain"])
def headless_args(headless, request):
    tmp_path, _ = headless
    (tmp_path / "plan.md").write_text("- [ ] task\n")
    return {
        "prompt": ["-p", "hello"],
        "loop": ["--loop", "plan.md"],
        "chain": ["--chain", "hello"],
    }[request.param]


def test_headless_prints_final_text_and_cost(headless, monkeypatch):
    _, use_script = headless
    use_script([{"text": "final answer", "usage": {"input_tokens": 10, "output_tokens": 5}}])
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )

    result = runner.invoke(app, ["-p", "do the thing"])

    assert result.exit_code == EXIT_OK
    assert result.stdout == "final answer\n"
    match = re.search(r"tokens: 10 in / 5 out · cost: \$(\d+\.\d{4})", result.stderr)
    assert match is not None
    # 10 * 0.09 + 5 * 0.18 per million (default model deepseek/deepseek-v4-flash)
    assert float(match.group(1)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("level", [None, "none", "low", "medium", "high"])
def test_headless_thinking_reaches_model_request(headless, monkeypatch, level):
    tmp_path, use_script = headless
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "cfg" / "config.toml"
    config_path.parent.mkdir(exist_ok=True)
    original = 'schema_version = 1\n\n[llm]\nthinking = "high"\n'
    config_path.write_text(original)
    provider = use_script([{"text": "ok"}])
    flags = [] if level is None else ["--thinking", level]
    result = runner.invoke(app, ["-p", "hi", *flags])
    assert result.exit_code == EXIT_OK, result.output
    expected = "high" if level is None else None if level == "none" else level
    assert provider.requests[0]["kwargs"]["reasoning_effort"] == expected
    assert config_path.read_text() == original


def test_headless_reads_prompt_from_stdin(headless):
    _, use_script = headless
    provider = use_script([{"text": "from stdin"}])

    result = runner.invoke(app, ["-p"], input="stdin prompt\n")

    assert result.exit_code == EXIT_OK
    assert result.stdout == "from stdin\n"
    user_messages = [m for m in provider.requests[0]["messages"] if m["role"] == "user"]
    assert user_messages == [{"role": "user", "content": "stdin prompt"}]


def test_headless_creates_auto_named_session(headless):
    tmp_path, use_script = headless
    use_script([{"text": "ok"}])

    result = runner.invoke(app, ["-p", "hello"])

    assert result.exit_code == EXIT_OK
    sessions = list((tmp_path / "cfg" / "sessions").glob("*.jsonl"))
    assert len(sessions) == 1
    meta = json.loads(sessions[0].read_text(encoding="utf-8").splitlines()[0])
    assert meta["name"].startswith("session-")
    lines = sessions[0].read_text(encoding="utf-8").splitlines()
    roles = [json.loads(line).get("role") for line in lines[1:]]
    assert roles == ["user", "assistant"]


@pytest.mark.parametrize(
    ("status", "retryable", "exit_code"),
    [
        (401, False, 4),
        (402, False, 5),
        (404, False, 6),
        (429, False, 7),
        (408, False, 8),
        (409, False, 8),
        (500, False, 8),
        (501, False, 8),
        (None, True, 8),
        (400, False, 1),
        (403, False, 1),
        (None, False, 1),
    ],
)
def test_provider_failure_exit_codes_in_all_modes(
    headless, headless_args, status, retryable, exit_code
):
    _, use_script = headless
    error = ProviderError("provider failed", status=status, retryable=retryable)
    use_script([{"error": error}] * 5)
    result = runner.invoke(app, headless_args)

    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert "provider failed" in result.stderr


@pytest.mark.parametrize(
    ("payload", "exit_code", "attempts"),
    [
        ('{"error": {"code": "401"}}', 4, 1),
        ('{"error": {"code": "402"}}', 5, 1),
        ('{"error": {"code": "model_not_found"}}', 6, 1),
        ('{"error": {"code": "429"}}', 7, 5),
        ('{"error": {"code": "server_error"}}', 8, 5),
        ('{"error": {"code": "insufficient_quota"}}', 5, 1),
        ('{"error": {"code": "unknown"}}', 9, 1),
        ("invalid json", 9, 1),
    ],
)
@respx.mock
def test_stream_failure_exit_codes_and_retries_in_all_modes(
    headless_args, monkeypatch, payload, exit_code, attempts
):
    base_url = "https://api.test/v1"
    monkeypatch.setattr(
        "lecode.cli.build_provider", lambda config, api_key=None: ChatClient(base_url)
    )
    respx.get(f"{base_url}/models").respond(200, json={"data": []})
    route = respx.post(f"{base_url}/chat/completions").respond(
        200, content=f"data: {payload}\n\n".encode()
    )

    result = runner.invoke(app, headless_args)

    assert result.exit_code == exit_code, result.output
    assert result.stdout == ""
    assert route.call_count == attempts


def test_headless_max_turns_exits_3(headless):
    _, use_script = headless
    script = [
        {"tool_calls": [{"id": f"c{i}", "name": "list_dir", "arguments": "{}"}]} for i in range(5)
    ]
    use_script(script)

    result = runner.invoke(app, ["-p", "loop forever", "--max-turns", "2"])

    assert result.exit_code == EXIT_MAX_TURNS


def test_headless_auth_required_without_key_is_startup_error(headless, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    args = ["-p", "hello", "--base-url", "http://localhost:9/v1", "--auth-policy", "required"]
    result = runner.invoke(app, args)

    assert result.exit_code == EXIT_STARTUP
    assert "required" in result.stderr


def test_no_args_launches_interactive(headless, monkeypatch):
    calls = []
    monkeypatch.setattr("lecode.cli.run_interactive", lambda **kw: calls.append(kw) or EXIT_OK)
    result = runner.invoke(app, [])
    assert result.exit_code == EXIT_OK
    assert len(calls) == 1


@pytest.mark.parametrize("case", ["short", "tools"])
def test_headless_does_not_load_interactive_modules(tmp_path, case):
    root = Path(__file__).resolve().parents[1]
    script = """
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from lecode import cli
from lecode.config.models import Config
from tests.fakes import FakeProvider

project = Path(sys.argv[1]) / "project"
project.mkdir()
(project / ".git").mkdir()
(project / "input.txt").write_text("line\\n" * 64)
os.chdir(project)
os.environ["LECODE_CONFIG_DIR"] = str(project / "cfg")
os.environ["LECODE_SKILLS_DIR"] = str(project / "skills")
for name in ("HERDR_ENV", "HERDR_BIN_PATH", "HERDR_PANE_ID"):
    os.environ.pop(name, None)
config = Config()
config.mcp.enable_exa = False
config.mcp.enable_context7 = False
cli.load_config = lambda: SimpleNamespace(config=config)
turns = 20 if sys.argv[2] == "tools" else 0
provider = FakeProvider([
    {"tool_calls": [{"id": f"read-{i}", "name": "read",
                     "arguments": f'{{"path":"input.txt","offset":{i},"limit":10}}'}]}
    for i in range(1, turns + 1)
] + [{"text": "final answer"}])
cli.build_provider = lambda config, api_key=None: provider
cli.check_dependencies = lambda: None
cli.app(args=["-p", "hello"], standalone_mode=False)
assert len(provider.requests) == turns + 1
results = [m for m in provider.requests[-1]["messages"] if m["role"] == "tool"]
assert len(results) == turns
if results:
    assert results[-1]["content"].startswith("20\\tline")
assert not any(m.startswith("lecode.tui.") for m in sys.modules)
assert "prompt_toolkit" not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            case,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout == "final answer\n"


@pytest.mark.parametrize("flag", ["--max-cost", "--timeout"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_limits_reject_invalid_values(headless, flag, value):
    result = runner.invoke(app, ["-p", "hello", flag, value])
    assert result.exit_code == EXIT_STARTUP
    assert "finite number greater than zero" in result.stderr


@pytest.mark.parametrize("args", [[], ["--loop", "plan.md"], ["--chain", "topic"], ["--setup"]])
def test_limits_require_single_prompt_mode(headless, args):
    result = runner.invoke(app, [*args, "--timeout", "1"])
    assert result.exit_code == EXIT_STARTUP
    assert "require --prompt" in result.stderr


def test_headless_unknown_price_rejects_before_request(headless, monkeypatch):
    _, use_script = headless
    provider = use_script([{"text": "never sent"}])
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(Catalog.default(), "empty")
    )
    result = runner.invoke(app, ["-p", "hello", "--max-cost", "1"])
    assert result.exit_code == EXIT_COST_LIMIT
    assert "price is unknown" in result.stderr
    assert not provider.requests


def test_unbudgeted_unknown_price_reports_only_known_cost(headless, monkeypatch):
    _, use_script = headless
    use_script([{"text": "answer", "usage": {"input_tokens": 10, "output_tokens": 5}}])
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(Catalog.default(), "empty")
    )
    result = runner.invoke(app, ["-p", "hello"])
    assert result.exit_code == EXIT_OK
    assert "cost: known $0.0000" in result.stderr


def test_headless_cost_stop_counts_child_and_blocks_next_request(headless, monkeypatch):
    tmp_path, use_script = headless
    provider = use_script(
        [
            {
                "tool_calls": [
                    {"id": "child", "name": "task", "arguments": '{"prompt":"research"}'}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.1},
            },
            {
                "text": "child answer",
                "usage": {"input_tokens": 20, "output_tokens": 8, "cost_usd": 0.5},
            },
            {"text": "must not send", "usage": {"cost_usd": 0.1}},
        ]
    )
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )
    result = runner.invoke(app, ["-p", "hello", "--max-cost", "0.5"])
    assert result.exit_code == EXIT_COST_LIMIT, result.stderr
    assert len(provider.requests) == 2
    assert "tokens: 30 in / 13 out · cost: $0.6000" in result.stderr
    records = _headless_records(tmp_path)
    assert any(
        record.get("kind") == "run_stopped" and record["data"]["reason"] == "cost_limit"
        for record in records
    )


def test_headless_empty_review_still_exhausts_budget(headless, monkeypatch):
    tmp_path, use_script = headless
    with (tmp_path / "cfg" / "config.toml").open("a") as config:
        config.write("[pierre]\nenabled = true\n")
    provider = use_script(
        [
            {"text": "answer", "usage": {"cost_usd": 0.1}},
            {"text": "", "usage": {"cost_usd": 0.5}},
        ]
    )
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )
    result = runner.invoke(app, ["-p", "hello", "--max-cost", "0.5"])
    assert result.exit_code == EXIT_COST_LIMIT, result.stderr
    assert len(provider.requests) == 2
    assert "cost: $0.6000" in result.stderr


def test_headless_context_overflow_has_distinct_exit(headless):
    tmp_path, use_script = headless
    with (tmp_path / "cfg" / "config.toml").open("a") as config:
        config.write("[agent]\ncontext_window = 10\n")
    provider = use_script([{"text": "not sent"}])
    result = runner.invoke(app, ["-p", "hello"])
    assert result.exit_code == EXIT_CONTEXT_OVERFLOW
    assert not provider.requests


@pytest.mark.parametrize("budgeted", [False, True])
def test_headless_timeout_persists_partial_and_closes_provider(headless, monkeypatch, budgeted):
    from lecode.providers.types import TokenDelta, Usage

    tmp_path, _ = headless

    class SlowProvider(FakeProvider):
        closed = False

        async def _stream(self, entry):
            yield TokenDelta("partial answer")
            if budgeted:
                yield Usage({"input_tokens": "invalid"})
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    provider = SlowProvider([{}])
    monkeypatch.setattr("lecode.cli.build_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )
    args = ["-p", "hello", "--timeout", "0.02"]
    if budgeted:
        args.extend(["--max-cost", "1"])
    result = runner.invoke(app, args)
    assert result.exit_code == EXIT_TIMEOUT, result.stderr
    assert provider.closed
    records = _headless_records(tmp_path)
    partial = next(record for record in records if record.get("role") == "assistant")
    assert partial["message"]["content"] == "partial answer" and partial["message"]["incomplete"]
    assert any(
        record.get("kind") == "run_stopped" and record["data"]["reason"] == "timeout"
        for record in records
    )


async def test_timeout_during_mcp_attachment_cleans_every_resource(monkeypatch):
    from lecode.cli import HeadlessTimeoutError

    closed = []

    class Resource:
        def __init__(self, name):
            self.name = name

        async def shutdown(self):
            # Cleanup is allowed to exceed the active execution deadline.
            await asyncio.sleep(0.01)
            closed.append(self.name)

        async def aclose(self):
            closed.append(self.name)

    ctx = SimpleNamespace(
        extras={"workers": Resource("workers"), "background": Resource("background")}
    )
    runtime = SimpleNamespace(ctx=ctx, registry=None, close=lambda: closed.append("runtime"))

    async def attach(*_args, **_kwargs):
        ctx.extras["mcp"] = Resource("mcp")
        await asyncio.Event().wait()

    monkeypatch.setattr("lecode.extras.mcp_client.attach_mcp", attach)
    with pytest.raises(HeadlessTimeoutError):
        await _run_with_mcp(runtime, Resource("provider"), None, [], timeout=0.01)
    assert closed == ["workers", "provider", "background", "mcp", "runtime"]


def test_headless_timeout_kills_foreground_shell(headless):
    tmp_path, use_script = headless
    pid_file = tmp_path / "shell.pid"
    use_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "shell",
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": f"echo $$ > {shlex.quote(str(pid_file))}; exec sleep 43"}
                        ),
                    }
                ],
            }
        ]
    )
    result = runner.invoke(app, ["-p", "run a command", "--timeout", "0.2"])
    assert result.exit_code == EXIT_TIMEOUT, result.stderr
    assert pid_file.exists(), "foreground shell did not start before timeout"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_headless_unknown_spend_stops_and_preserves_response(headless, monkeypatch):
    tmp_path, use_script = headless
    provider = use_script([{"text": "answer with no usage"}])
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )
    result = runner.invoke(app, ["-p", "hello", "--max-cost", "1"])
    assert result.exit_code == EXIT_COST_LIMIT, result.stderr
    assert "spend is unknown" in result.stderr
    assert len(provider.requests) == 1
    partial = next(
        record for record in _headless_records(tmp_path) if record.get("role") == "assistant"
    )
    assert partial["message"]["content"] == "answer with no usage"
    assert partial["usage"]["incomplete"]


def test_headless_malformed_http_usage_exits_cost_limit_and_preserves_response(
    headless, monkeypatch
):
    import httpx

    from lecode.providers.openai_compat import ChatClient

    tmp_path, _ = headless
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                'data: {"usage":true,"choices":[]}\n\n'
            ),
        )

    async_client = httpx.AsyncClient(
        base_url="https://provider.invalid/v1", transport=httpx.MockTransport(respond)
    )
    monkeypatch.setattr(
        "lecode.providers.openai_compat.httpx.AsyncClient", lambda **_: async_client
    )
    provider = ChatClient("https://provider.invalid/v1")
    monkeypatch.setattr("lecode.cli.build_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )

    result = runner.invoke(app, ["-p", "hello", "--max-cost", "1"])
    assert result.exit_code == EXIT_COST_LIMIT, result.exception
    assert "spend is unknown" in result.stderr
    assert len(requests) == 1 and async_client.is_closed
    records = _headless_records(tmp_path)
    partial = next(record for record in records if record.get("role") == "assistant")
    assert partial["message"]["content"] == "partial"
    assert partial["usage"]["incomplete"]
    assert any(
        record.get("kind") == "run_stopped" and record["data"]["reason"] == "cost_limit"
        for record in records
    )


async def test_provider_timeout_is_not_misreported_as_execution_deadline(monkeypatch):
    ctx = SimpleNamespace(extras={})
    runtime = SimpleNamespace(ctx=ctx, registry=None, close=lambda: None)

    async def attach(*_args, **_kwargs):
        raise TimeoutError("connection failed")

    monkeypatch.setattr("lecode.extras.mcp_client.attach_mcp", attach)
    with pytest.raises(TimeoutError, match="connection failed"):
        await _run_with_mcp(runtime, object(), None, [], timeout=1)


@pytest.mark.parametrize("with_review", [False, True])
def test_cost_stop_hook_reports_final_reason_once(headless, monkeypatch, with_review):
    from lecode.hooks.runner import HookDispatcher

    tmp_path, use_script = headless
    with (tmp_path / "cfg" / "config.toml").open("a") as config:
        config.write('[hooks]\nStop = ["true"]\n')
        if with_review:
            config.write("[pierre]\nenabled = true\n")
    use_script(
        [
            {"text": "answer", "usage": {"cost_usd": 0.1 if with_review else 0.6}},
            {"text": "review", "usage": {"cost_usd": 0.5}},
        ]
    )
    monkeypatch.setattr(
        "lecode.cli.fetch_catalog", lambda *_: LoadedCatalog(sample_catalog(), "live")
    )
    stops = []

    async def record(self, event, **payload):
        if event == "Stop":
            stops.append(payload["reason"])

    monkeypatch.setattr(HookDispatcher, "fire", record)
    result = runner.invoke(app, ["-p", "hello", "--max-cost", "0.5"])
    assert result.exit_code == EXIT_COST_LIMIT, result.stderr
    assert stops == ["cost_limit"]
