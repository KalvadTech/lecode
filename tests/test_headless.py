"""End-to-end tests for headless mode (``lecode -p``) with a fake provider."""

from __future__ import annotations

import json
import re

import pytest
from tests.fakes import FakeProvider
from typer.testing import CliRunner

from lecode.cli import EXIT_ERROR, EXIT_MAX_TURNS, EXIT_OK, EXIT_STARTUP, app
from lecode.providers.openai_compat import ProviderError

runner = CliRunner()


@pytest.fixture
def headless(tmp_path, monkeypatch):
    """Isolate config/session dirs, stub the dep check, return a script setter."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "global-skills"))
    monkeypatch.setattr("lecode.cli.find_missing_binaries", lambda: [])

    def use_script(script: list[dict]) -> FakeProvider:
        provider = FakeProvider(script)
        monkeypatch.setattr("lecode.cli.build_provider", lambda config, api_key=None: provider)
        return provider

    return tmp_path, use_script


def test_headless_prints_final_text_and_cost(headless):
    _, use_script = headless
    use_script([{"text": "final answer", "usage": {"input_tokens": 10, "output_tokens": 5}}])

    result = runner.invoke(app, ["-p", "do the thing"])

    assert result.exit_code == EXIT_OK
    assert result.stdout == "final answer\n"
    match = re.search(r"tokens: 10 in / 5 out · cost: \$(\d+\.\d{4})", result.stderr)
    assert match is not None
    # 10 * 0.09 + 5 * 0.18 per million (default model deepseek/deepseek-v4-flash)
    assert float(match.group(1)) == pytest.approx(0.0, abs=1e-9)


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


def test_headless_provider_error_exits_1(headless):
    _, use_script = headless
    use_script([{"error": ProviderError("invalid api key", status=401)}])

    result = runner.invoke(app, ["-p", "hello"])

    assert result.exit_code == EXIT_ERROR
    assert result.stdout == ""
    assert "invalid api key" in result.stderr


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


@pytest.fixture
def json_env(headless, monkeypatch):
    tmp_path, use_script = headless
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        "[mcp]\nenable_exa = false\nenable_context7 = false\n"
        "[memory]\nenabled = false\n[lsp]\nenabled = false\n",
        encoding="utf-8",
    )
    return tmp_path, use_script


def test_json_headless_emits_one_object(json_env):
    _, use_script = json_env
    answer = 'مرحبا\n"answer"\t\\ {"nested": true}'
    provider = use_script(
        [{"text": answer, "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.125}}]
    )

    result = runner.invoke(app, ["-p", "hello", "--output-format", "json", "--model", "test/model"])

    assert result.exit_code == EXIT_OK, result.output
    assert json.loads(result.stdout) == {
        "final_text": answer,
        "stop_reason": "done",
        "turns": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.125,
        "model": "test/model",
        "usage_incomplete": False,
    }
    assert provider.requests[0]["model"] == "test/model"
    assert result.stdout.count("\n") == 1
    assert "tokens:" in result.stderr


@pytest.mark.parametrize("mode", ["loop", "chain"])
def test_json_modes_aggregate_main_runs(json_env, mode):
    tmp_path, use_script = json_env
    script = [
        {
            "text": f"answer {number}",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.125},
        }
        for number in range(1, 5)
    ]
    use_script(script)
    if mode == "loop":
        (tmp_path / "plan.md").write_text("- [ ] remaining\n", encoding="utf-8")
        args = ["--loop", "plan.md", "--max-iterations", "4"]
        expected_exit, expected_reason = EXIT_MAX_TURNS, "max_iterations"
    else:
        args = ["--chain", "test topic"]
        expected_exit, expected_reason = EXIT_OK, "done"

    result = runner.invoke(app, [*args, "--output-format", "json"])

    assert result.exit_code == expected_exit, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == "answer 4"
    assert payload["stop_reason"] == expected_reason
    assert payload["turns"] == 4
    assert payload["input_tokens"] == 40
    assert payload["output_tokens"] == 20
    assert payload["cost_usd"] == 0.5
    assert payload["usage_incomplete"] is False
    assert "answer 1" in result.stderr


@pytest.mark.parametrize("mode", ["prompt", "loop", "chain"])
@pytest.mark.parametrize(
    "error", [ProviderError("provider unavailable", status=401), KeyboardInterrupt()]
)
def test_json_failures_retain_known_partial_usage(json_env, mode, error):
    tmp_path, use_script = json_env
    first = {
        "text": "earlier answer",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.125},
    }
    if mode == "prompt":
        first["tool_calls"] = [{"id": "read", "name": "list_dir", "arguments": "{}"}]
        args = ["-p", "hello"]
    elif mode == "loop":
        (tmp_path / "plan.md").write_text("- [ ] remaining\n", encoding="utf-8")
        args = ["--loop", "plan.md"]
    else:
        args = ["--chain", "topic"]
    use_script([first, {"error": error}])

    result = runner.invoke(app, [*args, "--output-format", "json"])

    assert result.exit_code == EXIT_ERROR, result.output
    payload = json.loads(result.stdout)
    assert payload["stop_reason"] == (
        "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
    )
    assert payload["turns"] == 1
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 5
    assert payload["cost_usd"] == 0.125
    assert payload["usage_incomplete"] is True
    assert "error:" in result.stderr


def test_json_startup_interruption_emits_interrupted(json_env, monkeypatch):
    def interrupt(config, api_key=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("lecode.cli.build_provider", interrupt)

    result = runner.invoke(app, ["-p", "hello", "--output-format", "json"])

    assert result.exit_code == EXIT_ERROR
    payload = json.loads(result.stdout)
    assert payload["stop_reason"] == "interrupted"
    assert payload["turns"] is None
    assert payload["cost_usd"] is None
    assert payload["usage_incomplete"] is True


def test_json_includes_delegated_worker_usage(json_env):
    _, use_script = json_env
    use_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "task",
                        "name": "task",
                        "arguments": json.dumps(
                            {"agent": "explore", "prompt": "inspect the directory"}
                        ),
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 1, "cost_usd": 0.125},
            },
            {
                "text": "child answer",
                "usage": {"input_tokens": 20, "output_tokens": 2, "cost_usd": 0.25},
            },
            {
                "text": "parent answer",
                "usage": {"input_tokens": 30, "output_tokens": 3, "cost_usd": 0.5},
            },
        ]
    )

    result = runner.invoke(app, ["-p", "delegate", "--output-format", "json"])

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == "parent answer"
    assert payload["turns"] == 2
    assert payload["input_tokens"] == 60
    assert payload["output_tokens"] == 6
    assert payload["cost_usd"] == 0.875
    assert payload["usage_incomplete"] is False


def test_json_nested_delegates_count_each_call_once(json_env):
    _, use_script = json_env
    task = [
        {
            "id": "task",
            "name": "task",
            "arguments": json.dumps({"agent": "explore", "prompt": "delegate the inspection"}),
        }
    ]
    use_script(
        [
            {
                **({"tool_calls": task} if number <= 2 else {"text": f"answer {number}"}),
                "usage": {"input_tokens": number, "output_tokens": number, "cost_usd": 0.125},
            }
            for number in range(1, 6)
        ]
    )

    result = runner.invoke(app, ["-p", "delegate twice", "--output-format", "json"])

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == "answer 5"
    assert payload["turns"] == 2
    assert payload["input_tokens"] == payload["output_tokens"] == 15
    assert payload["cost_usd"] == 0.625
    assert payload["usage_incomplete"] is False


@pytest.mark.parametrize("scenario", ["dependencies", "auth", "config", "provider", "empty_stdin"])
def test_json_failures_with_unavailable_metrics(json_env, monkeypatch, scenario):
    tmp_path, use_script = json_env
    args = ["-p", "hello", "--output-format", "json"]
    expected_exit = EXIT_STARTUP
    expected_reason = "startup_error"
    if scenario == "dependencies":
        from lecode.deps import MissingBinary

        monkeypatch.setattr(
            "lecode.cli.find_missing_binaries", lambda: [MissingBinary("fd", "install fd")]
        )
    elif scenario == "auth":

        def no_auth(config, api_key=None):
            raise ValueError("authentication required")

        monkeypatch.setattr("lecode.cli.build_provider", no_auth)
    elif scenario == "config":
        (tmp_path / "cfg" / "config.toml").write_text("[invalid", encoding="utf-8")
        expected_exit = EXIT_ERROR  # preserve the config-loading exception's exit
    elif scenario == "provider":
        use_script([{"error": ProviderError("unavailable", status=401)}])
        expected_exit, expected_reason = EXIT_ERROR, "error"
    else:
        args = ["-p", "--output-format", "json"]
        expected_exit = EXIT_ERROR

    result = runner.invoke(app, args, input="")

    assert result.exit_code == expected_exit, result.output
    payload = json.loads(result.stdout)
    assert payload["stop_reason"] == expected_reason
    assert payload["input_tokens"] is None
    assert payload["output_tokens"] is None
    assert payload["cost_usd"] is None
    assert payload["usage_incomplete"] is True


def test_json_already_complete_loop_has_zero_metrics(json_env):
    tmp_path, use_script = json_env
    provider = use_script([])
    (tmp_path / "plan.md").write_text("- [x] done\n", encoding="utf-8")

    result = runner.invoke(app, ["--loop", "plan.md", "--output-format", "json"])

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == ""
    assert payload["stop_reason"] == "done"
    assert payload["turns"] == payload["input_tokens"] == payload["output_tokens"] == 0
    assert payload["cost_usd"] == 0
    assert payload["usage_incomplete"] is False
    assert provider.requests == []


@pytest.mark.parametrize("args", [[], ["--setup"], ["--hooks-test"]])
def test_json_rejects_unsupported_modes(json_env, args):
    result = runner.invoke(app, [*args, "--output-format", "json"])

    assert result.exit_code == EXIT_STARTUP
    assert json.loads(result.stdout)["stop_reason"] == "startup_error"
    assert "requires --prompt, --loop, or --chain" in result.stderr


def test_output_format_rejects_unknown_choices(json_env):
    result = runner.invoke(app, ["-p", "hello", "--output-format", "xml"])

    assert result.exit_code == EXIT_STARTUP
    assert result.stdout == ""
    assert "Invalid value" in result.stderr


def test_json_piped_prompt_and_readonly_permission(json_env):
    tmp_path, use_script = json_env
    provider = use_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "write",
                        "name": "write",
                        "arguments": json.dumps(
                            {"file_path": "new.txt", "content": "must not be written"}
                        ),
                    }
                ]
            },
            {"text": "write denied"},
        ]
    )

    result = runner.invoke(
        app, ["-p", "--read-only", "--output-format", "json"], input="piped prompt"
    )

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == "write denied"
    assert payload["turns"] == 2
    assert payload["usage_incomplete"] is True
    assert not (tmp_path / "new.txt").exists()
    assert provider.requests[0]["messages"][-1]["content"] == "piped prompt"
    assert "denied" in provider.requests[1]["messages"][-1]["content"].lower()


@pytest.mark.parametrize("reason", ["max_turns", "context_overflow", "empty"])
def test_json_runner_stop_reasons_preserve_exit_codes(json_env, reason):
    tmp_path, use_script = json_env
    args = ["-p", "hello", "--output-format", "json"]
    if reason == "max_turns":
        use_script([{"tool_calls": [{"id": "list", "name": "list_dir", "arguments": "{}"}]}])
        args += ["--max-turns", "1"]
    elif reason == "context_overflow":
        use_script([])
        with (tmp_path / "cfg" / "config.toml").open("a", encoding="utf-8") as config:
            config.write("[agent]\ncontext_window = 1\n")
    else:
        use_script([{}] * 4)

    result = runner.invoke(app, args)

    assert result.exit_code == (EXIT_OK if reason == "empty" else EXIT_MAX_TURNS), result.output
    assert json.loads(result.stdout)["stop_reason"] == reason


def test_json_hook_denial_emits_blocked(json_env):
    tmp_path, use_script = json_env
    provider = use_script([])
    command = "echo '" + json.dumps({"verdict": "deny", "reason": "blocked for this test"}) + "'"
    with (tmp_path / "cfg" / "config.toml").open("a", encoding="utf-8") as config:
        config.write("[hooks]\nUserPromptSubmit = [" + json.dumps(command) + "]\n")

    result = runner.invoke(app, ["-p", "hello", "--output-format", "json"])

    assert result.exit_code == EXIT_ERROR, result.output
    payload = json.loads(result.stdout)
    assert payload["stop_reason"] == "blocked"
    assert payload["turns"] == payload["input_tokens"] == payload["output_tokens"] == 0
    assert payload["cost_usd"] == 0
    assert payload["usage_incomplete"] is False
    assert "prompt blocked by hook" in result.stderr
    assert provider.requests == []


def test_json_nonfinite_cost_is_unavailable(json_env):
    _, use_script = json_env
    use_script([{"text": "done", "usage": {"input_tokens": 5, "cost_usd": float("inf")}}])

    result = runner.invoke(app, ["-p", "hello", "--output-format", "json"])

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["cost_usd"] is None
    assert payload["usage_incomplete"] is True
    assert "Infinity" not in result.stdout


@pytest.mark.parametrize("usage", [None, {"input_tokens": 7, "output_tokens": 3}])
def test_json_missing_usage_and_pricing_are_null(json_env, usage):
    _, use_script = json_env
    use_script([{"text": "done", "usage": usage}])

    result = runner.invoke(
        app, ["-p", "hello", "--model", "unknown/model", "--output-format", "json"]
    )

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["turns"] == 1
    assert payload["input_tokens"] == (7 if usage else None)
    assert payload["output_tokens"] == (3 if usage else None)
    assert payload["cost_usd"] is None
    assert payload["usage_incomplete"] is True


def test_json_known_zero_usage_remains_available(json_env):
    _, use_script = json_env
    use_script([{"text": "done", "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0}}])

    result = runner.invoke(app, ["-p", "hello", "--output-format", "json"])

    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["input_tokens"] == payload["output_tokens"] == payload["cost_usd"] == 0
    assert payload["usage_incomplete"] is False


def test_json_usage_without_tokens_or_cost_is_unavailable(json_env):
    _, use_script = json_env
    use_script([{"text": "done", "usage": {"cached_tokens": 9}}])
    result = runner.invoke(app, ["-p", "hello", "--output-format", "json"])
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["input_tokens"] is payload["output_tokens"] is payload["cost_usd"] is None
    assert payload["usage_incomplete"] is True


@pytest.mark.parametrize("mode", ["prompt", "loop", "chain"])
def test_json_provider_failure_retains_streamed_usage(json_env, monkeypatch, mode):
    from lecode.providers.types import TokenDelta, Usage

    tmp_path, _ = json_env

    class PartialProvider(FakeProvider):
        async def stream_chat(self, *args, **kwargs):
            yield TokenDelta(text="partial")
            yield Usage(usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.25})
            raise ProviderError("stream failed", status=401)

    monkeypatch.setattr("lecode.cli.build_provider", lambda *args, **kwargs: PartialProvider([]))
    if mode == "loop":
        plan = tmp_path / "plan.md"
        plan.write_text("- [ ] work\n")
        args = ["--loop", str(plan)]
    else:
        args = ["-p", "work"] if mode == "prompt" else ["--chain", "work"]
    result = runner.invoke(app, [*args, "--output-format", "json"])
    assert result.exit_code == EXIT_ERROR, result.output
    payload = json.loads(result.stdout)
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 5
    assert payload["cost_usd"] == 0.25
    assert payload["turns"] == 0
    assert payload["usage_incomplete"] is True


@pytest.mark.parametrize("failed", [False, True])
def test_json_accounts_for_review_without_feedback(json_env, failed):
    tmp_path, use_script = json_env
    provider = use_script(
        [
            {"text": "done", "usage": {"input_tokens": 10, "output_tokens": 1, "cost_usd": 0.25}},
            {"error": ProviderError("review failed", status=401)}
            if failed
            else {"text": "", "usage": {"input_tokens": 20, "output_tokens": 0, "cost_usd": 0.5}},
        ]
    )
    with (tmp_path / "cfg" / "config.toml").open("a", encoding="utf-8") as config:
        config.write("[pierre]\nenabled = true\n")
    result = runner.invoke(app, ["-p", "work", "--output-format", "json"])
    assert result.exit_code == EXIT_OK, result.output
    assert len(provider.requests) == 2
    payload = json.loads(result.stdout)
    assert payload["input_tokens"] == (10 if failed else 30)
    assert payload["output_tokens"] == 1
    assert payload["cost_usd"] == (0.25 if failed else 0.75)
    assert payload["usage_incomplete"] is failed
    assert "pierre:" not in result.stderr


def test_json_successful_retry_retains_failed_attempt_usage(json_env, monkeypatch):
    from lecode.providers.types import Usage

    _, _ = json_env
    monkeypatch.setattr("lecode.providers.retry.uniform", lambda *args: 0)

    class RetryingProvider(FakeProvider):
        async def stream_chat(self, *args, **kwargs):
            if not self.requests:
                self.requests.append({})
                yield Usage(usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.25})
                raise ProviderError("retry", retryable=True)
            async for event in super().stream_chat(*args, **kwargs):
                yield event

    provider = RetryingProvider(
        [{"text": "done", "usage": {"input_tokens": 20, "output_tokens": 10, "cost_usd": 0.5}}]
    )
    monkeypatch.setattr("lecode.cli.build_provider", lambda *args, **kwargs: provider)
    result = runner.invoke(app, ["-p", "work", "--output-format", "json"])
    assert result.exit_code == EXIT_OK, result.output
    assert len(provider.requests) == 2
    payload = json.loads(result.stdout)
    assert payload["input_tokens"] == 30
    assert payload["output_tokens"] == 15
    assert payload["cost_usd"] == 0.75
    assert payload["turns"] == 1
    assert payload["usage_incomplete"] is True


def test_json_partial_call_retains_known_totals_as_incomplete(json_env):
    _, use_script = json_env
    use_script(
        [
            {
                "tool_calls": [{"id": "list", "name": "list_dir", "arguments": "{}"}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.25},
            },
            {"text": "done", "usage": {"input_tokens": 7, "cost_usd": 0.5}},
        ]
    )
    result = runner.invoke(app, ["-p", "hello", "--output-format", "json"])
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(result.stdout)
    assert payload["input_tokens"] == 17
    assert payload["output_tokens"] == 5
    assert payload["cost_usd"] == 0.75
    assert payload["usage_incomplete"] is True
