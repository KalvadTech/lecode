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
