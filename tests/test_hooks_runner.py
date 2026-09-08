"""Tests for hook handler execution and verdict merging."""

from __future__ import annotations

import json

import pytest

from lecode.config.models import Config
from lecode.hooks.events import (
    EVENTS,
    INTERRUPT,
    NOTIFICATION,
    PERMISSION_REQUEST,
    PERMISSION_RESULT,
    POST_COMPACT,
    POST_TOOL_USE,
    POST_TOOL_USE_FAILURE,
    PRE_COMPACT,
    PRE_TOOL_USE,
    SESSION_END,
    SESSION_START,
    STOP,
    SUBAGENT_END,
    SUBAGENT_START,
    USER_PROMPT_SUBMIT,
    build_envelope,
)
from lecode.hooks.runner import (
    HookDispatcher,
    HookHandler,
    dispatch_event,
    dispatcher_from_config,
    hooks_status,
    run_handler,
)


def envelope(event: str = PRE_TOOL_USE, **kwargs) -> dict:
    return build_envelope(event, "/tmp", **kwargs)


def echo_verdict(payload: dict) -> HookHandler:
    return HookHandler(f"echo '{json.dumps(payload)}'")


@pytest.mark.parametrize("verdict", ["allow", "defer", "ask", "deny"])
async def test_each_verdict_parsed(verdict):
    result = await run_handler(echo_verdict({"verdict": verdict}), envelope())
    assert result.verdict == verdict
    assert result.matched is True
    assert result.failed is False


async def test_reason_and_case_insensitive_verdict():
    result = await run_handler(echo_verdict({"verdict": "DENY", "reason": "no way"}), envelope())
    assert result.verdict == "deny"
    assert result.reason == "no way"


async def test_rewritten_input_parsed():
    result = await run_handler(
        echo_verdict({"verdict": "allow", "rewritten_input": {"text": "changed"}}), envelope()
    )
    assert result.rewritten_input == {"text": "changed"}


async def test_malformed_rewritten_input_ignored():
    result = await run_handler(
        echo_verdict({"verdict": "allow", "rewritten_input": "nope"}), envelope()
    )
    assert result.rewritten_input is None


async def test_empty_stdout_is_abstain():
    result = await run_handler(HookHandler("true"), envelope())
    assert result.matched is False
    assert result.verdict == "allow"
    assert result.failed is False


async def test_invalid_json_is_deny_safe_for_pre_tool_use():
    result = await run_handler(HookHandler("echo 'not json'"), envelope(PRE_TOOL_USE))
    assert result.verdict == "deny"
    assert result.failed is True
    assert "invalid JSON" in result.reason


async def test_invalid_json_is_noop_for_other_events():
    result = await run_handler(HookHandler("echo 'not json'"), envelope(STOP))
    assert result.matched is False
    assert result.failed is True


async def test_non_dict_json_is_failure():
    result = await run_handler(HookHandler("echo '[1, 2]'"), envelope(PRE_TOOL_USE))
    assert result.failed is True
    assert result.verdict == "deny"


async def test_missing_verdict_key_is_failure():
    result = await run_handler(echo_verdict({"reason": "no verdict"}), envelope(PRE_TOOL_USE))
    assert result.failed is True
    assert result.verdict == "deny"


async def test_non_zero_exit_is_failure_with_stderr():
    result = await run_handler(HookHandler("echo broken >&2; exit 3"), envelope(PRE_TOOL_USE))
    assert result.failed is True
    assert result.verdict == "deny"
    assert "exit code 3" in result.reason
    assert "broken" in result.reason


async def test_timeout_is_failure():
    handler = HookHandler("sleep 5", timeout_s=0.2)
    result = await run_handler(handler, envelope(PRE_TOOL_USE))
    assert result.failed is True
    assert result.verdict == "deny"
    assert "timed out" in result.reason


async def test_envelope_delivered_on_stdin(tmp_path):
    out = tmp_path / "envelope.json"
    handler = HookHandler(f"cat > {out}")
    env = envelope(
        PRE_TOOL_USE,
        tool_name="bash",
        tool_args={"command": "ls"},
        session=type("S", (), {"id": "sid", "name": "sname"})(),
    )
    await run_handler(handler, env)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["event"] == "PreToolUse"
    assert data["tool"] == {"name": "bash", "args": {"command": "ls"}}
    assert data["session"] == {"id": "sid", "name": "sname"}
    assert data["cwd"] == "/tmp"
    assert "ts" in data


async def test_merge_severity_order():
    merged = await dispatch_event(
        PRE_TOOL_USE,
        envelope(),
        [
            echo_verdict({"verdict": "allow"}),
            echo_verdict({"verdict": "deny", "reason": "denied"}),
            echo_verdict({"verdict": "ask"}),
            echo_verdict({"verdict": "defer"}),
        ],
    )
    assert merged.verdict == "deny"
    assert merged.reason == "denied"


async def test_merge_ask_beats_defer_beats_allow():
    merged = await dispatch_event(
        PRE_TOOL_USE,
        envelope(),
        [echo_verdict({"verdict": "defer"}), echo_verdict({"verdict": "ask"})],
    )
    assert merged.verdict == "ask"


async def test_merge_ignores_abstains():
    merged = await dispatch_event(
        PRE_TOOL_USE, envelope(), [HookHandler("true"), echo_verdict({"verdict": "defer"})]
    )
    assert merged.verdict == "defer"


async def test_merge_no_handlers_allows():
    merged = await dispatch_event(PRE_TOOL_USE, envelope(), [])
    assert merged.verdict == "allow"
    assert merged.failed is False


async def test_merge_first_reason_wins():
    merged = await dispatch_event(
        PRE_TOOL_USE,
        envelope(),
        [
            echo_verdict({"verdict": "ask", "reason": "first"}),
            echo_verdict({"verdict": "deny", "reason": "second"}),
        ],
    )
    assert merged.verdict == "deny"
    assert merged.reason == "first"


async def test_merge_rewritten_input_from_most_severe_provider():
    merged = await dispatch_event(
        PRE_TOOL_USE,
        envelope(),
        [
            echo_verdict({"verdict": "allow", "rewritten_input": {"from": "allow"}}),
            echo_verdict({"verdict": "deny"}),
            echo_verdict({"verdict": "ask", "rewritten_input": {"from": "ask"}}),
        ],
    )
    assert merged.verdict == "deny"
    assert merged.rewritten_input == {"from": "ask"}


async def test_merge_failure_flag_propagates():
    merged = await dispatch_event(
        STOP, envelope(STOP), [HookHandler("exit 1"), HookHandler("true")]
    )
    assert merged.verdict == "allow"
    assert merged.failed is True


def test_dispatcher_from_config_builds_handlers(tmp_path):
    config = Config()
    config.hooks = {"PreToolUse": ["echo hi"], "Stop": ["echo a", "echo b"]}
    dispatcher, warnings = dispatcher_from_config(config, tmp_path)
    assert warnings == []
    assert dispatcher is not None
    assert len(dispatcher.handlers["PreToolUse"]) == 1
    assert len(dispatcher.handlers["Stop"]) == 2


def test_dispatcher_from_config_empty_is_none(tmp_path):
    dispatcher, warnings = dispatcher_from_config(Config(), tmp_path)
    assert dispatcher is None
    assert warnings == []


def test_unknown_event_ignored_with_warning(tmp_path):
    config = Config()
    config.hooks = {"BogusEvent": ["echo hi"], "Stop": ["echo ok"]}
    dispatcher, warnings = dispatcher_from_config(config, tmp_path)
    assert warnings == ["unknown hook event: BogusEvent"]
    assert list(dispatcher.handlers) == ["Stop"]


def test_hooks_status_rows():
    config = Config()
    config.hooks = {"PreToolUse": ["echo hi"], "Bogus": ["echo nope"]}
    rows = hooks_status(config)
    assert rows == [{"event": "PreToolUse", "command": "echo hi", "timeout_s": 10.0}]


def test_events_tuple_covers_all_15_events():
    assert EVENTS == (
        PRE_TOOL_USE,
        POST_TOOL_USE,
        POST_TOOL_USE_FAILURE,
        PERMISSION_REQUEST,
        PERMISSION_RESULT,
        USER_PROMPT_SUBMIT,
        STOP,
        SESSION_START,
        SESSION_END,
        SUBAGENT_START,
        SUBAGENT_END,
        PRE_COMPACT,
        POST_COMPACT,
        INTERRUPT,
        NOTIFICATION,
    )


async def test_invalid_json_is_deny_safe_for_user_prompt_submit():
    result = await run_handler(HookHandler("echo 'not json'"), envelope(USER_PROMPT_SUBMIT))
    assert result.verdict == "deny"
    assert result.failed is True
    assert "hook failed" in result.reason


@pytest.mark.parametrize(
    "event",
    [SESSION_START, SESSION_END, NOTIFICATION, INTERRUPT, PRE_COMPACT, POST_TOOL_USE_FAILURE],
)
async def test_handler_failure_is_noop_for_observational_events(event):
    result = await run_handler(HookHandler("exit 1"), envelope(event))
    assert result.matched is False
    assert result.failed is True
    assert result.verdict == "allow"


async def test_dispatcher_fire_builds_envelope_with_payload(tmp_path):
    out = tmp_path / "envelope.json"
    dispatcher = HookDispatcher({"PermissionResult": [HookHandler(f"cat > {out}")]}, tmp_path)
    merged = await dispatcher.fire(
        PERMISSION_RESULT,
        tool_name="bash",
        tool_args={"command": "ls"},
        decision="allow_once",
    )
    assert merged.verdict == "allow"  # cat abstains
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["event"] == "PermissionResult"
    assert data["decision"] == "allow_once"
    assert data["tool"] == {"name": "bash", "args": {"command": "ls"}}
    assert data["cwd"] == str(tmp_path)


async def test_dispatcher_fire_without_handlers_is_noop(tmp_path):
    dispatcher = HookDispatcher({}, tmp_path)
    merged = await dispatcher.fire(STOP, reason="done")
    assert merged.verdict == "allow"
    assert merged.failed is False


def test_envelope_optional_fields(tmp_path):
    env = envelope(STOP, reason="max_turns", kind="finish", decision="deny")
    assert env["reason"] == "max_turns"
    assert env["kind"] == "finish"
    assert env["decision"] == "deny"
    env = envelope(SESSION_START)
    assert "reason" not in env
    assert "kind" not in env
    assert "decision" not in env


def test_new_events_accepted_from_config(tmp_path):
    config = Config()
    config.hooks = {
        "PreCompact": ["echo a"],
        "PostCompact": ["echo b"],
        "PermissionRequest": ["echo c"],
        "Notification": ["echo d"],
        "Interrupt": ["echo e"],
        "PostToolUseFailure": ["echo f"],
    }
    dispatcher, warnings = dispatcher_from_config(config, tmp_path)
    assert warnings == []
    assert set(dispatcher.handlers) == set(config.hooks)
