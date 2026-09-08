"""Tests for the hook tool decorator: narrow-only enforcement through dispatch."""

from __future__ import annotations

import json

from lecode.agent.builder import build_runtime
from lecode.agent.tools.base import Tool, ToolRegistry, ToolResult
from lecode.config.models import Config
from lecode.hooks.decorator import apply_hooks
from lecode.hooks.runner import HookDispatcher, HookHandler
from lecode.permission import Decision, PermissionChecker


class RecordingTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="rec", description="Record calls.", parameters={})
        self.calls: list[dict] = []

    async def run(self, args, ctx) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content=f"ran with {args}")


class FailingTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="fail", description="Return an error result.", parameters={})

    async def run(self, args, ctx) -> ToolResult:
        return ToolResult(content="boom", is_error=True)


class RaisingTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="raise", description="Raise out of run.", parameters={})

    async def run(self, args, ctx) -> ToolResult:
        raise RuntimeError("kaboom")


def make_ctx(
    tmp_path,
    config: Config | None = None,
    auto_approve: bool = False,
    mode="yolo",
    callback=None,
    ask_tool: str | None = None,
    hooks: HookDispatcher | None = None,
):
    from lecode.agent.tools.base import ToolContext
    from lecode.config.models import PermissionRule

    config = config or Config()
    if ask_tool is not None:
        config.permissions.rules.ask[ask_tool] = [PermissionRule(pattern="*")]
    checker = PermissionChecker(config, mode=mode, cwd=tmp_path)
    ctx = ToolContext(
        cwd=tmp_path,
        config=config,
        permission_checker=checker,
        auto_approve=auto_approve,
        approval_callback=callback,
    )
    if hooks is not None:
        ctx.extras["hooks"] = hooks
    return ctx


def dispatcher(tmp_path, pre: list[str] | None = None, post: list[str] | None = None):
    handlers = {}
    if pre is not None:
        handlers["PreToolUse"] = [HookHandler(c) for c in pre]
    if post is not None:
        handlers["PostToolUse"] = [HookHandler(c) for c in post]
    return HookDispatcher(handlers, tmp_path)


def deny_hook(reason: str = "hook says no") -> str:
    return f"echo '{json.dumps({'verdict': 'deny', 'reason': reason})}'"


async def test_no_hooks_runs_normally(tmp_path):
    tool = RecordingTool()
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path))
    ctx = make_ctx(tmp_path)
    message, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
    assert tool.calls == [{}]
    assert message["content"].startswith("ran with")


async def test_checker_allow_plus_hook_deny_blocks(tmp_path):
    tool = RecordingTool()
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, pre=[deny_hook()]))
    ctx = make_ctx(tmp_path)  # yolo: checker allows everything
    message, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is True
    assert "denied by hook" in message["content"]
    assert "hook says no" in message["content"]
    assert tool.calls == []  # tool never ran


async def test_checker_deny_never_consults_hooks(tmp_path):
    marker = tmp_path / "hook-ran"
    tool = RecordingTool()
    registry = apply_hooks(
        ToolRegistry([tool]),
        dispatcher(tmp_path, pre=[f'echo \'{{"verdict": "allow"}}\'; touch {marker}']),
    )
    ctx = make_ctx(tmp_path, mode="readonly")  # 'rec' is not a read tool → Deny
    assert ctx.permission_checker.check("rec", {}).decision == Decision.DENY
    message, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is True
    assert message["content"].startswith("denied:")
    assert not marker.exists()  # hook was never consulted
    assert tool.calls == []


async def test_hook_ask_requires_approval(tmp_path):
    tool = RecordingTool()
    ask = f"echo '{json.dumps({'verdict': 'ask', 'reason': 'confirm?'})}'"
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, pre=[ask]))
    ctx = make_ctx(tmp_path, auto_approve=False)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is True
    assert result.metadata["needs_approval"] is True
    assert tool.calls == []


async def test_hook_ask_auto_approved(tmp_path):
    tool = RecordingTool()
    ask = f"echo '{json.dumps({'verdict': 'ask'})}'"
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, pre=[ask]))
    ctx = make_ctx(tmp_path, auto_approve=True)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
    assert tool.calls == [{}]


async def test_rewritten_input_reaches_tool(tmp_path):
    tool = RecordingTool()
    rewrite = f"echo '{json.dumps({'verdict': 'allow', 'rewritten_input': {'text': 'rewritten'}})}'"
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, pre=[rewrite]))
    ctx = make_ctx(tmp_path)
    message, _ = await registry.dispatch_result("c1", "rec", '{"text": "original"}', ctx)
    assert tool.calls == [{"text": "rewritten"}]
    assert "rewritten" in message["content"]


async def test_defer_runs_with_original_args(tmp_path):
    tool = RecordingTool()
    defer = f"echo '{json.dumps({'verdict': 'defer'})}'"
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, pre=[defer]))
    ctx = make_ctx(tmp_path)
    await registry.dispatch_result("c1", "rec", '{"a": 1}', ctx)
    assert tool.calls == [{"a": 1}]


async def test_post_tool_use_is_informational(tmp_path):
    tool = RecordingTool()
    post = f"echo '{json.dumps({'verdict': 'deny', 'reason': 'too late'})}'"
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, post=[post]))
    ctx = make_ctx(tmp_path)
    message, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False  # cannot undo
    assert message["content"].startswith("ran with")
    assert result.metadata["post_hook"] == {"verdict": "deny", "reason": "too late"}


async def test_post_tool_use_abstain_records_nothing(tmp_path):
    tool = RecordingTool()
    registry = apply_hooks(ToolRegistry([tool]), dispatcher(tmp_path, post=["true"]))
    ctx = make_ctx(tmp_path)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert "post_hook" not in result.metadata


async def test_apply_hooks_is_idempotent(tmp_path):
    counter = tmp_path / "count"
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    disp = dispatcher(tmp_path, pre=[f"echo x >> {counter}"])  # abstains
    apply_hooks(registry, disp)
    apply_hooks(registry, disp)  # second application must not double-wrap
    ctx = make_ctx(tmp_path)
    await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 1


async def test_builder_decorates_when_hooks_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "no-skills"))
    config = Config()
    config.hooks = {"PreToolUse": [deny_hook("blocked by config hook")]}
    runtime = build_runtime(config, tmp_path, auto_approve=True)
    assert runtime.hooks is not None
    _, result = await runtime.registry.dispatch_result(
        "c1", "write", '{"file_path": "x.txt", "content": "hi"}', runtime.ctx
    )
    assert result.is_error is True
    assert "blocked by config hook" in result.content


async def test_builder_no_hooks_no_dispatcher(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "no-skills"))
    runtime = build_runtime(Config(), tmp_path)
    assert runtime.hooks is None
    assert runtime.warnings == []


async def test_builder_unknown_hook_event_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LECODE_SKILLS_DIR", str(tmp_path / "no-skills"))
    config = Config()
    config.hooks = {"Nope": ["echo hi"], "Stop": ["echo ok"]}
    runtime = build_runtime(config, tmp_path)
    assert runtime.warnings == ["unknown hook event: Nope"]
    assert "Stop" in runtime.hooks.handlers


# -- PostToolUseFailure (fires instead of PostToolUse on failure) ----------------


def event_log_dispatcher(tmp_path, *events: str) -> tuple[HookDispatcher, object]:
    """A dispatcher whose handlers append each envelope as a JSON line."""
    log = tmp_path / "events.jsonl"
    handlers = {event: [HookHandler(f"cat >> {log}; echo >> {log}")] for event in events}
    return HookDispatcher(handlers, tmp_path), log


def logged_events(log) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


async def test_failure_hook_fires_on_error_result_not_post(tmp_path):
    disp, log = event_log_dispatcher(tmp_path, "PostToolUse", "PostToolUseFailure")
    registry = apply_hooks(ToolRegistry([FailingTool()]), disp)
    ctx = make_ctx(tmp_path)
    _, result = await registry.dispatch_result("c1", "fail", "{}", ctx)
    assert result.is_error is True
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PostToolUseFailure"]
    assert events[0]["result"] == {"content": "boom", "is_error": True}
    assert events[0]["tool"]["name"] == "fail"


async def test_failure_hook_fires_on_exception_then_propagates(tmp_path):
    disp, log = event_log_dispatcher(tmp_path, "PostToolUse", "PostToolUseFailure")
    registry = apply_hooks(ToolRegistry([RaisingTool()]), disp)
    ctx = make_ctx(tmp_path)
    _, result = await registry.dispatch_result("c1", "raise", "{}", ctx)
    # _execute converts the re-raised exception into an error result
    assert result.is_error is True
    assert "RuntimeError: kaboom" in result.content
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PostToolUseFailure"]
    assert events[0]["reason"] == "RuntimeError: kaboom"


async def test_failure_hook_not_fired_on_success(tmp_path):
    tool = RecordingTool()
    disp, log = event_log_dispatcher(tmp_path, "PostToolUse", "PostToolUseFailure")
    registry = apply_hooks(ToolRegistry([tool]), disp)
    ctx = make_ctx(tmp_path)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PostToolUse"]


# -- PermissionRequest / PermissionResult ----------------------------------------


def permission_hooks(tmp_path) -> tuple[HookDispatcher, object]:
    return event_log_dispatcher(tmp_path, "PermissionRequest", "PermissionResult")


async def test_permission_request_result_allow_once(tmp_path):
    from lecode.permission import AllowOnce

    async def approve(name, args, reason):
        return AllowOnce()

    hooks, log = permission_hooks(tmp_path)
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    ctx = make_ctx(tmp_path, callback=approve, ask_tool="rec", hooks=hooks)
    _, result = await registry.dispatch_result("c1", "rec", '{"a": 1}', ctx)
    assert result.is_error is False
    assert tool.calls == [{"a": 1}]
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PermissionRequest", "PermissionResult"]
    assert events[0]["tool"] == {"name": "rec", "args": {"a": 1}}
    assert events[1]["decision"] == "allow_once"


async def test_permission_result_allow_always(tmp_path):
    from lecode.permission import AllowAlways

    async def approve(name, args, reason):
        return AllowAlways(pattern="*")

    hooks, log = permission_hooks(tmp_path)
    registry = ToolRegistry([RecordingTool()])
    ctx = make_ctx(tmp_path, callback=approve, ask_tool="rec", hooks=hooks)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PermissionRequest", "PermissionResult"]
    assert events[1]["decision"] == "allow_always"


async def test_permission_result_deny(tmp_path):
    from lecode.permission import Deny

    async def deny(name, args, reason):
        return Deny()

    hooks, log = permission_hooks(tmp_path)
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    ctx = make_ctx(tmp_path, callback=deny, ask_tool="rec", hooks=hooks)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is True
    assert "denied by user" in result.content
    assert tool.calls == []
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PermissionRequest", "PermissionResult"]
    assert events[1]["decision"] == "deny"


async def test_permission_result_auto_on_auto_approve(tmp_path):
    hooks, log = permission_hooks(tmp_path)
    registry = ToolRegistry([RecordingTool()])
    ctx = make_ctx(tmp_path, auto_approve=True, ask_tool="rec", hooks=hooks)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PermissionResult"]  # no request without a callback
    assert events[0]["decision"] == "auto"


async def test_permission_result_deny_without_callback(tmp_path):
    hooks, log = permission_hooks(tmp_path)
    registry = ToolRegistry([RecordingTool()])
    ctx = make_ctx(tmp_path, callback=None, ask_tool="rec", hooks=hooks)
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is True
    assert result.metadata["needs_approval"] is True
    events = logged_events(log)
    assert [e["event"] for e in events] == ["PermissionResult"]
    assert events[0]["decision"] == "deny"


async def test_permission_hooks_not_fired_without_ask(tmp_path):
    hooks, log = permission_hooks(tmp_path)
    registry = ToolRegistry([RecordingTool()])
    ctx = make_ctx(tmp_path, hooks=hooks)  # yolo allow: no permission round-trip
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
    assert logged_events(log) == []


async def test_permission_hooks_absent_dispatcher(tmp_path):
    async def approve(name, args, reason):
        from lecode.permission import AllowOnce

        return AllowOnce()

    registry = ToolRegistry([RecordingTool()])
    ctx = make_ctx(tmp_path, callback=approve, ask_tool="rec")  # no hooks in extras
    _, result = await registry.dispatch_result("c1", "rec", "{}", ctx)
    assert result.is_error is False
