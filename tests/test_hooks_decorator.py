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


def make_ctx(tmp_path, config: Config | None = None, auto_approve: bool = False, mode="yolo"):
    from lecode.agent.tools.base import ToolContext

    config = config or Config()
    checker = PermissionChecker(config, mode=mode, cwd=tmp_path)
    return ToolContext(
        cwd=tmp_path, config=config, permission_checker=checker, auto_approve=auto_approve
    )


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
