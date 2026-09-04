"""Tests for settings slash commands (model/theme/permissions/…) and prefixes."""

from __future__ import annotations

from tests.test_tui_app import make_app, make_blocking_app, wait_for

from lecode.config.models import Config
from lecode.permission import Decision
from lecode.slash.catalog import BUILTIN_COMMANDS

# -- /model /models /models-add /provider ------------------------------------------


async def test_model_shows_current(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/model")
    assert f"model: {app.config.llm.model}" in out.getvalue()


async def test_model_switch_updates_everywhere(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/model anthropic/claude-sonnet-4")
    assert app.config.llm.model == "anthropic/claude-sonnet-4"
    assert app.runner.model == "anthropic/claude-sonnet-4"
    assert app.status.model == "anthropic/claude-sonnet-4"
    assert "model: anthropic/claude-sonnet-4" in out.getvalue()


async def test_model_unique_prefix_resolves(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/model moonshotai")
    assert app.config.llm.model == "moonshotai/kimi-k2"


async def test_model_unknown_and_ambiguous_error(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/model nope/not-a-model")
    assert "unknown model: nope/not-a-model" in out.getvalue()
    await app.handle_command("/model openai/gpt-5-")
    assert "ambiguous model" in out.getvalue()
    assert app.config.llm.model == "openai/gpt-5-mini"  # unchanged


async def test_models_lists_catalog_and_marks_current(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/models")
    rendered = out.getvalue()
    assert "openai/gpt-5-mini (current)" in rendered
    assert "anthropic/claude-sonnet-4" in rendered


async def test_models_respects_hidden_models(tmp_path, monkeypatch):
    config = Config()
    config.ui.hidden_models = ["x-ai/"]
    app, _, out = make_app(tmp_path, monkeypatch, [], config=config)
    await app.handle_command("/models")
    rendered = out.getvalue()
    assert "x-ai/grok-4" not in rendered
    assert "openai/gpt-5" in rendered


async def test_models_add_then_switch(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/models-add local/my-finetune")
    assert "added model: local/my-finetune" in out.getvalue()
    await app.handle_command("/model local/my-finetune")
    assert app.config.llm.model == "local/my-finetune"
    await app.handle_command("/models-add local/my-finetune")
    assert "already known: local/my-finetune" in out.getvalue()


async def test_provider_shows_resolved_info(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/provider")
    rendered = out.getvalue()
    assert "provider: openrouter" in rendered
    assert "base url: https://openrouter.ai/api/v1" in rendered


# -- /thinking /reasoning -------------------------------------------------------------


async def test_thinking_show_and_set(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/thinking")
    assert "thinking: medium" in out.getvalue()
    await app.handle_command("/thinking high")
    assert app.config.llm.thinking == "high"
    await app.handle_command("/reasoning")
    assert "thinking: high" in out.getvalue()


async def test_thinking_invalid_level(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/thinking bogus")
    assert "unknown thinking level: bogus" in out.getvalue()
    assert app.config.llm.thinking == "medium"


async def test_thinking_passed_as_reasoning_effort(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "a"}, {"text": "b"}])
    await app.handle_command("/thinking high")
    await app._submit("hi")
    await app._turn_task
    assert provider.requests[-1]["kwargs"]["reasoning_effort"] == "high"
    await app.handle_command("/thinking none")
    await app._submit("again")
    await app._turn_task
    assert provider.requests[-1]["kwargs"]["reasoning_effort"] is None


# -- /permissions /mode /toggle ---------------------------------------------------------


async def test_permissions_shows_mode_and_rules(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/permissions")
    rendered = out.getvalue()
    assert "permission mode: standard" in rendered
    assert "rules: 0 allow · 0 ask · 0 deny" in rendered


async def test_permissions_switch_affects_checker(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    checker = app.runtime.ctx.permission_checker
    assert checker.check("bash", {"command": "ls"}).decision == Decision.ASK
    await app.handle_command("/permissions yolo")
    assert "permission mode: yolo" in out.getvalue()
    assert app.config.permissions.mode == "yolo"
    assert checker.check("bash", {"command": "ls"}).decision == Decision.ALLOW


async def test_permissions_unknown_mode_errors(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/permissions bogus")
    assert "unknown mode: bogus" in out.getvalue()
    assert app.runtime.ctx.permission_checker.mode == "standard"


async def test_mode_alias_switches(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/mode readonly")
    assert app.runtime.ctx.permission_checker.mode == "readonly"
    assert (
        app.runtime.ctx.permission_checker.check("write", {"path": "x"}).decision == Decision.DENY
    )


async def test_toggle_cycles_standard_yolo(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/toggle")
    assert app.runtime.ctx.permission_checker.mode == "yolo"
    await app.handle_command("/toggle")
    assert app.runtime.ctx.permission_checker.mode == "standard"
    assert out.getvalue().count("permission mode:") >= 2


# -- /theme /themes ----------------------------------------------------------------------


async def test_theme_show_and_switch(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/theme")
    assert "theme: default" in out.getvalue()
    old_theme = app._feed._theme
    await app.handle_command("/theme dracula")
    assert app.config.ui.theme == "dracula"
    assert app._feed._theme is not old_theme
    assert app._feed._theme.name == "dracula"
    assert "theme: dracula" in out.getvalue()


async def test_theme_unknown_errors(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/theme not-a-theme")
    assert "unknown theme: not-a-theme" in out.getvalue()
    assert app.config.ui.theme == "default"


async def test_themes_lists_and_marks_current(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/themes")
    rendered = out.getvalue()
    assert "default (current)" in rendered
    assert "dracula" in rendered


# -- /memory /hooks /agents /queue /btw /copy ----------------------------------------------


async def test_memory_routes_to_memory_command(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/memory")
    assert "long-term memory is empty" in out.getvalue()
    await app.handle_command("/memory search nope")
    assert "(no matches)" in out.getvalue()


async def test_memory_disabled(tmp_path, monkeypatch):
    config = Config()
    config.memory.enabled = False
    app, _, out = make_app(tmp_path, monkeypatch, [], config=config)
    await app.handle_command("/memory")
    assert "memory is disabled" in out.getvalue()


async def test_hooks_none_configured(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/hooks")
    assert "no hooks configured" in out.getvalue()


async def test_hooks_lists_handlers(tmp_path, monkeypatch):
    config = Config()
    config.hooks = {"PreToolUse": ["./check.sh"], "Stop": ["./notify.sh"]}
    app, _, out = make_app(tmp_path, monkeypatch, [], config=config)
    await app.handle_command("/hooks")
    rendered = out.getvalue()
    assert "PreToolUse: ./check.sh" in rendered
    assert "Stop: ./notify.sh" in rendered


async def test_agents_lists_primaries_and_subagents(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/agents")
    rendered = out.getvalue()
    assert "primaries:" in rendered
    assert "build (current)" in rendered
    assert "plan — Read-only planning agent." in rendered
    assert "subagents:" in rendered
    assert "explore" in rendered


async def test_queue_empty_and_pending(tmp_path, monkeypatch):
    app, provider, out = make_blocking_app(tmp_path, monkeypatch)
    await app.handle_command("/queue")
    assert "(queues empty)" in out.getvalue()
    await app._submit("go")
    await wait_for(lambda: len(provider.requests) == 1)
    await app._submit("later-msg")
    await app._submit("steer-msg", steer=True)
    out.truncate(0)
    out.seek(0)
    await app.handle_command("/queue")
    rendered = out.getvalue()
    assert "steered:" in rendered and "steer-msg" in rendered
    assert "queued:" in rendered and "later-msg" in rendered
    provider.blocked = False
    provider.release.set()
    await wait_for(lambda: not app._turn_running())


async def test_btw_note_prepended_to_next_submission(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "ok"}])
    await app.handle_command("/btw use python 3.12 syntax")
    assert "noted — included with your next message" in out.getvalue()
    await app._submit("write the script")
    await app._turn_task
    content = provider.requests[-1]["messages"][-1]["content"]
    assert "use python 3.12 syntax" in content
    assert "write the script" in content
    # consumed: the following submission is clean
    app2_script = [{"text": "ok"}]
    app._runner.provider.script = app2_script
    await app._submit("another")
    await app._turn_task
    assert provider.requests[-1]["messages"][-1]["content"] == "another"


async def test_btw_usage_error(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/btw")
    assert "usage: /btw <note>" in out.getvalue()


async def test_copy_nothing_and_copy_response(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "the answer"}])
    await app.handle_command("/copy")
    assert "nothing to copy" in out.getvalue()
    monkeypatch.setattr("lecode.tui.app.copy_to_clipboard", _copy_ok)
    await app._submit("hi")
    await app._turn_task
    await app.handle_command("/copy")
    assert "copied 10 chars" in out.getvalue()


async def _copy_ok(text) -> bool:
    return True


# -- /help /welcome /quit / stubs / dispatch ----------------------------------------------


async def test_help_lists_grouped_commands(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/help")
    rendered = out.getvalue()
    assert "Sessions:" in rendered
    assert "/new — Start a new session" in rendered
    assert "Permissions:" in rendered
    assert "Power features:" in rendered


async def test_help_one_command_shows_hint(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/help rewind")
    assert "/rewind [seq] — Rewind to an earlier point" in out.getvalue()


async def test_help_unknown_command(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/help bogus")
    assert "unknown command: /bogus" in out.getvalue()


async def test_welcome_cheat_sheet(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/welcome")
    rendered = out.getvalue()
    assert "cheat sheet" in rendered
    assert "Alt-Enter" in rendered


async def test_quit_and_exit_via_registry(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/quit")
    assert app._quit is True
    app._quit = False
    await app.handle_command("/exit")
    assert app._quit is True


async def test_every_catalog_command_dispatches(tmp_path, monkeypatch):
    """No stub handlers remain: every catalog name resolves to a real handler."""
    from lecode.slash.handlers import _HANDLERS, _make_stub

    app, _, out = make_app(tmp_path, monkeypatch, [])
    for name, _ in BUILTIN_COMMANDS:
        command = app.commands.get(name)
        assert command is not None
        assert name in _HANDLERS, f"/{name} is still a stub"
    stub = _make_stub("demo")
    await stub(app, [])
    assert "/demo is not yet available" in out.getvalue()


async def test_unique_prefix_dispatch(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/sess")
    assert "session: test-session" in out.getvalue()


async def test_ambiguous_prefix_message(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/mod")
    rendered = out.getvalue()
    assert "ambiguous command: /mod" in rendered
    assert "/model" in rendered and "/mode" in rendered


async def test_unknown_command_message(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/bogus arg")
    assert "unknown command: /bogus" in out.getvalue()


# -- prefixes (.persona, @agent) --------------------------------------------------------------


async def test_persona_prefix_adds_system_overlay(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "reviewed"}])
    await app._submit(".reviewer check this code")
    await app._turn_task
    messages = provider.requests[-1]["messages"]
    assert messages[0]["role"] == "system"  # base system prompt
    overlay = messages[1]
    assert overlay["role"] == "system"
    assert len(overlay["content"]) > 0  # the reviewer persona body
    assert messages[-1] == {"role": "user", "content": "check this code"}
    # overlay is not persisted to the session file
    stored = app.store.load_for_model(app.session)
    assert all(m.get("role") != "system" for m in stored)


async def test_persona_unknown_falls_through_as_text(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "ok"}])
    await app._submit(".notapersona hello")
    await app._turn_task
    assert provider.requests[-1]["messages"][-1] == {
        "role": "user",
        "content": ".notapersona hello",
    }


async def test_persona_without_text_is_usage_error(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app._submit(".reviewer")
    assert "usage: .reviewer <text>" in out.getvalue()


async def test_agent_mention_prepends_routing_note(tmp_path, monkeypatch):
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "ok"}])
    await app._submit("@plan design the schema")
    await app._turn_task
    content = provider.requests[-1]["messages"][-1]["content"]
    assert "@plan" in content
    assert "design the schema" in content
