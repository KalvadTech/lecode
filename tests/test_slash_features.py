"""Tests for the phase-11 handlers: init/tutor/review/notifications/prompt/
compress/editsys."""

from __future__ import annotations

from tests.test_tui_app import make_app
from tests.test_worktree import make_repo

# -- /init -------------------------------------------------------------------------


async def test_init_creates_agents_md(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/init")
    path = tmp_path / "AGENTS.md"
    assert path.is_file()
    text = path.read_text()
    assert "# AGENTS.md" in text
    assert "Build & test" in text
    assert "wrote" in out.getvalue()


async def test_init_refuses_to_overwrite(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# existing\n")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/init")
    assert "already exists" in out.getvalue()
    assert (tmp_path / "AGENTS.md").read_text() == "# existing\n"


# -- /tutor ------------------------------------------------------------------------


async def test_tutor_lists_topics(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tutor")
    rendered = out.getvalue()
    assert "usage: /tutor <topic>" in rendered
    for topic in ("permissions", "worktrees", "mcp"):
        assert topic in rendered


async def test_tutor_answers_a_topic(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tutor permissions")
    assert "readonly" in out.getvalue()


async def test_tutor_unknown_topic(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/tutor quantum")
    assert "unknown topic: quantum" in out.getvalue()


# -- /review ------------------------------------------------------------------------


async def test_review_clean_repo(tmp_path, monkeypatch):
    await make_repo(tmp_path)
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "ok"}])
    await app.handle_command("/review")
    assert "nothing to review" in out.getvalue()
    assert provider.requests == []


async def test_review_submits_diff_with_reviewer_persona(tmp_path, monkeypatch):
    await make_repo(tmp_path)
    (tmp_path / "file.txt").write_text("changed content\n")
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "review done"}])
    await app.handle_command("/review")
    await app._turn_task
    request = provider.requests[-1]
    prompt = request["messages"][-1]["content"]
    assert "meticulous code reviewer" in prompt  # the reviewer persona
    assert "changed content" in prompt  # the diff body


async def test_review_listed_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    app, provider, _ = make_app(tmp_path, monkeypatch, [{"text": "review done"}])
    await app.handle_command("/review a.py")
    await app._turn_task
    prompt = provider.requests[-1]["messages"][-1]["content"]
    assert "meticulous code reviewer" in prompt
    assert "### a.py" in prompt
    assert "x = 1" in prompt


async def test_review_outside_git_repo(tmp_path, monkeypatch):
    app, provider, out = make_app(tmp_path, monkeypatch, [{"text": "unused"}])
    await app.handle_command("/review")
    assert "not a git repository" in out.getvalue()
    assert provider.requests == []


# -- /notifications -------------------------------------------------------------------


async def test_notifications_shows_state(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])  # make_app disables sounds
    await app.handle_command("/notifications")
    rendered = out.getvalue()
    assert "notifications: off" in rendered
    assert "volume" in rendered and "approval" in rendered


async def test_notifications_toggle(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    assert app.config.notifications.enabled is False
    await app.handle_command("/notifications on")
    assert app.config.notifications.enabled is True
    await app.handle_command("/notifications")
    assert "notifications: on" in out.getvalue()
    await app.handle_command("/notifications bogus")
    assert "usage: /notifications" in out.getvalue()


# -- /prompt -------------------------------------------------------------------------


async def test_prompt_prints_assembled_system_prompt(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# Project\n\nAlways run the tests.\n")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/prompt")
    rendered = out.getvalue()
    assert "You are lecode" in rendered
    assert "Always run the tests." in rendered  # AGENTS.md made it in


# -- /compress -------------------------------------------------------------------------


async def test_compress_aliases_compact(tmp_path, monkeypatch):
    from lecode.slash.handlers import cmd_compact

    app, _, out = make_app(tmp_path, monkeypatch, [])
    assert app.commands.get("compress").handler is cmd_compact
    await app.handle_command("/compress")
    assert "not enough history to compact" in out.getvalue()


# -- /editsys ---------------------------------------------------------------------------


async def _fake_editor(tmp_path, monkeypatch, body: str) -> None:
    script = tmp_path / "editor.sh"
    script.write_text(f"#!/bin/sh\n{body}\n")
    monkeypatch.setenv("EDITOR", f"/bin/sh {script}")

    async def run_directly(func, **kwargs):
        func()

    monkeypatch.setattr("lecode.tui.input.run_in_terminal", run_directly)


async def test_editsys_saves_session_override(tmp_path, monkeypatch):
    await _fake_editor(tmp_path, monkeypatch, "printf 'CUSTOM SYSTEM PROMPT' > \"$1\"")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/editsys")
    assert app.config.llm.system_prompt.custom == "CUSTOM SYSTEM PROMPT"
    assert app.runtime.system_prompt == "CUSTOM SYSTEM PROMPT"
    assert "overridden for this session" in out.getvalue()


async def test_editsys_unchanged_is_noop(tmp_path, monkeypatch):
    await _fake_editor(tmp_path, monkeypatch, "true")
    app, _, out = make_app(tmp_path, monkeypatch, [])
    original = app.runtime.system_prompt
    await app.handle_command("/editsys")
    assert app.runtime.system_prompt == original
    assert app.config.llm.system_prompt.custom is None
    assert "unchanged" in out.getvalue()


# -- /doctor ---------------------------------------------------------------------------


def _patch_doctor_provider(monkeypatch, origin="live", count=2):
    """Patch the deferred fetch_catalog import used by /doctor."""
    from tests.fakes import sample_catalog

    import lecode.cli as cli
    from lecode.providers.live import LoadedCatalog

    monkeypatch.setattr(
        cli,
        "fetch_catalog",
        lambda config, api_key=None: LoadedCatalog(sample_catalog(), origin, count),
    )


async def test_doctor_all_sections(tmp_path, monkeypatch):
    """Every section renders with a mark; a healthy setup ends in 'all good'."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _patch_doctor_provider(monkeypatch)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/doctor")
    rendered = out.getvalue()
    for needle in (
        "fd:",
        "rg:",
        "rtk:",
        "config:",
        "provider:",
        "model:",
        "connectivity: reachable — 2 models",
        "mcp:",
        "session:",
        "memory:",
        "hooks:",
        "lsp:",
        "telemetry:",
        "permissions:",
        "tools:",
    ):
        assert needle in rendered, f"missing {needle!r}:\n{rendered}"
    # no API key in the test env → one warning, not "all good"
    assert "doctor: 1 issue(s)" in rendered


async def test_doctor_missing_binary_fails(tmp_path, monkeypatch):
    import shutil

    _patch_doctor_provider(monkeypatch)
    real_which = shutil.which

    def which(name, path=None):
        return None if name == "fd" else real_which(name, path=path)

    monkeypatch.setattr(shutil, "which", which)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/doctor")
    rendered = out.getvalue()
    assert "✗ fd: MISSING" in rendered
    assert "install:" in rendered


async def test_doctor_unreachable_provider(tmp_path, monkeypatch):
    _patch_doctor_provider(monkeypatch, origin="empty", count=0)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/doctor")
    rendered = out.getvalue()
    assert "✗ connectivity: catalog fetch failed" in rendered
