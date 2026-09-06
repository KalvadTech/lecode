"""Tests for session export (HTML/markdown), /import, and gist sharing."""

from __future__ import annotations

import json
from importlib import resources

import pytest
import respx
from tests.test_tui_app import make_app

from lecode.extras.export import (
    GITHUB_API_URL,
    ShareError,
    export_html,
    export_markdown,
    share_gist,
)
from lecode.session.storage import SessionStore

GIST_URL = "https://gist.github.com/u/abc123"


def make_session(tmp_path, monkeypatch, name="demo"):
    """A store+session rooted at tmp_path with an isolated config dir."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    store = SessionStore()
    session = store.create(name, tmp_path, model="openai/gpt-5-mini")
    return store, session


def populate(store, session):
    """A representative history: text, usage, a tool call pair, thinking."""
    store.append_message(session, {"role": "user", "content": "hello there"})
    store.append_message(
        session,
        {"role": "assistant", "content": "hi! let me look"},
        usage={"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.002},
    )
    store.append_message(
        session,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
    )
    store.append_message(
        session,
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "file.txt"},
    )
    store.append_message(
        session,
        {"role": "assistant", "content": "all done", "reasoning": "thinking hard"},
    )


# -- HTML export -----------------------------------------------------------------


def test_html_contains_messages(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    path = export_html(session, store)
    document = path.read_text(encoding="utf-8")
    assert "hello there" in document
    assert "all done" in document
    assert '<div class="msg user">' in document
    assert '<div class="msg assistant">' in document


def test_html_thinking_collapsed_in_details(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    document = export_html(session, store).read_text(encoding="utf-8")
    assert '<details class="thinking">' in document
    assert "thinking hard" in document


def test_html_tool_blocks(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    document = export_html(session, store).read_text(encoding="utf-8")
    assert '<div class="msg tool-call">' in document
    assert "bash(" in document
    assert '<div class="msg tool-result">' in document
    assert "file.txt" in document


def test_html_header_stats(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    document = export_html(session, store).read_text(encoding="utf-8")
    assert "<h1>demo</h1>" in document
    assert "openai/gpt-5-mini" in document
    assert "100 tokens in / 20 out" in document
    assert "$0.0020" in document


def test_html_uses_kalvad_palette(tmp_path, monkeypatch):
    """The standalone HTML follows the app's Kalvad palette (purple on dark)."""
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    document = export_html(session, store).read_text(encoding="utf-8")
    # Theme colors from lecode.tui.themes (accent / text / muted / thinking).
    for color in ("#a78bfa", "#ece7f7", "#8a80a3", "#6e6392"):
        assert color in document
    # No leftover GitHub-dark palette.
    for color in ("#0d1117", "#1f6feb", "#238636"):
        assert color not in document


def test_html_escapes_content(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    store.append_message(session, {"role": "user", "content": "<script>alert(1)</script>"})
    document = export_html(session, store).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document


def test_html_hides_tombstoned_messages(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    store.append_message(session, {"role": "user", "content": "visible question"})
    store.append_message(session, {"role": "assistant", "content": "visible answer"})
    store.append_message(session, {"role": "user", "content": "hidden question"})
    store.append_message(session, {"role": "assistant", "content": "hidden answer"})
    store.undo(session)
    document = export_html(session, store).read_text(encoding="utf-8")
    assert "visible answer" in document
    assert "hidden question" not in document
    assert "hidden answer" not in document


def test_template_is_self_contained():
    template = resources.files("lecode.data").joinpath("export_template.html").read_text("utf-8")
    assert "http://" not in template
    assert "https://" not in template
    assert "<script" not in template
    assert "<link" not in template


def test_default_out_path(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    path = export_html(session, store)
    assert path == tmp_path / "demo.html"
    assert path.is_file()


def test_explicit_out_path(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    out = tmp_path / "nested" / "report.html"
    path = export_html(session, store, out)
    assert path == out
    assert path.is_file()


def test_html_marks_attachments(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    store.append_message(
        session,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        },
    )
    document = export_html(session, store).read_text(encoding="utf-8")
    assert "look at this" in document
    assert "1 attachment(s)" in document


# -- markdown summary ---------------------------------------------------------------


def test_markdown_summary(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    markdown = export_markdown(session, store)
    assert markdown.startswith("# demo")
    assert "## user" in markdown
    assert "hello there" in markdown
    assert "`bash`" in markdown
    assert "$0.0020" in markdown


# -- /export and /import commands -----------------------------------------------------


async def test_export_command_writes_file(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "hi"}])
    await app._submit("hello")
    await app._turn_task
    await app.handle_command("/export")
    path = tmp_path / "test-session.html"
    assert path.is_file()
    assert "exported:" in out.getvalue()
    assert "hello" in path.read_text(encoding="utf-8")


async def test_export_command_explicit_path(tmp_path, monkeypatch):
    app, _, _ = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/export custom-out.html")
    assert (tmp_path / "custom-out.html").is_file()


async def test_import_command(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    other = SessionStore(config_dir=tmp_path / "other-cfg")
    external = other.create("ext-session", tmp_path)
    other.append_message(external, {"role": "user", "content": "from outside"})
    await app.handle_command(f"/import {external.path}")
    rendered = out.getvalue()
    assert "imported: ext-session" in rendered
    assert "/resume" in rendered
    assert [m.name for m in app.store.list_sessions()] == ["ext-session", "test-session"]


async def test_import_command_invalid_file(tmp_path, monkeypatch):
    app, _, out = make_app(tmp_path, monkeypatch, [])
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not a session\n", encoding="utf-8")
    await app.handle_command(f"/import {bad}")
    assert "import failed" in out.getvalue()


def test_import_roundtrip(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    copy = tmp_path / "copy.jsonl"
    copy.write_text(session.path.read_text(encoding="utf-8"), encoding="utf-8")
    imported = store.import_session(copy)
    assert imported.id != session.id  # id collision → fresh id
    assert imported.name == "demo-2"  # name collision → suffix
    original = [r.message for r in store.load_messages(session)]
    replayed = [r.message for r in store.load_messages(imported)]
    assert replayed == original


# -- gist sharing -----------------------------------------------------------------


@respx.mock
async def test_share_gist_creates_secret_gist(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    populate(store, session)
    route = respx.post(GITHUB_API_URL).respond(201, json={"html_url": GIST_URL})
    url = await share_gist(session, store, gh_token="tok")
    assert url == GIST_URL
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer tok"
    payload = json.loads(request.content)
    assert payload["public"] is False
    assert set(payload["files"]) == {"demo.jsonl", "demo.md"}
    assert '"type":"meta"' in payload["files"]["demo.jsonl"]["content"]
    assert "hello there" in payload["files"]["demo.md"]["content"]


@respx.mock
async def test_share_gist_401_is_clear_error(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    respx.post(GITHUB_API_URL).respond(401, json={"message": "Bad credentials"})
    with pytest.raises(ShareError, match="authentication failed"):
        await share_gist(session, store, gh_token="bad-token")


async def test_share_gist_without_token(tmp_path, monkeypatch):
    store, session = make_session(tmp_path, monkeypatch)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ShareError, match="no GitHub token"):
        await share_gist(session, store)


@respx.mock
async def test_share_command_prints_url(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "tok")
    respx.post(GITHUB_API_URL).respond(201, json={"html_url": GIST_URL})
    app, _, out = make_app(tmp_path, monkeypatch, [{"text": "hi"}])
    await app._submit("hello")
    await app._turn_task
    await app.handle_command("/share")
    assert GIST_URL in out.getvalue()


async def test_share_command_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    app, _, out = make_app(tmp_path, monkeypatch, [])
    await app.handle_command("/share")
    assert "no GitHub token" in out.getvalue()
