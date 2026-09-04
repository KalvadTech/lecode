"""Session export (standalone HTML + markdown) and secret-gist sharing.

The HTML export renders the logical session view (tombstones applied) into
a single self-contained file — inline CSS from ``data/export_template.html``,
no external assets. Sharing creates a **secret** gist via the GitHub API
with two files: the raw session JSONL (re-importable with ``/import``) and
a rendered markdown summary. The token comes from ``GH_TOKEN`` /
``GITHUB_TOKEN`` (or the explicit argument); without one, a clear error.
"""

from __future__ import annotations

import html
import os
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

import httpx

from lecode.session.stats import session_stats

if TYPE_CHECKING:
    from lecode.session.storage import Session, SessionStore

#: GitHub API endpoint for gist creation.
GITHUB_API_URL = "https://api.github.com/gists"

#: HTTP timeout for the gist request.
GIST_TIMEOUT_S = 30.0


class ShareError(Exception):
    """Sharing failed cleanly (no token, HTTP error, network down)."""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_cost(cost_usd: float) -> str:
    return f"${cost_usd:.4f}" if cost_usd < 1 else f"${cost_usd:.2f}"


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "-" for c in name).strip("-.")
    return cleaned or "session"


def _template() -> Template:
    text = resources.files("lecode.data").joinpath("export_template.html").read_text("utf-8")
    return Template(text)


def _text_and_attachments(content: Any) -> tuple[str, int]:
    """(text, attachment count) for plain-string or content-part messages."""
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        text = " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        count = sum(
            1
            for part in content
            if isinstance(part, dict) and part.get("type") in ("image_url", "file")
        )
        return text.strip(), count
    return str(content or "").strip(), 0


def _render_message(message: dict[str, Any]) -> str:
    """One message as an HTML block; thinking collapses into ``<details>``."""
    role = str(message.get("role", "unknown"))
    text, attachments = _text_and_attachments(message.get("content"))
    blocks: list[str] = []
    if message.get("reasoning"):
        thinking = html.escape(str(message["reasoning"]))
        blocks.append(
            f'<details class="thinking"><summary>thinking</summary><div>{thinking}</div></details>'
        )
    if role == "tool":
        name = html.escape(str(message.get("name", "tool")))
        body = html.escape(text) or "(no output)"
        blocks.append(
            f'<div class="msg tool-result"><div class="role">{name} result</div>{body}</div>'
        )
        return "\n".join(blocks)
    css = role if role in ("user", "assistant", "system") else "system"
    body = html.escape(text) or "<em>(no text)</em>"
    if attachments:
        body += f'\n<span class="attachment">📎 {attachments} attachment(s)</span>'
    blocks.append(f'<div class="msg {css}"><div class="role">{html.escape(role)}</div>{body}</div>')
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = html.escape(str(function.get("name", "?")))
        args = html.escape(_clip(str(function.get("arguments", "")), 500))
        blocks.append(
            f'<div class="msg tool-call"><div class="role">tool call</div>{name}({args})</div>'
        )
    return "\n".join(blocks)


def export_html(session: Session, store: SessionStore, out_path: Path | str | None = None) -> Path:
    """Write the session as a standalone HTML file; returns the path.

    Default output: ``<session cwd>/<session-name>.html``.
    """
    stats = session_stats(store, session)
    body = "\n".join(_render_message(r.message) for r in store.load_messages(session))
    document = _template().safe_substitute(
        title=html.escape(session.name),
        meta=html.escape(
            f"id: {session.id} · model: {session.meta.model or '?'} · "
            f"agent: {session.meta.agent} · cwd: {session.meta.cwd}"
        ),
        stats=html.escape(
            f"{stats.message_count} messages · {stats.input_tokens} tokens in / "
            f"{stats.output_tokens} out · cost {_format_cost(stats.cost_usd)} · "
            f"created {stats.created_at}"
        ),
        body=body,
        exported_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    path = (
        Path(out_path)
        if out_path is not None
        else Path(session.meta.cwd) / f"{_safe_filename(session.name)}.html"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def export_markdown(session: Session, store: SessionStore) -> str:
    """A compact markdown transcript (the gist's human-readable file)."""
    stats = session_stats(store, session)
    lines = [
        f"# {session.name}",
        "",
        f"- id: `{session.id}`",
        f"- model: {session.meta.model or '?'} · agent: {session.meta.agent}",
        f"- cwd: `{session.meta.cwd}`",
        f"- created: {stats.created_at}",
        f"- messages: {stats.message_count} · tokens: {stats.input_tokens} in / "
        f"{stats.output_tokens} out · cost {_format_cost(stats.cost_usd)}",
        "",
    ]
    for record in store.load_messages(session):
        message = record.message
        text, attachments = _text_and_attachments(message.get("content"))
        lines += [f"## {message.get('role', '?')}", ""]
        if text:
            lines += [text, ""]
        if attachments:
            lines += [f"📎 {attachments} attachment(s)", ""]
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name", "?")
            arguments = _clip(str(function.get("arguments", "")), 200)
            lines.append(f"- ⚙ `{name}` `{arguments}`")
        lines.append("")
    return "\n".join(lines)


async def share_gist(session: Session, store: SessionStore, gh_token: str | None = None) -> str:
    """Create a secret gist with the raw JSONL + a markdown summary.

    Returns the gist URL. Raises :class:`ShareError` on any failure.
    """
    token = gh_token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ShareError("no GitHub token — set GH_TOKEN or GITHUB_TOKEN")
    name = _safe_filename(session.name)
    payload = {
        "description": f"lecode session: {session.name}",
        "public": False,
        "files": {
            f"{name}.jsonl": {"content": session.path.read_text(encoding="utf-8")},
            f"{name}.md": {"content": export_markdown(session, store)},
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "lecode",
    }
    try:
        async with httpx.AsyncClient(timeout=GIST_TIMEOUT_S) as client:
            response = await client.post(GITHUB_API_URL, json=payload, headers=headers)
    except httpx.HTTPError as e:
        raise ShareError(f"gist request failed: {e}") from e
    if response.status_code == 401:
        raise ShareError("GitHub authentication failed (401) — check your token")
    if response.status_code != 201:
        detail = response.text[:200].strip()
        raise ShareError(f"gist creation failed: HTTP {response.status_code} {detail}".strip())
    url = response.json().get("html_url")
    if not url:
        raise ShareError("gist created but the response carried no URL")
    return str(url)
