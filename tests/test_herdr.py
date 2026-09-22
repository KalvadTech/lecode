"""Tests for the optional Herdr lifecycle reporter."""

from __future__ import annotations

import json

from tests.test_tui_app import make_app

from lecode.extras import herdr


def _fake_herdr(tmp_path):
    log = tmp_path / "calls.jsonl"
    binary = tmp_path / "herdr"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['HERDR_LOG'], 'a') as output:\n"
        "    json.dump(sys.argv[1:], output)\n"
        "    output.write('\\n')\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary, log


def _calls(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_reports_lifecycle_and_releases(tmp_path, monkeypatch):
    binary, log = _fake_herdr(tmp_path)
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_BIN_PATH", str(binary))
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p1")
    monkeypatch.setenv("HERDR_LOG", str(log))

    herdr.report("working", session_id="session-1")
    herdr.report("blocked", message="approval needed: bash")
    herdr.report("idle")
    herdr.release()

    calls = _calls(log)
    assert calls[0][:10] == [
        "pane",
        "report-agent",
        "w1:p1",
        "--source",
        "herdr:lecode",
        "--agent",
        "lecode",
        "--state",
        "working",
        "--seq",
    ]
    assert "--agent-session-id" in calls[0]
    assert "session-1" in calls[0]
    assert calls[1][calls[1].index("--state") + 1] == "blocked"
    assert calls[1][calls[1].index("--message") + 1] == "approval needed: bash"
    assert calls[2][calls[2].index("--state") + 1] == "idle"
    assert calls[3] == [
        "pane",
        "release-agent",
        "w1:p1",
        "--source",
        "herdr:lecode",
        "--agent",
        "lecode",
    ]
    sequences = [int(call[call.index("--seq") + 1]) for call in calls[:3]]
    assert sequences[0] < sequences[1] < sequences[2]


def test_is_inert_outside_herdr(tmp_path, monkeypatch):
    _, log = _fake_herdr(tmp_path)
    monkeypatch.setenv("HERDR_LOG", str(log))

    herdr.report("working")
    herdr.release()

    assert not log.exists()


async def test_tui_turn_reports_working_and_idle(tmp_path, monkeypatch):
    binary, log = _fake_herdr(tmp_path)
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_BIN_PATH", str(binary))
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p1")
    monkeypatch.setenv("HERDR_LOG", str(log))
    app, _, _ = make_app(tmp_path, monkeypatch, [{"text": "hi"}])

    await app._submit("hello")
    assert app._turn_task is not None
    await app._turn_task

    states = [call[call.index("--state") + 1] for call in _calls(log)]
    assert states == ["working", "idle"]
