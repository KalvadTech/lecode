"""Tests for the fuzzy trigger pickers (@ files/agents, / commands, . personas)."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from tests.fakes import FakeProvider

from lecode.agent.builder import build_runtime
from lecode.config.models import Config
from lecode.context.agents import load_agents
from lecode.context.skills import Skill, SkillRegistry
from lecode.extras.proc import ProcResult
from lecode.session.storage import SessionStore
from lecode.tui.app import TuiApp
from lecode.tui.input import FileLister
from lecode.tui.pickers import TriggerCompleter, fuzzy_score, persona_names

# -- fuzzy_score ---------------------------------------------------------------


def test_fuzzy_subsequence_match():
    assert fuzzy_score("qut", "quit") is not None
    assert fuzzy_score("xyz", "quit") is None


def test_fuzzy_empty_query_matches_everything():
    assert fuzzy_score("", "anything") == 0
    assert fuzzy_score("", "") == 0


def test_fuzzy_prefix_beats_mid_word():
    assert fuzzy_score("re", "review") > fuzzy_score("re", "share")


def test_fuzzy_contiguous_beats_scattered():
    assert fuzzy_score("ab", "abx") > fuzzy_score("ab", "axb")


def test_fuzzy_boundary_bonus():
    assert fuzzy_score("sub", "model-subagent") > fuzzy_score("sub", "asubx")


def test_fuzzy_case_insensitive():
    assert fuzzy_score("RE", "review") == fuzzy_score("re", "review")
    assert fuzzy_score("Re", "REVIEW") is not None


# -- TriggerCompleter -----------------------------------------------------------


@pytest.fixture
def lister(tmp_path, monkeypatch):
    """A FileLister over a fake fd listing."""
    files = "src/\nsrc/app.py\nsrc/lecode/\ndocs/\ndocs/build-plan.md\nREADME.md\n"

    async def _fd(*args, **kwargs):
        return ProcResult(exit_code=0, stdout=files, stderr="")

    monkeypatch.setattr("lecode.tui.input.run_proc", _fd)
    return FileLister(tmp_path)


@pytest.fixture
def agents():
    return load_agents()


@pytest.fixture
def skills():
    return SkillRegistry()


async def _complete(completer, text: str):
    doc = Document(text, len(text))
    return [c async for c in completer.get_completions_async(doc, CompleteEvent())]


async def test_at_lists_files_and_agents(lister, agents, skills):
    completions = await _complete(TriggerCompleter(lister, agents, skills), "@")
    metas = {str(c.display_meta_text) for c in completions}
    assert any("agent" in m for m in metas)
    assert "file" in metas
    texts = [c.text for c in completions]
    assert "@build " in texts
    assert "README.md" in texts


async def test_at_agent_keeps_mention_form(lister, agents, skills):
    completions = await _complete(TriggerCompleter(lister, agents, skills), "@pl")
    agent = next(c for c in completions if c.text.startswith("@"))
    assert agent.text == "@plan "
    assert agent.start_position == -len("@pl")
    assert "agent" in str(agent.display_meta_text)


async def test_at_file_inserts_path_without_at(lister, agents, skills):
    completions = await _complete(TriggerCompleter(lister, agents, skills), "@src/ap")
    file = next(c for c in completions if c.text == "src/app.py")
    assert file.start_position == -len("@src/ap")
    assert str(file.display_meta_text) == "file"


async def test_at_ranking_prefix_first(lister, agents, skills):
    completions = await _complete(TriggerCompleter(lister, agents, skills), "@src")
    file_texts = [c.text for c in completions if str(c.display_meta_text) == "file"]
    assert file_texts == sorted(file_texts, key=lambda t: -(fuzzy_score("src", t) or 0))


async def test_slash_lists_builtin_commands(lister, agents, skills):
    completions = await _complete(TriggerCompleter(lister, agents, skills), "/qu")
    texts = [c.text for c in completions]
    assert "/quit " in texts
    assert "/queue " in texts
    quit_completion = next(c for c in completions if c.text == "/quit ")
    assert str(quit_completion.display_meta_text) != ""


async def test_slash_includes_skill_commands(lister, agents):
    skills = SkillRegistry(
        {
            "deploy": Skill(
                name="deploy",
                description="Deploy the thing",
                body="do it",
                path=Path("deploy/SKILL.md"),
                register_cmd=True,
                cmd_info="/deploy <env>",
            )
        }
    )
    completions = await _complete(TriggerCompleter(lister, agents, skills), "/dep")
    assert [c.text for c in completions] == ["/deploy "]
    assert str(completions[0].display_meta_text) == "/deploy <env>"


async def test_slash_mid_word_does_not_trigger(lister, agents, skills):
    assert await _complete(TriggerCompleter(lister, agents, skills), "hey /qu") == []


async def test_dot_lists_personas(lister, agents, skills):
    completions = await _complete(TriggerCompleter(lister, agents, skills), ".rev")
    assert [c.text for c in completions] == [".reviewer "]
    assert str(completions[0].display_meta_text) == "persona"


async def test_dot_mid_word_does_not_trigger(lister, agents, skills):
    assert await _complete(TriggerCompleter(lister, agents, skills), "hey .rev") == []


async def test_plain_text_yields_nothing(lister, agents, skills):
    assert await _complete(TriggerCompleter(lister, agents, skills), "hello") == []


def test_persona_names_are_bundled():
    names = persona_names()
    assert "reviewer" in names
    assert "architect" in names
    assert len(names) >= 16


# -- app wiring -------------------------------------------------------------------


async def test_merged_completer_wired_in_app(tmp_path, monkeypatch):
    """The app's completer covers both path tokens and triggers."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(
        "lecode.tui.input.run_proc",
        lambda *a, **k: _async(ProcResult(exit_code=0, stdout="src/app.py\n", stderr="")),
    )
    config = Config()
    config.notifications.enabled = False  # never play sounds in tests
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    app = TuiApp(
        config,
        runtime,
        FakeProvider([]),
        session,
        store,
        console=Console(record=True, file=StringIO(), width=200),
    )
    doc = Document("/qu", 3)
    texts = [c.text async for c in app._completer.get_completions_async(doc, CompleteEvent())]
    assert "/quit " in texts
    doc = Document("open src/ap", 11)
    texts = [c.text async for c in app._completer.get_completions_async(doc, CompleteEvent())]
    assert "src/app.py" in texts


async def _async(value):
    return value


async def test_pipe_smoke_with_pickers(tmp_path, monkeypatch):
    """Completion machinery attached: submit flow still works end to end."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = Config()
    config.notifications.enabled = False  # never play sounds in tests
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    out = StringIO()
    app = TuiApp(
        config,
        runtime,
        FakeProvider([{"text": "picker smoke answer"}]),
        session,
        store,
        console=Console(record=True, file=out, width=200),
    )
    with create_pipe_input() as inp:
        inp.send_text("hello\r")
        task = asyncio.ensure_future(app.run(input=inp, output=DummyOutput()))
        for _ in range(500):
            if "picker smoke answer" in out.getvalue():
                break
            await asyncio.sleep(0.01)
        inp.send_text("/quit\r")
        assert await task == 0
    assert "picker smoke answer" in out.getvalue()
