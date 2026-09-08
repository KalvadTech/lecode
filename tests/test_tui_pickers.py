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
from tests.fakes import FakeProvider, sample_catalog

from lecode.agent.builder import build_runtime
from lecode.config.models import Config
from lecode.context.agents import load_agents
from lecode.context.skills import Skill, SkillRegistry
from lecode.extras.proc import ProcResult
from lecode.session.storage import SessionStore
from lecode.slash.handlers import build_registry
from lecode.tui.app import TuiApp
from lecode.tui.input import FileLister
from lecode.tui.pickers import (
    TriggerCompleter,
    arg_ranked,
    command_arg_context,
    fuzzy_score,
    persona_names,
)

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


async def test_slash_prefix_filter_is_case_insensitive_and_alphabetical(lister, agents, skills):
    """Slash completion matches command-name prefixes (not fuzzy), case-insensitively."""
    completions = await _complete(TriggerCompleter(lister, agents, skills), "/MOD")
    texts = [c.text for c in completions]
    assert texts, "several built-ins share the mod prefix"
    assert texts == sorted(texts), "alphabetical order"
    assert all(text.lower().startswith("/mod") for text in texts)
    assert texts == [
        c.text for c in await _complete(TriggerCompleter(lister, agents, skills), "/mod")
    ]


async def test_slash_requires_prefix_match(lister, agents, skills):
    """No fuzzy gap matching for slash: "qt" is not a prefix of "quit"."""
    assert await _complete(TriggerCompleter(lister, agents, skills), "/qt") == []


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


# -- command-argument pickers -----------------------------------------------------


def _doc(text: str) -> Document:
    return Document(text, len(text))


def _make_arg_app(tmp_path, monkeypatch, config=None):
    """An app over the sample catalog and the real command registry."""
    monkeypatch.setenv("LECODE_CONFIG_DIR", str(tmp_path / "cfg"))
    config = config or Config()
    config.notifications.enabled = False  # never play sounds in tests
    store = SessionStore()
    session = store.create("s", tmp_path, model=config.llm.model)
    runtime = build_runtime(config, tmp_path, session=session, store=store)
    return TuiApp(
        config,
        runtime,
        FakeProvider([]),
        session,
        store,
        console=Console(record=True, file=StringIO(), width=200),
        catalog=sample_catalog(),
    )


def test_arg_context_needs_known_command_and_space():
    registry = build_registry()
    assert command_arg_context(_doc("/model"), registry) is None  # still in the command word
    assert command_arg_context(_doc("/mod "), registry) is None  # ambiguous prefix
    assert command_arg_context(_doc("/nope "), registry) is None  # unknown command
    assert command_arg_context(_doc("/copy "), registry) is None  # free-text args
    assert command_arg_context(_doc("hello "), registry) is None
    assert command_arg_context(_doc("/model one\ntwo"), registry) is None  # multiline
    command, args, partial = command_arg_context(_doc("/resume --delete ab"), registry)
    assert (command.name, args, partial) == ("resume", ["--delete"], "ab")


def test_arg_ranked_keeps_provider_order_without_query():
    rows = [("3", "c", ""), ("1", "a", ""), ("2", "b", "")]
    assert arg_ranked("", rows) == rows


def test_arg_ranked_matches_insert_or_display():
    rows = [("an/id", "Fancy Name", ""), ("zz/x", "Plain", "")]
    assert arg_ranked("fancy", rows) == [("an/id", "Fancy Name", "")]  # by display
    assert arg_ranked("an/i", rows) == [("an/id", "Fancy Name", "")]  # by insert
    assert arg_ranked("q", rows) == []


async def test_model_argument_picker_lists_catalog(tmp_path, monkeypatch):
    app = _make_arg_app(tmp_path, monkeypatch)
    completions = await _complete(app._completer, "/model ")
    texts = [c.text for c in completions]
    assert "openai/gpt-5 " in texts and "anthropic/claude-sonnet-4 " in texts
    gpt = next(c for c in completions if c.text == "openai/gpt-5 ")
    assert gpt.start_position == 0  # nothing typed yet: rows insert at the cursor
    assert "GPT-5" in str(gpt.display_meta_text)  # friendly label in the meta


async def test_model_argument_picker_filters_and_hides(tmp_path, monkeypatch):
    config = Config()
    config.ui.hidden_models = ["moonshotai/kimi-k2.6"]
    app = _make_arg_app(tmp_path, monkeypatch, config)
    texts = [c.text for c in await _complete(app._completer, "/model gpt5")]
    assert "openai/gpt-5 " in texts  # fuzzy over the id
    assert "deepseek/deepseek-v4-flash " not in texts
    texts = [c.text for c in await _complete(app._completer, "/model ")]
    assert "moonshotai/kimi-k2.6 " not in texts  # hidden_models honored


async def test_argument_picker_beats_path_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lecode.tui.input.run_proc",
        lambda *a, **k: _async(ProcResult(exit_code=0, stdout="anthropic/claude.md\n", stderr="")),
    )
    app = _make_arg_app(tmp_path, monkeypatch)
    texts = [c.text for c in await _complete(app._completer, "/model anthropic/cl")]
    assert "anthropic/claude-sonnet-4 " in texts
    assert "anthropic/claude.md" not in texts  # a model ref is not a file path
    texts = [c.text for c in await _complete(app._completer, "anthropic/claude.md")]
    assert "anthropic/claude.md" in texts  # fallback intact outside the arg context


async def test_resume_argument_picker_lists_sessions_newest_first(tmp_path, monkeypatch):
    app = _make_arg_app(tmp_path, monkeypatch)
    app.store.create("older", tmp_path, model="m")
    newer = app.store.create("newer", tmp_path, model="m")
    completions = await _complete(app._completer, "/resume ")
    displays = [str(c.display_text) for c in completions]
    assert displays.index("newer") < displays.index("older")  # newest first
    by_text = {c.text: c for c in completions}
    assert f"{newer.id} " in by_text  # canonical ids insert
    assert "current" in str(by_text[f"{app.session.id} "].display_meta_text)


async def test_resume_delete_flag_completes_session_targets(tmp_path, monkeypatch):
    app = _make_arg_app(tmp_path, monkeypatch)
    other = app.store.create("other", tmp_path, model="m")
    texts = [c.text for c in await _complete(app._completer, "/resume --delete ")]
    assert f"{other.id} " in texts  # deletion targets offered after the flag


async def test_literal_argument_pickers(tmp_path, monkeypatch):
    app = _make_arg_app(tmp_path, monkeypatch)
    assert [c.text for c in await _complete(app._completer, "/thinking ")] == [
        "none ",
        "low ",
        "medium ",
        "high ",
    ]
    assert [c.text for c in await _complete(app._completer, "/mode ")] == ["readonly ", "yolo "]
    assert [c.text for c in await _complete(app._completer, "/notifications ")] == [
        "on ",
        "off ",
    ]


async def test_nested_argument_stages(tmp_path, monkeypatch):
    app = _make_arg_app(tmp_path, monkeypatch)
    assert "model " in [c.text for c in await _complete(app._completer, "/pierre ")]
    assert "openai/gpt-5 " in [c.text for c in await _complete(app._completer, "/pierre model ")]
    assert "default " in [c.text for c in await _complete(app._completer, "/model-subagent ")]
    assert "quit " in [c.text for c in await _complete(app._completer, "/help ")]


async def test_consumed_positions_offer_nothing(tmp_path, monkeypatch):
    app = _make_arg_app(tmp_path, monkeypatch)
    assert await _complete(app._completer, "/model openai/gpt-5 ") == []
    assert await _complete(app._completer, "/help quit ") == []
    assert await _complete(app._completer, "/thinking high ") == []


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
