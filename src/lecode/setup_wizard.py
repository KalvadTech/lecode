"""The ``--setup`` onboarding wizard: linear inline prompts, no fullscreen.

Steps: provider (openrouter / openai / custom base-url) → API key → default
model → theme → notifications → advisor opt-in. Writes TOML to
``<config_dir>/config.toml`` with **0600 permissions** because it contains
the API key (documented in the README; an env var stays the cleaner option).

Tests inject scripted answers by monkeypatching :func:`_ask_text`,
:func:`_ask_choice`, and :func:`_ask_yes_no`.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession

from lecode.config.loader import config_dir

#: Providers offered by the wizard: OpenRouter, or any custom
#: OpenRouter-compatible endpoint reached via a base URL.
PROVIDER_CHOICES = ("openrouter", "custom")

#: Catalog top picks offered as the default-model menu (all in models.json).
MODEL_PICKS = (
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-4.7",
)

#: Providers that cannot work without an API key.
_KEY_REQUIRED = ("openrouter",)


async def _ask_text(session: PromptSession, message: str, default: str = "") -> str:
    """One free-text prompt; the default applies to an empty answer."""
    suffix = f" [{default}]: " if default else ": "
    answer = (await session.prompt_async(f"{message}{suffix}")).strip()
    return answer or default


async def _ask_choice(
    session: PromptSession, message: str, choices: tuple[str, ...] | list[str]
) -> str:
    """A numbered menu; empty answer picks 1. Loops until a valid pick."""
    print(message)
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}) {choice}")
    while True:
        answer = (await session.prompt_async("Choose [1]: ")).strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in choices:
            return answer
        print(f"error: pick 1-{len(choices)} or one of {', '.join(choices)}")


async def _ask_yes_no(session: PromptSession, message: str, default: bool = True) -> bool:
    """A yes/no prompt; the default applies to an empty answer."""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = (await session.prompt_async(f"{message} {suffix} ")).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def build_config(answers: dict[str, Any]) -> dict[str, Any]:
    """Assemble the raw config dict from wizard answers."""
    llm: dict[str, Any] = {
        "provider": answers["provider"],
        "model": answers["model"],
    }
    if answers.get("api_key"):
        llm["api_key"] = answers["api_key"]
    if answers.get("base_url"):
        llm["base_url"] = answers["base_url"]
    config: dict[str, Any] = {
        "schema_version": 1,
        "llm": llm,
        "ui": {"theme": answers["theme"]},
        "notifications": {"enabled": answers["notifications"]},
    }
    if answers["advisor"]:
        config["advisor"] = {"enabled": True, "model": answers["model"]}
    return config


def write_config(raw: dict[str, Any], path: Path) -> Path:
    """Write the config as TOML with owner-only permissions (it holds a key)."""
    from lecode.config.loader import _dump_toml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_toml(raw), encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — the API key lives here
    return path


async def gather_answers(session: PromptSession) -> dict[str, Any]:
    """The linear question flow; Ctrl-C/Ctrl-D propagate to the caller."""
    provider = await _ask_choice(session, "Provider:", PROVIDER_CHOICES)
    base_url = ""
    if provider == "custom":
        base_url = await _ask_text(session, "Base URL (OpenRouter-compatible)")
        while not base_url.startswith(("http://", "https://")):
            print("error: the base URL must start with http:// or https://")
            base_url = await _ask_text(session, "Base URL (OpenRouter-compatible)")
    api_key = await _ask_text(session, "API key (stored in config.toml, chmod 0600)")
    while not api_key and provider in _KEY_REQUIRED:
        print(f"error: {provider} needs an API key")
        api_key = await _ask_text(session, "API key")
    model = await _ask_choice(session, "Default model:", MODEL_PICKS)
    from lecode.tui.themes import list_themes

    theme = await _ask_choice(session, "Theme:", list_themes())
    notifications = await _ask_yes_no(session, "Audio notifications?", default=True)
    advisor = await _ask_yes_no(
        session, "Enable the advisor (second-opinion model)?", default=False
    )
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "theme": theme,
        "notifications": notifications,
        "advisor": advisor,
    }


async def run_wizard(session: PromptSession | None = None) -> Path:
    """Run the wizard and write the config; returns the config file path."""
    prompt = session or PromptSession()
    print("lecode setup — answer a few questions to get started (Ctrl-C aborts)\n")
    answers = await gather_answers(prompt)
    path = write_config(build_config(answers), config_dir() / "config.toml")
    print(f"\nconfig written to {path} (permissions 0600 — it contains your API key)")
    return path


async def offer_first_run_setup() -> bool:
    """The first-run 'Run setup? [Y/n]' offer; True runs the wizard.

    On 'yes' the wizard runs inline; on 'no' the default config is left to
    the loader's auto-create. Ctrl-C/Ctrl-D counts as 'no'.
    """
    prompt = PromptSession()
    try:
        answer = await prompt.prompt_async("First run — no config found. Run setup? [Y/n] ")
    except (KeyboardInterrupt, EOFError):
        return False
    if answer.strip().lower() in ("n", "no"):
        return False
    try:
        await run_wizard(prompt)
    except (KeyboardInterrupt, EOFError):
        print("\nsetup cancelled — continuing with defaults")
    return True
