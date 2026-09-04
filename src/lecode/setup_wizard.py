"""The ``--setup`` onboarding wizard: linear inline prompts, no fullscreen.

Steps: import (pi / opencode / skip, only when a source config exists) →
provider (openrouter / custom base-url) → API key → default model →
notifications → advisor opt-in. Imported values prefill their questions.
Writes TOML to ``<config_dir>/config.toml`` with **0600 permissions** because
it contains the API key (documented in the README; an env var stays the
cleaner option).

Tests inject scripted answers by monkeypatching :func:`_ask_text`,
:func:`_ask_choice`, and :func:`_ask_yes_no`, or drive a ``FakeSession``.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession

from lecode.config.loader import config_dir
from lecode.providers.catalog import Catalog
from lecode.tui.statusline import human_tokens

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


# -- import from pi / opencode --------------------------------------------------


#: Auth-entry shapes both tools use: {"type": ..., "key": "..."} or a bare str.
def _auth_key(auth: dict[str, Any], provider: str) -> str:
    """Extract the API key for ``provider`` from a pi/opencode auth.json."""
    entry = auth.get(provider)
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("key"), str):
        return entry["key"]
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON/JSONC file; ``{}`` on any failure (import is best-effort)."""
    try:
        text = path.read_text(encoding="utf-8")
        # strip // comments outside strings (opencode uses .jsonc)
        out: list[str] = []
        i = 0
        in_str = False
        while i < len(text):
            c = text[i]
            if in_str:
                out.append(c)
                if c == "\\" and i + 1 < len(text):
                    out.append(text[i + 1])
                    i += 2
                    continue
                if c == '"':
                    in_str = False
                i += 1
            elif c == '"':
                in_str = True
                out.append(c)
                i += 1
            elif c == "/" and i + 1 < len(text) and text[i + 1] == "/":
                while i < len(text) and text[i] != "\n":
                    i += 1
            else:
                out.append(c)
                i += 1
        data = json.loads("".join(out), strict=False)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def import_from_pi(home: Path) -> dict[str, str]:
    """Map pi's ``~/.pi/agent/`` files onto wizard answers.

    ``settings.json`` gives defaultProvider/defaultModel, ``auth.json`` the
    API key, ``models.json`` custom-provider base URLs. Only providers lecode
    can speak to (openrouter, or an OpenAI-compatible baseUrl) are imported.
    """
    agent = home / ".pi" / "agent"
    settings = _load_json(agent / "settings.json")
    auth = _load_json(agent / "auth.json")
    models = _load_json(agent / "models.json")
    provider = str(settings.get("defaultProvider") or "")
    if not provider:
        return {}
    answers: dict[str, str] = {}
    if provider == "openrouter":
        answers["provider"] = "openrouter"
    else:
        custom = models.get("providers", {}).get(provider, {})
        base_url = custom.get("baseUrl", "") if isinstance(custom, dict) else ""
        if not base_url:
            return {}  # anthropic/openai/… — no equivalent in lecode
        answers["provider"] = "custom"
        answers["base_url"] = base_url
    key = _auth_key(auth, provider)
    if key:
        answers["api_key"] = key
    model = str(settings.get("defaultModel") or "")
    if model:
        answers["model"] = model
    return answers


def import_from_opencode(home: Path) -> dict[str, str]:
    """Map opencode's config onto wizard answers.

    ``~/.config/opencode/opencode.json(c)`` gives ``model``
    (``provider/model``) and per-provider ``baseURL``; auth lives in
    ``~/.local/share/opencode/auth.json``.
    """
    cfg = _load_json(home / ".config" / "opencode" / "opencode.json") or _load_json(
        home / ".config" / "opencode" / "opencode.jsonc"
    )
    auth = _load_json(home / ".local" / "share" / "opencode" / "auth.json")
    model_field = str(cfg.get("model") or "")
    provider, _, model = model_field.partition("/")
    if not provider:
        return {}
    answers: dict[str, str] = {}
    if provider == "openrouter":
        answers["provider"] = "openrouter"
        if model:
            answers["model"] = model
    else:
        custom = cfg.get("provider", {}).get(provider, {})
        options = custom.get("options", {}) if isinstance(custom, dict) else {}
        base_url = options.get("baseURL", "") if isinstance(options, dict) else ""
        if not base_url:
            return {}
        answers["provider"] = "custom"
        answers["base_url"] = base_url
        if model:
            answers["model"] = model
    key = _auth_key(auth, provider)
    if key:
        answers["api_key"] = key
    return answers


#: Import source name → (detector paths relative to home, importer).
_IMPORT_SOURCES: dict[str, tuple[tuple[str, ...], Callable[[Path], dict[str, str]]]] = {
    "pi": ((".pi/agent/settings.json", ".pi/agent/auth.json"), import_from_pi),
    "opencode": (
        (".config/opencode/opencode.json", ".config/opencode/opencode.jsonc"),
        import_from_opencode,
    ),
}


def detect_import_sources(home: Path) -> list[str]:
    """Import sources that have at least one config file present."""
    return [
        name
        for name, (markers, _) in _IMPORT_SOURCES.items()
        if any((home / marker).is_file() for marker in markers)
    ]


def _import_summary(answers: dict[str, str]) -> str:
    """One-line description of what an import found (key redacted)."""
    parts = [f"provider {answers['provider']}"]
    if answers.get("base_url"):
        parts.append(f"base_url {answers['base_url']}")
    if answers.get("model"):
        parts.append(f"model {answers['model']}")
    parts.append("api key found" if answers.get("api_key") else "no api key")
    return " · ".join(parts)


def _model_detail(model_id: str) -> str:
    """Menu annotation for a model pick: context size and per-million pricing."""
    try:
        info = Catalog.default().get(model_id)
    except Exception:  # catalog must never break the wizard
        return ""
    return (
        f"ctx {human_tokens(info.context_window)} · "
        f"${info.pricing.prompt}/M in · ${info.pricing.completion}/M out"
    )


async def _ask_text(session: PromptSession, message: str, default: str = "") -> str:
    """One free-text prompt; the default applies to an empty answer."""
    suffix = f" [{default}]: " if default else ": "
    answer = (await session.prompt_async(f"{message}{suffix}")).strip()
    return answer or default


async def _ask_choice(
    session: PromptSession,
    message: str,
    choices: tuple[str, ...] | list[str],
    describe: Callable[[str], str] | None = None,
    default: str | None = None,
) -> str:
    """A numbered menu; empty answer picks ``default`` (or 1). Loops until valid.

    ``describe`` adds a " — <detail>" suffix to each menu line (the return
    value is still the bare choice).
    """
    print(message)
    for i, choice in enumerate(choices, start=1):
        detail = describe(choice) if describe else ""
        suffix = f" — {detail}" if detail else ""
        marker = " (imported)" if choice == default else ""
        print(f"  {i}) {choice}{suffix}{marker}")
    fallback = default or choices[0]
    while True:
        answer = (await session.prompt_async(f"Choose [{fallback}]: ")).strip() or fallback
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


async def gather_answers(session: PromptSession, home: Path | None = None) -> dict[str, Any]:
    """The linear question flow; Ctrl-C/Ctrl-D propagate to the caller.

    Step 1 offers importing pi/opencode settings when their config files
    exist; imported values become the defaults of the later questions.
    """
    home = home or Path.home()
    imported: dict[str, str] = {}
    sources = detect_import_sources(home)
    if sources:
        choices = [*sources, "skip"]
        pick = await _ask_choice(session, "Import config from an existing agent?", choices)
        if pick != "skip":
            imported = _IMPORT_SOURCES[pick][1](home)
            if imported:
                print(f"imported from {pick}: {_import_summary(imported)}")
            else:
                print(f"nothing importable found in {pick}'s config — continuing manually")

    provider_choices: list[str] = list(PROVIDER_CHOICES)
    provider_default = imported.get("provider")
    provider = await _ask_choice(session, "Provider:", provider_choices, default=provider_default)
    base_url = imported.get("base_url", "")
    if provider == "custom":
        while True:
            base_url = await _ask_text(
                session, "Base URL (OpenRouter-compatible)", default=base_url
            )
            if base_url.startswith(("http://", "https://")):
                break
            print("error: the base URL must start with http:// or https://")
    else:
        base_url = ""
    api_key = await _ask_text(
        session,
        "API key (stored in config.toml, chmod 0600)",
        default=imported.get("api_key", ""),
    )
    while not api_key and provider in _KEY_REQUIRED:
        print(f"error: {provider} needs an API key")
        api_key = await _ask_text(session, "API key")
    model_choices: list[str] = list(MODEL_PICKS)
    model_default = imported.get("model")
    if model_default and model_default not in model_choices:
        model_choices.insert(0, model_default)
    model = await _ask_choice(
        session, "Default model:", model_choices, describe=_model_detail, default=model_default
    )
    notifications = await _ask_yes_no(session, "Audio notifications?", default=True)
    advisor = await _ask_yes_no(
        session, "Enable the advisor (second-opinion model)?", default=False
    )
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "notifications": notifications,
        "advisor": advisor,
    }


async def run_wizard(session: PromptSession | None = None, home: Path | None = None) -> Path:
    """Run the wizard and write the config; returns the config file path."""
    prompt = session or PromptSession()
    print("lecode setup — answer a few questions to get started (Ctrl-C aborts)\n")
    answers = await gather_answers(prompt, home=home)
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
