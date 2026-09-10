"""The ``--setup`` onboarding wizard: linear inline prompts, no fullscreen.

Steps: import (pi / opencode / skip, only when a source config exists) →
provider (openrouter / custom base-url) → API key → default model →
notifications. Imported values prefill their questions.
Writes TOML to ``<config_dir>/config.toml`` with **0600 permissions** because
it contains the API key (documented in the README; an env var stays the
cleaner option).

Tests inject scripted answers by monkeypatching :func:`_ask_text`,
:func:`_ask_choice`, and :func:`_ask_yes_no`, or drive a ``FakeSession``.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession

from lecode.config.loader import config_dir
from lecode.providers.catalog import ModelInfo
from lecode.tui.statusline import human_tokens

#: Providers offered by the wizard: OpenRouter, or any custom
#: OpenRouter-compatible endpoint reached via a base URL.
PROVIDER_CHOICES = ("openrouter", "custom")

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


def _opencode_mcp_servers(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Translate opencode's ``mcp`` block into lecode ``[mcp.servers]`` entries.

    ``type = "local"`` becomes stdio (the command list splits into command +
    args, ``environment`` becomes ``env``); ``type = "remote"`` becomes http
    (or sse when the URL ends in ``/sse``). Entries that don't parse are
    skipped — import is best-effort.
    """
    mcp = cfg.get("mcp")
    if not isinstance(mcp, dict):
        return {}
    servers: dict[str, dict[str, Any]] = {}
    for name, entry in mcp.items():
        if not isinstance(entry, dict):
            continue
        server: dict[str, Any] = {}
        kind = entry.get("type")
        if kind == "local":
            command = entry.get("command")
            if isinstance(command, str):
                command = [command]
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(part, str) for part in command)
            ):
                continue
            server["transport"] = "stdio"
            server["command"] = command[0]
            if len(command) > 1:
                server["args"] = command[1:]
            env = entry.get("environment")
            if isinstance(env, dict) and env:
                server["env"] = {str(k): str(v) for k, v in env.items()}
        elif kind == "remote":
            url = entry.get("url")
            if not isinstance(url, str) or not url:
                continue
            server["transport"] = "sse" if url.rstrip("/").endswith("/sse") else "http"
            server["url"] = url
            headers = entry.get("headers")
            if isinstance(headers, dict) and headers:
                server["headers"] = {str(k): str(v) for k, v in headers.items()}
        else:
            continue
        if entry.get("enabled") is False:
            server["enabled"] = False
        timeout = entry.get("timeout")  # opencode uses milliseconds
        if isinstance(timeout, (int, float)) and timeout > 0:
            server["timeout_s"] = timeout / 1000
        servers[name] = server
    return servers


def import_from_opencode(home: Path) -> dict[str, Any]:
    """Map opencode's config onto wizard answers.

    ``~/.config/opencode/opencode.json(c)`` gives ``model``
    (``provider/model``), per-provider ``baseURL``, and the ``mcp`` server
    block; auth lives in ``~/.local/share/opencode/auth.json``. MCP servers
    import even when the provider/model doesn't map onto lecode.
    """
    cfg = _load_json(home / ".config" / "opencode" / "opencode.json") or _load_json(
        home / ".config" / "opencode" / "opencode.jsonc"
    )
    auth = _load_json(home / ".local" / "share" / "opencode" / "auth.json")
    answers: dict[str, Any] = {}
    mcp_servers = _opencode_mcp_servers(cfg)
    if mcp_servers:
        answers["mcp_servers"] = mcp_servers
    model_field = str(cfg.get("model") or "")
    provider, _, model = model_field.partition("/")
    if not provider:
        return answers
    if provider == "openrouter":
        answers["provider"] = "openrouter"
        if model:
            answers["model"] = model
    else:
        custom = cfg.get("provider", {}).get(provider, {})
        options = custom.get("options", {}) if isinstance(custom, dict) else {}
        base_url = options.get("baseURL", "") if isinstance(options, dict) else ""
        if base_url:
            answers["provider"] = "custom"
            answers["base_url"] = base_url
            if model:
                answers["model"] = model
    if answers.get("provider"):
        key = _auth_key(auth, provider)
        if key:
            answers["api_key"] = key
    return answers


#: Import source name → (detector paths relative to home, importer).
_IMPORT_SOURCES: dict[str, tuple[tuple[str, ...], Callable[[Path], dict[str, Any]]]] = {
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


def _import_summary(answers: dict[str, Any]) -> str:
    """One-line description of what an import found (key redacted)."""
    parts = []
    if answers.get("provider"):
        parts.append(f"provider {answers['provider']}")
    if answers.get("base_url"):
        parts.append(f"base_url {answers['base_url']}")
    if answers.get("model"):
        parts.append(f"model {answers['model']}")
    if answers.get("provider"):
        parts.append("api key found" if answers.get("api_key") else "no api key")
    servers = answers.get("mcp_servers") or {}
    if servers:
        parts.append(f"{len(servers)} mcp server{'s' if len(servers) != 1 else ''}")
    return " · ".join(parts)


#: The model menu only lists models released in the last 3 months.
_MODEL_MENU_MAX_AGE_S = 90 * 24 * 60 * 60


def _recent_models(details: dict[str, ModelInfo]) -> dict[str, ModelInfo]:
    """Models released in the last 3 months; unknown release dates are kept.

    Falls back to the full list when nothing qualifies, so a stale or
    dateless catalog never dead-ends the menu.
    """
    cutoff = time.time() - _MODEL_MENU_MAX_AGE_S
    recent = {k: v for k, v in details.items() if v.created is None or v.created >= cutoff}
    return recent or details


def _model_detail(model_id: str, details: dict[str, ModelInfo]) -> str:
    """Menu annotation for a model pick: context size and per-million pricing."""
    info = details.get(model_id)
    if info is None:
        return ""
    return (
        f"ctx {human_tokens(info.context_window)} · "
        f"${info.pricing.prompt}/M in · ${info.pricing.completion}/M out"
    )


async def _live_model_details(provider: str, base_url: str, api_key: str) -> dict[str, ModelInfo]:
    """Fetch the provider's ``/models`` for menu annotations; ``{}`` on failure.

    Runs after the provider/key questions, so the menu annotates with real,
    current data. Any failure (offline, bad key, non-listing endpoint) just
    means bare model ids.
    """
    try:
        from lecode.providers.openai_compat import ChatClient
        from lecode.providers.openrouter import (
            fetch_remote_catalog,
            openrouter_client,
        )

        if provider == "openrouter":
            client = openrouter_client(api_key or None)
        else:
            client = ChatClient(base_url, api_key=api_key or None)
        async with client:
            entries = await asyncio.wait_for(fetch_remote_catalog(client), 5.0)
        return {e.id: e for e in entries}
    except Exception:
        return {}


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
    value is still the bare choice). Accepts a number, an exact choice, or a
    unique case-insensitive substring — handy when the menu is a provider's
    full model list.
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
        matches = [c for c in choices if answer.lower() in c.lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            shown = ", ".join(matches[:10])
            more = f" +{len(matches) - 10} more" if len(matches) > 10 else ""
            print(f"error: '{answer}' is ambiguous ({shown}{more}) — be more specific")
        else:
            print(f"error: pick 1-{len(choices)}, an id from the list, or a unique substring")


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
    if answers.get("mcp_servers"):
        config["mcp"] = {"servers": answers["mcp_servers"]}
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
    imported: dict[str, Any] = {}
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
    model_default = imported.get("model")
    details = await _live_model_details(provider, base_url, api_key)
    if details:
        # The provider list, sorted, annotated with ctx size and pricing.
        recent = _recent_models(details)
        if len(recent) < len(details):
            print(
                f"showing only models from the last 3 months ({len(recent)} of "
                f"{len(details)}) — the full list is at https://openrouter.ai/models"
            )
        model_choices = sorted(recent)
        if model_default and model_default not in recent:
            model_choices.insert(0, model_default)
        model = await _ask_choice(
            session,
            f"Default model ({len(recent)} from the provider):",
            model_choices,
            describe=lambda m: _model_detail(m, details),
            default=model_default,
        )
    else:
        # Fetch failed (offline, keyless non-listing endpoint): free text.
        model = ""
        while not model:
            model = await _ask_text(
                session, "Default model (provider list unavailable)", default=model_default or ""
            )
            if not model:
                print("error: a model id is required")
    notifications = await _ask_yes_no(session, "Audio notifications?", default=True)
    answers: dict[str, Any] = {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "notifications": notifications,
    }
    if imported.get("mcp_servers"):
        answers["mcp_servers"] = imported["mcp_servers"]
    return answers


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
