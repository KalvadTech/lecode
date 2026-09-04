"""Pydantic v2 models for the lecode configuration schema.

Every field has a sensible default so that an empty config file validates.
Sub-models use ``extra="ignore"``; the loader compares the raw parsed
mapping against the model fields to produce unknown-key warnings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Current on-disk schema version. Bump when adding a migration.
CURRENT_SCHEMA_VERSION = 1

ThinkingLevel = Literal["none", "low", "medium", "high"]
AuthPolicy = Literal["auto", "required", "none"]
PermissionMode = Literal["readonly", "yolo"]

#: Removed modes accepted from legacy configs; all coerced to ``yolo``.
LEGACY_PERMISSION_MODES = frozenset({"standard", "restrictive", "planwrite", "guarded"})


class SystemPromptConfig(BaseModel):
    """``[llm.system_prompt]`` — prompt style and overrides."""

    model_config = ConfigDict(extra="ignore")

    style: Literal["minimal", "rich"] = "minimal"
    custom: str | None = None
    persona: str | None = None


class LlmConfig(BaseModel):
    """``[llm]`` — provider selection, model, auth and network knobs."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "openrouter"
    model: str = "deepseek/deepseek-v4-flash"
    api_key: str | None = None
    base_url: str | None = None
    thinking: ThinkingLevel = "medium"
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 300.0
    auth_policy: AuthPolicy = "auto"
    tls_verify: bool = True
    system_prompt: SystemPromptConfig = Field(default_factory=SystemPromptConfig)


class CompactionConfig(BaseModel):
    """``[compaction]`` — automatic context compaction."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    buffer_tokens: int = 20000
    on_overflow: Literal["continue", "pause"] = "continue"
    mid_turn_threshold: float | None = None


class AgentConfig(BaseModel):
    """``[agent]`` — agent-loop limits."""

    model_config = ConfigDict(extra="ignore")

    max_turns: int = 500
    context_window: int = 200000
    turn_cooldown_ms: int = 0
    tool_idle_timeout_s: int = 300
    #: Model override for subagents (``/model-subagent``); None = inherit main.
    subagent_model: str | None = None


class ToolsConfig(BaseModel):
    """``[tools]`` — per-tool toggles and allowlisting."""

    model_config = ConfigDict(extra="ignore")

    enabled: dict[str, bool] = Field(default_factory=dict)
    allowlist: list[str] = Field(default_factory=list)


class UiConfig(BaseModel):
    """``[ui]`` — display preferences."""

    model_config = ConfigDict(extra="ignore")

    collapse_thinking: bool = True
    show_welcome: bool = True
    hidden_models: list[str] = Field(default_factory=list)
    no_color: bool = False


class PermissionRule(BaseModel):
    """A single allow/ask/deny rule: glob or regex matched against a tool-call target."""

    model_config = ConfigDict(extra="ignore")

    pattern: str
    kind: Literal["glob", "regex"] = "glob"


class PermissionRuleSet(BaseModel):
    """``[permissions.rules]`` — per-tool rule lists under allow/ask/deny keys."""

    model_config = ConfigDict(extra="ignore")

    allow: dict[str, list[PermissionRule]] = Field(default_factory=dict)
    ask: dict[str, list[PermissionRule]] = Field(default_factory=dict)
    deny: dict[str, list[PermissionRule]] = Field(default_factory=dict)


class PermissionsConfig(BaseModel):
    """``[permissions]`` — mode and rule tables."""

    model_config = ConfigDict(extra="ignore")

    mode: PermissionMode = "yolo"
    rules: PermissionRuleSet = Field(default_factory=PermissionRuleSet)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_legacy_mode(cls, value: object) -> object:
        # Backwards compatibility: modes removed in the two-mode system map
        # onto yolo (readonly loads unchanged).
        if isinstance(value, str) and value in LEGACY_PERMISSION_MODES:
            return "yolo"
        return value


class NotificationsConfig(BaseModel):
    """``[notifications]`` — audio notifications."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    volume: float = Field(default=0.5, ge=0.0, le=1.0)
    on_finish: bool = True
    on_error: bool = True
    on_approval: bool = True


class SignalsConfig(BaseModel):
    """``[signals]`` — status events over a Unix datagram socket."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    #: Default: ``<config_dir>/lecode.sock``.
    socket_path: str | None = None


class McpServerConfig(BaseModel):
    """One entry of ``[mcp.servers]``."""

    model_config = ConfigDict(extra="ignore")

    transport: Literal["stdio", "http"] = "stdio"
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # http
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # common
    timeout_s: float = 30.0
    enabled: bool = True


class McpConfig(BaseModel):
    """``[mcp]`` — MCP client configuration."""

    model_config = ConfigDict(extra="ignore")

    enable_exa: bool = True
    enable_context7: bool = False
    servers: dict[str, McpServerConfig] = Field(default_factory=dict)


class LspServerOverride(BaseModel):
    """One entry of ``[lsp.servers]`` — overrides the built-in registry."""

    model_config = ConfigDict(extra="ignore")

    command: list[str] = Field(default_factory=list)
    file_patterns: list[str] = Field(default_factory=list)


class LspConfig(BaseModel):
    """``[lsp]`` — LSP integration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    servers: dict[str, LspServerOverride] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """``[memory]`` — persistent Markdown memory store."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    max_bytes: int = 32768


class AdvisorConfig(BaseModel):
    """``[advisor]`` — second-opinion model."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    model: str | None = None
    max_uses: int = 5
    context_limit_kb: int = 32
    mode: Literal["model", "handoff"] = "model"


class CustomProvider(BaseModel):
    """One entry of ``[custom_providers]``."""

    model_config = ConfigDict(extra="ignore")

    base_url: str
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    auth_policy: AuthPolicy = "auto"


class Config(BaseModel):
    """Root configuration model (the whole config file)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = CURRENT_SCHEMA_VERSION
    llm: LlmConfig = Field(default_factory=LlmConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    lsp: LspConfig = Field(default_factory=LspConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    advisor: AdvisorConfig = Field(default_factory=AdvisorConfig)
    hooks: dict[str, list[str]] = Field(default_factory=dict)
    model_presets: dict[str, str] = Field(default_factory=dict)
    custom_providers: dict[str, CustomProvider] = Field(default_factory=dict)
