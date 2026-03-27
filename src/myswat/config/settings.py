"""Central configuration for MySwat."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


WORKFLOW_REVIEW_LIMIT_DEFAULTS = {
    "design_plan_review_limit": 10,
    "dev_plan_review_limit": 10,
    "dev_code_review_limit": 10,
    "ga_plan_review_limit": 2,
    "ga_test_review_limit": 2,
}


class TiDBSettings(BaseSettings):
    host: str = ""
    port: int = 4000
    user: str = ""
    password: str = ""
    ssl_ca: str = "/etc/ssl/certs/ca-certificates.crt"
    database: str = "myswat"
    connect_timeout_seconds: int = 180

    model_config = {"env_prefix": "MYSWAT_TIDB_"}


class AgentSettings(BaseSettings):
    codex_path: str = "codex"
    kimi_path: str = "kimi"
    claude_path: str = "claude"
    claude_required_ip: str = ""
    claude_ip_check_timeout_seconds: int = 10
    architect_backend: str = "codex"
    developer_backend: str = "codex"
    qa_main_backend: str = "claude"
    qa_vice_enabled: bool = False
    qa_vice_backend: str = "kimi"
    developer_model: str = "gpt-5.4"
    architect_model: str = "gpt-5.4"
    qa_main_model: str = "claude-opus-4-6"
    qa_fallback_model: str = "gpt-5.4"
    qa_vice_model: str = "kimi-code/kimi-for-coding"
    codex_default_flags: list[str] = Field(
        default=["--full-auto", "--json"],
    )
    kimi_default_flags: list[str] = Field(
        default=["--print", "--output-format", "text", "--yolo", "--final-message-only"],
    )
    # Security tradeoff: MySwat runs non-interactive agent workflows and relies
    # on outer sandbox/proxy controls rather than Claude permission prompts.
    # Override this with an explicit Claude permission flag if needed.
    claude_default_flags: list[str] = Field(
        default=["--print", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"],
    )
    # Kill agent subprocess if no stdout/stderr output for this many seconds.
    # 0 = disabled. Default: 1200 (20 min).
    agent_stall_timeout_seconds: int = 1200

    model_config = {"env_prefix": "MYSWAT_AGENTS_"}


def get_workflow_review_limit(
    workflow_settings: object,
    field_name: str,
    default: int | None = None,
) -> int:
    resolved_default = default if default is not None else WORKFLOW_REVIEW_LIMIT_DEFAULTS.get(field_name, 10)
    value = getattr(workflow_settings, field_name, resolved_default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return resolved_default
    try:
        return int(value)
    except (TypeError, ValueError):
        return resolved_default


class WorkflowSettings(BaseSettings):
    design_plan_review_limit: int = WORKFLOW_REVIEW_LIMIT_DEFAULTS["design_plan_review_limit"]
    dev_plan_review_limit: int = WORKFLOW_REVIEW_LIMIT_DEFAULTS["dev_plan_review_limit"]
    dev_code_review_limit: int = WORKFLOW_REVIEW_LIMIT_DEFAULTS["dev_code_review_limit"]
    ga_plan_review_limit: int = WORKFLOW_REVIEW_LIMIT_DEFAULTS["ga_plan_review_limit"]
    ga_test_review_limit: int = WORKFLOW_REVIEW_LIMIT_DEFAULTS["ga_test_review_limit"]
    assignment_poll_interval_seconds: float = 1.0
    assignment_timeout_seconds: int = 0
    requirements_skills_root: str = ""

    model_config = {"env_prefix": "MYSWAT_WORKFLOW_"}


class ServerSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8765

    model_config = {"env_prefix": "MYSWAT_SERVER_"}


class EmbeddingSettings(BaseSettings):
    # "auto" = try local BGE-M3 first, fall back to TiDB built-in EMBEDDING().
    # "local" = only use local model (None if unavailable).
    # "tidb" = always use TiDB built-in EMBEDDING().
    backend: str = "auto"
    # TiDB built-in model name passed to EMBEDDING(model, text).
    tidb_model: str = "built-in"

    model_config = {"env_prefix": "MYSWAT_EMBEDDING_"}


class CompactionSettings(BaseSettings):
    # Compaction triggers when the uncompacted turn count reaches the threshold.
    threshold_turns: int = 50
    compaction_backend: str = "codex"  # which agent backend to use for compaction

    model_config = {"env_prefix": "MYSWAT_COMPACTION_", "extra": "ignore"}


class MemoryWorkerSettings(BaseSettings):
    backend: str = "codex"
    model: str = "gpt-5.4"
    role_name: str = "_memory_worker"
    async_enabled: bool = True
    trigger_mode: str = "events_only"

    model_config = {"env_prefix": "MYSWAT_MEMORY_WORKER_", "extra": "ignore"}


class MySwatSettings(BaseSettings):
    tidb: TiDBSettings = Field(default_factory=TiDBSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
    memory_worker: MemoryWorkerSettings = Field(default_factory=MemoryWorkerSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    config_path: Path = Path("~/.myswat/config.toml").expanduser()

    model_config = {"env_prefix": "MYSWAT_"}

    @model_validator(mode="before")
    @classmethod
    def load_toml(cls, values: dict[str, Any]) -> dict[str, Any]:
        config_path = values.get("config_path") or Path("~/.myswat/config.toml").expanduser()
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "rb") as f:
                toml_data = tomllib.load(f)
            # Merge TOML data under values (env vars take precedence)
            for section, section_data in toml_data.items():
                if section not in values or values[section] is None:
                    values[section] = section_data
                elif isinstance(section_data, dict) and isinstance(values[section], dict):
                    for k, v in section_data.items():
                        if k not in values[section]:
                            values[section][k] = v
        return values
