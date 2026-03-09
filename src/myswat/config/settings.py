"""Central configuration for MySwat."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class TiDBSettings(BaseSettings):
    host: str = ""
    port: int = 4000
    user: str = ""
    password: str = ""
    ssl_ca: str = "/etc/ssl/certs/ca-certificates.crt"
    database: str = "myswat"

    model_config = {"env_prefix": "MYSWAT_TIDB_"}


class AgentSettings(BaseSettings):
    codex_path: str = "codex"
    kimi_path: str = "kimi"
    developer_model: str = "gpt-5.4"
    architect_model: str = "gpt-5.4"
    qa_main_model: str = "kimi-code/kimi-for-coding"
    qa_vice_model: str = "kimi-code/kimi-for-coding"
    codex_default_flags: list[str] = Field(
        default=["--full-auto", "--json"],
    )
    kimi_default_flags: list[str] = Field(
        default=["--print", "--output-format", "text", "--yolo", "--final-message-only"],
    )

    model_config = {"env_prefix": "MYSWAT_AGENTS_"}


class WorkflowSettings(BaseSettings):
    max_review_iterations: int = 5

    model_config = {"env_prefix": "MYSWAT_WORKFLOW_"}


class CompactionSettings(BaseSettings):
    # Compaction triggers when EITHER threshold is exceeded.
    # Codex context is ~400k tokens, kimi ~200k tokens.
    # We compact well before the CLI's own limit so the distilled knowledge
    # is available for the next round, while leaving headroom for the current
    # prompt + system context (~8k tokens from retriever).
    threshold_turns: int = 60       # ~30 user+assistant exchanges
    threshold_tokens: int = 100000  # ~100k tokens accumulated in one session
    compaction_backend: str = "codex"  # which agent backend to use for compaction

    model_config = {"env_prefix": "MYSWAT_COMPACTION_"}


class MySwatSettings(BaseSettings):
    tidb: TiDBSettings = Field(default_factory=TiDBSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)
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
