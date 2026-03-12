"""Shared AgentRunner factory helpers."""

from __future__ import annotations

import json
import os
from typing import Any

import typer

from myswat.agents.base import AgentRunner
from myswat.agents.claude_runner import ClaudeRunner
from myswat.agents.codex_runner import CodexRunner
from myswat.agents.kimi_runner import KimiRunner
from myswat.config.settings import MySwatSettings

_DEFAULT_CLAUDE_IP_CHECK_TIMEOUT_SECONDS = 10


def _parse_extra_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _nested_attr(obj: Any, *names: str) -> Any:
    current = obj
    for name in names:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _claude_required_ip(settings: MySwatSettings | None) -> str:
    value = _nested_attr(settings, "agents", "claude_required_ip")
    if isinstance(value, str) and value:
        return value
    # Env fallback is mainly for callers that do not provide a populated
    # MySwatSettings object. When settings comes from MySwatSettings(), pydantic
    # will already have folded this env var into settings.agents.
    env_value = os.environ.get("MYSWAT_AGENTS_CLAUDE_REQUIRED_IP")
    if env_value:
        return env_value
    raise typer.BadParameter(
        "Claude runner requires `claude_required_ip` to be set in "
        "`~/.myswat/config.toml` under [agents] or via "
        "`MYSWAT_AGENTS_CLAUDE_REQUIRED_IP`."
    )


def _claude_ip_check_timeout_seconds(settings: MySwatSettings | None) -> int:
    value = _nested_attr(settings, "agents", "claude_ip_check_timeout_seconds")
    if isinstance(value, int) and value > 0:
        return value
    env_value = os.environ.get("MYSWAT_AGENTS_CLAUDE_IP_CHECK_TIMEOUT_SECONDS")
    if env_value:
        try:
            parsed = int(env_value)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return _DEFAULT_CLAUDE_IP_CHECK_TIMEOUT_SECONDS


def make_runner_from_row(
    agent_row: dict,
    *,
    settings: MySwatSettings | None = None,
    workdir: str | None = None,
) -> AgentRunner:
    """Create the appropriate AgentRunner from a DB agent row."""
    backend = agent_row["cli_backend"]
    cli_path = agent_row["cli_path"]
    model = agent_row["model_name"]
    extra_flags = _parse_extra_flags(agent_row.get("cli_extra_args"))

    if backend == "codex":
        return CodexRunner(
            cli_path=cli_path,
            model=model,
            workdir=workdir,
            extra_flags=extra_flags,
        )
    if backend == "kimi":
        return KimiRunner(
            cli_path=cli_path,
            model=model,
            workdir=workdir,
            extra_flags=extra_flags,
        )
    if backend == "claude":
        return ClaudeRunner(
            cli_path=cli_path,
            model=model,
            workdir=workdir,
            extra_flags=extra_flags,
            required_ip=_claude_required_ip(settings),
            ip_check_timeout_seconds=_claude_ip_check_timeout_seconds(settings),
        )
    raise typer.BadParameter(f"Unknown CLI backend: {backend}")
