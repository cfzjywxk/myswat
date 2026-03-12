"""Agent/Role model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Agent(BaseModel):
    id: int | None = None
    project_id: int
    role: str  # 'architect', 'developer', 'qa_main', 'qa_vice'
    display_name: str
    cli_backend: str  # 'codex', 'kimi', or 'claude'
    model_name: str
    cli_path: str
    cli_extra_args: list[str] | None = None
    system_prompt: str | None = None
    created_at: datetime | None = None
