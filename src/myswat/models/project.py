"""Project model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Project(BaseModel):
    id: int | None = None
    slug: str
    name: str
    description: str | None = None
    repo_path: str | None = None
    config_json: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
