"""Session and SessionTurn models."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator


class Session(BaseModel):
    id: int | None = None
    agent_id: int
    session_uuid: str
    parent_session_id: int | None = None
    status: str = "active"  # 'active', 'completed', 'compacted', 'archived'
    purpose: str | None = None
    work_item_id: int | None = None
    token_count_est: int = 0
    compacted_through_turn_index: int = -1  # turns <= this index are "recycled"
    compacted_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SessionTurn(BaseModel):
    id: int | None = None
    session_id: int
    turn_index: int
    role: str  # 'system', 'user', 'assistant'
    content: str
    token_count_est: int = 0
    metadata_json: dict[str, Any] | None = None
    created_at: datetime | None = None

    @field_validator("metadata_json", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> dict[str, Any] | None:
        if isinstance(v, str):
            return json.loads(v)
        return v
