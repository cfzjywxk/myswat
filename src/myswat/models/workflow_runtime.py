"""Models for the server-first workflow runtime state."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator


class StageRun(BaseModel):
    id: int | None = None
    work_item_id: int
    stage_name: str
    stage_index: int = 0
    iteration: int = 1
    owner_agent_id: int | None = None
    owner_role: str | None = None
    status: str = "pending"
    summary: str | None = None
    metadata_json: dict[str, Any] | None = None
    started_at: datetime | None = None
    claimed_by_runtime_id: int | None = None
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    output_artifact_id: int | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("metadata_json", mode="before")
    @classmethod
    def parse_metadata(cls, value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            return json.loads(value)
        return value


class CoordinationEvent(BaseModel):
    id: int | None = None
    work_item_id: int
    stage_run_id: int | None = None
    stage_name: str | None = None
    event_type: str
    title: str | None = None
    summary: str
    from_agent_id: int | None = None
    from_role: str | None = None
    to_agent_id: int | None = None
    to_role: str | None = None
    payload_json: dict[str, Any] | None = None
    created_at: datetime | None = None

    @field_validator("payload_json", mode="before")
    @classmethod
    def parse_payload(cls, value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            return json.loads(value)
        return value


class RuntimeRegistration(BaseModel):
    id: int | None = None
    project_id: int
    agent_id: int | None = None
    agent_role: str | None = None
    runtime_name: str
    runtime_kind: str
    endpoint: str | None = None
    status: str = "online"
    capabilities_json: dict[str, Any] | None = None
    metadata_json: dict[str, Any] | None = None
    last_heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("capabilities_json", "metadata_json", mode="before")
    @classmethod
    def parse_json_fields(cls, value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            return json.loads(value)
        return value
