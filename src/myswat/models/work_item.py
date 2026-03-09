"""WorkItem, Artifact, and ReviewCycle models."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator


class WorkItem(BaseModel):
    id: int | None = None
    project_id: int
    title: str
    description: str | None = None
    item_type: str  # 'task', 'design', 'code_change', 'review', 'benchmark'
    status: str = "pending"  # 'pending', 'in_progress', 'review', 'approved', 'completed', 'blocked'
    assigned_agent_id: int | None = None
    parent_item_id: int | None = None
    priority: int = 3  # 1=critical, 5=low
    metadata_json: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("metadata_json", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> dict[str, Any] | None:
        if isinstance(v, str):
            return json.loads(v)
        return v


class Artifact(BaseModel):
    id: int | None = None
    work_item_id: int
    agent_id: int
    iteration: int
    artifact_type: str  # 'proposal', 'diff', 'patch', 'test_plan', 'design_doc'
    title: str | None = None
    content: str
    metadata_json: dict[str, Any] | None = None
    created_at: datetime | None = None


class ReviewVerdict(BaseModel):
    """Structured verdict from a reviewer — validated on parse."""
    verdict: str  # 'lgtm' or 'changes_requested'
    issues: list[str] = []
    summary: str = ""


class ReviewCycle(BaseModel):
    id: int | None = None
    work_item_id: int
    artifact_id: int | None = None
    iteration: int = 1
    proposer_agent_id: int
    reviewer_agent_id: int
    proposal_session_id: int | None = None
    review_session_id: int | None = None
    verdict: str = "pending"  # 'pending', 'changes_requested', 'lgtm'
    verdict_json: ReviewVerdict | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
