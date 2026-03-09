"""Knowledge entry model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class KnowledgeEntry(BaseModel):
    id: int | None = None
    project_id: int
    agent_id: int | None = None
    source_session_id: int | None = None
    source_turn_ids: list[int] | None = None
    category: str  # 'decision', 'architecture', 'pattern', 'bug_fix', 'review_feedback', 'progress'
    title: str
    content: str
    embedding: list[float] | None = None
    tags: list[str] | None = None
    relevance_score: float = 1.0
    confidence: float = 1.0
    ttl_days: int | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
