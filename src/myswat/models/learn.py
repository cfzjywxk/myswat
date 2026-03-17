"""Unified learn pipeline models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KnowledgeLocator(BaseModel):
    """Exact knowledge row locator used by worker actions."""

    model_config = ConfigDict(extra="forbid")

    category: str
    title: str
    source_type: str | None = None
    source_file: str | None = None

    @field_validator("category", "title")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("source_type", "source_file")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def cache_key(self) -> tuple[str, str, str, str]:
        return (
            self.category.casefold(),
            self.title.casefold(),
            (self.source_type or "").casefold(),
            (self.source_file or "").casefold(),
        )


class LearnRequest(BaseModel):
    """Durable learn trigger before execution."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    project_id: int
    source_kind: str
    trigger_kind: str
    source_session_id: int | None = None
    source_work_item_id: int | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "started", "completed", "failed"] = "pending"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("source_kind", "trigger_kind")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned


class LearnRun(BaseModel):
    """Worker execution record for audit and replay."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    learn_request_id: int
    worker_backend: str
    worker_model: str
    input_context_json: dict[str, Any]
    output_envelope_json: dict[str, Any] | None = None
    status: Literal["started", "completed", "failed"] = "started"
    error_text: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeAction(BaseModel):
    """Knowledge row mutation requested by the hidden worker."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["create", "update", "delete"]
    knowledge_id: int | None = None
    match: KnowledgeLocator | None = None
    category: str | None = None
    title: str | None = None
    content: str | None = None
    source_type: str | None = None
    source_file: str | None = None
    tags: list[str] = Field(default_factory=list)
    relevance_score: float = 1.0
    confidence: float = 1.0
    ttl_days: int | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("category", "title", "content", "source_type", "source_file")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            cleaned = str(item).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> "KnowledgeAction":
        if self.op == "create":
            if not self.category or not self.title or not self.content:
                raise ValueError("create actions require category, title, and content")
            return self

        if self.knowledge_id is None and self.match is None:
            raise ValueError("update/delete actions require knowledge_id or match")

        if self.op == "update" and not (
            self.model_fields_set
            & {
                "category",
                "title",
                "content",
                "source_type",
                "source_file",
                "tags",
                "relevance_score",
                "confidence",
                "ttl_days",
                "metadata_json",
            }
        ):
            raise ValueError("update actions must include at least one field to change")
        return self

    def target_locator(self) -> KnowledgeLocator | None:
        if self.category and self.title:
            return KnowledgeLocator(
                category=self.category,
                title=self.title,
                source_type=self.source_type,
                source_file=self.source_file,
            )
        return None


class RelationAction(BaseModel):
    """Explicit relation write requested by the hidden worker."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["add", "delete"]
    knowledge_id: int | None = None
    knowledge_match: KnowledgeLocator | None = None
    source_entity: str
    relation: str
    target_entity: str
    confidence: float = 1.0

    @field_validator("source_entity", "relation", "target_entity")
    @classmethod
    def validate_relation_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_target(self) -> "RelationAction":
        if self.knowledge_id is None and self.knowledge_match is None:
            raise ValueError("relation actions require knowledge_id or knowledge_match")
        return self


class IndexTerm(BaseModel):
    """Lexical index term written directly from worker hints."""

    model_config = ConfigDict(extra="forbid")

    term: str
    field: str = "content"
    weight: float = 1.0

    @field_validator("term", "field")
    @classmethod
    def validate_term_fields(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned


class IndexEntity(BaseModel):
    """Entity hint written directly from worker hints."""

    model_config = ConfigDict(extra="forbid")

    entity_name: str
    confidence: float = 1.0

    @field_validator("entity_name")
    @classmethod
    def validate_entity_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("entity_name must not be blank")
        return cleaned


class IndexHint(BaseModel):
    """Direct index writes for terms and entities."""

    model_config = ConfigDict(extra="forbid")

    knowledge_id: int | None = None
    knowledge_match: KnowledgeLocator | None = None
    terms: list[IndexTerm] = Field(default_factory=list)
    entities: list[IndexEntity] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target(self) -> "IndexHint":
        if self.knowledge_id is None and self.knowledge_match is None:
            raise ValueError("index hints require knowledge_id or knowledge_match")
        if not self.terms and not self.entities:
            raise ValueError("index hints require at least one term or entity")
        return self


class LearnActionEnvelope(BaseModel):
    """Strict JSON envelope returned by the hidden worker."""

    model_config = ConfigDict(extra="forbid")

    knowledge_actions: list[KnowledgeAction] = Field(default_factory=list)
    relation_actions: list[RelationAction] = Field(default_factory=list)
    index_hints: list[IndexHint] = Field(default_factory=list)

    @field_validator("index_hints", mode="before")
    @classmethod
    def coerce_index_hints(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value
