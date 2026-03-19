"""Typed request and response models for the MySwat MCP server."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AssignmentKind = Literal["none", "stage", "review"]
ReviewVerdictValue = Literal["lgtm", "changes_requested"]


class ProjectLookupRequest(BaseModel):
    project_slug: str


class ProjectLookupResult(BaseModel):
    project_id: int
    project_slug: str
    name: str
    repo_path: str | None = None


class KnowledgeSearchRequest(BaseModel):
    project_id: int
    query: str
    agent_id: int | None = None
    category: str | None = None
    source_type: str | None = None
    limit: int = 10
    use_vector: bool = True
    use_fulltext: bool = True


class RecentArtifactsRequest(BaseModel):
    project_id: int
    limit: int = 5


class WorkItemSnapshotRequest(BaseModel):
    work_item_id: int
    stage_name: str | None = None
    focus: str = ""


class StatusReport(BaseModel):
    work_item_id: int
    agent_id: int
    agent_role: str
    stage: str
    summary: str
    next_todos: list[str] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)
    title: str | None = None


class ArtifactSubmission(BaseModel):
    work_item_id: int
    agent_id: int
    agent_role: str
    iteration: int
    artifact_type: str
    content: str
    title: str | None = None
    stage: str = ""
    summary: str | None = None
    metadata_json: dict[str, Any] | None = None


class ArtifactSubmissionResult(BaseModel):
    artifact_id: int


class RuntimeRegistrationRequest(BaseModel):
    project_id: int
    runtime_name: str
    runtime_kind: str
    agent_role: str
    agent_id: int | None = None
    endpoint: str | None = None
    capabilities_json: dict[str, Any] | None = None
    metadata_json: dict[str, Any] | None = None
    lease_seconds: int = 300


class RuntimeRegistrationResult(BaseModel):
    runtime_registration_id: int


class RuntimeHeartbeatRequest(BaseModel):
    runtime_registration_id: int
    lease_seconds: int = 300
    metadata_json: dict[str, Any] | None = None


class RuntimeStatusUpdateRequest(BaseModel):
    runtime_registration_id: int
    status: str
    metadata_json: dict[str, Any] | None = None


class StageRunStart(BaseModel):
    work_item_id: int
    stage_name: str
    stage_index: int = 0
    iteration: int = 1
    owner_agent_id: int | None = None
    owner_role: str | None = None
    status: str = "pending"
    summary: str | None = None
    task_prompt: str | None = None
    task_focus: str = ""
    artifact_type: str | None = None
    artifact_title: str | None = None
    metadata_json: dict[str, Any] | None = None


class StageRunResult(BaseModel):
    stage_run_id: int


class StageRunUpdate(BaseModel):
    stage_run_id: int
    status: str | None = None
    summary: str | None = None
    completed: bool = False
    claimed_by_runtime_id: int | None = None
    lease_seconds: int | None = None
    output_artifact_id: int | None = None
    metadata_json: dict[str, Any] | None = None


class StageRunLeaseRenewalRequest(BaseModel):
    stage_run_id: int
    runtime_registration_id: int
    lease_seconds: int = 300


class StageRunWaitRequest(BaseModel):
    stage_run_id: int
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None


class StageRunCompletion(BaseModel):
    stage_run_id: int
    work_item_id: int
    stage_name: str
    status: str
    summary: str = ""
    artifact_id: int | None = None
    artifact_content: str = ""
    metadata_json: dict[str, Any] | None = None


class ReviewRequest(BaseModel):
    work_item_id: int
    artifact_id: int
    iteration: int
    proposer_agent_id: int
    proposer_role: str
    reviewer_agent_id: int
    reviewer_role: str
    proposal_session_id: int | None = None
    stage: str = ""
    summary: str | None = None
    task_prompt: str | None = None
    task_focus: str = ""
    task_json: dict[str, Any] | None = None


class ReviewRequestResult(BaseModel):
    cycle_id: int


class ReviewWaitRequest(BaseModel):
    cycle_ids: list[int]
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None


class ReviewVerdictEnvelope(BaseModel):
    cycle_id: int
    reviewer_role: str
    verdict: ReviewVerdictValue
    issues: list[str] = Field(default_factory=list)
    summary: str = ""


class ReviewVerdictSubmission(BaseModel):
    cycle_id: int
    work_item_id: int
    reviewer_agent_id: int
    reviewer_role: str
    verdict: ReviewVerdictValue
    issues: list[str] = Field(default_factory=list)
    summary: str = ""
    review_session_id: int | None = None
    stage: str = ""
    runtime_registration_id: int | None = None


class ReviewCycleLeaseRenewalRequest(BaseModel):
    cycle_id: int
    runtime_registration_id: int
    lease_seconds: int = 300


class ClaimNextAssignmentRequest(BaseModel):
    project_id: int
    agent_role: str
    runtime_registration_id: int
    lease_seconds: int = 300


class AssignmentEnvelope(BaseModel):
    assignment_kind: AssignmentKind = "none"
    runtime_registration_id: int
    project_id: int
    work_item_id: int | None = None
    stage_run_id: int | None = None
    review_cycle_id: int | None = None
    stage_name: str | None = None
    agent_id: int | None = None
    agent_role: str | None = None
    iteration: int | None = None
    prompt: str = ""
    focus: str = ""
    system_context: str = ""
    artifact_type: str | None = None
    artifact_title: str | None = None
    artifact_id: int | None = None
    metadata_json: dict[str, Any] | None = None


class CompleteStageTaskRequest(BaseModel):
    stage_run_id: int
    runtime_registration_id: int
    work_item_id: int
    agent_id: int
    agent_role: str
    iteration: int
    stage_name: str
    artifact_type: str
    content: str
    title: str | None = None
    summary: str = ""
    metadata_json: dict[str, Any] | None = None


class FailStageTaskRequest(BaseModel):
    stage_run_id: int
    runtime_registration_id: int
    work_item_id: int
    agent_id: int
    agent_role: str
    stage_name: str
    summary: str
    metadata_json: dict[str, Any] | None = None


class CoordinationEventRecord(BaseModel):
    work_item_id: int
    event_type: str
    summary: str
    stage_run_id: int | None = None
    stage_name: str | None = None
    title: str | None = None
    from_agent_id: int | None = None
    from_role: str | None = None
    to_agent_id: int | None = None
    to_role: str | None = None
    payload_json: dict[str, Any] | None = None


class DecisionPersistenceRequest(BaseModel):
    project_id: int
    title: str
    content: str
    work_item_id: int | None = None
    agent_id: int | None = None
    agent_role: str | None = None
    category: str = "decision"
    source_type: str = "manual"
    source_session_id: int | None = None
    source_turn_ids: list[int] | None = None
    source_file: str | None = None
    tags: list[str] | None = None
    relevance_score: float = 1.0
    confidence: float = 1.0
    ttl_days: int | None = None
    compute_embedding: bool = True
    stage: str = ""
    search_metadata_json: dict[str, Any] | None = None


class DecisionPersistenceResult(BaseModel):
    knowledge_id: int
    action: str
