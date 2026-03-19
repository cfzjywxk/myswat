"""Store-backed workflow and MCP tool service for MySwat."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from myswat.memory.store import MemoryStore
from myswat.server.contracts import (
    ArtifactSubmission,
    ArtifactSubmissionResult,
    AssignmentEnvelope,
    ClaimNextAssignmentRequest,
    CompleteStageTaskRequest,
    CoordinationEventRecord,
    DecisionPersistenceRequest,
    DecisionPersistenceResult,
    FailStageTaskRequest,
    KnowledgeSearchRequest,
    ProjectLookupRequest,
    ProjectLookupResult,
    RecentArtifactsRequest,
    ReviewRequest,
    ReviewRequestResult,
    ReviewVerdictEnvelope,
    ReviewVerdictSubmission,
    ReviewWaitRequest,
    RuntimeHeartbeatRequest,
    RuntimeRegistrationRequest,
    RuntimeRegistrationResult,
    RuntimeStatusUpdateRequest,
    StageRunCompletion,
    StageRunResult,
    StageRunStart,
    StageRunUpdate,
    StageRunWaitRequest,
    StatusReport,
    WorkItemSnapshotRequest,
)


class MySwatToolService:
    """Canonical coordination surface for workflow orchestration and MCP tools."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    @staticmethod
    def _stage_or_none(stage: str) -> str | None:
        stage = stage.strip()
        return stage or None

    @staticmethod
    def _sleep_interval(seconds: float) -> float:
        return max(0.05, seconds)

    def resolve_project(self, request: ProjectLookupRequest) -> ProjectLookupResult:
        project = self._store.get_project_by_slug(request.project_slug)
        if not project:
            raise ValueError(f"Project not found: {request.project_slug}")
        return ProjectLookupResult(
            project_id=int(project["id"]),
            project_slug=str(project["slug"]),
            name=str(project["name"]),
            repo_path=project.get("repo_path"),
        )

    def search_knowledge(self, request: KnowledgeSearchRequest) -> list[dict[str, Any]]:
        return self._store.search_knowledge(
            project_id=request.project_id,
            query=request.query,
            agent_id=request.agent_id,
            category=request.category,
            source_type=request.source_type,
            limit=request.limit,
            use_vector=request.use_vector,
            use_fulltext=request.use_fulltext,
        )

    def get_recent_artifacts(self, request: RecentArtifactsRequest) -> list[dict[str, Any]]:
        return self._store.get_recent_artifacts_for_project(
            project_id=request.project_id,
            limit=request.limit,
        )

    def get_work_item_snapshot(self, request: WorkItemSnapshotRequest) -> dict[str, Any]:
        work_item = self._store.get_work_item(request.work_item_id) or {}
        project_id = int(work_item.get("project_id") or 0)
        project = self._store.get_project(project_id) or {}
        task_state = self._store.get_work_item_state(request.work_item_id)
        artifacts = self._store.list_artifacts(request.work_item_id)
        events = self._store.list_coordination_events(
            request.work_item_id,
            stage_name=request.stage_name,
            limit=8,
        )
        knowledge: list[dict[str, Any]] = []
        if project_id and request.focus.strip():
            try:
                knowledge = self._store.search_knowledge(
                    project_id=project_id,
                    query=request.focus[:400],
                    limit=5,
                    use_vector=False,
                    use_fulltext=True,
                )
            except Exception:
                knowledge = []

        return {
            "project": project,
            "work_item": work_item,
            "task_state": task_state,
            "recent_artifacts": artifacts[-5:],
            "recent_events": [event.model_dump() for event in reversed(events)],
            "knowledge": knowledge,
            "system_context": self._build_system_context(
                work_item_id=request.work_item_id,
                stage_name=request.stage_name,
                focus=request.focus,
            ),
        }

    def _build_system_context(
        self,
        *,
        work_item_id: int,
        stage_name: str | None,
        focus: str,
    ) -> str:
        work_item = self._store.get_work_item(work_item_id) or {}
        project = self._store.get_project(int(work_item.get("project_id") or 0)) or {}
        parts: list[str] = [
            "You are working through MySwat MCP. "
            "Do not assume hidden chat history. "
            "Rely on the repository state, persisted artifacts, and coordination records returned here.",
        ]

        if project:
            parts.append(
                "## Project\n"
                f"- Name: {project.get('name', '')}\n"
                f"- Repo: {project.get('repo_path', '')}\n"
                f"- Current stage: {stage_name or ''}"
            )

        task_state = self._store.get_work_item_state(work_item_id)
        if task_state:
            parts.append(
                "## Work Item State\n"
                f"- Recorded stage: {task_state.get('current_stage', '')}\n"
                f"- Latest summary: {str(task_state.get('latest_summary') or '')}\n"
                f"- Next todos: {(task_state.get('next_todos') or [])[:5]}\n"
                f"- Open issues: {(task_state.get('open_issues') or [])[:5]}"
            )

        artifacts = self._store.list_artifacts(work_item_id)
        if artifacts:
            rendered = []
            for artifact in artifacts[-3:]:
                rendered.append(
                    f"- {artifact.get('artifact_type')} / {artifact.get('title') or 'untitled'} "
                    f"(iteration {artifact.get('iteration')}): "
                    f"{str(artifact.get('content') or '')}"
                )
            parts.append("## Recent Artifacts\n" + "\n".join(rendered))

        events = self._store.list_coordination_events(work_item_id, stage_name=stage_name, limit=8)
        if events:
            rendered = []
            for event in reversed(events):
                rendered.append(
                    f"- [{event.stage_name or '-'}] {event.event_type}: {event.summary}"
                )
            parts.append("## Recent Coordination Events\n" + "\n".join(rendered))

        project_id = int(work_item.get("project_id") or 0)
        if project_id and focus.strip():
            try:
                knowledge = self._store.search_knowledge(
                    project_id=project_id,
                    query=focus[:400],
                    limit=5,
                    use_vector=False,
                    use_fulltext=True,
                )
            except Exception:
                knowledge = []
            if knowledge:
                rendered = []
                for row in knowledge:
                    rendered.append(
                        f"- {row.get('title', '')}: {str(row.get('content') or '')}"
                    )
                parts.append("## Relevant Project Knowledge\n" + "\n".join(rendered))

        return "\n\n".join(part for part in parts if part.strip())

    def register_runtime(self, request: RuntimeRegistrationRequest) -> RuntimeRegistrationResult:
        runtime_registration_id = self._store.register_runtime(
            project_id=request.project_id,
            runtime_name=request.runtime_name,
            runtime_kind=request.runtime_kind,
            agent_role=request.agent_role,
            agent_id=request.agent_id,
            endpoint=request.endpoint,
            capabilities_json=request.capabilities_json,
            metadata_json=request.metadata_json,
            lease_seconds=request.lease_seconds,
        )
        return RuntimeRegistrationResult(runtime_registration_id=runtime_registration_id)

    def heartbeat_runtime(self, request: RuntimeHeartbeatRequest) -> None:
        self._store.heartbeat_runtime(
            request.runtime_registration_id,
            lease_seconds=request.lease_seconds,
            metadata_json=request.metadata_json,
        )

    def update_runtime_status(self, request: RuntimeStatusUpdateRequest) -> dict[str, Any]:
        self._store.update_runtime_status(
            request.runtime_registration_id,
            status=request.status,
            metadata_json=request.metadata_json,
        )
        return {
            "runtime_registration_id": request.runtime_registration_id,
            "status": request.status,
        }

    def start_stage_run(self, request: StageRunStart) -> StageRunResult:
        metadata_json = dict(request.metadata_json or {})
        if request.task_prompt is not None:
            metadata_json["task_prompt"] = request.task_prompt
        if request.task_focus:
            metadata_json["task_focus"] = request.task_focus
        if request.artifact_type:
            metadata_json["artifact_type"] = request.artifact_type
        if request.artifact_title:
            metadata_json["artifact_title"] = request.artifact_title

        stage_run_id = self._store.create_stage_run(
            work_item_id=request.work_item_id,
            stage_name=request.stage_name,
            stage_index=request.stage_index,
            iteration=request.iteration,
            owner_agent_id=request.owner_agent_id,
            owner_role=request.owner_role,
            status=request.status,
            summary=request.summary,
            metadata_json=metadata_json or None,
        )
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=request.stage_name,
            latest_summary=request.summary,
            updated_by_agent_id=request.owner_agent_id,
        )
        event_type = "stage_queued" if request.status == "pending" else "stage_started"
        self._store.append_coordination_event(
            work_item_id=request.work_item_id,
            stage_run_id=stage_run_id,
            stage_name=request.stage_name,
            event_type=event_type,
            summary=request.summary or f"{request.stage_name} queued",
            from_agent_id=request.owner_agent_id,
            from_role=request.owner_role or "myswat",
            payload_json={"status": request.status},
        )
        return StageRunResult(stage_run_id=stage_run_id)

    def update_stage_run(self, request: StageRunUpdate) -> None:
        lease_expires_at = None
        if request.lease_seconds is not None:
            lease_expires_at = datetime.now() + timedelta(seconds=max(request.lease_seconds, 30))
        self._store.update_stage_run(
            request.stage_run_id,
            status=request.status,
            summary=request.summary,
            completed=request.completed,
            claimed_by_runtime_id=request.claimed_by_runtime_id,
            lease_expires_at=lease_expires_at,
            output_artifact_id=request.output_artifact_id,
            metadata_json=request.metadata_json,
        )

    def wait_for_stage_run_completion(
        self,
        request: StageRunWaitRequest,
    ) -> StageRunCompletion:
        started_at = time.monotonic()
        poll = self._sleep_interval(request.poll_interval_seconds)
        while True:
            stage_run = self._store.get_stage_run(request.stage_run_id)
            if not stage_run:
                raise ValueError(f"Stage run not found: {request.stage_run_id}")

            if stage_run.status in {"completed", "blocked", "cancelled", "failed"}:
                artifact_id = stage_run.output_artifact_id
                artifact_content = ""
                if artifact_id is not None:
                    artifact = self._store.get_artifact(artifact_id) or {}
                    artifact_content = str(artifact.get("content") or "")
                elif stage_run.metadata_json and stage_run.metadata_json.get("artifact_type"):
                    artifact = self._store.get_latest_artifact_by_type(
                        stage_run.work_item_id,
                        str(stage_run.metadata_json.get("artifact_type")),
                    )
                    if artifact:
                        artifact_id = int(artifact["id"])
                        artifact_content = str(artifact.get("content") or "")
                return StageRunCompletion(
                    stage_run_id=int(stage_run.id or request.stage_run_id),
                    work_item_id=int(stage_run.work_item_id),
                    stage_name=str(stage_run.stage_name),
                    status=str(stage_run.status),
                    summary=str(stage_run.summary or ""),
                    artifact_id=artifact_id,
                    artifact_content=artifact_content,
                    metadata_json=stage_run.metadata_json,
                )

            if (
                request.timeout_seconds is not None
                and time.monotonic() - started_at > request.timeout_seconds
            ):
                raise TimeoutError(f"Timed out waiting for stage run {request.stage_run_id}")
            time.sleep(poll)

    def claim_next_assignment(
        self,
        request: ClaimNextAssignmentRequest,
    ) -> AssignmentEnvelope:
        stage_run = self._store.claim_stage_run(
            project_id=request.project_id,
            owner_role=request.agent_role,
            runtime_registration_id=request.runtime_registration_id,
            lease_seconds=request.lease_seconds,
        )
        if stage_run:
            metadata = dict(stage_run.metadata_json or {})
            prompt = str(metadata.get("task_prompt") or "")
            focus = str(metadata.get("task_focus") or "")
            artifact_type = metadata.get("artifact_type")
            artifact_title = metadata.get("artifact_title")
            return AssignmentEnvelope(
                assignment_kind="stage",
                runtime_registration_id=request.runtime_registration_id,
                project_id=request.project_id,
                work_item_id=int(stage_run.work_item_id),
                stage_run_id=int(stage_run.id or 0),
                stage_name=str(stage_run.stage_name),
                agent_id=stage_run.owner_agent_id,
                agent_role=stage_run.owner_role,
                iteration=int(stage_run.iteration),
                prompt=prompt,
                focus=focus,
                system_context=self._build_system_context(
                    work_item_id=int(stage_run.work_item_id),
                    stage_name=str(stage_run.stage_name),
                    focus=focus,
                ),
                artifact_type=str(artifact_type) if artifact_type else None,
                artifact_title=str(artifact_title) if artifact_title else None,
                metadata_json=metadata,
            )

        review = self._store.claim_review_cycle(
            project_id=request.project_id,
            reviewer_role=request.agent_role,
            runtime_registration_id=request.runtime_registration_id,
            lease_seconds=request.lease_seconds,
        )
        if review:
            task_json = dict(review.get("task_json") or {})
            prompt = str(task_json.get("task_prompt") or "")
            focus = str(task_json.get("task_focus") or "")
            artifact = self._store.get_artifact(int(review.get("artifact_id") or 0)) or {}
            metadata_json = dict(task_json)
            if artifact:
                metadata_json["artifact_content"] = str(artifact.get("content") or "")
                metadata_json["artifact_title"] = artifact.get("title")
            return AssignmentEnvelope(
                assignment_kind="review",
                runtime_registration_id=request.runtime_registration_id,
                project_id=request.project_id,
                work_item_id=int(review["work_item_id"]),
                review_cycle_id=int(review["id"]),
                stage_name=str(review.get("stage_name") or ""),
                agent_id=int(review.get("reviewer_agent_id") or 0) or None,
                agent_role=str(review.get("reviewer_role") or request.agent_role),
                iteration=int(review.get("iteration") or 1),
                prompt=prompt,
                focus=focus,
                system_context=self._build_system_context(
                    work_item_id=int(review["work_item_id"]),
                    stage_name=str(review.get("stage_name") or ""),
                    focus=focus,
                ),
                artifact_id=int(review.get("artifact_id") or 0) or None,
                metadata_json=metadata_json,
            )

        return AssignmentEnvelope(
            assignment_kind="none",
            runtime_registration_id=request.runtime_registration_id,
            project_id=request.project_id,
        )

    def report_status(self, request: StatusReport) -> dict[str, Any]:
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage),
            latest_summary=request.summary,
            next_todos=request.next_todos,
            open_issues=request.open_issues,
            updated_by_agent_id=request.agent_id,
        )
        return self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="status_report",
            summary=request.summary,
            from_role=request.agent_role,
            title=request.title,
            updated_by_agent_id=request.agent_id,
        )

    def submit_artifact(self, request: ArtifactSubmission) -> ArtifactSubmissionResult:
        artifact_id = self._store.create_artifact(
            work_item_id=request.work_item_id,
            agent_id=request.agent_id,
            iteration=request.iteration,
            artifact_type=request.artifact_type,
            content=request.content,
            title=request.title,
            metadata_json=request.metadata_json,
        )
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage),
            latest_summary=request.summary,
            last_artifact_id=artifact_id,
            updated_by_agent_id=request.agent_id,
        )
        summary = request.summary or f"Submitted {request.artifact_type} artifact"
        self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="artifact_submitted",
            summary=summary,
            from_role=request.agent_role,
            title=request.title,
            updated_by_agent_id=request.agent_id,
        )
        return ArtifactSubmissionResult(artifact_id=artifact_id)

    def complete_stage_task(self, request: CompleteStageTaskRequest) -> StageRunCompletion:
        artifact_id = self.submit_artifact(
            ArtifactSubmission(
                work_item_id=request.work_item_id,
                agent_id=request.agent_id,
                agent_role=request.agent_role,
                iteration=request.iteration,
                artifact_type=request.artifact_type,
                content=request.content,
                title=request.title,
                stage=request.stage_name,
                summary=request.summary,
                metadata_json=request.metadata_json,
            )
        ).artifact_id
        self._store.update_stage_run(
            request.stage_run_id,
            status="completed",
            summary=request.summary,
            completed=True,
            output_artifact_id=artifact_id,
            metadata_json=request.metadata_json,
        )
        self._store.append_coordination_event(
            work_item_id=request.work_item_id,
            stage_run_id=request.stage_run_id,
            stage_name=request.stage_name,
            event_type="stage_completed",
            summary=request.summary or f"{request.stage_name} completed",
            from_agent_id=request.agent_id,
            from_role=request.agent_role,
            payload_json={"artifact_id": artifact_id},
        )
        return StageRunCompletion(
            stage_run_id=request.stage_run_id,
            work_item_id=request.work_item_id,
            stage_name=request.stage_name,
            status="completed",
            summary=request.summary,
            artifact_id=artifact_id,
            artifact_content=request.content,
            metadata_json=request.metadata_json,
        )

    def fail_stage_task(self, request: FailStageTaskRequest) -> StageRunCompletion:
        self._store.update_stage_run(
            request.stage_run_id,
            status="blocked",
            summary=request.summary,
            completed=True,
            metadata_json=request.metadata_json,
        )
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage_name),
            latest_summary=request.summary,
            updated_by_agent_id=request.agent_id,
        )
        self._store.append_coordination_event(
            work_item_id=request.work_item_id,
            stage_run_id=request.stage_run_id,
            stage_name=request.stage_name,
            event_type="stage_blocked",
            summary=request.summary,
            from_agent_id=request.agent_id,
            from_role=request.agent_role,
        )
        return StageRunCompletion(
            stage_run_id=request.stage_run_id,
            work_item_id=request.work_item_id,
            stage_name=request.stage_name,
            status="blocked",
            summary=request.summary,
            metadata_json=request.metadata_json,
        )

    def request_review(self, request: ReviewRequest) -> ReviewRequestResult:
        task_json = dict(request.task_json or {})
        if request.task_prompt is not None:
            task_json["task_prompt"] = request.task_prompt
        if request.task_focus:
            task_json["task_focus"] = request.task_focus
        cycle_id = self._store.create_review_cycle(
            work_item_id=request.work_item_id,
            iteration=request.iteration,
            proposer_agent_id=request.proposer_agent_id,
            reviewer_agent_id=request.reviewer_agent_id,
            reviewer_role=request.reviewer_role,
            artifact_id=request.artifact_id,
            proposal_session_id=request.proposal_session_id,
            stage_name=request.stage,
            status="pending",
            task_json=task_json or None,
        )
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage),
            updated_by_agent_id=request.proposer_agent_id,
        )
        summary = request.summary or f"Requested review for artifact {request.artifact_id}"
        self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="review_requested",
            summary=summary,
            from_role=request.proposer_role,
            to_role=request.reviewer_role,
            updated_by_agent_id=request.proposer_agent_id,
        )
        return ReviewRequestResult(cycle_id=cycle_id)

    def wait_for_review_verdicts(
        self,
        request: ReviewWaitRequest,
    ) -> list[ReviewVerdictEnvelope]:
        started_at = time.monotonic()
        poll = self._sleep_interval(request.poll_interval_seconds)
        while True:
            cycles = self._store.get_review_cycles_by_ids(request.cycle_ids)
            if len(cycles) == len(request.cycle_ids):
                terminal = True
                verdicts: list[ReviewVerdictEnvelope] = []
                for cycle in cycles:
                    verdict = str(cycle.get("verdict") or "pending")
                    status = str(cycle.get("status") or "pending")
                    if verdict == "pending" or status not in {"completed", "blocked", "cancelled"}:
                        terminal = False
                        break
                    verdict_json = cycle.get("verdict_json") or {}
                    issues = verdict_json.get("issues") if isinstance(verdict_json, dict) else []
                    summary = verdict_json.get("summary") if isinstance(verdict_json, dict) else ""
                    verdicts.append(
                        ReviewVerdictEnvelope(
                            cycle_id=int(cycle["id"]),
                            reviewer_role=str(cycle.get("reviewer_role") or ""),
                            verdict=verdict,  # type: ignore[arg-type]
                            issues=[str(item) for item in (issues or [])],
                            summary=str(summary or ""),
                        )
                    )
                if terminal:
                    return verdicts

            if (
                request.timeout_seconds is not None
                and time.monotonic() - started_at > request.timeout_seconds
            ):
                raise TimeoutError("Timed out waiting for review verdicts")
            time.sleep(poll)

    def publish_review_verdict(self, request: ReviewVerdictSubmission) -> dict[str, Any]:
        verdict_json = {
            "verdict": request.verdict,
            "issues": request.issues,
            "summary": request.summary,
        }
        self._store.update_review_verdict(
            cycle_id=request.cycle_id,
            verdict=request.verdict,
            verdict_json=verdict_json,
            review_session_id=request.review_session_id,
            status="completed",
            claimed_by_runtime_id=request.runtime_registration_id,
        )
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage),
            latest_summary=request.summary or f"Review verdict: {request.verdict}",
            open_issues=request.issues if request.verdict == "changes_requested" else [],
            updated_by_agent_id=request.reviewer_agent_id,
        )
        return self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="review_verdict",
            summary=request.summary or f"Review verdict: {request.verdict}",
            from_role=request.reviewer_role,
            updated_by_agent_id=request.reviewer_agent_id,
        )

    def append_coordination_event(self, request: CoordinationEventRecord) -> dict[str, Any]:
        event = self._store.append_coordination_event(
            work_item_id=request.work_item_id,
            stage_run_id=request.stage_run_id,
            stage_name=request.stage_name,
            event_type=request.event_type,
            title=request.title,
            summary=request.summary,
            from_agent_id=request.from_agent_id,
            from_role=request.from_role,
            to_agent_id=request.to_agent_id,
            to_role=request.to_role,
            payload_json=request.payload_json,
        )
        self._store.append_work_item_process_event(
            request.work_item_id,
            event_type=request.event_type,
            summary=request.summary,
            from_role=request.from_role,
            to_role=request.to_role,
            title=request.title,
            updated_by_agent_id=request.from_agent_id,
        )
        return event.model_dump()

    def persist_decision(
        self,
        request: DecisionPersistenceRequest,
    ) -> DecisionPersistenceResult:
        knowledge_id, action = self._store.upsert_knowledge(
            project_id=request.project_id,
            category=request.category,
            title=request.title,
            content=request.content,
            source_type=request.source_type,
            agent_id=request.agent_id,
            source_session_id=request.source_session_id,
            source_turn_ids=request.source_turn_ids,
            source_file=request.source_file,
            tags=request.tags,
            relevance_score=request.relevance_score,
            confidence=request.confidence,
            ttl_days=request.ttl_days,
            compute_embedding=request.compute_embedding,
            search_metadata_json=request.search_metadata_json,
        )
        if request.work_item_id is not None:
            self._store.update_work_item_state(
                request.work_item_id,
                current_stage=self._stage_or_none(request.stage),
                updated_by_agent_id=request.agent_id,
            )
            self._store.append_work_item_process_event(
                request.work_item_id,
                event_type="decision_recorded",
                summary=f"Recorded decision: {request.title}",
                from_role=request.agent_role,
                title=request.title,
                updated_by_agent_id=request.agent_id,
            )
        return DecisionPersistenceResult(knowledge_id=knowledge_id, action=action)
