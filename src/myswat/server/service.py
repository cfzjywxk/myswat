"""Store-backed workflow and MCP tool service for MySwat."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from myswat.agents.factory import make_runner_from_row
from myswat.agents.session_manager import SessionManager as AgentSessionManager
from myswat.config.settings import MySwatSettings
from myswat.large_payloads import build_agent_context_usage_prompt
from myswat.memory.store import MemoryStore
from myswat.models.session import Session
from myswat.server.contracts import (
    ArtifactSubmission,
    ArtifactSubmissionResult,
    AssignmentEnvelope,
    ChatMessageRequest,
    ChatMessageResult,
    ChatSessionMutationRequest,
    ChatSessionMutationResult,
    ChatSessionOpenRequest,
    ChatSessionResult,
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
    ReviewCycleCancellationRequest,
    ReviewCycleLeaseRenewalRequest,
    ReviewFailureSubmission,
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
    StageRunLeaseRenewalRequest,
    StageRunResult,
    StageRunStart,
    StageRunUpdate,
    StageRunWaitRequest,
    StatusReport,
    WorkItemSnapshotRequest,
)


_SYSTEM_CONTEXT_EVENT_LIMIT = 20
_SYSTEM_CONTEXT_ARTIFACT_LIMIT = 3
_SYSTEM_CONTEXT_KNOWLEDGE_LIMIT = 5
_SNAPSHOT_ARTIFACT_LIMIT = 5
_SYSTEM_CONTEXT_SUMMARY_TEXT_LIMIT = 4_000
_SYSTEM_CONTEXT_ARTIFACT_TEXT_LIMIT = 8_192
_SYSTEM_CONTEXT_EVENT_TEXT_LIMIT = 2_000
_SYSTEM_CONTEXT_KNOWLEDGE_TEXT_LIMIT = 2_000
_SYSTEM_CONTEXT_REVIEW_HISTORY_ROUND_LIMIT = 3
_SYSTEM_CONTEXT_REVIEW_HISTORY_ISSUE_LIMIT = 3
_SYSTEM_CONTEXT_REVIEW_HISTORY_TEXT_LIMIT = 1_000
_REVIEW_HISTORY_CACHE_MAX_ENTRIES = 256
_TERMINAL_STAGE_STATUSES = frozenset({"completed", "blocked", "cancelled", "failed", "paused"})
_TERMINAL_REVIEW_STATUSES = frozenset({"completed", "blocked", "cancelled", "paused"})
_VISIBLE_REVIEW_VERDICTS = frozenset({"lgtm", "changes_requested", "failed", "paused"})


@dataclass(frozen=True)
class _ContextBundle:
    work_item: dict[str, Any]
    project: dict[str, Any]
    task_state: dict[str, Any]
    artifacts: list[dict[str, Any]]
    events: list[Any]
    knowledge: list[dict[str, Any]]
    review_history: list[dict[str, Any]]


class _CoordinationNotifier:
    """Condition-based notifier for in-process orchestrator waits."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._stage_tokens: dict[int, int] = {}
        self._review_tokens: dict[int, int] = {}
        self._stage_waiters: dict[int, int] = {}
        self._review_waiters: dict[int, int] = {}

    def snapshot_stage(self, stage_run_id: int) -> int:
        with self._condition:
            self._stage_waiters[stage_run_id] = self._stage_waiters.get(stage_run_id, 0) + 1
            return self._stage_tokens.get(stage_run_id, 0)

    def release_stage(self, stage_run_id: int) -> None:
        with self._condition:
            remaining = self._stage_waiters.get(stage_run_id, 0) - 1
            if remaining > 0:
                self._stage_waiters[stage_run_id] = remaining
                return
            self._stage_waiters.pop(stage_run_id, None)
            self._stage_tokens.pop(stage_run_id, None)

    def has_stage_waiter(self, stage_run_id: int) -> bool:
        with self._condition:
            return self._stage_waiters.get(stage_run_id, 0) > 0

    def wait_for_stage_change(
        self,
        *,
        stage_run_id: int,
        observed_token: int,
        timeout_seconds: float | None,
    ) -> int:
        def _changed() -> bool:
            return self._stage_tokens.get(stage_run_id, 0) != observed_token

        with self._condition:
            if not _changed():
                self._condition.wait_for(_changed, timeout=timeout_seconds)
            return self._stage_tokens.get(stage_run_id, 0)

    def notify_stage(self, stage_run_id: int) -> None:
        with self._condition:
            if self._stage_waiters.get(stage_run_id, 0) <= 0:
                return
            self._stage_tokens[stage_run_id] = self._stage_tokens.get(stage_run_id, 0) + 1
            self._condition.notify_all()

    def snapshot_reviews(self, cycle_ids: list[int]) -> dict[int, int]:
        with self._condition:
            unique_cycle_ids = list(dict.fromkeys(cycle_ids))
            for cycle_id in unique_cycle_ids:
                self._review_waiters[cycle_id] = self._review_waiters.get(cycle_id, 0) + 1
            return {cycle_id: self._review_tokens.get(cycle_id, 0) for cycle_id in unique_cycle_ids}

    def release_reviews(self, cycle_ids: list[int]) -> None:
        with self._condition:
            for cycle_id in dict.fromkeys(cycle_ids):
                remaining = self._review_waiters.get(cycle_id, 0) - 1
                if remaining > 0:
                    self._review_waiters[cycle_id] = remaining
                    continue
                self._review_waiters.pop(cycle_id, None)
                self._review_tokens.pop(cycle_id, None)

    def has_review_waiter(self, cycle_id: int) -> bool:
        with self._condition:
            return self._review_waiters.get(cycle_id, 0) > 0

    def wait_for_review_change(
        self,
        *,
        cycle_ids: list[int],
        observed_tokens: dict[int, int],
        timeout_seconds: float | None,
    ) -> dict[int, int]:
        cycle_ids = list(cycle_ids)

        def _changed() -> bool:
            return any(
                self._review_tokens.get(cycle_id, 0) != observed_tokens.get(cycle_id, 0)
                for cycle_id in cycle_ids
            )

        with self._condition:
            if not _changed():
                self._condition.wait_for(_changed, timeout=timeout_seconds)
            return {cycle_id: self._review_tokens.get(cycle_id, 0) for cycle_id in cycle_ids}

    def notify_review(self, cycle_id: int) -> None:
        with self._condition:
            if self._review_waiters.get(cycle_id, 0) <= 0:
                return
            self._review_tokens[cycle_id] = self._review_tokens.get(cycle_id, 0) + 1
            self._condition.notify_all()


def _clip_context_text(value: str, limit: int) -> str:
    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text

    base_marker = "\n...[truncated]...\n"
    budget = max(0, limit - len(base_marker))
    if budget <= 0:
        return text[:limit]

    head = budget // 2
    tail = budget - head
    truncated_chars = max(0, len(text) - head - tail)
    marker = f"\n...[truncated {truncated_chars} chars]...\n"
    budget = max(0, limit - len(marker))
    head = budget // 2
    tail = budget - head
    truncated_chars = max(0, len(text) - head - tail)
    marker = f"\n...[truncated {truncated_chars} chars]...\n"
    return text[:head] + marker + text[-tail:]


class MySwatToolService:
    """Canonical coordination surface for workflow orchestration and MCP tools."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._notifier = _CoordinationNotifier()
        self._review_history_cache: OrderedDict[tuple[int, str], list[dict[str, Any]]] = OrderedDict()
        self._review_history_cache_lock = threading.Lock()

    @staticmethod
    def _stage_or_none(stage: str) -> str | None:
        stage = stage.strip()
        return stage or None

    @staticmethod
    def _prefer_saved_stage(current_stage: str | None, fallback_stage: str | None) -> str | None:
        if current_stage and fallback_stage:
            if current_stage == fallback_stage or current_stage.startswith(f"{fallback_stage}_"):
                return current_stage
        return fallback_stage or current_stage

    @staticmethod
    def _sleep_interval(seconds: float) -> float:
        return max(0.05, seconds)

    @staticmethod
    def _remaining_timeout(started_at: float, timeout_seconds: float | None) -> float | None:
        if timeout_seconds is None:
            return None
        remaining = timeout_seconds - (time.monotonic() - started_at)
        return max(0.0, remaining)

    def notify_work_item_coordination_changed(self, work_item_id: int) -> None:
        self._invalidate_review_history_cache_for_work_item(work_item_id)
        for stage_run in self._store.list_stage_runs(work_item_id):
            if stage_run.id is None:
                continue
            stage_run_id = int(stage_run.id)
            status = str(stage_run.status or "")
            if status not in _TERMINAL_STAGE_STATUSES or self._notifier.has_stage_waiter(stage_run_id):
                self._notifier.notify_stage(stage_run_id)
        for review_cycle in self._store.get_review_cycles(work_item_id):
            cycle_id = int(review_cycle.get("id") or 0)
            if cycle_id <= 0:
                continue
            status = str(review_cycle.get("status") or "")
            if status not in _TERMINAL_REVIEW_STATUSES or self._notifier.has_review_waiter(cycle_id):
                self._notifier.notify_review(cycle_id)

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

    @staticmethod
    def _string_or_empty(value: object) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _require_positive_id(value: object, *, label: str) -> int:
        try:
            resolved = int(value or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} is missing a valid id") from exc
        if resolved <= 0:
            raise ValueError(f"{label} is missing a valid id")
        return resolved

    @staticmethod
    def _require_workdir(workdir: str | None) -> str:
        resolved = str(workdir or "").strip()
        if not resolved:
            raise ValueError("Chat session requires a workdir.")
        return resolved

    def _chat_session_result(
        self,
        *,
        session_row: dict[str, Any],
        agent_row: dict[str, Any],
    ) -> ChatSessionResult:
        session_id = self._require_positive_id(session_row.get("id"), label="Chat session")
        agent_id = self._require_positive_id(agent_row.get("id"), label="Chat agent")
        session_uuid = str(session_row.get("session_uuid") or "").strip()
        if not session_uuid:
            raise ValueError(f"Chat session {session_id} is missing a session UUID")
        return ChatSessionResult(
            session_id=session_id,
            session_uuid=session_uuid,
            agent_id=agent_id,
            agent_role=str(agent_row.get("role") or ""),
            display_name=str(agent_row.get("display_name") or ""),
            cli_backend=str(agent_row.get("cli_backend") or ""),
            model_name=str(agent_row.get("model_name") or ""),
        )

    def _build_chat_session_manager(
        self,
        *,
        project_row: dict[str, Any],
        agent_row: dict[str, Any],
        workdir: str,
    ) -> AgentSessionManager:
        settings = MySwatSettings()
        runner = make_runner_from_row(agent_row, settings=settings)
        runner.workdir = workdir
        return AgentSessionManager(
            store=self._store,
            runner=runner,
            agent_row=agent_row,
            project_id=int(project_row["id"]),
            settings=settings,
        )

    def _load_chat_session_rows(
        self,
        session_id: int,
        *,
        require_active: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        session_row = self._store.get_session(session_id)
        if not session_row:
            raise ValueError(f"Chat session not found: {session_id}")
        if require_active and str(session_row.get("status") or "") != "active":
            raise ValueError(f"Chat session is not active: {session_id}")

        agent_id = self._require_positive_id(session_row.get("agent_id"), label="Chat session agent")
        agent_row = self._store.get_agent_by_id(agent_id)
        if not agent_row:
            raise ValueError(f"Chat agent not found: {agent_id}")

        project_id = self._require_positive_id(agent_row.get("project_id"), label="Chat project")
        project_row = self._store.get_project(project_id)
        if not project_row:
            raise ValueError(f"Project not found: {project_id}")

        return session_row, agent_row, project_row

    @staticmethod
    def _prime_chat_session_manager(
        manager: AgentSessionManager,
        *,
        session_row: dict[str, Any],
    ) -> None:
        manager._session = Session.model_validate(session_row)
        manager._memory_revision_warned = False
        manager._restore_cli_session(int(session_row["id"]))

    def open_chat_session(self, request: ChatSessionOpenRequest) -> ChatSessionResult:
        project_row = self._store.get_project(request.project_id)
        if not project_row:
            raise ValueError(f"Project not found: {request.project_id}")
        agent_row = self._store.get_agent(request.project_id, request.agent_role)
        if not agent_row:
            raise ValueError(
                f"Agent role '{request.agent_role}' not found for project {request.project_id}"
            )
        workdir = self._require_workdir(request.workdir or project_row.get("repo_path"))
        manager = self._build_chat_session_manager(
            project_row=project_row,
            agent_row=agent_row,
            workdir=workdir,
        )
        session = manager.create_or_resume(purpose=request.purpose)
        return self._chat_session_result(
            session_row={
                "id": session.id,
                "session_uuid": session.session_uuid,
            },
            agent_row=agent_row,
        )

    def send_chat_message(self, request: ChatMessageRequest) -> ChatMessageResult:
        session_row, agent_row, project_row = self._load_chat_session_rows(request.session_id)
        workdir = self._require_workdir(request.workdir or project_row.get("repo_path"))
        manager = self._build_chat_session_manager(
            project_row=project_row,
            agent_row=agent_row,
            workdir=workdir,
        )
        self._prime_chat_session_manager(manager, session_row=session_row)
        response = manager.send(request.prompt, task_description=request.task_description)
        chat_session = self._chat_session_result(session_row=session_row, agent_row=agent_row)
        return ChatMessageResult(
            session_id=chat_session.session_id,
            session_uuid=chat_session.session_uuid,
            agent_id=chat_session.agent_id,
            agent_role=chat_session.agent_role,
            content=self._string_or_empty(response.content),
            exit_code=int(response.exit_code or 0),
            raw_stdout=self._string_or_empty(response.raw_stdout),
            raw_stderr=self._string_or_empty(response.raw_stderr),
            token_usage=response.token_usage if isinstance(response.token_usage, dict) else {},
            cancelled=bool(response.cancelled),
        )

    def reset_chat_session(
        self,
        request: ChatSessionMutationRequest,
    ) -> ChatSessionMutationResult:
        session_row, _, _ = self._load_chat_session_rows(request.session_id)
        self._store.append_turn(
            session_id=request.session_id,
            role="system",
            content="AI session reset.",
            metadata={"cli_session_reset": True},
        )
        return ChatSessionMutationResult(
            session_id=int(session_row["id"]),
            session_uuid=str(session_row.get("session_uuid") or ""),
            ok=True,
        )

    def close_chat_session(
        self,
        request: ChatSessionMutationRequest,
    ) -> ChatSessionMutationResult:
        session_row, _, _ = self._load_chat_session_rows(request.session_id, require_active=False)
        self._store.close_session(request.session_id)
        return ChatSessionMutationResult(
            session_id=int(session_row["id"]),
            session_uuid=str(session_row.get("session_uuid") or ""),
            ok=True,
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
        bundle = self._load_context_bundle(
            work_item_id=request.work_item_id,
            stage_name=request.stage_name,
            focus=request.focus,
        )

        return {
            "project": bundle.project,
            "work_item": bundle.work_item,
            "task_state": bundle.task_state,
            "recent_artifacts": bundle.artifacts[-_SNAPSHOT_ARTIFACT_LIMIT:],
            "recent_events": [event.model_dump() for event in reversed(bundle.events)],
            "knowledge": bundle.knowledge,
            "system_context": self._build_system_context(
                bundle,
                stage_name=request.stage_name,
            ),
        }

    def _load_context_bundle(
        self,
        *,
        work_item_id: int,
        stage_name: str | None,
        focus: str,
        review_stage_name: str | None = None,
        review_before_iteration: int | None = None,
    ) -> _ContextBundle:
        work_item = self._store.get_work_item(work_item_id) or {}
        project = self._store.get_project(int(work_item.get("project_id") or 0)) or {}
        task_state = self._store.get_work_item_state(work_item_id) or {}
        artifacts = self._store.list_artifacts(work_item_id)
        events = self._store.list_coordination_events(
            work_item_id,
            stage_name=stage_name,
            limit=_SYSTEM_CONTEXT_EVENT_LIMIT,
        )
        knowledge: list[dict[str, Any]] = []
        project_id = int(work_item.get("project_id") or 0)
        if project_id and focus.strip():
            try:
                knowledge = self._store.search_knowledge(
                    project_id=project_id,
                    query=focus[:400],
                    limit=_SYSTEM_CONTEXT_KNOWLEDGE_LIMIT,
                    use_vector=False,
                    use_fulltext=True,
                )
            except Exception:
                knowledge = []
        review_history: list[dict[str, Any]] = []
        if review_stage_name and review_before_iteration and review_before_iteration > 1:
            review_history = self._get_recent_review_history(
                work_item_id=work_item_id,
                stage_name=review_stage_name,
                before_iteration=review_before_iteration,
                round_limit=_SYSTEM_CONTEXT_REVIEW_HISTORY_ROUND_LIMIT,
            )
        return _ContextBundle(
            work_item=work_item,
            project=project,
            task_state=task_state,
            artifacts=artifacts,
            events=events,
            knowledge=knowledge,
            review_history=review_history,
        )

    def _build_system_context(
        self,
        bundle: _ContextBundle,
        *,
        stage_name: str | None,
    ) -> str:
        parts: list[str] = [
            "You are working through MySwat MCP. "
            "Do not assume hidden chat history. "
            "Rely on the repository state, persisted artifacts, and coordination records returned here.",
            build_agent_context_usage_prompt(heading="## Context Handling"),
        ]

        if bundle.project:
            parts.append(
                "## Project\n"
                f"- Name: {bundle.project.get('name', '')}\n"
                f"- Repo: {bundle.project.get('repo_path', '')}\n"
                f"- Current stage: {stage_name or ''}"
            )

        if bundle.task_state:
            next_todos = [str(item) for item in (bundle.task_state.get("next_todos") or [])[:5] if str(item).strip()]
            open_issues = [str(item) for item in (bundle.task_state.get("open_issues") or [])[:5] if str(item).strip()]
            work_state_lines = [
                "## Work Item State",
                f"- Recorded stage: {bundle.task_state.get('current_stage', '')}",
                "- Latest summary: "
                + _clip_context_text(
                    str(bundle.task_state.get("latest_summary") or ""),
                    _SYSTEM_CONTEXT_SUMMARY_TEXT_LIMIT,
                ),
            ]
            if next_todos:
                work_state_lines.append("- Next todos:")
                work_state_lines.extend(f"  - {item}" for item in next_todos)
            else:
                work_state_lines.append("- Next todos: []")
            if open_issues:
                work_state_lines.append("- Open issues:")
                work_state_lines.extend(f"  - {item}" for item in open_issues)
            else:
                work_state_lines.append("- Open issues: []")
            parts.append(
                "\n".join(work_state_lines)
            )

        if bundle.artifacts:
            rendered = []
            for artifact in bundle.artifacts[-_SYSTEM_CONTEXT_ARTIFACT_LIMIT:]:
                rendered.append(
                    f"- {artifact.get('artifact_type')} / {artifact.get('title') or 'untitled'} "
                    f"(iteration {artifact.get('iteration')}): "
                    f"{_clip_context_text(str(artifact.get('content') or ''), _SYSTEM_CONTEXT_ARTIFACT_TEXT_LIMIT)}"
                )
            parts.append("## Recent Artifacts\n" + "\n".join(rendered))

        if bundle.events:
            rendered = []
            for event in reversed(bundle.events):
                rendered.append(
                    f"- [{event.stage_name or '-'}] {event.event_type}: "
                    f"{_clip_context_text(str(event.summary or ''), _SYSTEM_CONTEXT_EVENT_TEXT_LIMIT)}"
                )
            parts.append("## Recent Coordination Events\n" + "\n".join(rendered))

        if bundle.review_history:
            rendered = []
            for row in bundle.review_history:
                verdict = str(row.get("verdict") or "pending")
                reviewer_role = str(row.get("reviewer_role") or "reviewer")
                iteration = int(row.get("iteration") or 0)
                verdict_json = row.get("verdict_json") or {}
                summary = str(verdict_json.get("summary") or "") if isinstance(verdict_json, dict) else ""
                issues = (
                    [str(item) for item in (verdict_json.get("issues") or [])]
                    if isinstance(verdict_json, dict)
                    else []
                )
                lines = [f"- Iteration {iteration} / {reviewer_role} / {verdict}"]
                if summary.strip():
                    lines.append(
                        "  Summary: "
                        + _clip_context_text(summary, _SYSTEM_CONTEXT_REVIEW_HISTORY_TEXT_LIMIT)
                    )
                clipped_issues = [
                    _clip_context_text(issue, _SYSTEM_CONTEXT_REVIEW_HISTORY_TEXT_LIMIT)
                    for issue in issues[:_SYSTEM_CONTEXT_REVIEW_HISTORY_ISSUE_LIMIT]
                    if issue.strip()
                ]
                if clipped_issues:
                    lines.append("  Issues:")
                    lines.extend(f"  - {issue}" for issue in clipped_issues)
                rendered.append("\n".join(lines))
            parts.append("## Prior Review Rounds For This Stage\n" + "\n".join(rendered))

        if bundle.knowledge:
            rendered = []
            for row in bundle.knowledge:
                rendered.append(
                    f"- {row.get('title', '')}: "
                    f"{_clip_context_text(str(row.get('content') or ''), _SYSTEM_CONTEXT_KNOWLEDGE_TEXT_LIMIT)}"
                )
            parts.append("## Relevant Project Knowledge\n" + "\n".join(rendered))

        return "\n\n".join(part for part in parts if part.strip())

    @staticmethod
    def _review_history_cache_key(work_item_id: int, stage_name: str | None) -> tuple[int, str] | None:
        normalized_stage = str(stage_name or "").strip()
        if work_item_id <= 0 or not normalized_stage:
            return None
        return (work_item_id, normalized_stage)

    @staticmethod
    def _review_history_row(cycle: dict[str, Any]) -> dict[str, Any] | None:
        work_item_id = int(cycle.get("work_item_id") or 0)
        stage_name = str(cycle.get("stage_name") or "").strip()
        verdict = str(cycle.get("verdict") or "pending")
        status = str(cycle.get("status") or "pending")
        if (
            work_item_id <= 0
            or not stage_name
            or status not in _TERMINAL_REVIEW_STATUSES
            or verdict not in _VISIBLE_REVIEW_VERDICTS
        ):
            return None
        normalized = dict(cycle)
        normalized["work_item_id"] = work_item_id
        normalized["stage_name"] = stage_name
        normalized["iteration"] = int(cycle.get("iteration") or 0)
        normalized["id"] = int(cycle.get("id") or 0)
        normalized["reviewer_role"] = str(cycle.get("reviewer_role") or "")
        normalized["verdict"] = verdict
        verdict_json = cycle.get("verdict_json")
        normalized["verdict_json"] = dict(verdict_json) if isinstance(verdict_json, dict) else {}
        return normalized

    @staticmethod
    def _sort_review_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                -int(row.get("iteration") or 0),
                str(row.get("reviewer_role") or ""),
                int(row.get("id") or 0),
            ),
        )

    def _invalidate_review_history_cache_for_work_item(self, work_item_id: int) -> None:
        with self._review_history_cache_lock:
            keys = [key for key in self._review_history_cache if key[0] == work_item_id]
            for key in keys:
                self._review_history_cache.pop(key, None)

    def _invalidate_review_history_cache_for_cycle_rows(self, cycles: list[dict[str, Any]]) -> None:
        work_item_ids = {int(cycle.get("work_item_id") or 0) for cycle in cycles if int(cycle.get("work_item_id") or 0) > 0}
        for work_item_id in work_item_ids:
            self._invalidate_review_history_cache_for_work_item(work_item_id)

    def _store_review_history_cache_entry(
        self,
        key: tuple[int, str],
        rows: list[dict[str, Any]],
    ) -> None:
        self._review_history_cache[key] = rows
        self._review_history_cache.move_to_end(key)
        while len(self._review_history_cache) > _REVIEW_HISTORY_CACHE_MAX_ENTRIES:
            self._review_history_cache.popitem(last=False)

    def _load_review_history_cache(self, *, work_item_id: int, stage_name: str) -> list[dict[str, Any]]:
        key = self._review_history_cache_key(work_item_id, stage_name)
        if key is None:
            return []
        with self._review_history_cache_lock:
            cached = self._review_history_cache.get(key)
            if cached is not None:
                self._review_history_cache.move_to_end(key)
                return list(cached)

        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for cycle in self._store.get_review_cycles(work_item_id):
            normalized = self._review_history_row(cycle)
            if normalized is None:
                continue
            grouped_rows.setdefault(normalized["stage_name"], []).append(normalized)
        for cached_stage_name in list(grouped_rows):
            grouped_rows[cached_stage_name] = self._sort_review_history_rows(grouped_rows[cached_stage_name])
        grouped_rows.setdefault(stage_name, [])

        with self._review_history_cache_lock:
            for cached_stage_name, rows in grouped_rows.items():
                cached_key = (work_item_id, cached_stage_name)
                if cached_key not in self._review_history_cache:
                    self._store_review_history_cache_entry(cached_key, rows)
            cached = self._review_history_cache.get(key)
            if cached is None:
                self._store_review_history_cache_entry(key, [])
                return []
            self._review_history_cache.move_to_end(key)
            return list(cached)

    def _get_recent_review_history(
        self,
        *,
        work_item_id: int,
        stage_name: str,
        before_iteration: int,
        round_limit: int,
    ) -> list[dict[str, Any]]:
        if before_iteration <= 1 or round_limit <= 0:
            return []
        rows = self._load_review_history_cache(work_item_id=work_item_id, stage_name=stage_name)
        selected_iterations: list[int] = []
        selected_set: set[int] = set()
        for row in rows:
            iteration = int(row.get("iteration") or 0)
            if iteration <= 0 or iteration >= before_iteration or iteration in selected_set:
                continue
            selected_iterations.append(iteration)
            selected_set.add(iteration)
            if len(selected_iterations) >= round_limit:
                break
        if not selected_set:
            return []
        return [dict(row) for row in rows if int(row.get("iteration") or 0) in selected_set]

    def _cache_terminal_review_cycle(self, cycle_id: int) -> None:
        cycle = self._store.get_review_cycle(cycle_id)
        if not isinstance(cycle, dict):
            return
        normalized = self._review_history_row(cycle or {})
        if normalized is None:
            self._invalidate_review_history_cache_for_cycle_rows([cycle])
            return
        key = self._review_history_cache_key(normalized["work_item_id"], normalized["stage_name"])
        if key is None:
            return
        with self._review_history_cache_lock:
            cached = self._review_history_cache.get(key)
            if cached is None:
                return
            updated_rows = [row for row in cached if int(row.get("id") or 0) != normalized["id"]]
            updated_rows.append(normalized)
            self._store_review_history_cache_entry(key, self._sort_review_history_rows(updated_rows))

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
        self._notifier.notify_stage(request.stage_run_id)

    def renew_stage_run_lease(self, request: StageRunLeaseRenewalRequest) -> None:
        renewed = self._store.renew_stage_run_lease(
            request.stage_run_id,
            runtime_registration_id=request.runtime_registration_id,
            lease_seconds=request.lease_seconds,
        )
        if not renewed:
            raise ValueError(
                "Stage run lease renewal failed: "
                f"stage_run_id={request.stage_run_id} runtime_registration_id={request.runtime_registration_id}"
            )

    def wait_for_stage_run_completion(
        self,
        request: StageRunWaitRequest,
    ) -> StageRunCompletion:
        started_at = time.monotonic()
        observed_token = self._notifier.snapshot_stage(request.stage_run_id)
        try:
            while True:
                stage_run = self._store.get_stage_run(request.stage_run_id)
                if not stage_run:
                    raise ValueError(f"Stage run not found: {request.stage_run_id}")

                if stage_run.status in _TERMINAL_STAGE_STATUSES:
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

                remaining = self._remaining_timeout(started_at, request.timeout_seconds)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for stage run {request.stage_run_id}")
                observed_token = self._notifier.wait_for_stage_change(
                    stage_run_id=request.stage_run_id,
                    observed_token=observed_token,
                    timeout_seconds=remaining,
                )
        finally:
            self._notifier.release_stage(request.stage_run_id)

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
            bundle = self._load_context_bundle(
                work_item_id=int(stage_run.work_item_id),
                stage_name=str(stage_run.stage_name),
                focus=focus,
            )
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
                    bundle,
                    stage_name=str(stage_run.stage_name),
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
            bundle = self._load_context_bundle(
                work_item_id=int(review["work_item_id"]),
                stage_name=str(review.get("stage_name") or ""),
                focus=focus,
                review_stage_name=str(review.get("stage_name") or "") or None,
                review_before_iteration=int(review.get("iteration") or 1),
            )
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
                    bundle,
                    stage_name=str(review.get("stage_name") or ""),
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
        self._notifier.notify_stage(request.stage_run_id)
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
        task_state = self._store.get_work_item_state(request.work_item_id) or {}
        saved_stage = self._stage_or_none(str(task_state.get("current_stage") or ""))
        blocked_stage = self._prefer_saved_stage(saved_stage, self._stage_or_none(request.stage_name))
        self._store.update_stage_run(
            request.stage_run_id,
            status="blocked",
            summary=request.summary,
            completed=True,
            metadata_json=request.metadata_json,
        )
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=blocked_stage,
            latest_summary=request.summary,
            updated_by_agent_id=request.agent_id,
        )
        self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="stage_blocked",
            summary=request.summary,
            from_role=request.agent_role,
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
            payload_json=request.metadata_json,
        )
        self._notifier.notify_stage(request.stage_run_id)
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
        cycle = self._store.get_review_cycle(cycle_id)
        if isinstance(cycle, dict):
            cycle_status = str(cycle.get("status") or "")
            cycle_verdict = str(cycle.get("verdict") or "")
            if cycle_status in {"paused", "cancelled"} and cycle_verdict in {"pending", "paused", "cancelled"}:
                reactivated = self._store.reactivate_review_cycle(
                    cycle_id,
                    iteration=request.iteration,
                    stage_name=request.stage,
                    proposal_session_id=request.proposal_session_id,
                    task_json=task_json or None,
                )
                if not reactivated:
                    refreshed_cycle = self._store.get_review_cycle(cycle_id) or {}
                    refreshed_status = str(refreshed_cycle.get("status") or "")
                    if refreshed_status not in {"pending", "claimed", "completed", "blocked"}:
                        raise RuntimeError(
                            "Existing review cycle could not be reactivated: "
                            f"cycle_id={cycle_id} status={refreshed_status or 'unknown'}"
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

    @staticmethod
    def _review_verdict_envelope(cycle: dict[str, Any]) -> ReviewVerdictEnvelope | None:
        verdict = str(cycle.get("verdict") or "pending")
        status = str(cycle.get("status") or "pending")
        if verdict == "pending" or status not in _TERMINAL_REVIEW_STATUSES:
            return None
        if verdict in {"paused", "cancelled"}:
            verdict_json = cycle.get("verdict_json") or {}
            summary = verdict_json.get("summary") if isinstance(verdict_json, dict) else ""
            return ReviewVerdictEnvelope(
                cycle_id=int(cycle["id"]),
                reviewer_role=str(cycle.get("reviewer_role") or ""),
                verdict="paused",
                issues=[],
                summary=str(summary or f"Review {verdict}."),
            )
        if verdict not in _VISIBLE_REVIEW_VERDICTS:
            raise ValueError(
                "Unsupported terminal review verdict: "
                f"cycle_id={int(cycle.get('id') or 0)} verdict={verdict!r} status={status!r}"
            )
        verdict_json = cycle.get("verdict_json") or {}
        issues = verdict_json.get("issues") if isinstance(verdict_json, dict) else []
        summary = verdict_json.get("summary") if isinstance(verdict_json, dict) else ""
        return ReviewVerdictEnvelope(
            cycle_id=int(cycle["id"]),
            reviewer_role=str(cycle.get("reviewer_role") or ""),
            verdict=verdict,  # type: ignore[arg-type]
            issues=[str(item) for item in (issues or [])],
            summary=str(summary or ""),
        )

    def wait_for_review_verdicts(
        self,
        request: ReviewWaitRequest,
    ) -> list[ReviewVerdictEnvelope]:
        started_at = time.monotonic()
        observed_tokens = self._notifier.snapshot_reviews(request.cycle_ids)
        try:
            while True:
                cycles = self._store.get_review_cycles_by_ids(request.cycle_ids)
                if len(cycles) == len(request.cycle_ids):
                    all_terminal = True
                    verdicts: list[ReviewVerdictEnvelope] = []
                    saw_stop_verdict = False
                    for cycle in cycles:
                        verdict = str(cycle.get("verdict") or "pending")
                        status = str(cycle.get("status") or "pending")
                        if verdict == "pending" or status not in _TERMINAL_REVIEW_STATUSES:
                            all_terminal = False
                            continue
                        envelope = self._review_verdict_envelope(cycle)
                        if envelope is None:
                            continue
                        verdicts.append(envelope)
                        if envelope.verdict in {"failed", "paused"}:
                            saw_stop_verdict = True
                    if all_terminal or (request.return_on_failed and saw_stop_verdict):
                        return verdicts

                remaining = self._remaining_timeout(started_at, request.timeout_seconds)
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Timed out waiting for review verdicts")
                observed_tokens = self._notifier.wait_for_review_change(
                    cycle_ids=request.cycle_ids,
                    observed_tokens=observed_tokens,
                    timeout_seconds=remaining,
                )
        finally:
            self._notifier.release_reviews(request.cycle_ids)

    def publish_review_verdict(self, request: ReviewVerdictSubmission) -> dict[str, Any]:
        verdict_json = {
            "verdict": request.verdict,
            "issues": request.issues,
            "summary": request.summary,
        }
        updated = self._store.update_review_verdict(
            cycle_id=request.cycle_id,
            verdict=request.verdict,
            verdict_json=verdict_json,
            review_session_id=request.review_session_id,
            status="completed",
            claimed_by_runtime_id=request.runtime_registration_id,
            only_if_active=True,
        )
        if not updated:
            return {"ok": False, "ignored": True, "cycle_id": request.cycle_id}
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage),
            latest_summary=request.summary or f"Review verdict: {request.verdict}",
            open_issues=request.issues if request.verdict == "changes_requested" else [],
            updated_by_agent_id=request.reviewer_agent_id,
        )
        event = self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="review_verdict",
            summary=request.summary or f"Review verdict: {request.verdict}",
            from_role=request.reviewer_role,
            updated_by_agent_id=request.reviewer_agent_id,
        )
        self._cache_terminal_review_cycle(request.cycle_id)
        self._notifier.notify_review(request.cycle_id)
        return event

    def fail_review_cycle(self, request: ReviewFailureSubmission) -> dict[str, Any]:
        diagnostics = dict(request.diagnostics or {})
        verdict_json = {
            "verdict": "failed",
            "issues": [],
            "summary": request.summary,
            "failure_kind": request.failure_kind,
            "attempts": request.attempts,
            "diagnostics": diagnostics,
        }
        updated = self._store.update_review_verdict(
            cycle_id=request.cycle_id,
            verdict="failed",
            verdict_json=verdict_json,
            review_session_id=None,
            status="blocked",
            claimed_by_runtime_id=request.runtime_registration_id,
            only_if_active=True,
        )
        if not updated:
            return {"ok": False, "ignored": True, "cycle_id": request.cycle_id}
        self._store.update_work_item_state(
            request.work_item_id,
            current_stage=self._stage_or_none(request.stage),
            latest_summary=request.summary,
            open_issues=[],
            updated_by_agent_id=request.reviewer_agent_id,
        )
        self._store.append_coordination_event(
            work_item_id=request.work_item_id,
            stage_name=self._stage_or_none(request.stage),
            event_type="review_failed",
            summary=request.summary,
            from_agent_id=request.reviewer_agent_id,
            from_role=request.reviewer_role,
            payload_json=verdict_json,
        )
        event = self._store.append_work_item_process_event(
            request.work_item_id,
            event_type="review_failed",
            summary=request.summary,
            from_role=request.reviewer_role,
            updated_by_agent_id=request.reviewer_agent_id,
        )
        self._cache_terminal_review_cycle(request.cycle_id)
        self._notifier.notify_review(request.cycle_id)
        return event

    def cancel_review_cycles(self, request: ReviewCycleCancellationRequest) -> dict[str, Any]:
        existing_cycles = self._store.get_review_cycles_by_ids(request.cycle_ids)
        updated = self._store.cancel_review_cycles_by_ids(
            request.cycle_ids,
            summary=request.summary,
            status=request.status,
        )
        self._invalidate_review_history_cache_for_cycle_rows(existing_cycles)
        for cycle_id in dict.fromkeys(request.cycle_ids):
            self._notifier.notify_review(cycle_id)
        return {
            "ok": True,
            "updated": updated,
            "cycle_ids": list(dict.fromkeys(request.cycle_ids)),
            "status": request.status,
        }

    def renew_review_cycle_lease(self, request: ReviewCycleLeaseRenewalRequest) -> None:
        renewed = self._store.renew_review_cycle_lease(
            request.cycle_id,
            runtime_registration_id=request.runtime_registration_id,
            lease_seconds=request.lease_seconds,
        )
        if not renewed:
            raise ValueError(
                "Review cycle lease renewal failed: "
                f"cycle_id={request.cycle_id} runtime_registration_id={request.runtime_registration_id}"
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
