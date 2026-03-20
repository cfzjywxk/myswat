"""Tests for the store-backed MCP service layer."""

from __future__ import annotations

import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import myswat.server.service as service_module
from myswat.large_payloads import build_agent_context_usage_prompt
from myswat.memory.store import MemoryStore
from myswat.models.workflow_runtime import StageRun
from myswat.server.contracts import (
    ClaimNextAssignmentRequest,
    CompleteStageTaskRequest,
    CoordinationEventRecord,
    DecisionPersistenceRequest,
    KnowledgeSearchRequest,
    RecentArtifactsRequest,
    ReviewCycleCancellationRequest,
    ReviewCycleLeaseRenewalRequest,
    ReviewFailureSubmission,
    ReviewRequest,
    ReviewVerdictSubmission,
    ReviewWaitRequest,
    RuntimeRegistrationRequest,
    RuntimeStatusUpdateRequest,
    StageRunLeaseRenewalRequest,
    StageRunStart,
    StageRunUpdate,
    StatusReport,
    WorkItemSnapshotRequest,
)
from myswat.server.service import MySwatToolService


def test_search_knowledge_delegates_to_store():
    store = Mock(spec=MemoryStore)
    store.search_knowledge.return_value = [{"title": "txn"}]
    service = MySwatToolService(store)

    result = service.search_knowledge(KnowledgeSearchRequest(project_id=7, query="txn"))

    assert result == [{"title": "txn"}]
    store.search_knowledge.assert_called_once_with(
        project_id=7,
        query="txn",
        agent_id=None,
        category=None,
        source_type=None,
        limit=10,
        use_vector=True,
        use_fulltext=True,
    )


def test_get_recent_artifacts_delegates_to_store():
    store = Mock(spec=MemoryStore)
    store.get_recent_artifacts_for_project.return_value = [{"artifact_type": "patch"}]
    service = MySwatToolService(store)

    result = service.get_recent_artifacts(RecentArtifactsRequest(project_id=9, limit=2))

    assert result == [{"artifact_type": "patch"}]
    store.get_recent_artifacts_for_project.assert_called_once_with(project_id=9, limit=2)


def test_get_work_item_snapshot_reuses_single_context_bundle_fetch():
    store = Mock(spec=MemoryStore)
    store.get_work_item.return_value = {"id": 17, "project_id": 9}
    store.get_project.return_value = {"id": 9, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {"current_stage": "design"}
    store.list_artifacts.return_value = [{"artifact_type": "design_doc", "content": "..."}, {"artifact_type": "plan", "content": "Phase 1"}]
    store.list_coordination_events.return_value = []
    store.search_knowledge.return_value = [{"title": "decision", "content": "Use staged review."}]
    service = MySwatToolService(store)

    result = service.get_work_item_snapshot(
        WorkItemSnapshotRequest(
            work_item_id=17,
            stage_name="design",
            focus="staged review",
        )
    )

    assert result["work_item"]["id"] == 17
    assert store.get_work_item.call_count == 1
    assert store.get_project.call_count == 1
    assert store.get_work_item_state.call_count == 1
    assert store.list_artifacts.call_count == 1
    assert store.list_coordination_events.call_count == 1
    assert store.search_knowledge.call_count == 1


def test_claim_next_assignment_clips_large_context_entries_but_preserves_tail_markers(monkeypatch):
    monkeypatch.setattr(service_module, "_SYSTEM_CONTEXT_SUMMARY_TEXT_LIMIT", 128)
    monkeypatch.setattr(service_module, "_SYSTEM_CONTEXT_ARTIFACT_TEXT_LIMIT", 128)
    monkeypatch.setattr(service_module, "_SYSTEM_CONTEXT_EVENT_TEXT_LIMIT", 128)
    monkeypatch.setattr(service_module, "_SYSTEM_CONTEXT_KNOWLEDGE_TEXT_LIMIT", 128)

    store = Mock(spec=MemoryStore)
    store.claim_stage_run.return_value = StageRun(
        id=56,
        work_item_id=18,
        stage_name="phase_1",
        stage_index=40,
        iteration=1,
        owner_agent_id=3,
        owner_role="developer",
        status="claimed",
        summary="Implement phase 1",
        metadata_json={
            "task_prompt": "Implement the Rust fibonacci generator",
            "task_focus": "fibonacci generator",
            "artifact_type": "phase_result",
            "artifact_title": "Phase 1 result",
        },
        started_at=datetime.now(),
    )
    store.get_work_item.return_value = {"id": 18, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {
        "current_stage": "phase_1",
        "latest_summary": ("summary-body " * 20) + "SUMMARY-TAIL-MARKER",
        "next_todos": ["finish implementation"],
        "open_issues": ["none"],
    }
    store.list_artifacts.return_value = [
        {
            "artifact_type": "design_doc",
            "title": "Technical design",
            "iteration": 1,
            "content": ("artifact-body " * 20) + "ARTIFACT-TAIL-MARKER",
        }
    ]
    store.list_coordination_events.return_value = [
        CoordinationEventRecord(
            work_item_id=18,
            stage_name="phase_1",
            event_type="review_feedback",
            summary=("event-body " * 20) + "EVENT-TAIL-MARKER",
        )
    ]
    store.search_knowledge.return_value = [
        {
            "title": "design-note",
            "content": ("knowledge-body " * 20) + "KNOWLEDGE-TAIL-MARKER",
        }
    ]
    service = MySwatToolService(store)

    result = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="developer",
            runtime_registration_id=92,
        )
    )

    assert "truncated" in result.system_context
    assert "SUMMARY-TAIL-MARKER" in result.system_context
    assert "ARTIFACT-TAIL-MARKER" in result.system_context
    assert "EVENT-TAIL-MARKER" in result.system_context
    assert "KNOWLEDGE-TAIL-MARKER" in result.system_context


def test_wait_for_stage_run_completion_releases_notifier_state():
    store = Mock(spec=MemoryStore)
    pending = SimpleNamespace(
        id=55,
        work_item_id=17,
        stage_name="plan",
        status="claimed",
        output_artifact_id=None,
        metadata_json=None,
        summary="pending",
    )
    completed = SimpleNamespace(
        id=55,
        work_item_id=17,
        stage_name="plan",
        status="completed",
        output_artifact_id=None,
        metadata_json=None,
        summary="done",
    )
    store.get_stage_run.side_effect = [pending, completed]
    service = MySwatToolService(store)
    result_holder: dict[str, object] = {}

    def _wait() -> None:
        result_holder["result"] = service.wait_for_stage_run_completion(
            service_module.StageRunWaitRequest(stage_run_id=55, timeout_seconds=1)
        )

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()
    assert service._notifier.has_stage_waiter(55) is True
    service._notifier.notify_stage(55)
    thread.join(timeout=1)

    assert thread.is_alive() is False
    assert result_holder["result"].status == "completed"
    assert service._notifier.has_stage_waiter(55) is False
    assert service._notifier._stage_tokens == {}


def test_notify_work_item_coordination_changed_skips_terminal_items_without_waiters():
    store = Mock(spec=MemoryStore)
    service = MySwatToolService(store)
    waiting_stage = StageRun(
        id=11,
        work_item_id=17,
        stage_name="phase_1",
        stage_index=10,
        iteration=1,
        owner_agent_id=3,
        owner_role="developer",
        status="cancelled",
        summary="stopped",
        started_at=datetime.now(),
    )
    completed_stage = StageRun(
        id=12,
        work_item_id=17,
        stage_name="plan",
        stage_index=20,
        iteration=1,
        owner_agent_id=3,
        owner_role="developer",
        status="completed",
        summary="done",
        started_at=datetime.now(),
    )
    active_stage = StageRun(
        id=13,
        work_item_id=17,
        stage_name="design",
        stage_index=5,
        iteration=1,
        owner_agent_id=3,
        owner_role="architect",
        status="pending",
        summary="queued",
        started_at=datetime.now(),
    )
    store.list_stage_runs.return_value = [waiting_stage, completed_stage, active_stage]
    store.get_review_cycles.return_value = [
        {"id": 21, "status": "cancelled"},
        {"id": 22, "status": "completed"},
        {"id": 23, "status": "pending"},
    ]

    service._notifier.snapshot_stage(11)
    service._notifier.snapshot_reviews([21])
    stage_calls: list[int] = []
    review_calls: list[int] = []
    service._notifier.notify_stage = lambda stage_run_id: stage_calls.append(stage_run_id)
    service._notifier.notify_review = lambda cycle_id: review_calls.append(cycle_id)

    service.notify_work_item_coordination_changed(17)

    assert stage_calls == [11, 13]
    assert review_calls == [21, 23]
    service._notifier.release_stage(11)
    service._notifier.release_reviews([21])


def test_register_runtime_creates_runtime_registration():
    store = Mock(spec=MemoryStore)
    store.register_runtime.return_value = 55
    service = MySwatToolService(store)

    result = service.register_runtime(
        RuntimeRegistrationRequest(
            project_id=11,
            runtime_name="codex-daemon",
            runtime_kind="mcp",
            agent_role="developer",
            agent_id=3,
        )
    )

    assert result.runtime_registration_id == 55
    store.register_runtime.assert_called_once()


def test_update_runtime_status_delegates_to_store():
    store = Mock(spec=MemoryStore)
    service = MySwatToolService(store)

    result = service.update_runtime_status(
        RuntimeStatusUpdateRequest(
            runtime_registration_id=55,
            status="offline",
            metadata_json={"stop_reason": "idle_exit"},
        )
    )

    assert result == {"runtime_registration_id": 55, "status": "offline"}
    store.update_runtime_status.assert_called_once_with(
        55,
        status="offline",
        metadata_json={"stop_reason": "idle_exit"},
    )


def test_start_stage_run_creates_pending_stage_and_updates_work_item_state():
    store = Mock(spec=MemoryStore)
    store.create_stage_run.return_value = 55
    service = MySwatToolService(store)

    result = service.start_stage_run(
        StageRunStart(
            work_item_id=11,
            stage_name="design",
            stage_index=10,
            iteration=1,
            owner_agent_id=3,
            owner_role="architect",
            status="pending",
            summary="Produce technical design",
            task_prompt="Write a design",
            task_focus="coordination",
            artifact_type="design_doc",
            artifact_title="Technical design",
        )
    )

    assert result.stage_run_id == 55
    store.create_stage_run.assert_called_once_with(
        work_item_id=11,
        stage_name="design",
        stage_index=10,
        iteration=1,
        owner_agent_id=3,
        owner_role="architect",
        status="pending",
        summary="Produce technical design",
        metadata_json={
            "task_prompt": "Write a design",
            "task_focus": "coordination",
            "artifact_type": "design_doc",
            "artifact_title": "Technical design",
        },
    )
    store.update_work_item_state.assert_called_once_with(
        11,
        current_stage="design",
        latest_summary="Produce technical design",
        updated_by_agent_id=3,
    )


def test_update_stage_run_delegates_to_store():
    store = Mock(spec=MemoryStore)
    service = MySwatToolService(store)

    service.update_stage_run(
        StageRunUpdate(
            stage_run_id=55,
            status="completed",
            summary="done",
            completed=True,
            output_artifact_id=999,
            metadata_json={"iterations": 2},
        )
    )

    store.update_stage_run.assert_called_once_with(
        55,
        status="completed",
        summary="done",
        completed=True,
        claimed_by_runtime_id=None,
        lease_expires_at=None,
        output_artifact_id=999,
        metadata_json={"iterations": 2},
    )


def test_renew_stage_run_lease_delegates_to_store():
    store = Mock(spec=MemoryStore)
    store.renew_stage_run_lease.return_value = True
    service = MySwatToolService(store)

    service.renew_stage_run_lease(
        StageRunLeaseRenewalRequest(
            stage_run_id=55,
            runtime_registration_id=91,
            lease_seconds=120,
        )
    )

    store.renew_stage_run_lease.assert_called_once_with(
        55,
        runtime_registration_id=91,
        lease_seconds=120,
    )


def test_claim_next_assignment_returns_stage_assignment_bundle():
    store = Mock(spec=MemoryStore)
    store.claim_stage_run.return_value = StageRun(
        id=55,
        work_item_id=17,
        stage_name="plan",
        stage_index=30,
        iteration=1,
        owner_agent_id=3,
        owner_role="developer",
        status="claimed",
        summary="Draft the plan",
        metadata_json={
            "task_prompt": "Write the implementation plan",
            "task_focus": "workflow",
            "artifact_type": "implementation_plan",
            "artifact_title": "Implementation plan",
        },
        started_at=datetime.now(),
    )
    store.get_work_item.return_value = {"id": 17, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {}
    store.list_artifacts.return_value = []
    store.list_coordination_events.return_value = []
    store.search_knowledge.return_value = []
    service = MySwatToolService(store)

    result = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="developer",
            runtime_registration_id=91,
        )
    )

    assert result.assignment_kind == "stage"
    assert result.stage_run_id == 55
    assert result.prompt == "Write the implementation plan"
    assert result.artifact_type == "implementation_plan"
    assert build_agent_context_usage_prompt(heading="## Context Handling") in result.system_context


def test_claim_next_assignment_preserves_large_context_details_for_worker_externalization():
    store = Mock(spec=MemoryStore)
    long_summary = ("summary-body " * 260) + "SUMMARY-TAIL-MARKER"
    long_artifact = ("artifact-body " * 260) + "ARTIFACT-TAIL-MARKER"
    long_event = ("event-body " * 260) + "EVENT-TAIL-MARKER"
    long_knowledge = ("knowledge-body " * 260) + "KNOWLEDGE-TAIL-MARKER"
    store.claim_stage_run.return_value = StageRun(
        id=56,
        work_item_id=18,
        stage_name="phase_1",
        stage_index=40,
        iteration=1,
        owner_agent_id=3,
        owner_role="developer",
        status="claimed",
        summary="Implement phase 1",
        metadata_json={
            "task_prompt": "Implement the Rust fibonacci generator",
            "task_focus": "fibonacci generator",
            "artifact_type": "phase_result",
            "artifact_title": "Phase 1 result",
        },
        started_at=datetime.now(),
    )
    store.get_work_item.return_value = {"id": 18, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {
        "current_stage": "phase_1",
        "latest_summary": long_summary,
        "next_todos": ["finish implementation"],
        "open_issues": ["none"],
    }
    store.list_artifacts.return_value = [
        {
            "artifact_type": "design_doc",
            "title": "Technical design",
            "iteration": 1,
            "content": long_artifact,
        }
    ]
    store.list_coordination_events.return_value = [
        CoordinationEventRecord(
            work_item_id=18,
            stage_name="phase_1",
            event_type="review_feedback",
            summary=long_event,
        )
    ]
    store.search_knowledge.return_value = [{"title": "design-note", "content": long_knowledge}]
    service = MySwatToolService(store)

    result = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="developer",
            runtime_registration_id=92,
        )
    )

    assert result.assignment_kind == "stage"
    assert "SUMMARY-TAIL-MARKER" in result.system_context
    assert "ARTIFACT-TAIL-MARKER" in result.system_context
    assert "EVENT-TAIL-MARKER" in result.system_context
    assert "KNOWLEDGE-TAIL-MARKER" in result.system_context


def test_claim_next_assignment_review_includes_prior_review_history(monkeypatch):
    monkeypatch.setattr(service_module, "_SYSTEM_CONTEXT_REVIEW_HISTORY_TEXT_LIMIT", 96)

    store = Mock(spec=MemoryStore)
    store.claim_stage_run.return_value = None
    store.claim_review_cycle.return_value = {
        "id": 88,
        "work_item_id": 17,
        "artifact_id": 42,
        "stage_name": "plan_review",
        "reviewer_agent_id": 4,
        "reviewer_role": "qa_main",
        "iteration": 3,
        "task_json": {
            "task_prompt": "Review the plan",
            "task_focus": "rollback safety",
        },
    }
    store.get_artifact.return_value = {"id": 42, "title": "Implementation plan", "content": "artifact body"}
    store.get_work_item.return_value = {"id": 17, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {}
    store.list_artifacts.return_value = []
    store.list_coordination_events.return_value = []
    store.search_knowledge.return_value = []
    store.get_review_cycles.return_value = [
        {
            "id": 1,
            "work_item_id": 17,
            "stage_name": "plan_review",
            "status": "completed",
            "iteration": 2,
            "reviewer_role": "qa_main",
            "verdict": "changes_requested",
            "verdict_json": {
                "summary": ("summary-body " * 20) + "SUMMARY-TAIL-MARKER",
                "issues": [("issue-body " * 20) + "ISSUE-TAIL-MARKER"],
            },
        },
        {
            "id": 2,
            "work_item_id": 17,
            "stage_name": "plan_review",
            "status": "completed",
            "iteration": 2,
            "reviewer_role": "security",
            "verdict": "lgtm",
            "verdict_json": {
                "summary": "Looks safe.",
                "issues": [],
            },
        },
        {
            "id": 3,
            "work_item_id": 17,
            "stage_name": "plan_review",
            "status": "completed",
            "iteration": 1,
            "reviewer_role": "qa_main",
            "verdict": "changes_requested",
            "verdict_json": {
                "summary": "Need a rollback step.",
                "issues": ["Add rollback documentation."],
            },
        },
    ]
    service = MySwatToolService(store)

    result = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="qa_main",
            runtime_registration_id=91,
        )
    )

    assert result.assignment_kind == "review"
    assert "## Prior Review Rounds For This Stage" in result.system_context
    assert "Iteration 2 / qa_main / changes_requested" in result.system_context
    assert "Iteration 2 / security / lgtm" in result.system_context
    assert "Iteration 1 / qa_main / changes_requested" in result.system_context
    assert "SUMMARY-TAIL-MARKER" in result.system_context
    assert "ISSUE-TAIL-MARKER" in result.system_context
    store.get_review_cycles.assert_called_once_with(17)


def test_claim_next_assignment_first_review_round_skips_prior_review_history_lookup():
    store = Mock(spec=MemoryStore)
    store.claim_stage_run.return_value = None
    store.claim_review_cycle.return_value = {
        "id": 89,
        "work_item_id": 17,
        "artifact_id": 42,
        "stage_name": "plan_review",
        "reviewer_agent_id": 4,
        "reviewer_role": "qa_main",
        "iteration": 1,
        "task_json": {
            "task_prompt": "Review the plan",
            "task_focus": "rollback safety",
        },
    }
    store.get_artifact.return_value = {"id": 42, "title": "Implementation plan", "content": "artifact body"}
    store.get_work_item.return_value = {"id": 17, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {}
    store.list_artifacts.return_value = []
    store.list_coordination_events.return_value = []
    store.search_knowledge.return_value = []
    service = MySwatToolService(store)

    result = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="qa_main",
            runtime_registration_id=91,
        )
    )

    assert result.assignment_kind == "review"
    assert "## Prior Review Rounds For This Stage" not in result.system_context
    store.get_review_cycles.assert_not_called()


def test_claim_next_assignment_reuses_cached_review_history_for_later_rounds():
    store = Mock(spec=MemoryStore)
    store.claim_stage_run.return_value = None
    store.claim_review_cycle.side_effect = [
        {
            "id": 88,
            "work_item_id": 17,
            "artifact_id": 42,
            "stage_name": "plan_review",
            "reviewer_agent_id": 4,
            "reviewer_role": "qa_main",
            "iteration": 3,
            "task_json": {
                "task_prompt": "Review the plan",
                "task_focus": "rollback safety",
            },
        },
        {
            "id": 89,
            "work_item_id": 17,
            "artifact_id": 43,
            "stage_name": "plan_review",
            "reviewer_agent_id": 4,
            "reviewer_role": "qa_main",
            "iteration": 4,
            "task_json": {
                "task_prompt": "Review the revised plan",
                "task_focus": "rollback safety",
            },
        },
    ]
    store.get_artifact.side_effect = [
        {"id": 42, "title": "Implementation plan", "content": "artifact body"},
        {"id": 43, "title": "Implementation plan", "content": "artifact body v2"},
    ]
    store.get_work_item.return_value = {"id": 17, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {}
    store.list_artifacts.return_value = []
    store.list_coordination_events.return_value = []
    store.search_knowledge.return_value = []
    store.get_review_cycles.return_value = [
        {
            "id": 1,
            "work_item_id": 17,
            "stage_name": "plan_review",
            "status": "completed",
            "iteration": 2,
            "reviewer_role": "qa_main",
            "verdict": "changes_requested",
            "verdict_json": {
                "summary": "Need a rollback step.",
                "issues": ["Add rollback documentation."],
            },
        }
    ]
    service = MySwatToolService(store)

    first = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="qa_main",
            runtime_registration_id=91,
        )
    )
    second = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="qa_main",
            runtime_registration_id=92,
        )
    )

    assert "Iteration 2 / qa_main / changes_requested" in first.system_context
    assert "Iteration 2 / qa_main / changes_requested" in second.system_context
    store.get_review_cycles.assert_called_once_with(17)


def test_claim_next_assignment_warms_review_history_cache_for_all_stages_from_one_fetch():
    store = Mock(spec=MemoryStore)
    store.claim_stage_run.return_value = None
    store.claim_review_cycle.side_effect = [
        {
            "id": 88,
            "work_item_id": 17,
            "artifact_id": 42,
            "stage_name": "plan_review",
            "reviewer_agent_id": 4,
            "reviewer_role": "qa_main",
            "iteration": 3,
            "task_json": {
                "task_prompt": "Review the plan",
                "task_focus": "rollback safety",
            },
        },
        {
            "id": 89,
            "work_item_id": 17,
            "artifact_id": 43,
            "stage_name": "design_review",
            "reviewer_agent_id": 5,
            "reviewer_role": "architect",
            "iteration": 3,
            "task_json": {
                "task_prompt": "Review the design",
                "task_focus": "rollback safety",
            },
        },
    ]
    store.get_artifact.side_effect = [
        {"id": 42, "title": "Implementation plan", "content": "artifact body"},
        {"id": 43, "title": "Technical design", "content": "design artifact"},
    ]
    store.get_work_item.return_value = {"id": 17, "project_id": 1}
    store.get_project.return_value = {"id": 1, "name": "proj", "repo_path": "/tmp/repo"}
    store.get_work_item_state.return_value = {}
    store.list_artifacts.return_value = []
    store.list_coordination_events.return_value = []
    store.search_knowledge.return_value = []
    store.get_review_cycles.return_value = [
        {
            "id": 1,
            "work_item_id": 17,
            "stage_name": "plan_review",
            "status": "completed",
            "iteration": 2,
            "reviewer_role": "qa_main",
            "verdict": "changes_requested",
            "verdict_json": {
                "summary": "Need a rollback step.",
                "issues": ["Add rollback documentation."],
            },
        },
        {
            "id": 2,
            "work_item_id": 17,
            "stage_name": "design_review",
            "status": "completed",
            "iteration": 2,
            "reviewer_role": "architect",
            "verdict": "lgtm",
            "verdict_json": {
                "summary": "Design looks good.",
                "issues": [],
            },
        },
    ]
    service = MySwatToolService(store)

    first = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="qa_main",
            runtime_registration_id=91,
        )
    )
    second = service.claim_next_assignment(
        ClaimNextAssignmentRequest(
            project_id=1,
            agent_role="architect",
            runtime_registration_id=92,
        )
    )

    assert "Iteration 2 / qa_main / changes_requested" in first.system_context
    assert "Iteration 2 / architect / lgtm" in second.system_context
    store.get_review_cycles.assert_called_once_with(17)


def test_review_history_cache_evicts_oldest_entry_when_bounded(monkeypatch):
    monkeypatch.setattr(service_module, "_REVIEW_HISTORY_CACHE_MAX_ENTRIES", 2)

    store = Mock(spec=MemoryStore)
    store.get_review_cycles.side_effect = [
        [
            {
                "id": 1,
                "work_item_id": 17,
                "stage_name": "plan_review",
                "status": "completed",
                "iteration": 1,
                "reviewer_role": "qa_main",
                "verdict": "lgtm",
                "verdict_json": {"summary": "ok", "issues": []},
            }
        ],
        [
            {
                "id": 2,
                "work_item_id": 18,
                "stage_name": "plan_review",
                "status": "completed",
                "iteration": 1,
                "reviewer_role": "qa_main",
                "verdict": "lgtm",
                "verdict_json": {"summary": "ok", "issues": []},
            }
        ],
        [
            {
                "id": 3,
                "work_item_id": 19,
                "stage_name": "plan_review",
                "status": "completed",
                "iteration": 1,
                "reviewer_role": "qa_main",
                "verdict": "lgtm",
                "verdict_json": {"summary": "ok", "issues": []},
            }
        ],
    ]
    service = MySwatToolService(store)

    assert service._get_recent_review_history(work_item_id=17, stage_name="plan_review", before_iteration=2, round_limit=3)
    assert service._get_recent_review_history(work_item_id=18, stage_name="plan_review", before_iteration=2, round_limit=3)
    assert service._get_recent_review_history(work_item_id=19, stage_name="plan_review", before_iteration=2, round_limit=3)

    assert (17, "plan_review") not in service._review_history_cache
    assert (18, "plan_review") in service._review_history_cache
    assert (19, "plan_review") in service._review_history_cache


def test_report_status_updates_task_state_and_process_log():
    store = Mock(spec=MemoryStore)
    store.append_work_item_process_event.return_value = {"type": "status_report"}
    service = MySwatToolService(store)

    result = service.report_status(
        StatusReport(
            work_item_id=11,
            agent_id=3,
            agent_role="developer",
            stage="develop",
            summary="Implemented phase 1 and started validation.",
            next_todos=["add retry path"],
            open_issues=["need QA on deadlock edge case"],
            title="phase-1",
        )
    )

    assert result == {"type": "status_report"}
    store.update_work_item_state.assert_called_once_with(
        11,
        current_stage="develop",
        latest_summary="Implemented phase 1 and started validation.",
        next_todos=["add retry path"],
        open_issues=["need QA on deadlock edge case"],
        updated_by_agent_id=3,
    )


def test_complete_stage_task_persists_artifact_and_marks_stage_completed():
    store = Mock(spec=MemoryStore)
    store.create_artifact.return_value = 42
    service = MySwatToolService(store)

    result = service.complete_stage_task(
        CompleteStageTaskRequest(
            stage_run_id=55,
            runtime_registration_id=91,
            work_item_id=11,
            agent_id=3,
            agent_role="developer",
            iteration=2,
            stage_name="plan",
            artifact_type="implementation_plan",
            title="Implementation plan",
            content="Phase 1: Do the work",
            summary="Plan ready",
        )
    )

    assert result.status == "completed"
    assert result.artifact_id == 42
    store.update_stage_run.assert_called_once_with(
        55,
        status="completed",
        summary="Plan ready",
        completed=True,
        output_artifact_id=42,
        metadata_json=None,
    )


def test_request_review_creates_pending_cycle_and_logs_event():
    store = Mock(spec=MemoryStore)
    store.create_review_cycle.return_value = 88
    service = MySwatToolService(store)

    result = service.request_review(
        ReviewRequest(
            work_item_id=17,
            artifact_id=42,
            iteration=2,
            proposer_agent_id=3,
            proposer_role="developer",
            reviewer_agent_id=4,
            reviewer_role="qa_main",
            stage="review",
            summary="Ready for QA review.",
            task_prompt="Please review this artifact",
            task_focus="rollback safety",
        )
    )

    assert result.cycle_id == 88
    store.create_review_cycle.assert_called_once_with(
        work_item_id=17,
        iteration=2,
        proposer_agent_id=3,
        reviewer_agent_id=4,
        reviewer_role="qa_main",
        artifact_id=42,
        proposal_session_id=None,
        stage_name="review",
        status="pending",
        task_json={
            "task_prompt": "Please review this artifact",
            "task_focus": "rollback safety",
        },
    )


def test_publish_review_verdict_updates_cycle_and_work_item():
    store = Mock(spec=MemoryStore)
    store.append_work_item_process_event.return_value = {"type": "review_verdict"}
    service = MySwatToolService(store)

    result = service.publish_review_verdict(
        ReviewVerdictSubmission(
            cycle_id=88,
            work_item_id=17,
            reviewer_agent_id=4,
            reviewer_role="qa_main",
            verdict="changes_requested",
            issues=["missing rollback coverage"],
            summary="Need rollback coverage before LGTM.",
            runtime_registration_id=91,
            stage="review",
        )
    )

    assert result == {"type": "review_verdict"}
    store.update_review_verdict.assert_called_once_with(
        cycle_id=88,
        verdict="changes_requested",
        verdict_json={
            "verdict": "changes_requested",
            "issues": ["missing rollback coverage"],
            "summary": "Need rollback coverage before LGTM.",
        },
        review_session_id=None,
        status="completed",
        claimed_by_runtime_id=91,
        only_if_active=True,
    )


def test_wait_for_review_verdicts_returns_partial_result_on_failed_cycle():
    store = Mock(spec=MemoryStore)
    store.get_review_cycles_by_ids.return_value = [
        {
            "id": 88,
            "reviewer_role": "qa_main",
            "status": "blocked",
            "verdict": "failed",
            "verdict_json": {
                "summary": "Reviewer crashed twice.",
                "issues": [],
            },
        },
        {
            "id": 89,
            "reviewer_role": "security",
            "status": "claimed",
            "verdict": "pending",
            "verdict_json": None,
        },
    ]
    service = MySwatToolService(store)

    result = service.wait_for_review_verdicts(
        ReviewWaitRequest(
            cycle_ids=[88, 89],
            timeout_seconds=1,
            return_on_failed=True,
        )
    )

    assert result == [
        service_module.ReviewVerdictEnvelope(
            cycle_id=88,
            reviewer_role="qa_main",
            verdict="failed",
            issues=[],
            summary="Reviewer crashed twice.",
        )
    ]


def test_wait_for_review_verdicts_rejects_unknown_terminal_verdict():
    store = Mock(spec=MemoryStore)
    store.get_review_cycles_by_ids.return_value = [
        {
            "id": 88,
            "reviewer_role": "qa_main",
            "status": "completed",
            "verdict": "timeout",
            "verdict_json": {
                "summary": "Timed out.",
                "issues": [],
            },
        }
    ]
    service = MySwatToolService(store)

    with pytest.raises(ValueError, match="Unsupported terminal review verdict"):
        service.wait_for_review_verdicts(
            ReviewWaitRequest(
                cycle_ids=[88],
                timeout_seconds=1,
            )
        )


def test_publish_review_verdict_ignores_already_terminal_cycle():
    store = Mock(spec=MemoryStore)
    store.update_review_verdict.return_value = False
    service = MySwatToolService(store)

    result = service.publish_review_verdict(
        ReviewVerdictSubmission(
            cycle_id=88,
            work_item_id=17,
            reviewer_agent_id=4,
            reviewer_role="qa_main",
            verdict="lgtm",
            summary="Looks good.",
            runtime_registration_id=91,
            stage="review",
        )
    )

    assert result == {"ok": False, "ignored": True, "cycle_id": 88}
    store.update_work_item_state.assert_not_called()
    store.append_work_item_process_event.assert_not_called()


def test_fail_review_cycle_marks_review_blocked_and_records_diagnostics():
    store = Mock(spec=MemoryStore)
    store.update_review_verdict.return_value = True
    store.append_work_item_process_event.return_value = {"type": "review_failed"}
    service = MySwatToolService(store)

    result = service.fail_review_cycle(
        ReviewFailureSubmission(
            cycle_id=88,
            work_item_id=17,
            reviewer_agent_id=4,
            reviewer_role="qa_main",
            stage="plan_review",
            runtime_registration_id=91,
            summary="Reviewer returned malformed output twice.",
            failure_kind="malformed_output",
            attempts=2,
            diagnostics={"attempt": 2, "stderr_tail": "traceback"},
        )
    )

    assert result == {"type": "review_failed"}
    store.update_review_verdict.assert_called_once_with(
        cycle_id=88,
        verdict="failed",
        verdict_json={
            "verdict": "failed",
            "issues": [],
            "summary": "Reviewer returned malformed output twice.",
            "failure_kind": "malformed_output",
            "attempts": 2,
            "diagnostics": {"attempt": 2, "stderr_tail": "traceback"},
        },
        review_session_id=None,
        status="blocked",
        claimed_by_runtime_id=91,
        only_if_active=True,
    )
    store.append_coordination_event.assert_called_once()


def test_cancel_review_cycles_delegates_to_store_and_notifies_waiters():
    store = Mock(spec=MemoryStore)
    store.cancel_review_cycles_by_ids.return_value = 2
    store.get_review_cycles_by_ids.return_value = []
    service = MySwatToolService(store)
    notified: list[int] = []
    service._notifier.notify_review = lambda cycle_id: notified.append(cycle_id)

    result = service.cancel_review_cycles(
        ReviewCycleCancellationRequest(
            cycle_ids=[88, 89, 88],
            summary="Sibling review failed.",
        )
    )

    assert result == {
        "ok": True,
        "updated": 2,
        "cycle_ids": [88, 89],
        "status": "cancelled",
    }
    store.cancel_review_cycles_by_ids.assert_called_once_with(
        [88, 89, 88],
        summary="Sibling review failed.",
        status="cancelled",
    )
    assert notified == [88, 89]


def test_renew_review_cycle_lease_delegates_to_store():
    store = Mock(spec=MemoryStore)
    store.renew_review_cycle_lease.return_value = True
    service = MySwatToolService(store)

    service.renew_review_cycle_lease(
        ReviewCycleLeaseRenewalRequest(
            cycle_id=88,
            runtime_registration_id=91,
            lease_seconds=120,
        )
    )

    store.renew_review_cycle_lease.assert_called_once_with(
        88,
        runtime_registration_id=91,
        lease_seconds=120,
    )


def test_append_coordination_event_updates_both_event_streams():
    store = Mock(spec=MemoryStore)
    appended = Mock()
    appended.model_dump.return_value = {"event_type": "handoff"}
    store.append_coordination_event.return_value = appended
    service = MySwatToolService(store)

    result = service.append_coordination_event(
        CoordinationEventRecord(
            work_item_id=17,
            stage_run_id=55,
            stage_name="plan_review",
            event_type="handoff",
            title="Hand off to QA",
            summary="Developer submitted the reviewed plan to QA.",
            from_agent_id=3,
            from_role="developer",
            to_agent_id=4,
            to_role="qa_main",
            payload_json={"artifact_id": 42},
        )
    )

    assert result == {"event_type": "handoff"}


def test_persist_decision_upserts_knowledge_and_logs_when_work_item_is_present():
    store = Mock(spec=MemoryStore)
    store.upsert_knowledge.return_value = (123, "created")
    service = MySwatToolService(store)

    result = service.persist_decision(
        DecisionPersistenceRequest(
            project_id=5,
            work_item_id=17,
            agent_id=4,
            agent_role="architect",
            title="Use staged review artifacts",
            content="Keep artifacts and review cycles as the system of record.",
            tags=["workflow", "review"],
            stage="design",
            search_metadata_json={"subsystem": "workflow"},
        )
    )

    assert result.knowledge_id == 123
    assert result.action == "created"
