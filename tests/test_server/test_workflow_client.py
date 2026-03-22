"""Tests for MCP-backed workflow coordination helpers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from myswat.server.contracts import (
    ReviewCycleCancellationRequest,
    ReviewRequest,
    ReviewWaitRequest,
    StageRunStart,
    StageRunWaitRequest,
    StatusReport,
)
from myswat.server.workflow_client import LocalMCPToolClient, MCPWorkflowCoordinator


class _FakeCaller:
    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, name: str, arguments: dict[str, object]):
        self.calls.append((name, arguments))
        return self._responses.pop(0)


def test_local_mcp_tool_client_returns_structured_content():
    dispatcher = Mock()
    dispatcher.call_tool.return_value = {"structuredContent": {"ok": True}}

    client = LocalMCPToolClient(dispatcher)

    assert client.call_tool("ping", {"project_id": 1}) == {"ok": True}
    dispatcher.call_tool.assert_called_once_with("ping", {"project_id": 1})


def test_local_mcp_tool_client_rejects_non_mapping_dispatcher_result():
    dispatcher = Mock()
    dispatcher.call_tool.return_value = ["bad"]

    client = LocalMCPToolClient(dispatcher)

    with pytest.raises(RuntimeError, match="Invalid dispatcher result"):
        client.call_tool("ping", {})


def test_report_status_normalizes_non_mapping_results():
    caller = _FakeCaller({"ok": True}, "ignored")
    coordinator = MCPWorkflowCoordinator(caller)
    request = StatusReport(
        work_item_id=11,
        agent_id=3,
        agent_role="developer",
        stage="implement",
        summary="done",
    )

    assert coordinator.report_status(request) == {"ok": True}
    assert coordinator.report_status(request) == {}
    assert caller.calls == [
        ("report_status", request.model_dump(exclude_none=True)),
        ("report_status", request.model_dump(exclude_none=True)),
    ]


def test_stage_run_and_review_requests_are_validated():
    caller = _FakeCaller(
        {"stage_run_id": 7},
        {
            "stage_run_id": 7,
            "work_item_id": 11,
            "stage_name": "implement",
            "status": "completed",
            "summary": "ready",
        },
        {"cycle_id": 13},
        [{"cycle_id": 13, "reviewer_role": "qa_main", "verdict": "lgtm"}],
        {"cancelled": 1},
    )
    coordinator = MCPWorkflowCoordinator(caller)
    stage_request = StageRunStart(work_item_id=11, stage_name="implement")
    wait_request = StageRunWaitRequest(stage_run_id=7)
    review_request = ReviewRequest(
        work_item_id=11,
        artifact_id=5,
        iteration=1,
        proposer_agent_id=3,
        proposer_role="developer",
        reviewer_agent_id=4,
        reviewer_role="qa_main",
    )
    review_wait_request = ReviewWaitRequest(cycle_ids=[13])
    cancel_request = ReviewCycleCancellationRequest(cycle_ids=[13], summary="done")

    assert coordinator.start_stage_run(stage_request).stage_run_id == 7
    completion = coordinator.wait_for_stage_run_completion(wait_request)
    assert completion.status == "completed"
    assert coordinator.request_review(review_request).cycle_id == 13
    verdicts = coordinator.wait_for_review_verdicts(review_wait_request)
    assert [verdict.verdict for verdict in verdicts] == ["lgtm"]
    assert coordinator.cancel_review_cycles(cancel_request) == {"cancelled": 1}

    assert caller.calls == [
        ("start_stage_run", stage_request.model_dump(exclude_none=True)),
        ("wait_for_stage_run_completion", wait_request.model_dump(exclude_none=True)),
        ("request_review", review_request.model_dump(exclude_none=True)),
        ("wait_for_review_verdicts", review_wait_request.model_dump(exclude_none=True)),
        ("cancel_review_cycles", cancel_request.model_dump(exclude_none=True)),
    ]


def test_wait_for_review_verdicts_handles_empty_and_invalid_payloads():
    caller = _FakeCaller(None, {"not": "a list"})
    coordinator = MCPWorkflowCoordinator(caller)
    request = ReviewWaitRequest(cycle_ids=[1])

    assert coordinator.wait_for_review_verdicts(request) == []
    with pytest.raises(RuntimeError, match="Invalid review verdict payload"):
        coordinator.wait_for_review_verdicts(request)


def test_cancel_review_cycles_normalizes_non_mapping_results():
    caller = _FakeCaller("ignored")
    coordinator = MCPWorkflowCoordinator(caller)
    request = ReviewCycleCancellationRequest(cycle_ids=[1], summary="cleanup")

    assert coordinator.cancel_review_cycles(request) == {}
