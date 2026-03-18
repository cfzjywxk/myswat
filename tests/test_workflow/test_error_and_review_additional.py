"""Small coverage fillers for workflow error handling and review loop."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from myswat.agents.base import AgentResponse
from myswat.workflow.error_handler import WorkflowError, _consult_architect, handle_workflow_error
from myswat.workflow.review_loop import run_review_loop


def test_consult_architect_returns_none_when_runner_cannot_be_built():
    store = MagicMock()
    store.get_agent.return_value = {"id": 1, "role": "architect"}
    err = WorkflowError(error=RuntimeError("boom"), stage="design")

    with patch("myswat.workflow.error_handler._build_runner", return_value=None):
        assert _consult_architect(err, store, project_id=1) is None


def test_handle_workflow_error_prints_traceback_when_available():
    store = MagicMock()
    err = WorkflowError(error=RuntimeError("boom"), stage="design", traceback_str="x" * 900)

    with patch("myswat.workflow.error_handler._consult_architect", return_value=None):
        with patch("myswat.workflow.error_handler.console.print") as mock_print:
            handle_workflow_error(err, store=store, project_id=1)

    assert store.store_knowledge.called
    assert any("x" * 50 in str(call) for call in mock_print.call_args_list)


def test_review_loop_uses_verdict_name_when_summary_and_issues_are_empty():
    store = MagicMock()
    store.create_artifact.return_value = 42
    store.create_review_cycle.return_value = 77
    dev_sm = MagicMock(agent_id=10, agent_role="developer")
    reviewer_sm = MagicMock(agent_id=20, agent_role="qa_main")
    dev_sm.send.return_value = AgentResponse(content="proposal", exit_code=0)
    reviewer_sm.send.return_value = AgentResponse(
        content=json.dumps({"verdict": "lgtm", "issues": [], "summary": ""}),
        exit_code=0,
    )

    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task="ship auth",
        project_id=1,
        work_item_id=1,
        max_iterations=1,
    )

    assert verdict.verdict == "lgtm"
    assert any(
        call.kwargs.get("summary") == "lgtm"
        for call in store.append_work_item_process_event.call_args_list
    )
