"""Extra review_loop tests for cancellation and failure branches."""

from __future__ import annotations

from unittest.mock import MagicMock

from myswat.agents.base import AgentResponse
from myswat.workflow.review_loop import run_review_loop


def _sm(agent_id: int, role: str, responses: list[AgentResponse]):
    sm = MagicMock()
    sm.agent_id = agent_id
    sm.agent_role = role
    sm.session = MagicMock(id=agent_id * 10)
    sm.send = MagicMock(side_effect=responses)
    return sm


def test_run_review_loop_returns_cancelled_verdict_before_first_iteration():
    store = MagicMock()
    dev_sm = _sm(1, "developer", [])
    reviewer_sm = _sm(2, "qa_main", [])

    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task="implement feature",
        project_id=1,
        work_item_id=7,
        should_cancel=lambda: True,
    )

    assert verdict.summary == "Review loop cancelled by user."
    dev_sm.send.assert_not_called()
    reviewer_sm.send.assert_not_called()


def test_run_review_loop_returns_default_verdict_when_developer_response_fails():
    store = MagicMock()
    dev_sm = _sm(1, "developer", [AgentResponse(content="boom", exit_code=1)])
    reviewer_sm = _sm(2, "qa_main", [])

    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task="implement feature",
        project_id=1,
        work_item_id=7,
    )

    assert verdict.verdict == "changes_requested"
    reviewer_sm.send.assert_not_called()


def test_run_review_loop_returns_cancelled_verdict_after_reviewer_send():
    store = MagicMock()
    dev_sm = _sm(1, "developer", [AgentResponse(content="draft", exit_code=0)])
    reviewer_sm = _sm(2, "qa_main", [AgentResponse(content='{"verdict":"lgtm"}', exit_code=0)])
    cancel_checks = iter([False, True])

    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task="implement feature",
        project_id=1,
        work_item_id=7,
        should_cancel=lambda: next(cancel_checks),
    )

    assert verdict.summary == "Review loop cancelled by user."
    assert reviewer_sm.send.call_count == 1


def test_run_review_loop_returns_default_verdict_when_reviewer_response_fails():
    store = MagicMock()
    dev_sm = _sm(1, "developer", [AgentResponse(content="draft", exit_code=0)])
    reviewer_sm = _sm(2, "qa_main", [AgentResponse(content="boom", exit_code=1)])

    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task="implement feature",
        project_id=1,
        work_item_id=7,
    )

    assert verdict.verdict == "changes_requested"


def test_run_review_loop_ignores_persist_failures_and_still_returns_lgtm():
    store = MagicMock()
    store.update_work_item_state.side_effect = RuntimeError("state down")
    store.append_work_item_process_event.side_effect = RuntimeError("event down")
    store.create_artifact.side_effect = RuntimeError("artifact down")
    store.create_review_cycle.side_effect = RuntimeError("cycle down")
    dev_sm = _sm(1, "developer", [AgentResponse(content="draft", exit_code=0)])
    reviewer_sm = _sm(2, "qa_main", [AgentResponse(content='{"verdict":"lgtm","summary":"ok"}', exit_code=0)])

    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task="implement feature",
        project_id=1,
        work_item_id=7,
    )

    assert verdict.verdict == "lgtm"
