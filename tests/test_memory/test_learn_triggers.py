"""Tests for best-effort unified learn triggers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from myswat.memory.learn_triggers import (
    is_explicit_learn_request,
    submit_chat_learn_request,
    submit_session_summary_learn_request,
    submit_workflow_summary_learn_request,
)


def test_is_explicit_learn_request_matches_imperative_phrases() -> None:
    assert is_explicit_learn_request("Learn this design doc.")
    assert is_explicit_learn_request("Please remember this architecture note.")
    assert is_explicit_learn_request("Read the subsystem and learn it.")
    assert not is_explicit_learn_request("How do people learn Rust effectively?")


@patch("myswat.memory.learn_triggers.LearnOrchestrator")
def test_submit_chat_learn_request_skips_non_learning_messages(mock_orchestrator_cls) -> None:
    result = submit_chat_learn_request(
        store=MagicMock(),
        settings=MagicMock(),
        project_id=1,
        user_message="hello there",
        assistant_response="hi",
    )

    assert result is None
    mock_orchestrator_cls.assert_not_called()


@patch("myswat.memory.learn_triggers.LearnOrchestrator")
def test_submit_chat_learn_request_builds_explicit_request(mock_orchestrator_cls) -> None:
    store = MagicMock()
    settings = MagicMock()
    orchestrator = MagicMock()
    orchestrator.submit.return_value = object()
    mock_orchestrator_cls.return_value = orchestrator

    submit_chat_learn_request(
        store=store,
        settings=settings,
        project_id=7,
        user_message="Remember this fix.",
        assistant_response="I will use the new retry path.",
        source_session_id=11,
        workdir="/tmp/project",
    )

    request = orchestrator.submit.call_args.args[0]
    assert request.project_id == 7
    assert request.source_kind == "chat"
    assert request.trigger_kind == "explicit_user_request"
    assert request.source_session_id == 11
    assert request.payload_json["user_message"] == "Remember this fix."
    mock_orchestrator_cls.assert_called_once_with(store=store, settings=settings, workdir="/tmp/project")


@patch("myswat.memory.learn_triggers.LearnOrchestrator")
def test_submit_workflow_summary_learn_request_builds_work_request(mock_orchestrator_cls) -> None:
    orchestrator = MagicMock()
    mock_orchestrator_cls.return_value = orchestrator

    submit_workflow_summary_learn_request(
        store=MagicMock(),
        settings=MagicMock(),
        project_id=3,
        source_work_item_id=44,
        source_session_id=18,
        requirement="Ship the fix",
        final_status="completed",
        final_summary="Workflow completed successfully.",
        mode="full",
        payload_json={"review_iterations": 2},
    )

    request = orchestrator.submit.call_args.args[0]
    assert request.project_id == 3
    assert request.source_kind == "work"
    assert request.trigger_kind == "workflow_summary"
    assert request.source_work_item_id == 44
    assert request.source_session_id == 18
    assert request.payload_json["final_status"] == "completed"
    assert request.payload_json["review_iterations"] == 2


@patch("myswat.memory.learn_triggers.LearnOrchestrator")
def test_submit_session_summary_learn_request_builds_session_request(mock_orchestrator_cls) -> None:
    orchestrator = MagicMock()
    mock_orchestrator_cls.return_value = orchestrator

    submit_session_summary_learn_request(
        store=MagicMock(),
        settings=MagicMock(),
        project_id=5,
        source_session_id=19,
        source_work_item_id=23,
        agent_role="developer",
        purpose="investigate panic",
        payload_json={"turn_count": 6},
    )

    request = orchestrator.submit.call_args.args[0]
    assert request.project_id == 5
    assert request.source_kind == "session"
    assert request.trigger_kind == "session_termination"
    assert request.source_session_id == 19
    assert request.source_work_item_id == 23
    assert request.payload_json["agent_role"] == "developer"
    assert request.payload_json["purpose"] == "investigate panic"
    assert request.payload_json["turn_count"] == 6

    _, kwargs = orchestrator.submit.call_args
    assert kwargs == {"asynchronous": None}
