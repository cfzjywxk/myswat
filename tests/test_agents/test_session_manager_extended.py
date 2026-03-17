"""Extended tests for SessionManager progress tracking and session-end learning."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, PropertyMock, patch

from myswat.agents.base import AgentResponse
from myswat.agents.session_manager import SessionManager
from myswat.models.session import Session


def _make_sm(store=None, runner=None, agent_row=None):
    store = store or MagicMock()
    runner = runner or MagicMock()
    agent_row = agent_row or {"id": 1, "role": "developer", "system_prompt": "Be helpful."}
    type(runner).is_session_started = PropertyMock(return_value=False)
    runner.workdir = "/tmp/test"
    runner.cli_session_id = None
    sm = SessionManager(store=store, runner=runner, agent_row=agent_row, project_id=1)
    return sm, store, runner


def _attach_session(sm, session_id=42, *, work_item_id=None, purpose=None):
    session = Session(
        id=session_id,
        agent_id=1,
        session_uuid="abcdef1234567890",
        work_item_id=work_item_id,
        purpose=purpose,
    )
    sm._session = session
    return session


class TestUpdateProgress:
    def test_returns_immediately_when_session_is_none(self):
        sm, store, _ = _make_sm()
        response = MagicMock(spec=AgentResponse)
        sm._update_progress("do something", response, 1.5)
        store.update_session_progress.assert_not_called()

    def test_cancelled_response_builds_cancelled_note(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=10)
        store.count_session_turns.return_value = 5

        response = MagicMock(spec=AgentResponse)
        response.cancelled = True
        response.success = False

        sm._update_progress("fix the bug", response, 2.0)

        note = store.update_session_progress.call_args[0][1]
        assert "CANCELLED" in note
        assert "turn 5" in note
        assert "2s" in note

    def test_success_response_builds_success_note_with_agent_summary(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=7)
        store.count_session_turns.return_value = 2

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "ok\nThis is a meaningful response line from the agent"

        sm._update_progress("write tests", response, 3.5)

        note = store.update_session_progress.call_args[0][1]
        assert "turn 2" in note
        assert "3s" in note
        assert "write tests" in note
        assert "This is a meaningful response line" in note

    def test_error_response_builds_error_note(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=99)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = False

        sm._update_progress("deploy", response, 0.8)

        note = store.update_session_progress.call_args[0][1]
        assert "ERROR" in note
        assert "deploy" in note

    def test_note_is_truncated_to_512_chars(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "A" * 1000

        sm._update_progress("B" * 1000, response, 1.0)

        note = store.update_session_progress.call_args[0][1]
        assert len(note) <= 512

    def test_catches_exceptions_silently(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)
        store.count_session_turns.side_effect = RuntimeError("db down")

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "ok"

        sm._update_progress("prompt", response, 1.0)


class TestSubmitSessionSummary:
    def test_skips_when_no_session(self):
        sm, _, _ = _make_sm()
        with patch("myswat.agents.session_manager.submit_session_summary_learn_request") as mock_submit:
            sm._submit_session_summary()
        mock_submit.assert_not_called()

    def test_skips_when_turn_count_too_small(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=12)
        store.count_session_turns.return_value = 1

        with patch("myswat.agents.session_manager.submit_session_summary_learn_request") as mock_submit:
            sm._submit_session_summary()

        mock_submit.assert_not_called()

    def test_submits_session_summary_for_real_session(self):
        sm, store, runner = _make_sm()
        _attach_session(sm, session_id=12, work_item_id=77, purpose="triage")
        store.count_session_turns.return_value = 6
        runner.workdir = "/tmp/project"

        with patch("myswat.agents.session_manager.submit_session_summary_learn_request") as mock_submit:
            sm._submit_session_summary()

        mock_submit.assert_called_once_with(
            store=store,
            settings=sm._settings,
            project_id=1,
            source_session_id=12,
            source_work_item_id=77,
            agent_role="developer",
            purpose="triage",
            workdir="/tmp/project",
            payload_json={"turn_count": 6},
            asynchronous=False,
        )

    def test_logs_failure_when_count_lookup_raises(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=12)
        store.count_session_turns.side_effect = RuntimeError("db down")

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._submit_session_summary()

        assert "[session learn] Failed: db down" in captured.getvalue()

    def test_logs_failure_when_submit_raises(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=12)
        store.count_session_turns.return_value = 5

        captured = io.StringIO()
        with patch(
            "myswat.agents.session_manager.submit_session_summary_learn_request",
            side_effect=RuntimeError("worker down"),
        ):
            with patch("sys.stderr", captured):
                sm._submit_session_summary()

        assert "[session learn] Failed: worker down" in captured.getvalue()
