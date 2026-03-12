"""Tests for SessionManager lifecycle and error handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from myswat.agents.base import AgentResponse
from myswat.agents.session_manager import SessionManager
from myswat.models.session import Session, SessionTurn


def _make_sm(store=None, runner=None, agent_row=None, compactor=None):
    store = store or MagicMock()
    runner = runner or MagicMock()
    agent_row = agent_row or {
        "id": 1, "role": "developer", "system_prompt": "Be helpful.",
    }
    # Configure runner defaults
    type(runner).is_session_started = PropertyMock(return_value=False)
    runner.workdir = "/tmp/test"
    runner.cli_session_id = None

    sm = SessionManager(
        store=store, runner=runner, agent_row=agent_row,
        project_id=1, compactor=compactor,
    )
    return sm, store, runner


class TestCreateOrResume:
    def test_creates_new_session(self):
        sm, store, _ = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(
            id=10, agent_id=1, session_uuid="abc-123",
        )

        session = sm.create_or_resume(purpose="test")

        assert session.id == 10
        store.get_active_session.assert_called_once_with(1, work_item_id=None)
        store.create_session.assert_called_once()

    def test_resumes_existing_session(self):
        sm, store, _ = _make_sm()
        existing = Session(id=5, agent_id=1, session_uuid="existing-uuid")
        store.get_active_session.return_value = existing
        store.get_session_turns.return_value = []

        session = sm.create_or_resume()

        assert session.id == 5
        store.get_active_session.assert_called_once_with(1, work_item_id=None)
        store.create_session.assert_not_called()

    def test_resumes_existing_session_scoped_by_work_item(self):
        sm, store, _ = _make_sm()
        existing = Session(id=8, agent_id=1, session_uuid="work-item-uuid", work_item_id=42)
        store.get_active_session.return_value = existing
        store.get_session_turns.return_value = []

        session = sm.create_or_resume(work_item_id=42)

        assert session.id == 8
        store.get_active_session.assert_called_once_with(1, work_item_id=42)
        store.create_session.assert_not_called()

    def test_resumes_existing_session_restores_cli_session_id(self):
        sm, store, runner = _make_sm()
        existing = Session(id=9, agent_id=1, session_uuid="resume-uuid")
        store.get_active_session.return_value = existing
        store.get_session_turns.return_value = [
            SessionTurn(
                session_id=9,
                turn_index=0,
                role="assistant",
                content="hello",
                metadata_json={"cli_session_id": "restored-session-id"},
            )
        ]

        sm.create_or_resume()

        runner.restore_session.assert_called_once_with("restored-session-id")


class TestSend:
    def test_first_turn_builds_context(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(
            id=1, agent_id=1, session_uuid="uuid-1",
        )
        type(runner).is_session_started = PropertyMock(return_value=False)
        runner.invoke.return_value = AgentResponse(
            content="response", exit_code=0,
        )
        store.count_session_turns.return_value = 2

        sm.send("hello", task_description="greet")

        # invoke called with system_context (not None)
        call_args = runner.invoke.call_args
        assert call_args[1].get("system_context") is not None or len(call_args[0]) > 1

    def test_records_user_and_assistant_turns(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(
            id=1, agent_id=1, session_uuid="uuid-1",
        )
        type(runner).is_session_started = PropertyMock(return_value=True)
        runner.invoke.return_value = AgentResponse(
            content="hi there", exit_code=0,
        )
        store.count_session_turns.return_value = 2

        sm.send("hello")

        # Two append_turn calls: user + assistant
        assert store.append_turn.call_count == 2
        roles = [c[1]["role"] for c in store.append_turn.call_args_list]
        assert roles == ["user", "assistant"]

    def test_cancelled_response_recorded(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(
            id=1, agent_id=1, session_uuid="uuid-1",
        )
        type(runner).is_session_started = PropertyMock(return_value=True)
        runner.invoke.return_value = AgentResponse(
            content="cancelled", exit_code=-1, cancelled=True,
        )
        store.count_session_turns.return_value = 2

        resp = sm.send("do something")

        assert resp.cancelled


class TestClose:
    def test_close_marks_completed(self):
        sm, store, _ = _make_sm()
        sm._session = Session(id=5, agent_id=1, session_uuid="uuid")
        store.get_session.return_value = None

        sm.close()

        store.close_session.assert_called_once_with(5)

    def test_close_without_session(self):
        sm, store, _ = _make_sm()
        sm._session = None

        sm.close()  # Should not raise

        store.close_session.assert_not_called()


class TestResetSession:
    def test_reset_clears_cli_session(self):
        sm, _, runner = _make_sm()
        sm.reset_ai_session()

        runner.reset_session.assert_called_once()


class TestMidSessionCompaction:
    def test_compaction_triggered_when_threshold_exceeded(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = [1, 2]

        sm, store, runner = _make_sm(compactor=compactor)
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(
            id=1, agent_id=1, session_uuid="uuid-1",
        )
        type(runner).is_session_started = PropertyMock(return_value=True)
        runner.invoke.return_value = AgentResponse(content="ok", exit_code=0)
        store.count_session_turns.return_value = 2
        store.delete_compacted_turns.return_value = 10

        sm.send("test")

        compactor.should_compact.assert_called()
        compactor.compact_session.assert_called_once()

    def test_compaction_failure_does_not_crash(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.side_effect = Exception("compaction failed")

        sm, store, runner = _make_sm(compactor=compactor)
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(
            id=1, agent_id=1, session_uuid="uuid-1",
        )
        type(runner).is_session_started = PropertyMock(return_value=True)
        runner.invoke.return_value = AgentResponse(content="ok", exit_code=0)
        store.count_session_turns.return_value = 2

        # Should not raise — compaction failure is non-fatal
        resp = sm.send("test")
        assert resp.content == "ok"


class TestProperties:
    def test_agent_role(self):
        sm, _, _ = _make_sm()
        assert sm.agent_role == "developer"

    def test_agent_id(self):
        sm, _, _ = _make_sm()
        assert sm.agent_id == 1

    def test_session_initially_none(self):
        sm, _, _ = _make_sm()
        assert sm.session is None

    def test_fmt_duration(self):
        assert SessionManager._fmt_duration(30) == "30s"
        assert SessionManager._fmt_duration(90) == "1m30s"
        assert SessionManager._fmt_duration(3661) == "1h01m01s"
