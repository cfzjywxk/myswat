"""Extended tests for SessionManager progress tracking and session-end learning."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pymysql.err
import pytest

from myswat.agents.base import AgentResponse
from myswat.agents.session_manager import SessionManager
from myswat.large_payloads import extract_markdown_path
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


class TestMemoryFallbacks:
    def test_restore_cli_session_ignores_lookup_failure(self):
        sm, store, runner = _make_sm()
        store.get_session_turns.side_effect = RuntimeError("db down")

        sm._restore_cli_session(12)

        runner.restore_session.assert_not_called()

    def test_restore_cli_session_ignores_non_assistant_and_non_dict_metadata(self):
        sm, store, runner = _make_sm()
        store.get_session_turns.return_value = [
            SimpleNamespace(role="user", metadata_json={"cli_session_id": "skip-user"}),
            SimpleNamespace(role="assistant", metadata_json="not-a-dict"),
            SimpleNamespace(role="assistant", metadata_json={}),
        ]

        sm._restore_cli_session(12)

        runner.restore_session.assert_not_called()

    def test_memory_revision_hint_failure_is_best_effort(self, caplog):
        sm, store, _ = _make_sm()
        sm._session = Session(
            id=1,
            agent_id=1,
            session_uuid="uuid-1",
            memory_revision_at_context_build=1,
        )
        store.get_project_memory_revision.side_effect = pymysql.err.OperationalError(
            2003, "revision down",
        )

        with caplog.at_level(logging.WARNING):
            prompt = sm._maybe_prefix_memory_revision_hint("continue")

        assert prompt == "continue"
        assert "[session memory] Revision check failed" in caplog.text
        assert "revision down" in caplog.text

    def test_memory_revision_hint_skips_when_revision_not_newer(self):
        sm, store, _ = _make_sm()
        sm._session = Session(
            id=1,
            agent_id=1,
            session_uuid="uuid-1",
            memory_revision_at_context_build=3,
        )
        store.get_project_memory_revision.return_value = 3

        assert sm._maybe_prefix_memory_revision_hint("continue") == "continue"

    def test_send_continues_when_memory_context_build_fails(self, caplog):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.get_project_memory_revision.return_value = 5
        store.count_session_turns.return_value = 2
        type(runner).is_session_started = PropertyMock(return_value=False)
        runner.invoke.return_value = AgentResponse(content="response", exit_code=0)
        sm._retriever.build_context_for_agent = MagicMock(
            side_effect=pymysql.err.OperationalError(2003, "search down"),
        )

        with caplog.at_level(logging.WARNING):
            response = sm.send("hello", task_description="greet")

        assert response.content == "response"
        system_context = runner.invoke.call_args.kwargs["system_context"]
        assert "Be helpful." in system_context
        assert "Large Payload Handling" in system_context
        assert "[session memory] Context build failed" in caplog.text
        assert "search down" in caplog.text

    def test_build_system_context_logs_revision_tracking_failure(self, caplog):
        sm, store, _ = _make_sm()
        sm._session = Session(id=1, agent_id=1, session_uuid="uuid-1")
        sm._retriever.build_context_for_agent = MagicMock(return_value="ctx")
        store.get_project_memory_revision.side_effect = pymysql.err.OperationalError(
            2003, "revision write down",
        )

        with caplog.at_level(logging.WARNING):
            context = sm._build_system_context_and_track_revision("task", "prompt")

        assert "Be helpful." in context
        assert "ctx" in context
        assert "Large Payload Handling" in context
        assert "[session memory] Revision tracking failed" in caplog.text
        assert "revision write down" in caplog.text

    def test_send_propagates_memory_layer_programming_errors(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.count_session_turns.return_value = 2
        type(runner).is_session_started = PropertyMock(return_value=False)
        sm._retriever.build_context_for_agent = MagicMock(side_effect=TypeError("bug"))

        with pytest.raises(TypeError, match="bug"):
            sm.send("hello", task_description="greet")

    def test_send_retries_stalled_runner_with_backoff_and_restores_timeout(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.get_project_memory_revision.return_value = 3
        store.count_session_turns.return_value = 2
        runner.timeout = 20
        type(runner).is_session_started = PropertyMock(return_value=False)
        sm._retriever.build_context_for_agent = MagicMock(return_value="ctx")
        runner.invoke.side_effect = [
            AgentResponse(content="", exit_code=-1, cancelled=False),
            AgentResponse(content="", exit_code=-1, cancelled=False),
            AgentResponse(content="ok", exit_code=0, cancelled=False),
        ]

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            response = sm.send("hello", task_description="greet")

        assert response.content == "ok"
        assert runner.reset_session.call_count == 2
        assert sm._retriever.build_context_for_agent.call_count == 3
        assert runner.timeout == 20
        assert store.set_session_memory_revision.call_count == 3
        assert "Agent stalled (attempt 1/3)" in captured.getvalue()
        assert "Agent stalled (attempt 2/3)" in captured.getvalue()

    def test_send_returns_last_stall_response_after_retry_exhaustion(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.get_project_memory_revision.return_value = 3
        store.count_session_turns.return_value = 2
        runner.timeout = None
        type(runner).is_session_started = PropertyMock(return_value=False)
        sm._retriever.build_context_for_agent = MagicMock(return_value="ctx")
        runner.invoke.side_effect = [
            AgentResponse(content="stall-1", exit_code=-1, cancelled=False),
            AgentResponse(content="stall-2", exit_code=-1, cancelled=False),
            AgentResponse(content="stall-3", exit_code=-1, cancelled=False),
        ]

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            response = sm.send("hello", task_description="greet")

        assert response.content == "stall-3"
        assert runner.reset_session.call_count == 2
        assert runner.timeout is None
        assert "Agent stalled (attempt 3/3)" in captured.getvalue()

    def test_send_externalizes_large_prompt_and_response_to_temp_markdown(self):
        sm, store, runner = _make_sm()
        _attach_session(sm, session_id=1)
        type(runner).is_session_started = PropertyMock(return_value=True)
        store.count_session_turns.return_value = 2
        large_prompt = "prompt section\n" + ("A" * 1500)
        large_response = "response section\n" + ("B" * 1500)
        runner.invoke.return_value = AgentResponse(content=large_response, exit_code=0)

        response = sm.send(large_prompt, task_description="large")

        assert response.content == large_response
        sent_prompt = runner.invoke.call_args.args[0]
        prompt_path = extract_markdown_path(sent_prompt)
        assert prompt_path is not None
        assert Path(prompt_path).exists()
        assert large_prompt in Path(prompt_path).read_text(encoding="utf-8")

        user_call = store.append_turn.call_args_list[0]
        assistant_call = store.append_turn.call_args_list[1]
        assert extract_markdown_path(user_call.kwargs["content"]) is not None
        response_path = assistant_call.kwargs["metadata"]["externalized_response_path"]
        assert Path(response_path).exists()
        assert large_response in Path(response_path).read_text(encoding="utf-8")
        assert "The detailed response is in" in assistant_call.kwargs["content"]

    @pytest.mark.parametrize("role", ["architect", "developer", "qa_main", "qa_vice"])
    def test_first_turn_externalizes_large_system_context_for_primary_roles(self, role):
        large_system_prompt = f"You are {role}.\n" + ("S" * 900)
        sm, store, runner = _make_sm(
            agent_row={"id": 1, "role": role, "system_prompt": large_system_prompt},
        )
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.get_project_memory_revision.return_value = 3
        store.count_session_turns.return_value = 2
        sm._retriever.build_context_for_agent = MagicMock(return_value="C" * 900)
        type(runner).is_session_started = PropertyMock(return_value=False)
        runner.invoke.return_value = AgentResponse(content="ok", exit_code=0)

        sm.send("hello", task_description="bootstrap")

        sent_system_context = runner.invoke.call_args.kwargs["system_context"]
        context_path = extract_markdown_path(sent_system_context)
        assert context_path is not None
        context_text = Path(context_path).read_text(encoding="utf-8")
        assert "Large Payload Handling" in context_text
        assert large_system_prompt in context_text
        assert "C" * 900 in context_text

    def test_send_status_callback_reports_stall_and_retry(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.get_project_memory_revision.return_value = 3
        store.count_session_turns.return_value = 2
        runner.timeout = 20
        type(runner).is_session_started = PropertyMock(return_value=False)
        sm._retriever.build_context_for_agent = MagicMock(return_value="ctx")
        runner.invoke.side_effect = [
            AgentResponse(content="", exit_code=-1, cancelled=False),
            AgentResponse(content="ok", exit_code=0, cancelled=False),
        ]
        callback = MagicMock()

        response = sm.send("hello", task_description="greet", status_callback=callback)

        assert response.content == "ok"
        callback.assert_any_call(
            "agent_stalled",
            {"attempt": 1, "max_attempts": 3, "timeout": 20},
        )
        callback.assert_any_call(
            "agent_retry",
            {"attempt": 1, "next_attempt": 2, "max_attempts": 3, "next_timeout": 30},
        )

    def test_send_retries_empty_output_runner_response(self):
        sm, store, runner = _make_sm()
        store.get_active_session.return_value = None
        store.create_session.return_value = Session(id=1, agent_id=1, session_uuid="uuid-1")
        store.get_project_memory_revision.return_value = 3
        store.count_session_turns.return_value = 2
        runner.timeout = 20
        type(runner).is_session_started = PropertyMock(return_value=False)
        sm._retriever.build_context_for_agent = MagicMock(return_value="ctx")
        runner.invoke.side_effect = [
            AgentResponse(content="Claude CLI returned empty output.", exit_code=-3, cancelled=False),
            AgentResponse(content="ok", exit_code=0, cancelled=False),
        ]

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            response = sm.send("hello", task_description="greet")

        assert response.content == "ok"
        assert runner.reset_session.call_count == 1
        assert sm._retriever.build_context_for_agent.call_count == 2
        assert "Agent returned empty output (attempt 1/3)" in captured.getvalue()
