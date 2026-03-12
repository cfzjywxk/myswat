"""Tests for workflow error handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from myswat.agents.base import AgentResponse
from myswat.workflow.error_handler import (
    WorkflowError,
    _build_runner,
    _consult_architect,
    handle_workflow_error,
)


class TestWorkflowError:
    def test_summary_format(self):
        err = WorkflowError(
            error=ValueError("bad value"),
            stage="review_loop",
        )
        assert err.summary() == "[review_loop] ValueError: bad value"

    def test_to_record_structure(self):
        err = WorkflowError(
            error=RuntimeError("test"),
            stage="workflow_execution",
            context={"project": "myproj", "task": "implement X"},
        )
        record = err.to_record()

        assert record["error_type"] == "RuntimeError"
        assert record["message"] == "test"
        assert record["stage"] == "workflow_execution"
        assert "project" in record["context"]
        assert "traceback" in record

    def test_context_values_truncated(self):
        err = WorkflowError(
            error=ValueError("x"),
            stage="test",
            context={"long": "A" * 1000},
        )
        record = err.to_record()
        assert len(record["context"]["long"]) <= 500

    def test_traceback_captured(self):
        try:
            raise ValueError("deliberate")
        except ValueError as e:
            err = WorkflowError(error=e, stage="test")

        assert "deliberate" in err.traceback_str

    def test_default_traceback_when_no_exception(self):
        err = WorkflowError(
            error=ValueError("no trace"),
            stage="test",
            traceback_str="custom trace",
        )
        assert err.traceback_str == "custom trace"


class TestBuildRunner:
    def test_codex_runner(self):
        row = {
            "cli_backend": "codex",
            "cli_path": "/usr/bin/codex",
            "model_name": "gpt-5.4",
            "cli_extra_args": None,
        }
        runner = _build_runner(row)
        assert runner is not None
        assert runner.model == "gpt-5.4"

    def test_kimi_runner(self):
        row = {
            "cli_backend": "kimi",
            "cli_path": "/usr/bin/kimi",
            "model_name": "kimi-code",
            "cli_extra_args": '["--fast"]',
        }
        runner = _build_runner(row)
        assert runner is not None
        assert "--fast" in runner.extra_flags

    def test_claude_runner(self):
        row = {
            "cli_backend": "claude",
            "cli_path": "/usr/bin/claude",
            "model_name": "claude-sonnet-4-6",
            "cli_extra_args": '["--print"]',
        }
        settings = MagicMock()
        settings.agents.claude_required_ip = "154.28.2.59"
        settings.agents.claude_ip_check_timeout_seconds = 10
        runner = _build_runner(row, settings=settings)
        assert runner is not None
        assert "--print" in runner.extra_flags

    def test_unknown_backend(self):
        row = {
            "cli_backend": "unknown",
            "cli_path": "foo",
            "model_name": "bar",
            "cli_extra_args": None,
        }
        assert _build_runner(row) is None


class TestConsultArchitect:
    @patch("myswat.workflow.error_handler.MySwatSettings")
    @patch("myswat.workflow.error_handler._build_runner")
    def test_passes_settings_to_runner_builder(self, mock_build, mock_settings_cls):
        store = MagicMock()
        agent_row = {
            "id": 1, "cli_backend": "claude", "cli_path": "claude",
            "model_name": "claude-sonnet-4-6", "cli_extra_args": None,
        }
        store.get_agent.return_value = agent_row
        err = WorkflowError(error=ValueError("test"), stage="test")

        settings = MagicMock()
        mock_settings_cls.return_value = settings
        mock_runner = MagicMock()
        mock_runner.invoke.return_value = AgentResponse(
            content="Root cause: X. Fix: do Y.", exit_code=0,
        )
        mock_build.return_value = mock_runner

        _consult_architect(err, store, project_id=1)

        mock_build.assert_called_once_with(agent_row, settings=settings)

    def test_returns_suggestion_on_success(self):
        store = MagicMock()
        store.get_agent.return_value = {
            "id": 1, "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5.4", "cli_extra_args": None,
        }
        err = WorkflowError(error=ValueError("test"), stage="test")

        with patch("myswat.workflow.error_handler._build_runner") as mock_build:
            mock_runner = MagicMock()
            mock_runner.invoke.return_value = AgentResponse(
                content="Root cause: X. Fix: do Y.", exit_code=0,
            )
            mock_build.return_value = mock_runner

            result = _consult_architect(err, store, project_id=1)

        assert result is not None
        assert "Root cause" in result

    def test_returns_none_when_no_architect(self):
        store = MagicMock()
        store.get_agent.return_value = None

        err = WorkflowError(error=ValueError("test"), stage="test")
        assert _consult_architect(err, store, 1) is None

    def test_returns_none_on_runner_failure(self):
        store = MagicMock()
        store.get_agent.return_value = {
            "id": 1, "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5.4", "cli_extra_args": None,
        }
        err = WorkflowError(error=ValueError("test"), stage="test")

        with patch("myswat.workflow.error_handler._build_runner") as mock_build:
            mock_runner = MagicMock()
            mock_runner.invoke.return_value = AgentResponse(
                content="error", exit_code=1,
            )
            mock_build.return_value = mock_runner

            assert _consult_architect(err, store, 1) is None

    def test_survives_exception(self):
        store = MagicMock()
        store.get_agent.side_effect = Exception("db down")

        err = WorkflowError(error=ValueError("test"), stage="test")
        assert _consult_architect(err, store, 1) is None


class TestHandleWorkflowError:
    def test_records_to_knowledge(self):
        store = MagicMock()
        store.get_agent.return_value = None  # No architect

        err = WorkflowError(error=ValueError("bad"), stage="test")
        handle_workflow_error(err, store=store, project_id=1)

        store.store_knowledge.assert_called_once()
        call_kwargs = store.store_knowledge.call_args[1]
        assert call_kwargs["category"] == "error_log"
        assert call_kwargs["ttl_days"] == 30
        assert call_kwargs["compute_embedding"] is False

    def test_survives_recording_failure(self):
        store = MagicMock()
        store.store_knowledge.side_effect = Exception("db error")
        store.get_agent.return_value = None

        err = WorkflowError(error=ValueError("bad"), stage="test")
        # Should not raise
        result = handle_workflow_error(err, store=store, project_id=1)
        assert result is None

    def test_works_without_store(self):
        err = WorkflowError(error=ValueError("bad"), stage="test")
        # Should not raise
        result = handle_workflow_error(err)
        assert result is None

    def test_returns_architect_suggestion(self):
        store = MagicMock()
        store.get_agent.return_value = {
            "id": 1, "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5.4", "cli_extra_args": None,
        }

        err = WorkflowError(error=ValueError("bad"), stage="test")

        with patch("myswat.workflow.error_handler._consult_architect") as mock_consult:
            mock_consult.return_value = "Fix: restart the process"
            result = handle_workflow_error(err, store=store, project_id=1)

        assert result == "Fix: restart the process"
