"""Tests for myswat.cli.work_cmd."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.agents.base import AgentResponse
from myswat.cli.work_cmd import _make_runner, run_work


# ---------------------------------------------------------------------------
# _make_runner
# ---------------------------------------------------------------------------
class TestMakeRunner:
    def test_codex(self):
        row = {
            "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5", "cli_extra_args": None,
        }
        assert _make_runner(row) is not None

    def test_kimi(self):
        row = {
            "cli_backend": "kimi", "cli_path": "kimi",
            "model_name": "k2", "cli_extra_args": None,
        }
        assert _make_runner(row) is not None

    def test_unknown_raises(self):
        row = {
            "cli_backend": "unknown", "cli_path": "x",
            "model_name": "m", "cli_extra_args": None,
        }
        with pytest.raises(typer.BadParameter):
            _make_runner(row)

    def test_extra_args(self):
        row = {
            "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5",
            "cli_extra_args": json.dumps(["--flag"]),
        }
        assert _make_runner(row) is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _agent_row(role="developer", backend="codex"):
    return {
        "id": 1, "role": role, "display_name": f"Agent-{role}",
        "cli_backend": backend, "model_name": "gpt-5",
        "cli_path": backend, "cli_extra_args": None,
    }


# ---------------------------------------------------------------------------
# run_work
# ---------------------------------------------------------------------------
class TestRunWork:
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                                mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_work("missing", "do stuff")

    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_dev_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                            mock_settings_cls, mock_learn):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_work("proj", "do stuff")

    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_no_qa_agents(self, mock_store_cls, mock_mig, mock_pool_cls,
                           mock_settings_cls, mock_learn):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_work("proj", "do stuff")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_success(self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
                      mock_pool_cls, mock_settings_cls, mock_learn,
                      mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "completed")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_failure(self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
                      mock_pool_cls, mock_settings_cls, mock_learn,
                      mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=False)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "review")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_exception(self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
                        mock_pool_cls, mock_settings_cls, mock_learn,
                        mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.side_effect = RuntimeError("engine crash")
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "blocked")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_migrations_applied_printed(self, mock_comp, mock_sm_cls,
                                         mock_store_cls, mock_mig,
                                         mock_pool_cls, mock_settings_cls,
                                         mock_learn, mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings
        mock_mig.return_value = ["v001"]

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
