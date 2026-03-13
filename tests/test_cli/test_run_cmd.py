"""Tests for myswat.cli.run_cmd."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.agents.base import AgentResponse
from myswat.cli.run_cmd import (
    _find_compaction_runner,
    run_single,
    run_with_review,
)


# ---------------------------------------------------------------------------
# _find_compaction_runner
# ---------------------------------------------------------------------------
class TestFindCompactionRunner:
    def test_returns_matching_backend(self):
        store = MagicMock()
        store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "codex", "cli_path": "codex",
                "model_name": "gpt-5", "cli_extra_args": None,
            }
        ]
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        proj = {"id": 1}

        result = _find_compaction_runner(store, proj, settings)
        assert result is not None

    def test_fallback_to_first_agent(self):
        store = MagicMock()
        store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "kimi", "cli_path": "kimi",
                "model_name": "k2", "cli_extra_args": None,
            }
        ]
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"  # no match
        proj = {"id": 1}

        result = _find_compaction_runner(store, proj, settings)
        assert result is not None

    def test_no_agents_returns_none(self):
        store = MagicMock()
        store.list_agents.return_value = []
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        proj = {"id": 1}

        result = _find_compaction_runner(store, proj, settings)
        assert result is None


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------
class TestRunSingle:
    def _setup_mocks(self):
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200

        pool = MagicMock()
        store = MagicMock()
        proj = {
            "id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp/r",
        }
        agent_row = {
            "id": 1, "role": "developer", "display_name": "Dev",
            "cli_backend": "codex", "model_name": "gpt-5",
            "cli_path": "codex", "cli_extra_args": None,
        }
        return settings, pool, store, proj, agent_row

    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    def test_project_not_found(self, mock_comp, mock_sm_cls,
                               mock_store_cls, mock_mig, mock_pool_cls,
                               mock_settings_cls, mock_learn):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_single("missing", "do stuff")

    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    def test_agent_not_found(self, mock_comp, mock_sm_cls,
                              mock_store_cls, mock_mig, mock_pool_cls,
                              mock_settings_cls, mock_learn):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_single("proj", "do stuff")

    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_success_response(self, mock_learn, mock_comp, mock_sm_cls,
                               mock_store_cls, mock_mig, mock_pool_cls,
                               mock_settings_cls):
        settings, pool, store, proj, agent_row = self._setup_mocks()
        mock_settings_cls.return_value = settings
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="test-uuid")
        sm.send.return_value = AgentResponse(content="done", exit_code=0)
        mock_sm_cls.return_value = sm

        run_single("proj", "do stuff")
        sm.send.assert_called_once()
        sm.close.assert_called_once()
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["threshold_turns"] == 200
        assert "threshold_tokens" not in kwargs

    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_failure_response(self, mock_learn, mock_comp, mock_sm_cls,
                               mock_store_cls, mock_mig, mock_pool_cls,
                               mock_settings_cls):
        settings, pool, store, proj, agent_row = self._setup_mocks()
        mock_settings_cls.return_value = settings
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="test-uuid")
        sm.send.return_value = AgentResponse(
            content="error", exit_code=1, raw_stderr="some err",
        )
        mock_sm_cls.return_value = sm

        run_single("proj", "do stuff")
        sm.close.assert_called_once()

    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_exception_during_send(self, mock_learn, mock_comp, mock_sm_cls,
                                    mock_store_cls, mock_mig, mock_pool_cls,
                                    mock_settings_cls):
        settings, pool, store, proj, agent_row = self._setup_mocks()
        mock_settings_cls.return_value = settings
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="test-uuid")
        sm.send.side_effect = RuntimeError("agent crash")
        mock_sm_cls.return_value = sm

        run_single("proj", "do stuff")
        sm.close.assert_called_once()

    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_workdir_override(self, mock_learn, mock_comp, mock_sm_cls,
                               mock_store_cls, mock_mig, mock_pool_cls,
                               mock_settings_cls):
        settings, pool, store, proj, agent_row = self._setup_mocks()
        mock_settings_cls.return_value = settings
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="test-uuid")
        sm.send.return_value = AgentResponse(content="done", exit_code=0)
        mock_sm_cls.return_value = sm

        run_single("proj", "do stuff", workdir="/custom/dir")
        sm.close.assert_called_once()


# ---------------------------------------------------------------------------
# run_with_review
# ---------------------------------------------------------------------------
class TestRunWithReview:
    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_project_not_found(self, mock_learn, mock_store_cls, mock_mig,
                                mock_pool_cls, mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_with_review("missing", "do stuff")

    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_dev_agent_not_found(self, mock_learn, mock_store_cls, mock_mig,
                                  mock_pool_cls, mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_with_review("proj", "do stuff")

    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_reviewer_not_found(self, mock_learn, mock_store_cls, mock_mig,
                                 mock_pool_cls, mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        # Return dev but not reviewer
        def side_effect(pid, role):
            if role == "developer":
                return {
                    "id": 1, "role": "developer", "display_name": "Dev",
                    "cli_backend": "codex", "model_name": "gpt-5",
                    "cli_path": "codex", "cli_extra_args": None,
                }
            return None
        mock_store.get_agent.side_effect = side_effect
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_with_review("proj", "do stuff")

    @patch("myswat.workflow.review_loop.run_review_loop")
    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_lgtm_verdict(self, mock_learn, mock_comp, mock_sm_cls,
                           mock_store_cls, mock_mig, mock_pool_cls,
                           mock_settings_cls, mock_review_loop):
        from myswat.models.work_item import ReviewVerdict

        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        agent_row = {
            "id": 1, "role": "developer", "display_name": "Dev",
            "cli_backend": "codex", "model_name": "gpt-5",
            "cli_path": "codex", "cli_extra_args": None,
        }
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        mock_sm_cls.return_value = sm

        mock_review_loop.return_value = ReviewVerdict(
            verdict="lgtm", issues=[], summary="looks good",
        )

        run_with_review("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "approved")

    @patch("myswat.workflow.review_loop.run_review_loop")
    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_changes_requested_verdict(self, mock_learn, mock_comp, mock_sm_cls,
                                        mock_store_cls, mock_mig, mock_pool_cls,
                                        mock_settings_cls, mock_review_loop):
        from myswat.models.work_item import ReviewVerdict

        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        agent_row = {
            "id": 1, "role": "developer", "display_name": "Dev",
            "cli_backend": "codex", "model_name": "gpt-5",
            "cli_path": "codex", "cli_extra_args": None,
        }
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        mock_sm_cls.return_value = sm

        mock_review_loop.return_value = ReviewVerdict(
            verdict="changes_requested", issues=["fix x"], summary="needs work",
        )

        run_with_review("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "review")

    @patch("myswat.workflow.review_loop.run_review_loop")
    @patch("myswat.cli.run_cmd.MySwatSettings")
    @patch("myswat.cli.run_cmd.TiDBPool")
    @patch("myswat.cli.run_cmd.run_migrations")
    @patch("myswat.cli.run_cmd.MemoryStore")
    @patch("myswat.cli.run_cmd.SessionManager")
    @patch("myswat.cli.run_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_exception_during_review_loop(self, mock_learn, mock_comp, mock_sm_cls,
                                           mock_store_cls, mock_mig, mock_pool_cls,
                                           mock_settings_cls, mock_review_loop):
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        agent_row = {
            "id": 1, "role": "developer", "display_name": "Dev",
            "cli_backend": "codex", "model_name": "gpt-5",
            "cli_path": "codex", "cli_extra_args": None,
        }
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        mock_sm_cls.return_value = sm

        mock_review_loop.side_effect = RuntimeError("review crash")

        run_with_review("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "blocked")
