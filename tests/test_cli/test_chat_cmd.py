"""Tests for myswat.cli.chat_cmd."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.agents.base import AgentResponse
from myswat.cli.chat_cmd import (
    _make_compaction_runner,
    _show_status,
    _run_inline_review,
    _run_inline_review_interactive,
    _run_workflow,
    _run_workflow_interactive,
    _check_esc,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _agent_row(role="developer", backend="codex"):
    return {
        "id": 1, "role": role, "display_name": f"Agent-{role}",
        "cli_backend": backend, "model_name": "gpt-5",
        "cli_path": backend, "cli_extra_args": None,
    }


def _proj():
    return {"id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp"}


# ---------------------------------------------------------------------------
# _make_compaction_runner
# ---------------------------------------------------------------------------
class TestMakeCompactionRunner:
    def test_matching_backend(self):
        store = MagicMock()
        store.list_agents.return_value = [_agent_row("dev", "codex")]
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"

        r = _make_compaction_runner(store, _proj(), settings)
        assert r is not None

    def test_fallback(self):
        store = MagicMock()
        store.list_agents.return_value = [_agent_row("dev", "kimi")]
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"

        r = _make_compaction_runner(store, _proj(), settings)
        assert r is not None

    def test_no_agents(self):
        store = MagicMock()
        store.list_agents.return_value = []
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"

        r = _make_compaction_runner(store, _proj(), settings)
        assert r is None


# ---------------------------------------------------------------------------
# _check_esc
# ---------------------------------------------------------------------------
class TestCheckEsc:
    @patch("myswat.cli.progress.select")
    @patch("myswat.cli.progress.sys")
    def test_esc_pressed(self, mock_sys, mock_select):
        mock_select.select.return_value = ([True], [], [])
        mock_sys.stdin.read.return_value = "\x1b"
        assert _check_esc() is True

    @patch("myswat.cli.progress.select")
    def test_no_input(self, mock_select):
        mock_select.select.return_value = ([], [], [])
        assert _check_esc() is False

    @patch("myswat.cli.progress.select")
    def test_exception(self, mock_select):
        mock_select.select.side_effect = Exception("bad fd")
        assert _check_esc() is False

    @patch("myswat.cli.progress.select")
    @patch("myswat.cli.progress.sys")
    def test_non_esc_char(self, mock_sys, mock_select):
        mock_select.select.return_value = ([True], [], [])
        mock_sys.stdin.read.return_value = "a"
        assert _check_esc() is False


# ---------------------------------------------------------------------------
# _show_status
# ---------------------------------------------------------------------------
class TestShowStatus:
    def test_with_active_items(self):
        store = MagicMock()
        store.list_work_items.return_value = [
            {"id": 1, "status": "in_progress", "title": "Do stuff"},
        ]
        pool = MagicMock()
        proj = _proj()

        _show_status(store, pool, proj)
        store.list_work_items.assert_called_once()

    def test_no_active_items(self):
        store = MagicMock()
        store.list_work_items.return_value = [
            {"id": 1, "status": "completed", "title": "Done"},
        ]
        pool = MagicMock()
        proj = _proj()

        _show_status(store, pool, proj)

    def test_empty_items(self):
        store = MagicMock()
        store.list_work_items.return_value = []
        pool = MagicMock()
        proj = _proj()

        _show_status(store, pool, proj)


# ---------------------------------------------------------------------------
# _run_inline_review
# ---------------------------------------------------------------------------
class TestRunInlineReview:
    @patch("myswat.workflow.review_loop.run_review_loop")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_missing_agents(self, mock_sm_cls, mock_review):
        store = MagicMock()
        store.get_agent.return_value = None
        compactor = MagicMock()

        _run_inline_review(store, _proj(), compactor, "/tmp", MagicMock(), "do stuff")
        mock_review.assert_not_called()

    @patch("myswat.workflow.review_loop.run_review_loop")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_lgtm_verdict(self, mock_sm_cls, mock_review):
        from myswat.models.work_item import ReviewVerdict

        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_agent
            if role == "qa_main":
                return qa_agent
            return None
        store.get_agent.side_effect = get_agent_side
        store.create_work_item.return_value = 42

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        mock_sm_cls.return_value = sm

        settings = MagicMock()
        settings.workflow.max_review_iterations = 5

        mock_review.return_value = ReviewVerdict(
            verdict="lgtm", issues=[], summary="ok",
        )
        compactor = MagicMock()

        _run_inline_review(store, _proj(), compactor, "/tmp", settings, "do stuff")
        store.update_work_item_status.assert_any_call(42, "approved")

    @patch("myswat.workflow.review_loop.run_review_loop")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_changes_requested(self, mock_sm_cls, mock_review):
        from myswat.models.work_item import ReviewVerdict

        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_agent
            if role == "qa_main":
                return qa_agent
            return None
        store.get_agent.side_effect = get_agent_side
        store.create_work_item.return_value = 42

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        mock_sm_cls.return_value = sm

        settings = MagicMock()
        settings.workflow.max_review_iterations = 5

        mock_review.return_value = ReviewVerdict(
            verdict="changes_requested", issues=["fix x"], summary="needs work",
        )
        compactor = MagicMock()

        _run_inline_review(store, _proj(), compactor, "/tmp", settings, "do stuff")
        store.update_work_item_status.assert_any_call(42, "review")


class TestRunInlineReviewInteractive:
    @patch("myswat.cli.chat_cmd._run_with_task_monitor")
    def test_uses_task_monitor(self, mock_task_monitor):
        _run_inline_review_interactive(
            store=MagicMock(),
            proj=_proj(),
            compactor=MagicMock(),
            workdir="/tmp",
            settings=MagicMock(),
            task="do stuff",
        )

        mock_task_monitor.assert_called_once()
        kwargs = mock_task_monitor.call_args.kwargs
        assert kwargs["label"] == "Running dev+QA review loop"
        assert kwargs["proj"] == _proj()


# ---------------------------------------------------------------------------
# _run_workflow
# ---------------------------------------------------------------------------
class TestRunWorkflow:
    @patch("myswat.workflow.engine.WorkflowEngine")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_missing_dev_agent(self, mock_sm_cls, mock_engine_cls):
        store = MagicMock()
        store.get_agent.return_value = None
        compactor = MagicMock()

        _run_workflow(store, _proj(), compactor, "/tmp", MagicMock(), "do stuff")
        mock_engine_cls.assert_not_called()

    @patch("myswat.workflow.engine.WorkflowEngine")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_missing_qa_agents(self, mock_sm_cls, mock_engine_cls):
        store = MagicMock()
        dev_agent = _agent_row("developer")

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_agent
            return None
        store.get_agent.side_effect = get_agent_side
        compactor = MagicMock()

        _run_workflow(store, _proj(), compactor, "/tmp", MagicMock(), "do stuff")
        mock_engine_cls.assert_not_called()

    @patch("myswat.workflow.engine.WorkflowEngine")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_success(self, mock_sm_cls, mock_engine_cls):
        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_agent
            if role == "qa_main":
                return qa_agent
            return None
        store.get_agent.side_effect = get_agent_side
        store.create_work_item.return_value = 42

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_agent
        mock_sm_cls.return_value = sm

        settings = MagicMock()
        settings.workflow.max_review_iterations = 5

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        compactor = MagicMock()

        _run_workflow(store, _proj(), compactor, "/tmp", settings, "do stuff")
        store.update_work_item_status.assert_any_call(42, "completed")

    @patch("myswat.workflow.engine.WorkflowEngine")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_failure(self, mock_sm_cls, mock_engine_cls):
        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_agent
            if role == "qa_main":
                return qa_agent
            return None
        store.get_agent.side_effect = get_agent_side
        store.create_work_item.return_value = 42

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_agent
        mock_sm_cls.return_value = sm

        settings = MagicMock()
        settings.workflow.max_review_iterations = 5

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=False)
        mock_engine_cls.return_value = engine

        compactor = MagicMock()

        _run_workflow(store, _proj(), compactor, "/tmp", settings, "do stuff")
        store.update_work_item_status.assert_any_call(42, "review")

    @patch("myswat.workflow.engine.WorkflowEngine")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_with_ask_user_callback(self, mock_sm_cls, mock_engine_cls):
        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_agent
            if role == "qa_main":
                return qa_agent
            return None
        store.get_agent.side_effect = get_agent_side
        store.create_work_item.return_value = 42

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_agent
        mock_sm_cls.return_value = sm

        settings = MagicMock()
        settings.workflow.max_review_iterations = 5

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        compactor = MagicMock()
        prompt_session = MagicMock()

        _run_workflow(
            store, _proj(), compactor, "/tmp", settings, "do stuff",
            prompt_session=prompt_session,
        )


class TestRunWorkflowInteractive:
    @patch("myswat.cli.chat_cmd._run_with_task_monitor")
    def test_uses_task_monitor(self, mock_task_monitor):
        prompt_session = MagicMock()

        _run_workflow_interactive(
            store=MagicMock(),
            proj=_proj(),
            compactor=MagicMock(),
            workdir="/tmp",
            settings=MagicMock(),
            requirement="ship feature",
            prompt_session=prompt_session,
        )

        mock_task_monitor.assert_called_once()
        kwargs = mock_task_monitor.call_args.kwargs
        assert kwargs["label"] == "Running full teamwork workflow"
        assert kwargs["proj"] == _proj()


# ---------------------------------------------------------------------------
# run_chat (REPL) — test slash command paths via mocking
# ---------------------------------------------------------------------------
class TestRunChat:
    def _setup_mocks(self):
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        settings.workflow.max_review_iterations = 5
        return settings

    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_project_not_found(self, mock_learn, mock_store_cls, mock_mig,
                                mock_pool_cls, mock_settings_cls,
                                mock_preload):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            from myswat.cli.chat_cmd import run_chat
            run_chat("missing")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_quit_command(self, mock_learn, mock_comp, mock_sm_cls,
                          mock_store_cls, mock_mig, mock_pool_cls,
                          mock_settings_cls, mock_preload,
                          mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.return_value = "/quit"
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        sm.close.assert_called()
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["threshold_turns"] == 200
        assert "threshold_tokens" not in kwargs

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_help_command(self, mock_learn, mock_comp, mock_sm_cls,
                          mock_store_cls, mock_mig, mock_pool_cls,
                          mock_settings_cls, mock_preload,
                          mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/help", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_status_command(self, mock_learn, mock_comp, mock_sm_cls,
                            mock_store_cls, mock_mig, mock_pool_cls,
                            mock_settings_cls, mock_preload,
                            mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store.list_work_items.return_value = []
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/status", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_role_switch(self, mock_learn, mock_comp, mock_sm_cls,
                         mock_store_cls, mock_mig, mock_pool_cls,
                         mock_settings_cls, mock_preload,
                         mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/role architect", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_role_no_arg(self, mock_learn, mock_comp, mock_sm_cls,
                          mock_store_cls, mock_mig, mock_pool_cls,
                          mock_settings_cls, mock_preload,
                          mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/role", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_agents_command(self, mock_learn, mock_comp, mock_sm_cls,
                            mock_store_cls, mock_mig, mock_pool_cls,
                            mock_settings_cls, mock_preload,
                            mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/agents", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_sessions_command(self, mock_learn, mock_comp, mock_sm_cls,
                              mock_store_cls, mock_mig, mock_pool_cls,
                              mock_settings_cls, mock_preload,
                              mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.return_value = [
            {
                "session_uuid": "uuid-1234", "role": "developer",
                "display_name": "Dev", "id": 1, "purpose": "test",
            }
        ]
        mock_pool_cls.return_value = pool

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/sessions", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_history_command(self, mock_learn, mock_comp, mock_sm_cls,
                             mock_store_cls, mock_mig, mock_pool_cls,
                             mock_settings_cls, mock_preload,
                             mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store.get_session_turns.return_value = [
            SimpleNamespace(role="user", content="hello"),
            SimpleNamespace(role="assistant", content="hi"),
        ]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234", id=1)
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/history", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_reset_command(self, mock_learn, mock_comp, mock_sm_cls,
                           mock_store_cls, mock_mig, mock_pool_cls,
                           mock_settings_cls, mock_preload,
                           mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/reset", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        sm.reset_ai_session.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_new_command(self, mock_learn, mock_comp, mock_sm_cls,
                         mock_store_cls, mock_mig, mock_pool_cls,
                         mock_settings_cls, mock_preload,
                         mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/new", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_unknown_command(self, mock_learn, mock_comp, mock_sm_cls,
                             mock_store_cls, mock_mig, mock_pool_cls,
                             mock_settings_cls, mock_preload,
                             mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/foobar", "/exit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_eof_exits(self, mock_learn, mock_comp, mock_sm_cls,
                       mock_store_cls, mock_mig, mock_pool_cls,
                       mock_settings_cls, mock_preload,
                       mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = EOFError()
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_keyboard_interrupt(self, mock_learn, mock_comp, mock_sm_cls,
                                mock_store_cls, mock_mig, mock_pool_cls,
                                mock_settings_cls, mock_preload,
                                mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = [KeyboardInterrupt(), EOFError()]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_empty_input_skipped(self, mock_learn, mock_comp, mock_sm_cls,
                                  mock_store_cls, mock_mig, mock_pool_cls,
                                  mock_settings_cls, mock_preload,
                                  mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["", "  ", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        sm.send.assert_not_called()

    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_regular_message_success(self, mock_learn, mock_comp, mock_sm_cls,
                                      mock_store_cls, mock_mig, mock_pool_cls,
                                      mock_settings_cls, mock_preload,
                                      mock_prompt_session_cls,
                                      mock_send_timer):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(content="hello back", exit_code=0), 2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        mock_send_timer.assert_called_once()

    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_architect_delegation_auto_starts_review_loop(
        self,
        mock_learn,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
    ):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        architect_row = _agent_row("architect")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = architect_row
        mock_store.list_agents.return_value = [architect_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        sm.agent_id = 9
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(
                content="Plan\n```delegate\nTASK: update the design doc\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="architect")

        mock_review.assert_called_once()
        review_args = mock_review.call_args.args
        review_kwargs = mock_review.call_args.kwargs
        assert review_args[5] == "update the design doc"
        assert review_kwargs["initial_process_events"][0]["from_role"] == "architect"
        assert review_kwargs["initial_process_events"][0]["to_role"] == "developer"

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_architect_design_delegation_warns_until_supported(
        self,
        mock_learn,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_console_print,
    ):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        architect_row = _agent_row("architect")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = architect_row
        mock_store.list_agents.return_value = [architect_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        sm.agent_id = 9
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(
                content="Plan\n```delegate\nMODE: design\nTASK: update the design doc\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="architect")

        mock_review.assert_not_called()
        sm.close.assert_called_once()
        assert any(
            "not available yet" in str(call)
            for call in mock_console_print.call_args_list
        )

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_qa_testplan_delegation_warns_until_supported(
        self,
        mock_learn,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_console_print,
    ):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        qa_row = _agent_row("qa_main")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = qa_row
        mock_store.list_agents.return_value = [qa_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        sm.agent_id = 9
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(
                content="Plan\n```delegate\nMODE: testplan\nTASK: finalize the test plan\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="qa_main")

        mock_review.assert_not_called()
        sm.close.assert_called_once()
        assert any(
            "not available yet" in str(call)
            for call in mock_console_print.call_args_list
        )


    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_qa_code_delegation_warns_for_unsupported_role(
        self,
        mock_learn,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_console_print,
    ):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        qa_row = _agent_row("qa_main")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = qa_row
        mock_store.list_agents.return_value = [qa_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        sm.agent_id = 9
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(
                content="Plan\n```delegate\nTASK: implement the fix\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="qa_main")

        mock_review.assert_not_called()
        sm.close.assert_called_once()
        assert any(
            "not available for role 'qa_main' yet" in str(call)
            for call in mock_console_print.call_args_list
        )

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_unknown_delegation_mode_warns_as_unsupported(
        self,
        mock_learn,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_console_print,
    ):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        architect_row = _agent_row("architect")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = architect_row
        mock_store.list_agents.return_value = [architect_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        sm.agent_id = 9
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(
                content="Plan\n```delegate\nMODE: unknown_mode\nTASK: investigate\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="architect")

        mock_review.assert_not_called()
        sm.close.assert_called_once()
        assert any(
            "not supported" in str(call)
            for call in mock_console_print.call_args_list
        )
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_regular_message_cancelled(self, mock_learn, mock_comp, mock_sm_cls,
                                        mock_store_cls, mock_mig, mock_pool_cls,
                                        mock_settings_cls, mock_preload,
                                        mock_prompt_session_cls,
                                        mock_send_timer):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(content="cancelled", exit_code=-1, cancelled=True), 1.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_regular_message_error(self, mock_learn, mock_comp, mock_sm_cls,
                                    mock_store_cls, mock_mig, mock_pool_cls,
                                    mock_settings_cls, mock_preload,
                                    mock_prompt_session_cls,
                                    mock_send_timer):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        mock_send_timer.return_value = (
            AgentResponse(content="error", exit_code=1, raw_stderr="oops"), 2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_review_command(self, mock_learn, mock_comp, mock_sm_cls,
                            mock_store_cls, mock_mig, mock_pool_cls,
                            mock_settings_cls, mock_preload,
                            mock_prompt_session_cls, mock_review):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/review fix bug", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        mock_review.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_review_no_arg(self, mock_learn, mock_comp, mock_sm_cls,
                           mock_store_cls, mock_mig, mock_pool_cls,
                           mock_settings_cls, mock_preload,
                           mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/review", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_work_command(self, mock_learn, mock_comp, mock_sm_cls,
                          mock_store_cls, mock_mig, mock_pool_cls,
                          mock_settings_cls, mock_preload,
                          mock_prompt_session_cls, mock_workflow):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/work add feature", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        mock_workflow.assert_called_once()

    @patch("myswat.cli.chat_cmd._show_task_details")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_task_command(self, mock_learn, mock_comp, mock_sm_cls,
                          mock_store_cls, mock_mig, mock_pool_cls,
                          mock_settings_cls, mock_preload,
                          mock_prompt_session_cls, mock_show_task):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/task 42", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        mock_show_task.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_work_no_arg(self, mock_learn, mock_comp, mock_sm_cls,
                          mock_store_cls, mock_mig, mock_pool_cls,
                          mock_settings_cls, mock_preload,
                          mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/work", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_history_no_session(self, mock_learn, mock_comp, mock_sm_cls,
                                mock_store_cls, mock_mig, mock_pool_cls,
                                mock_settings_cls, mock_preload,
                                mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        # Return sm with a valid session, but test /history when no turns
        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="test-uuid", id=1)
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/history", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_sessions_no_active(self, mock_learn, mock_comp, mock_sm_cls,
                                 mock_store_cls, mock_mig, mock_pool_cls,
                                 mock_settings_cls, mock_preload,
                                 mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.return_value = []
        mock_pool_cls.return_value = pool

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/sessions", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_reset_no_session(self, mock_learn, mock_comp, mock_sm_cls,
                               mock_store_cls, mock_mig, mock_pool_cls,
                               mock_settings_cls, mock_preload,
                               mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        # Agent not found on first _switch_agent, will set sm = None
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        # First call returns agent, second returns None for /reset path
        agent_row = _agent_row()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store_cls.return_value = mock_store

        # sm that returns falsy for reset condition
        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234")
        sm.__bool__ = lambda self: False  # Makes `if sm:` evaluate False
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/reset", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.run_migrations")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.KnowledgeCompactor")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    def test_history_with_n_param(self, mock_learn, mock_comp, mock_sm_cls,
                                   mock_store_cls, mock_mig, mock_pool_cls,
                                   mock_settings_cls, mock_preload,
                                   mock_prompt_session_cls):
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        agent_row = _agent_row()
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.return_value = agent_row
        mock_store.list_agents.return_value = [agent_row]
        mock_store.get_session_turns.return_value = [
            SimpleNamespace(role="user", content="a" * 300),
        ]
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid-1234", id=1)
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/history 5", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
