"""Tests for myswat.cli.chat_cmd."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import typer
from click.exceptions import Exit as ClickExit
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import FileHistory

from myswat.agents.base import AgentResponse
from myswat.cli.progress import _check_esc
from myswat.server.control_client import DaemonClientError
from myswat.server.mcp_http_client import MCPHTTPClientError
from myswat.workflow.engine import WorkMode
from myswat.cli.chat_cmd import (
    _show_status,
    _run_ga_test_interactive,
    _run_inline_review,
    _run_inline_review_interactive,
    _run_prd_workflow_interactive,
    _run_design_review,
    _run_design_review_interactive,
    _run_testplan_review,
    _run_testplan_review_interactive,
    _run_workflow,
    _run_workflow_interactive,
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
# daemon-backed workflow helpers
# ---------------------------------------------------------------------------
class TestRunInlineReview:
    @patch("myswat.cli.chat_cmd._run_workflow")
    def test_delegates_to_develop_workflow(self, mock_run_workflow):
        created = []

        _run_inline_review(
            MagicMock(),
            _proj(),
            "/tmp",
            MagicMock(),
            "do stuff",
            should_cancel=lambda: True,
            on_work_item_created=created.append,
            register_cancel_target=MagicMock(),
            initial_process_events=[{"event_type": "ignored"}],
        )

        kwargs = mock_run_workflow.call_args.kwargs
        assert kwargs["requirement"] == "do stuff"
        assert kwargs["mode"] == WorkMode.develop
        assert kwargs["should_cancel"]()
        kwargs["on_work_item_created"](42)
        assert created == [42]


class TestRunInlineReviewInteractive:
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    def test_uses_task_monitor(self, mock_run_workflow_interactive):
        _run_inline_review_interactive(
            store=MagicMock(),
            proj=_proj(),
            workdir="/tmp",
            settings=MagicMock(),
            task="do stuff",
        )

        kwargs = mock_run_workflow_interactive.call_args.kwargs
        assert kwargs["label"] == "Running development workflow"
        assert kwargs["mode"] == WorkMode.develop


class TestRunGATestInteractive:
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    def test_uses_task_monitor(self, mock_run_workflow_interactive):
        _run_ga_test_interactive(
            store=MagicMock(),
            proj=_proj(),
            workdir="/tmp",
            settings=MagicMock(),
            task="test thing",
        )

        kwargs = mock_run_workflow_interactive.call_args.kwargs
        assert kwargs["label"] == "Running GA test workflow"
        assert kwargs["mode"] == WorkMode.test


class TestRunDesignReview:
    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_workflow")
    def test_missing_architect_session(self, mock_run_workflow, mock_console_print):
        _run_design_review(MagicMock(), _proj(), None, "/tmp", MagicMock(), "do stuff")
        mock_run_workflow.assert_not_called()
        assert any("Missing architect session." in str(call) for call in mock_console_print.call_args_list)

    @patch("myswat.cli.chat_cmd._run_workflow")
    def test_delegates_architect_design_mode(self, mock_run_workflow):
        proposer_sm = MagicMock()

        _run_design_review(
            MagicMock(),
            _proj(),
            proposer_sm,
            "/tmp",
            MagicMock(),
            "do stuff",
            should_cancel=lambda: False,
        )

        kwargs = mock_run_workflow.call_args.kwargs
        assert kwargs["mode"] == WorkMode.architect_design
        assert kwargs["proposer_sm"] is proposer_sm
        assert kwargs["requirement"] == "do stuff"


class TestRunTestplanReview:
    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_workflow")
    def test_missing_qa_session(self, mock_run_workflow, mock_console_print):
        _run_testplan_review(MagicMock(), _proj(), None, "/tmp", MagicMock(), "do stuff")
        mock_run_workflow.assert_not_called()
        assert any("Missing QA session." in str(call) for call in mock_console_print.call_args_list)

    @patch("myswat.cli.chat_cmd._run_workflow")
    def test_delegates_testplan_mode(self, mock_run_workflow):
        proposer_sm = MagicMock()

        _run_testplan_review(
            MagicMock(),
            _proj(),
            proposer_sm,
            "/tmp",
            MagicMock(),
            "plan tests",
            should_cancel=lambda: True,
            auto_approve=False,
        )

        kwargs = mock_run_workflow.call_args.kwargs
        assert kwargs["mode"] == WorkMode.testplan_design
        assert kwargs["proposer_sm"] is proposer_sm
        assert kwargs["requirement"] == "plan tests"
        assert kwargs["should_cancel"]()


class TestRunDesignReviewInteractive:
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    def test_uses_task_monitor(self, mock_run_workflow_interactive):
        proposer_sm = MagicMock()

        _run_design_review_interactive(
            store=MagicMock(),
            proj=_proj(),
            proposer_sm=proposer_sm,
            workdir="/tmp",
            settings=MagicMock(),
            task="design thing",
            prompt_session=MagicMock(),
        )

        kwargs = mock_run_workflow_interactive.call_args.kwargs
        assert kwargs["label"] == "Running design workflow"
        assert kwargs["mode"] == WorkMode.architect_design
        assert kwargs["proposer_sm"] is proposer_sm


class TestRunTestplanReviewInteractive:
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    def test_uses_task_monitor(self, mock_run_workflow_interactive):
        proposer_sm = MagicMock()

        _run_testplan_review_interactive(
            store=MagicMock(),
            proj=_proj(),
            proposer_sm=proposer_sm,
            workdir="/tmp",
            settings=MagicMock(),
            task="plan tests",
            prompt_session=MagicMock(),
        )

        kwargs = mock_run_workflow_interactive.call_args.kwargs
        assert kwargs["label"] == "Running QA test-plan workflow"
        assert kwargs["mode"] == WorkMode.testplan_design
        assert kwargs["proposer_sm"] is proposer_sm


class TestRunPRDWorkflowInteractive:
    @patch("myswat.cli.chat_cmd._send_with_timer")
    def test_persists_approved_prd_artifact(self, mock_send_with_timer):
        store = MagicMock()
        store.create_work_item.return_value = 41
        store.create_artifact.return_value = 99
        proposer_sm = MagicMock()
        proposer_sm.agent_id = 7
        prompt_session = MagicMock()
        prompt_session.prompt.return_value = ""
        mock_send_with_timer.return_value = (
            AgentResponse(
                content=(
                    "Here is the draft.\n"
                    "```prd\n"
                    "# PRD: Billing Revamp\n\n"
                    "## Problem Statement\n\n"
                    "Legacy billing is brittle.\n"
                    "```"
                ),
                exit_code=0,
            ),
            1.5,
        )

        _run_prd_workflow_interactive(
            store=store,
            proj=_proj(),
            proposer_sm=proposer_sm,
            workdir="/tmp",
            settings=MagicMock(),
            task="Write a PRD for billing",
            prompt_session=prompt_session,
        )

        first_prompt = mock_send_with_timer.call_args.args[2]
        assert "interactive PRD workflow" in first_prompt
        store.create_work_item.assert_called_once()
        store.create_artifact.assert_called_once_with(
            work_item_id=41,
            agent_id=7,
            iteration=1,
            artifact_type="prd_doc",
            title="PRD: Billing Revamp",
            content="# PRD: Billing Revamp\n\n## Problem Statement\n\nLegacy billing is brittle.",
            metadata_json={"approved": True, "source": "chat_prd", "work_mode": WorkMode.prd.value},
        )
        store.update_work_item_status.assert_any_call(41, "approved")

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._send_with_timer", side_effect=RuntimeError("boom"))
    def test_blocks_work_item_on_unexpected_exception(
        self,
        mock_send_with_timer,
        mock_console_print,
    ):
        store = MagicMock()
        store.create_work_item.return_value = 41
        proposer_sm = MagicMock()
        proposer_sm.agent_id = 7

        _run_prd_workflow_interactive(
            store=store,
            proj=_proj(),
            proposer_sm=proposer_sm,
            workdir="/tmp",
            settings=MagicMock(),
            task="Write a PRD for billing",
            prompt_session=MagicMock(),
        )

        mock_send_with_timer.assert_called_once()
        store.update_work_item_status.assert_any_call(41, "blocked")
        assert any("PRD workflow crashed" in str(call) for call in mock_console_print.call_args_list)


class TestRunWorkflow:
    @patch("myswat.cli.chat_cmd._run_remote_workflow")
    def test_missing_dev_agent(self, mock_run_remote):
        store = MagicMock()
        store.get_agent.return_value = None

        _run_workflow(store, _proj(), "/tmp", MagicMock(), "do stuff")

        mock_run_remote.assert_not_called()

    @patch("myswat.cli.chat_cmd._run_remote_workflow")
    def test_missing_qa_agents(self, mock_run_remote):
        store = MagicMock()
        dev_agent = _agent_row("developer")

        def get_agent_side(_pid, role):
            if role == "developer":
                return dev_agent
            return None

        store.get_agent.side_effect = get_agent_side

        _run_workflow(store, _proj(), "/tmp", MagicMock(), "do stuff")

        mock_run_remote.assert_not_called()

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_remote_workflow")
    def test_missing_architect_for_architect_design_mode(self, mock_run_remote, mock_print):
        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(_pid, role):
            return {
                "developer": dev_agent,
                "qa_main": qa_agent,
                "architect": None,
            }.get(role)

        store.get_agent.side_effect = get_agent_side

        _run_workflow(store, _proj(), "/tmp", MagicMock(), "design auth", mode=WorkMode.architect_design)

        mock_run_remote.assert_not_called()
        assert any("Missing architect agent." in str(call) for call in mock_print.call_args_list)

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_remote_workflow")
    def test_missing_architect_or_developer_for_testplan_mode(self, mock_run_remote, mock_print):
        store = MagicMock()
        dev_agent = _agent_row("developer")

        def get_agent_side(_pid, role):
            return {"developer": dev_agent, "architect": None}.get(role)

        store.get_agent.side_effect = get_agent_side

        _run_workflow(
            store,
            _proj(),
            "/tmp",
            MagicMock(),
            "plan tests",
            mode=WorkMode.testplan_design,
            proposer_sm=MagicMock(),
        )

        mock_run_remote.assert_not_called()
        assert any("Missing architect or developer agent." in str(call) for call in mock_print.call_args_list)

    @patch("myswat.cli.chat_cmd._run_remote_workflow")
    def test_delegates_to_remote_workflow(self, mock_run_remote):
        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(_pid, role):
            return {"developer": dev_agent, "qa_main": qa_agent}.get(role)

        store.get_agent.side_effect = get_agent_side
        created = []

        _run_workflow(
            store,
            _proj(),
            "/tmp",
            MagicMock(),
            "do stuff",
            should_cancel=lambda: False,
            on_work_item_created=created.append,
        )

        kwargs = mock_run_remote.call_args.kwargs
        assert kwargs["mode"] == WorkMode.full
        assert kwargs["requirement"] == "do stuff"
        assert kwargs["should_cancel"]() is False
        kwargs["on_work_item_created"](42)
        assert created == [42]

    @patch("myswat.cli.chat_cmd._print_daemon_error")
    @patch("myswat.cli.chat_cmd._run_remote_workflow", side_effect=DaemonClientError("boom"))
    def test_reports_daemon_errors(self, mock_run_remote, mock_print_daemon_error):
        store = MagicMock()
        dev_agent = _agent_row("developer")
        qa_agent = _agent_row("qa_main", "kimi")

        def get_agent_side(_pid, role):
            return {"developer": dev_agent, "qa_main": qa_agent}.get(role)

        store.get_agent.side_effect = get_agent_side

        _run_workflow(store, _proj(), "/tmp", MagicMock(), "do stuff")

        mock_run_remote.assert_called_once()
        mock_print_daemon_error.assert_called_once()


class TestRunWorkflowInteractive:
    @patch("myswat.cli.chat_cmd._run_with_task_monitor")
    @patch("myswat.cli.chat_cmd._run_workflow")
    def test_uses_task_monitor(self, mock_run_workflow, mock_task_monitor):
        def _run_monitor(**kwargs):
            kwargs["worker_fn"]()
            assert kwargs["work_item_ref"]["id"] == 66
            assert kwargs["cancel_targets"] == []

        def _run_workflow_side_effect(**kwargs):
            kwargs["on_work_item_created"](66)
            assert kwargs["mode"] == WorkMode.full
            assert kwargs["should_cancel"]() is False

        mock_task_monitor.side_effect = _run_monitor
        mock_run_workflow.side_effect = _run_workflow_side_effect

        _run_workflow_interactive(
            store=MagicMock(),
            proj=_proj(),
            workdir="/tmp",
            settings=MagicMock(),
            requirement="ship feature",
            prompt_session=MagicMock(),
        )

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

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_chat_uses_emacs_prompt_session(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
    ):
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

        kwargs = mock_prompt_session_cls.call_args.kwargs
        assert kwargs["editing_mode"] == EditingMode.EMACS
        assert kwargs["enable_history_search"] is True
        assert isinstance(kwargs["history"], FileHistory)

    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_mig,
                                mock_pool_cls, mock_settings_cls,
                                mock_preload):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            from myswat.cli.chat_cmd import run_chat
            run_chat("missing")

    @patch("myswat.cli.chat_cmd._print_daemon_error")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_initial_daemon_session_open_failure_exits(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_print_daemon_error,
    ):
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
        sm.create_or_resume.side_effect = MCPHTTPClientError("daemon unavailable")
        mock_sm_cls.return_value = sm
        mock_prompt_session_cls.return_value = MagicMock()

        with pytest.raises(ClickExit):
            run_chat("proj")

        mock_print_daemon_error.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_quit_command(self, mock_sm_cls,
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

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.console.print")
    def test_help_command(self, mock_console_print, mock_sm_cls,
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

        printed = "\n".join(
            str(call.args[0]) for call in mock_console_print.call_args_list if call.args
        )
        assert "/task" in printed
        assert "/history" in printed
        assert "qa_vice" not in printed

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_status_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_role_switch(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_role_no_arg(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_agents_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_sessions_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_history_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_reset_command(self, mock_sm_cls,
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

    @patch("myswat.cli.chat_cmd._print_daemon_error")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_reset_command_reports_daemon_error(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_print_daemon_error,
    ):
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
        sm.reset_ai_session.side_effect = MCPHTTPClientError("reset failed")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/reset", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

        mock_print_daemon_error.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_new_command(self, mock_sm_cls,
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

    @patch("myswat.cli.chat_cmd._print_daemon_error")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_cleanup_close_error_is_best_effort(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_print_daemon_error,
    ):
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
        sm.close.side_effect = MCPHTTPClientError("close failed")
        mock_sm_cls.return_value = sm

        prompt_session = MagicMock()
        prompt_session.prompt.return_value = "/quit"
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

        mock_print_daemon_error.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_unknown_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_eof_exits(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_keyboard_interrupt(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_empty_input_skipped(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.submit_chat_learn_request")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_regular_message_success(self, mock_sm_cls,
                                      mock_store_cls, mock_mig, mock_pool_cls,
                                      mock_settings_cls, mock_submit_learn, mock_preload,
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
        mock_submit_learn.assert_called_once()

    @patch("myswat.cli.chat_cmd._print_daemon_error")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_regular_message_daemon_error_continues(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_print_daemon_error,
    ):
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

        mock_send_timer.side_effect = MCPHTTPClientError("send failed")

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["hello", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

        mock_print_daemon_error.assert_called_once()

    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_architect_delegation_auto_starts_develop_workflow(
        self,
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
        assert review_args[4] == "update the design doc"
        assert review_kwargs["mode"] == WorkMode.develop


    @patch("myswat.cli.chat_cmd._run_design_review_interactive")
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_architect_design_delegation_starts_design_workflow(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_design_review,
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
        mock_design_review.assert_called_once()
        assert mock_design_review.call_args.args[2] is sm

    @patch("myswat.cli.chat_cmd._run_testplan_review_interactive")
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_qa_testplan_delegation_starts_testplan_workflow(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_testplan_review,
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
        mock_testplan_review.assert_called_once()
        assert mock_testplan_review.call_args.args[2] is sm

    @patch("myswat.cli.chat_cmd._run_prd_workflow_interactive")
    @patch("myswat.cli.chat_cmd._run_full_workflow_interactive")
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_architect_full_delegation_starts_full_workflow(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_full_workflow,
        mock_prd_workflow,
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
                content="Sure\n```delegate\nMODE: full\nTASK: design and implement the auth module\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["finish the design and implementation", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="architect")

        mock_review.assert_not_called()
        mock_full_workflow.assert_called_once()
        mock_prd_workflow.assert_not_called()
        # The proposer_sm (3rd positional arg) should be the architect session
        assert mock_full_workflow.call_args.args[2] is sm
        # The requirement (7th positional arg) should be the task
        assert mock_full_workflow.call_args.args[5] == "design and implement the auth module"

    @patch("myswat.cli.chat_cmd._run_prd_workflow_interactive")
    @patch("myswat.cli.chat_cmd._run_full_workflow_interactive")
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_architect_prd_delegation_starts_prd_workflow(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_review,
        mock_full_workflow,
        mock_prd_workflow,
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
                content="Sure\n```delegate\nMODE: prd\nTASK: write a PRD for billing revamp\n```",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["write a proper PRD", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="architect")

        mock_review.assert_not_called()
        mock_full_workflow.assert_not_called()
        mock_prd_workflow.assert_called_once()
        assert mock_prd_workflow.call_args.args[2] is sm
        assert mock_prd_workflow.call_args.args[5] == "write a PRD for billing revamp"

    @patch("myswat.cli.chat_cmd._run_full_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_architect_team_request_is_sent_to_architect_instead_of_python_auto_routing(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_send_timer,
        mock_full_workflow,
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
                content="I will think through the approach first.",
                exit_code=0,
            ),
            2.0,
        )

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = [
            '"Design and implement the auth module with your team"',
            "/quit",
        ]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj", role="architect")

        mock_send_timer.assert_called_once()
        assert mock_send_timer.call_args.args[2] == "Design and implement the auth module with your team"
        mock_full_workflow.assert_not_called()

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_qa_develop_delegation_warns_for_unsupported_role(
        self,
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
            "Delegation mode 'develop' is only available for role(s): architect. Current role: 'qa_main'." in str(call)
            for call in mock_console_print.call_args_list
        )

    @patch("myswat.cli.chat_cmd.console.print")
    @patch("myswat.cli.chat_cmd._run_inline_review_interactive")
    @patch("myswat.cli.chat_cmd._send_with_timer")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_unknown_delegation_mode_warns_as_unsupported(
        self,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_regular_message_cancelled(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_regular_message_error(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_dev_command(self, mock_sm_cls,
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
        prompt_session.prompt.side_effect = ["/dev fix bug", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        mock_review.assert_called_once()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_dev_no_arg(self, mock_sm_cls,
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
        prompt_session.prompt.side_effect = ["/dev", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

    @patch("myswat.cli.chat_cmd._run_ga_test_interactive")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_ga_test_command(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_ga_test,
    ):
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
        prompt_session.prompt.side_effect = ["/ga-test verify bugfix", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")
        mock_ga_test.assert_called_once()

    @patch("myswat.cli.chat_cmd._run_workflow_interactive")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_work_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_task_command(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_work_no_arg(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_history_no_session(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_sessions_no_active(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_reset_no_session(self, mock_sm_cls,
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
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_history_with_n_param(self, mock_sm_cls,
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

    @patch("myswat.cli.chat_cmd._run_prd_workflow_interactive")
    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    def test_prd_command_switches_to_architect_and_restores(
        self,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
        mock_prd_workflow,
    ):
        """``/prd`` from a non-architect role temporarily switches to architect,
        runs the PRD workflow, then restores the original role."""
        from myswat.cli.chat_cmd import run_chat

        settings = self._setup_mocks()
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        arch_row = _agent_row("architect")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = _proj()
        mock_store.get_agent.side_effect = lambda _pid, role: {
            "developer": dev_row,
            "architect": arch_row,
        }.get(role, dev_row)
        mock_store.list_agents.return_value = [dev_row, arch_row]
        mock_store_cls.return_value = mock_store

        sm_instances = []

        def _make_sm(*args, **kwargs):
            sm = MagicMock()
            sm.session = SimpleNamespace(session_uuid=f"uuid-{len(sm_instances)}")
            sm.agent_id = len(sm_instances) + 1
            sm_instances.append(sm)
            return sm

        mock_sm_cls.side_effect = _make_sm

        prompt_session = MagicMock()
        prompt_session.prompt.side_effect = ["/prd billing revamp", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

        mock_prd_workflow.assert_called_once()
        prd_task_arg = mock_prd_workflow.call_args.args[5]
        assert prd_task_arg == "billing revamp"
        # The initial developer SM is closed before switching to architect.
        sm_instances[0].close.assert_called()

    @patch("myswat.cli.chat_cmd.PromptSession")
    @patch("myswat.cli.chat_cmd.preload_model")
    @patch("myswat.cli.chat_cmd.MySwatSettings")
    @patch("myswat.cli.chat_cmd.TiDBPool")
    @patch("myswat.cli.chat_cmd.ensure_schema")
    @patch("myswat.cli.chat_cmd.MemoryStore")
    @patch("myswat.cli.chat_cmd.SessionManager")
    @patch("myswat.cli.chat_cmd.console.print")
    def test_prd_no_arg(
        self,
        mock_console_print,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_preload,
        mock_prompt_session_cls,
    ):
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
        prompt_session.prompt.side_effect = ["/prd", "/quit"]
        mock_prompt_session_cls.return_value = prompt_session

        run_chat("proj")

        printed = "\n".join(
            str(call.args[0]) for call in mock_console_print.call_args_list if call.args
        )
        assert "Usage: /prd" in printed
