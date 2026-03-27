"""Tests for myswat.cli.main — status command and _print_teamwork_details."""

from __future__ import annotations

import io
import re
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit
from rich.console import Console

from myswat.cli.main import (
    _display_mode,
    _follow_work_item_until_terminal,
    _infer_stage_labels,
    _print_teamwork_details,
)
from myswat.server.control_client import DaemonClientError
from myswat.workflow.engine import WorkMode


# ---------------------------------------------------------------------------
# _print_teamwork_details
# ---------------------------------------------------------------------------
class TestPrintTeamworkDetails:
    def test_with_review_cycles(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "changes_requested",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
                {
                    "iteration": 2, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
            ],
            [
                {
                    "artifact_type": "proposal", "title": "Design",
                    "iteration": 1, "created_at": "2026-03-07",
                    "agent_role": "developer", "agent_name": "Dev",
                },
            ],
        ]

        item = {
            "id": 1, "title": "Implement auth", "status": "completed",
        }
        console = MagicMock()

        _print_teamwork_details(pool, item, console)
        console.print.assert_called()

    def test_with_no_review_cycles(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [],  # no cycles
            [],  # no artifacts
        ]

        item = {
            "id": 1, "title": "Simple task", "status": "in_progress",
        }
        console = MagicMock()

        _print_teamwork_details(pool, item, console)
        console.print.assert_called()

    def test_review_rounds_grouping(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
                {
                    "iteration": 2, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "test_plan",
                    "proposer_role": "qa_main",
                    "reviewer_role": "developer",
                    "artifact_title": "Test plan",
                    "artifact_type": "test_plan",
                },
            ],
            [],  # artifacts
        ]

        item = {
            "id": 1, "title": "Complex task", "status": "completed",
        }
        console = MagicMock()

        _print_teamwork_details(pool, item, console)

    def test_multiple_iterations_same_pair(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "changes_requested",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
                {
                    "iteration": 2, "verdict": "changes_requested",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
                {
                    "iteration": 3, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
            ],
            [],  # artifacts
        ]

        item = {
            "id": 1, "title": "Iterated task", "status": "completed",
        }
        console = MagicMock()

        _print_teamwork_details(pool, item, console)

    def test_details_mode_shows_message_flow(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "changes_requested",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Iteration 1",
                    "artifact_type": "proposal",
                    "verdict_json": '{"verdict":"changes_requested","summary":"Needs more detail","issues":["add phase scope"]}',
                },
            ],
            [],
        ]

        item = {
            "id": 1,
            "title": "Implement auth",
            "status": "completed",
            "description": "Architect request to developer",
        }
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        _print_teamwork_details(pool, item, console, details=True)

        rendered = output.getvalue()
        assert "Message Flow" in rendered
        assert "Developer -> QA" in rendered
        assert "QA -> Developer" in rendered
        assert "CHANGES REQUESTED" in rendered

    def test_architect_review_flow_uses_role_labels(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "design",
                    "proposer_role": "architect",
                    "reviewer_role": "developer",
                    "artifact_title": "Design",
                    "artifact_type": "proposal",
                },
            ],
            [],
        ]
        item = {"id": 1, "title": "Design task", "status": "completed"}
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        _print_teamwork_details(pool, item, console)

        rendered = output.getvalue()
        assert "Architect -> Developer" in rendered
        assert "LGTM" in rendered

    def test_test_plan_flow_with_architect_reviewer_rendered(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "updated_at": "2026-03-07",
                    "completed_at": "2026-03-07",
                    "stage_name": "test_plan",
                    "proposer_role": "qa_main",
                    "reviewer_role": "architect",
                    "artifact_title": "Test plan",
                    "artifact_type": "test_plan",
                },
            ],
            [],
        ]
        item = {"id": 2, "title": "Test plan task", "status": "completed"}
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        _print_teamwork_details(pool, item, console)

        rendered = output.getvalue()
        assert "QA -> Architect" in rendered
        assert "Test plan" in rendered

    def test_artifacts_table_renders(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [],  # cycles
            [
                {
                    "artifact_type": "design_doc",
                    "title": "Technical design",
                    "iteration": 2,
                    "created_at": "2026-03-07",
                    "agent_role": "architect",
                    "agent_name": "Architect",
                },
            ],
        ]

        item = {
            "id": 1, "title": "Small task", "status": "completed",
        }
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        _print_teamwork_details(pool, item, console)

        rendered = output.getvalue()
        assert "Artifacts" in rendered
        assert "Technical design" in rendered


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------
class TestStatusCommand:
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "missing"])
        assert result.exit_code == 1

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_agents_and_items(self, mock_store_cls, mock_pool_cls,
                                    mock_settings_cls, mock_teamwork):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp",
        }
        mock_store.list_agents.return_value = [
            {"role": "developer", "model_name": "gpt-5", "cli_backend": "codex"},
        ]
        mock_store.list_work_items.return_value = [
            {
                "id": 1, "status": "completed", "item_type": "code_change",
                "title": "Do stuff", "assigned_agent_id": 1,
            },
        ]
        mock_store.count_session_turns.return_value = 5
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        # Cycles query for work mode detection
        pool.fetch_all.side_effect = [
            [],  # No cycles = solo mode
            [],  # Active sessions
        ]
        pool.fetch_one.side_effect = [
            {"role": "developer"},  # Agent role for solo mode
            {"cnt": 10},  # Knowledge count
            {"cnt": 0},   # Compacted sessions count
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj"])
        assert result.exit_code == 0

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_default_status_is_compact_and_shows_current_item_alerts_and_recent_messages(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
    ):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1,
            "slug": "proj",
            "name": "Proj",
            "repo_path": "/tmp/repo",
        }
        mock_store.list_work_items.return_value = [
            {
                "id": 1,
                "status": "completed",
                "item_type": "code_change",
                "title": "Older task",
                "metadata_json": {
                    "task_state": {
                        "current_stage": "report",
                        "process_log": [],
                    }
                },
            },
            {
                "id": 2,
                "status": "in_progress",
                "item_type": "code_change",
                "title": "Current task",
                "metadata_json": {
                    "task_state": {
                        "current_stage": "phase_1",
                        "latest_summary": "Implementing rollback plumbing",
                        "process_log": [
                            {
                                "at": "2026-03-21T20:50:00",
                                "type": "status_report",
                                "title": "oldest_excluded",
                                "summary": "oldest",
                                "from_role": "developer",
                            },
                            {
                                "at": "2026-03-21T20:51:00",
                                "type": "status_report",
                                "title": "shown_1",
                                "summary": "second",
                                "from_role": "developer",
                            },
                            {
                                "at": "2026-03-21T20:52:00",
                                "type": "status_report",
                                "title": "shown_2",
                                "summary": "third",
                                "from_role": "developer",
                            },
                            {
                                "at": "2026-03-21T20:53:00",
                                "type": "review_requested",
                                "title": "shown_3",
                                "summary": "review requested",
                                "from_role": "developer",
                                "to_role": "qa_main",
                            },
                            {
                                "at": "2026-03-21T20:54:00",
                                "type": "review_verdict",
                                "title": "shown_4",
                                "summary": "lgtm",
                                "from_role": "qa_main",
                                "to_role": "developer",
                                "verdict": "lgtm",
                            },
                            {
                                "at": "2026-03-21T20:55:00",
                                "type": "stage_blocked",
                                "title": "shown_5",
                                "summary": "Borrow checker regression",
                                "from_role": "myswat",
                                "to_role": "user",
                            },
                        ],
                    }
                },
            },
        ]
        mock_store.list_runtime_registrations.return_value = []
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.return_value = []
        mock_pool_cls.return_value = pool

        result = CliRunner().invoke(app, ["status", "--project", "proj"])

        assert result.exit_code == 0
        assert "Current Work Item" in result.stdout
        assert "Current phase:" in result.stdout
        assert "phase_1" in result.stdout
        assert "Alerts" in result.stdout
        assert "Borrow checker regression" in result.stdout
        assert "Recent Messages" in result.stdout
        assert "oldest_excluded" not in result.stdout
        assert "shown_1" in result.stdout
        assert "shown_5" in result.stdout
        assert "Agents" not in result.stdout
        assert "Work Items" not in result.stdout
        assert "Active Sessions" not in result.stdout
        assert "Worker Health" not in result.stdout
        assert "Project:" not in result.stdout

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_default_status_skips_runtime_queries_without_active_items(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
    ):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1,
            "slug": "proj",
            "name": "Proj",
        }
        mock_store.list_work_items.return_value = [
            {
                "id": 1,
                "status": "completed",
                "item_type": "code_change",
                "title": "Completed task",
                "metadata_json": {
                    "task_state": {
                        "current_stage": "report",
                        "process_log": [
                            {
                                "at": "2026-03-21T20:55:00",
                                "type": "status_report",
                                "title": "Workflow report",
                                "summary": "Completed task summary",
                                "from_role": "developer",
                                "to_role": "user",
                            },
                        ],
                    }
                },
            },
        ]
        mock_store.list_runtime_registrations.return_value = []
        mock_store_cls.return_value = mock_store
        pool = MagicMock()
        pool.fetch_all.return_value = []
        mock_pool_cls.return_value = pool

        result = CliRunner().invoke(app, ["status", "--project", "proj"])

        assert result.exit_code == 0
        assert "No active work item." in result.stdout
        assert "Recent Messages" in result.stdout
        assert "Completed task summary" in result.stdout
        mock_store.list_runtime_registrations.assert_not_called()


class TestTaskCommand:
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import task

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            task(work_item_id=1, project="missing")

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_work_item_not_found(self, mock_store_cls, mock_pool_cls, mock_settings_cls, mock_teamwork):
        from myswat.cli.main import task

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj", "name": "Proj"}
        mock_store.get_work_item.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            task(work_item_id=1, project="proj")

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_success(self, mock_store_cls, mock_pool_cls, mock_settings_cls, mock_teamwork):
        from myswat.cli.main import task

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj", "name": "Proj"}
        mock_store.get_work_item.return_value = {
            "id": 7,
            "project_id": 1,
            "status": "in_progress",
            "item_type": "code_change",
            "title": "Implement feature",
            "description": "Detailed",
            "metadata_json": {
                "task_state": {
                    "current_stage": "phase_1",
                    "latest_summary": "working",
                    "next_todos": ["do x"],
                    "open_issues": ["issue y"],
                }
            },
        }
        mock_store_cls.return_value = mock_store

        task(work_item_id=7, project="proj")
        mock_teamwork.assert_called_once()

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_task_prints_workflow_mode(self, mock_store_cls, mock_pool_cls, mock_settings_cls, mock_teamwork):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj", "name": "Proj"}
        mock_store.get_work_item.return_value = {
            "id": 7,
            "project_id": 1,
            "status": "in_progress",
            "item_type": "code_change",
            "title": "Implement feature",
            "description": "Detailed",
            "metadata_json": {
                "work_mode": "develop",
                "task_state": {
                    "current_stage": "phase_1",
                    "latest_summary": "working",
                },
            },
        }
        mock_store_cls.return_value = mock_store
        mock_pool_cls.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(app, ["task", "7", "--project", "proj"])
        assert result.exit_code == 0
        assert "Workflow mode:" in result.stdout
        assert "develop" in result.stdout
        mock_teamwork.assert_called_once()

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_no_items(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj",
        }
        mock_store.list_agents.return_value = []
        mock_store.list_work_items.return_value = []
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.return_value = []
        pool.fetch_one.side_effect = [
            {"cnt": 0},  # knowledge
            {"cnt": 0},  # compacted
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj"])
        assert result.exit_code == 0

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_teamwork_items(self, mock_store_cls, mock_pool_cls,
                                  mock_settings_cls, mock_teamwork):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp",
        }
        mock_store.list_agents.return_value = []
        mock_store.list_work_items.return_value = [
            {
                "id": 1, "status": "completed", "item_type": "code_change",
                "title": "Teamwork task", "assigned_agent_id": 1,
            },
        ]
        mock_store.count_session_turns.return_value = 5
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        # cycles query returns data -> teamwork mode
        pool.fetch_all.side_effect = [
            [
                {
                    "proposer_agent_id": 1, "reviewer_agent_id": 2,
                    "proposer_role": "developer", "reviewer_role": "qa_main",
                },
            ],
            [],  # active sessions
        ]
        pool.fetch_one.side_effect = [
            {"cnt": 5},  # knowledge
            {"cnt": 1},  # compacted
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj"])
        assert result.exit_code == 0
        mock_teamwork.assert_not_called()

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_teamwork_items_details(self, mock_store_cls, mock_pool_cls,
                                         mock_settings_cls, mock_teamwork):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp",
        }
        mock_store.list_agents.return_value = []
        mock_store.list_work_items.return_value = [
            {
                "id": 1, "status": "completed", "item_type": "code_change",
                "title": "Teamwork task", "assigned_agent_id": 1,
            },
        ]
        mock_store.count_session_turns.return_value = 5
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "proposer_agent_id": 1, "reviewer_agent_id": 2,
                    "proposer_role": "developer", "reviewer_role": "qa_main",
                },
            ],
            [],
        ]
        pool.fetch_one.side_effect = [
            {"cnt": 5},
            {"cnt": 1},
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj", "--details"])
        assert result.exit_code == 0
        mock_teamwork.assert_called_once()

    def test_status_prefers_work_mode_metadata(self):
        item = {"metadata_json": {"work_mode": "develop"}}
        assert _display_mode(item, "[cyan]team[/cyan]") == "develop"

    def test_status_falls_back_when_no_work_mode(self):
        item = {"metadata_json": {}}
        assert _display_mode(item, "[dim]solo[/dim]") == "[dim]solo[/dim]"

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_active_sessions(self, mock_store_cls, mock_pool_cls,
                                   mock_settings_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app
        import json as _json

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj",
        }
        mock_store.list_agents.return_value = []
        mock_store.list_work_items.return_value = []
        mock_store.count_session_turns.return_value = 5
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.side_effect = [
            # Active sessions
            [
                {
                    "session_uuid": "uuid-1234", "display_name": "Dev",
                    "id": 1, "purpose": "coding", "role": "developer",
                    "token_count_est": 5000,
                },
            ],
            # recent_turns for first session
            [
                {
                    "role": "user", "content": "hello",
                    "metadata_json": None, "created_at": "2026-03-07",
                },
                {
                    "role": "assistant", "content": "hi back",
                    "metadata_json": _json.dumps({"elapsed_seconds": 125}),
                    "created_at": "2026-03-07",
                },
            ],
        ]
        pool.fetch_one.side_effect = [
            {"role": "user"},  # last_turn (is_thinking=True)
            {"content": "hello world"},  # pending_turn (shown when thinking)
            {"cnt": 0},  # knowledge
            {"cnt": 0},  # compacted
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj"])
        assert result.exit_code == 0

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_solo_item_with_session_agents(self, mock_store_cls, mock_pool_cls,
                                            mock_settings_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj",
        }
        mock_store.list_agents.return_value = []
        mock_store.list_work_items.return_value = [
            {
                "id": 1, "status": "in_progress", "item_type": "code_change",
                "title": "Solo task", "assigned_agent_id": None,
            },
        ]
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [],  # No cycles
            [{"role": "developer"}],  # Session agents fallback
            [],  # Active sessions
        ]
        pool.fetch_one.side_effect = [
            {"cnt": 0},  # knowledge
            {"cnt": 0},  # compacted
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj"])
        assert result.exit_code == 0

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_repo_path(self, mock_store_cls, mock_pool_cls,
                             mock_settings_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "slug": "proj", "name": "Proj", "repo_path": "/home/user/repo",
        }
        mock_store.list_agents.return_value = []
        mock_store.list_work_items.return_value = []
        mock_store_cls.return_value = mock_store

        pool = MagicMock()
        pool.fetch_all.return_value = []
        pool.fetch_one.side_effect = [
            {"cnt": 0},  # knowledge
            {"cnt": 0},  # compacted
        ]
        mock_pool_cls.return_value = pool

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--project", "proj"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Typer command routing
# ---------------------------------------------------------------------------
class TestCommandRouting:
    @patch("myswat.cli.main.time.sleep", return_value=None)
    def test_follow_work_item_until_terminal_prints_progress_until_completed(self, _mock_sleep):
        client = MagicMock()
        client.get_work_item.side_effect = [
            {
                "work_item": {
                    "id": 41,
                    "status": "in_progress",
                    "metadata_json": {
                        "task_state": {
                            "current_stage": "design",
                            "latest_summary": "Produce technical design",
                            "process_log": [
                                {
                                    "at": "2026-03-19T18:00:00",
                                    "event_type": "daemon_queued",
                                    "title": "Workflow queued",
                                    "summary": "MySwat daemon accepted the work item.",
                                    "from_role": "user",
                                    "to_role": "myswat",
                                }
                            ],
                        }
                    },
                }
            },
            {
                "work_item": {
                    "id": 41,
                    "status": "completed",
                    "metadata_json": {
                        "task_state": {
                            "current_stage": "report",
                            "latest_summary": "Workflow completed successfully.",
                            "process_log": [
                                {
                                    "at": "2026-03-19T18:00:00",
                                    "event_type": "daemon_queued",
                                    "title": "Workflow queued",
                                    "summary": "MySwat daemon accepted the work item.",
                                    "from_role": "user",
                                    "to_role": "myswat",
                                },
                                {
                                    "at": "2026-03-19T18:00:05",
                                    "event_type": "workflow_completed",
                                    "title": "Workflow completed",
                                    "summary": "Final report generated.",
                                    "from_role": "myswat",
                                    "to_role": "user",
                                },
                            ],
                        }
                    },
                }
            },
        ]
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        item = _follow_work_item_until_terminal(
            client=client,
            project="proj",
            work_item_id=41,
            poll_interval_seconds=0.01,
            console=console,
        )

        assert item["status"] == "completed"
        rendered = output.getvalue()
        assert "Workflow queued" in rendered
        assert "status=in_progress stage=design" in rendered
        assert "status=completed stage=report" in rendered

    @patch("myswat.cli.main.time.sleep", return_value=None)
    def test_follow_work_item_until_terminal_retries_retryable_daemon_errors(self, _mock_sleep):
        client = MagicMock()
        client.get_work_item.side_effect = [
            DaemonClientError(
                "MySwat daemon request timed out: POST http://127.0.0.1:8765/api/work-item",
                retryable=True,
            ),
            {
                "work_item": {
                    "id": 41,
                    "status": "completed",
                    "metadata_json": {
                        "task_state": {
                            "current_stage": "report",
                            "latest_summary": "Workflow completed successfully.",
                            "process_log": [],
                        }
                    },
                }
            },
        ]
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        item = _follow_work_item_until_terminal(
            client=client,
            project="proj",
            work_item_id=41,
            poll_interval_seconds=0.01,
            console=console,
        )

        assert item["status"] == "completed"
        rendered = output.getvalue()
        assert "Waiting for daemon response" in rendered
        assert "status=completed stage=report" in rendered

    @patch("myswat.cli.main.time.sleep", return_value=None)
    def test_follow_work_item_until_terminal_raises_non_retryable_daemon_errors(self, _mock_sleep):
        client = MagicMock()
        client.get_work_item.side_effect = DaemonClientError("Work item 41 not found")
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        with pytest.raises(DaemonClientError, match="Work item 41 not found"):
            _follow_work_item_until_terminal(
                client=client,
                project="proj",
                work_item_id=41,
                poll_interval_seconds=0.01,
                console=console,
            )

    @patch("myswat.cli.chat_cmd.run_chat")
    def test_chat_command(self, mock_run_chat):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--project", "proj"])
        mock_run_chat.assert_called_once()

    @patch("myswat.cli.chat_cmd.run_chat")
    def test_run_no_task_calls_chat(self, mock_run_chat):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--project", "proj"])
        mock_run_chat.assert_called_once()

    @patch("myswat.cli.run_cmd.run_single")
    def test_run_single_mode(self, mock_run_single):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "do stuff", "--project", "proj", "--single"])
        mock_run_single.assert_called_once()

    @patch("myswat.cli.run_cmd.run_with_review")
    def test_run_review_mode(self, mock_run_review):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "do stuff", "--project", "proj"])
        mock_run_review.assert_called_once()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_detaches_by_default(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 41, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="add feature",
        )
        mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_with_ga_test(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 47, "workers": ["architect", "developer", "qa_main"]}
        mock_client_cls.return_value = mock_client
        mock_follow.return_value = {"status": "completed"}

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--with-ga-test"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="add feature",
            with_ga_test=True,
        )
        mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_mode_aliases(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 42, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client
        mock_follow.return_value = {"status": "completed"}

        runner = CliRunner()
        cases = [
            ("--design", WorkMode.design),
            ("--plan", WorkMode.design),
            ("--develop", WorkMode.develop),
            ("--dev", WorkMode.develop),
            ("--test", WorkMode.test),
            ("--ga-test", WorkMode.test),
        ]
        for flag, expected_mode in cases:
            mock_client.submit_work.reset_mock()
            result = runner.invoke(app, ["work", "add feature", "--project", "proj", flag])
            assert result.exit_code == 0
            mock_client.submit_work.assert_called_once_with(
                workdir=None,
                mode=expected_mode.value,
                project="proj",
                requirement="add feature",
            )
            mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_follow_opt_in(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 41, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client
        mock_follow.return_value = {"status": "completed"}

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--follow"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="add feature",
        )
        mock_follow.assert_called_once()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_resume_submits_existing_work_item(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 41, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client
        mock_follow.return_value = {"status": "completed"}

        runner = CliRunner()
        result = runner.invoke(app, ["work", "--project", "proj", "--resume", "41"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="",
            resume_work_item_id=41,
        )
        mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_resume_follow_attaches_to_existing_work_item(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 41, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client
        mock_follow.return_value = {"status": "completed"}

        runner = CliRunner()
        result = runner.invoke(app, ["work", "--project", "proj", "--resume", "41", "--follow"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="",
            resume_work_item_id=41,
        )
        mock_follow.assert_called_once()

    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_rejects_multiple_mode_flags(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["work", "add feature", "--project", "proj", "--design", "--test"],
        )
        assert result.exit_code != 0
        assert result.exception is not None
        mock_client_cls.return_value.submit_work.assert_not_called()

    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_rejects_with_ga_test_for_non_full_mode(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["work", "add feature", "--project", "proj", "--test", "--with-ga-test"],
        )
        assert result.exit_code != 0
        mock_client_cls.return_value.submit_work.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_background(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 42, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--background"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="add feature",
        )
        mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_design_background_submits_design_mode(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 43, "workers": ["architect", "developer", "qa_main"]}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--background", "--design"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="design",
            project="proj",
            requirement="add feature",
        )
        mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_background_mode_threads(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 44, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--background", "--dev"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="develop",
            project="proj",
            requirement="add feature",
        )
        mock_follow.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_auto_approve(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 45, "workers": ["developer", "qa_main"]}
        mock_client_cls.return_value = mock_client
        mock_follow.return_value = {"status": "completed"}

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--auto-approve"])
        assert result.exit_code == 0
        mock_client.submit_work.assert_called_once_with(
            workdir=None,
            mode="full",
            project="proj",
            requirement="add feature",
        )
        mock_follow.assert_not_called()

    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_interactive_checkpoints(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--interactive-checkpoints"])
        assert result.exit_code != 0
        mock_client_cls.return_value.submit_work.assert_not_called()

    @patch("myswat.cli.main._follow_work_item_until_terminal")
    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_ctrl_c_requests_cancel(self, mock_client_cls, mock_follow):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.base_url = "http://127.0.0.1:8765"
        mock_client.submit_work.return_value = {"work_item_id": 46, "workers": ["developer", "qa_main"]}
        mock_client.control_work.return_value = {"work_item_id": 46, "status": "cancelled"}
        mock_client_cls.return_value = mock_client
        mock_follow.side_effect = [KeyboardInterrupt(), {"status": "cancelled"}]

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--follow"])

        assert result.exit_code == 130
        mock_client.control_work.assert_called_once_with(project="proj", work_item_id=46, action="cancel")

    @patch("myswat.server.control_client.DaemonClient")
    def test_work_command_rejects_background_with_follow(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["work", "add feature", "--project", "proj", "--background", "--follow"],
        )
        assert result.exit_code != 0
        mock_client_cls.return_value.submit_work.assert_not_called()

    def test_work_background_worker_command_removed(self):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "work-background-worker",
                "add feature",
                "--project",
                "proj",
                "--work-item-id",
                "42",
                "--mode",
                "develop",
            ],
        )
        assert result.exit_code != 0
        assert "No such command 'work-background-worker'" in result.output

    @patch("myswat.server.control_client.DaemonClient")
    def test_stop_command(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.control_work.return_value = {"work_item_id": 42}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["stop", "42", "--project", "proj"])
        assert result.exit_code == 0
        mock_client.control_work.assert_called_once_with(project="proj", work_item_id=42, action="cancel")

    @patch("myswat.server.control_client.DaemonClient")
    def test_stop_command_requires_daemon(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.control_work.side_effect = DaemonClientError(
            "MySwat daemon is unavailable at http://127.0.0.1:8765: <urlopen error [Errno 111] connection refused>",
            retryable=True,
        )
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["stop", "42", "--project", "proj"])

        assert result.exit_code == 1
        assert "unavailable" in result.output
        assert "Start the daemon first" in result.output

    @patch("myswat.server.control_client.DaemonClient")
    def test_pause_command(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.control_work.return_value = {"work_item_id": 42}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["pause", "42", "--project", "proj"])
        assert result.exit_code == 0
        mock_client.control_work.assert_called_once_with(project="proj", work_item_id=42, action="pause")

    @patch("myswat.server.control_client.DaemonClient")
    def test_pause_command_requires_daemon(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.control_work.side_effect = DaemonClientError(
            "MySwat daemon is unavailable at http://127.0.0.1:8765: <urlopen error [Errno 111] connection refused>",
            retryable=True,
        )
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["pause", "42", "--project", "proj"])

        assert result.exit_code == 1
        assert "unavailable" in result.output
        assert "Start the daemon first" in result.output

    @patch("myswat.server.control_client.DaemonClient")
    def test_cleanup_command(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.cleanup_project.return_value = {
            "project": "proj",
            "deleted": {"projects": 1, "agents": 3},
            "removed_runtime_paths": ["/tmp/workers/proj"],
        }
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["cleanup", "--project", "proj", "--yes"])

        assert result.exit_code == 0
        mock_client.cleanup_project.assert_called_once_with(project="proj")
        assert "Project 'proj' removed." in result.output
        assert "projects=1" in result.output

    def test_help_lists_project_introspection_commands(self):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        commands = set(re.findall(r"│\s+([a-z][a-z0-9-]*)\s+", result.output))
        for visible_name in ("chat", "work", "status", "search", "history", "task", "memory", "gc", "stop", "pause", "init", "cleanup", "reset"):
            assert visible_name in commands

    def test_feed_command_removed(self):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["feed", "/tmp/doc.md", "--project", "proj"])
        assert result.exit_code != 0
        assert "No such command 'feed'" in result.output

    def test_learn_command_removed(self):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["learn", "--project", "proj"])
        assert result.exit_code != 0
        assert "No such command 'learn'" in result.output

    @patch("myswat.server.control_client.DaemonClient")
    def test_init_command(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.init_project.return_value = {"project": "my-project"}
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-project"])
        assert result.exit_code == 0
        mock_client.init_project.assert_called_once_with(
            name="my-project",
            repo_path=None,
            description=None,
        )

    @patch("myswat.server.control_client.DaemonClient")
    def test_init_command_timeout_reports_in_progress_hint(self, mock_client_cls):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        mock_client = MagicMock()
        mock_client.init_project.side_effect = DaemonClientError(
            "MySwat daemon request timed out: POST http://127.0.0.1:8765/api/init",
            retryable=True,
        )
        mock_client_cls.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-project"])

        assert result.exit_code == 1
        assert "timed out" in result.output
        assert "still in progress or blocked" in result.output
        assert "Start the daemon first" not in result.output
