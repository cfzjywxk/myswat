"""Tests for myswat.cli.main — status command and _print_teamwork_details."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit
from rich.console import Console

from myswat.cli.main import _display_mode, _print_teamwork_details, _infer_stage_labels
from myswat.workflow.engine import WorkMode


# ---------------------------------------------------------------------------
# _print_teamwork_details
# ---------------------------------------------------------------------------
class TestPrintTeamworkDetails:
    def test_with_review_cycles(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            # all_cycles
            [
                {
                    "iteration": 1, "verdict": "changes_requested",
                    "created_at": "2026-03-07",
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
                {
                    "iteration": 2, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
            ],
            # artifacts
            [
                {
                    "artifact_type": "proposal", "title": "Design",
                    "iteration": 1, "created_at": "2026-03-07",
                    "agent_role": "developer", "agent_name": "Dev",
                },
            ],
            # agent_effort
            [
                {
                    "role": "developer", "display_name": "Dev",
                    "session_count": 1, "turn_count": 10,
                    "total_tokens": 50000,
                },
                {
                    "role": "qa_main", "display_name": "QA",
                    "session_count": 1, "turn_count": 5,
                    "total_tokens": 800,
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
            [],  # no agent_effort
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
            # Cycles with different proposer/reviewer pairs
            [
                {
                    "iteration": 1, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
                {
                    "iteration": 2, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "proposer_role": "qa_main", "proposer_name": "QA",
                    "reviewer_role": "developer", "reviewer_name": "Dev",
                },
            ],
            [],  # artifacts
            [],  # agent effort
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
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
                {
                    "iteration": 2, "verdict": "changes_requested",
                    "created_at": "2026-03-07",
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
                {
                    "iteration": 3, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
            ],
            [],  # artifacts
            [],  # agent effort
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
                    "proposer_role": "developer", "proposer_name": "Dev",
                    "reviewer_role": "qa_main", "reviewer_name": "QA",
                },
            ],
            [
                {
                    "iteration": 1,
                    "verdict": "changes_requested",
                    "verdict_json": '{"verdict":"changes_requested","summary":"Needs more detail","issues":["add phase scope"]}',
                    "proposer_role": "developer",
                    "reviewer_role": "qa_main",
                    "artifact_title": "Iteration 1",
                    "artifact_type": "proposal",
                    "artifact_content": "Initial design draft",
                },
            ],
            [],
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
        assert "developer -> qa_main" in rendered
        assert "qa_main -> developer" in rendered

    def test_architect_design_round_label_rendered(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "proposer_role": "architect", "proposer_name": "Architect",
                    "reviewer_role": "developer", "reviewer_name": "Dev",
                },
            ],
            [],
            [],
        ]
        item = {"id": 1, "title": "Design task", "status": "completed"}
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        _print_teamwork_details(pool, item, console)

        rendered = output.getvalue()
        assert "Architect Design Review" in rendered

    def test_test_plan_round_label_with_architect_reviewer_rendered(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [
                {
                    "iteration": 1, "verdict": "lgtm",
                    "created_at": "2026-03-07",
                    "proposer_role": "qa_main", "proposer_name": "QA",
                    "reviewer_role": "architect", "reviewer_name": "Architect",
                },
            ],
            [],
            [],
        ]
        item = {"id": 2, "title": "Test plan task", "status": "completed"}
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)

        _print_teamwork_details(pool, item, console)

        rendered = output.getvalue()
        assert "Test Plan Review" in rendered

    def test_tokens_under_1000(self):
        pool = MagicMock()
        pool.fetch_all.side_effect = [
            [],  # cycles
            [],  # artifacts
            [
                {
                    "role": "developer", "display_name": "Dev",
                    "session_count": 1, "turn_count": 2,
                    "total_tokens": 500,  # under 1000
                },
            ],
        ]

        item = {
            "id": 1, "title": "Small task", "status": "completed",
        }
        console = MagicMock()

        _print_teamwork_details(pool, item, console)


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
                "work_mode": "development",
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
        assert "development" in result.stdout
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
        item = {"metadata_json": {"work_mode": "development"}}
        assert _display_mode(item, "[cyan]team[/cyan]") == "development"

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

    @patch("myswat.cli.work_cmd.run_work")
    def test_work_command(self, mock_run_work):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj"])
        assert result.exit_code == 0
        mock_run_work.assert_called_once_with(
            "proj",
            "add feature",
            workdir=None,
            background=False,
            mode=WorkMode.full,
        )

    @patch("myswat.cli.work_cmd.run_work")
    def test_work_command_mode_aliases(self, mock_run_work):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        cases = [
            ("--design", WorkMode.design),
            ("--plan", WorkMode.design),
            ("--development", WorkMode.development),
            ("--dev", WorkMode.development),
            ("--test", WorkMode.test),
            ("--ga-test", WorkMode.test),
        ]
        for flag, expected_mode in cases:
            mock_run_work.reset_mock()
            result = runner.invoke(app, ["work", "add feature", "--project", "proj", flag])
            assert result.exit_code == 0
            mock_run_work.assert_called_once_with(
                "proj",
                "add feature",
                workdir=None,
                background=False,
                mode=expected_mode,
            )

    @patch("myswat.cli.work_cmd.run_work")
    def test_work_command_rejects_multiple_mode_flags(self, mock_run_work):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["work", "add feature", "--project", "proj", "--design", "--test"],
        )
        assert result.exit_code != 0
        assert result.exception is not None
        mock_run_work.assert_not_called()

    @patch("myswat.cli.work_cmd.run_work")
    def test_work_command_background(self, mock_run_work):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--background"])
        assert result.exit_code == 0
        mock_run_work.assert_called_once_with(
            "proj",
            "add feature",
            workdir=None,
            background=True,
            mode=WorkMode.full,
        )

    @patch("myswat.cli.work_cmd.run_work")
    def test_work_command_rejects_design_background(self, mock_run_work):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--background", "--design"])
        assert result.exit_code != 0
        mock_run_work.assert_not_called()

    @patch("myswat.cli.work_cmd.run_work")
    def test_work_command_background_mode_threads(self, mock_run_work):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["work", "add feature", "--project", "proj", "--background", "--dev"])
        assert result.exit_code == 0
        mock_run_work.assert_called_once_with(
            "proj",
            "add feature",
            workdir=None,
            background=True,
            mode=WorkMode.development,
        )

    @patch("myswat.cli.work_cmd.run_background_work_item")
    def test_work_background_worker_command(self, mock_run_background_work_item):
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
                "development",
            ],
        )
        assert result.exit_code == 0
        mock_run_background_work_item.assert_called_once_with(
            "proj",
            "add feature",
            work_item_id=42,
            workdir=None,
            mode=WorkMode.development,
        )

    @patch("myswat.cli.work_cmd.stop_work_item")
    def test_stop_command(self, mock_stop_work_item):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["stop", "42", "--project", "proj"])
        mock_stop_work_item.assert_called_once_with("proj", 42)

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

    @patch("myswat.cli.init_cmd.run_init")
    def test_init_command(self, mock_run_init):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-project"])
        mock_run_init.assert_called_once()
