"""Tests for myswat.cli.main — status command and _print_teamwork_details."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit

from myswat.cli.main import _print_teamwork_details, _infer_stage_labels


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
        from myswat.cli.main import status

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            status(project="missing")

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_agents_and_items(self, mock_store_cls, mock_pool_cls,
                                    mock_settings_cls, mock_teamwork):
        from myswat.cli.main import status

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

        status(project="proj")


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

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_no_items(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import status

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

        status(project="proj")

    @patch("myswat.cli.main._print_teamwork_details")
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_teamwork_items(self, mock_store_cls, mock_pool_cls,
                                  mock_settings_cls, mock_teamwork):
        from myswat.cli.main import status

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

        status(project="proj")
        mock_teamwork.assert_called_once()

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_active_sessions(self, mock_store_cls, mock_pool_cls,
                                   mock_settings_cls):
        from myswat.cli.main import status
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

        status(project="proj")

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_solo_item_with_session_agents(self, mock_store_cls, mock_pool_cls,
                                            mock_settings_cls):
        from myswat.cli.main import status

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

        status(project="proj")

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_with_repo_path(self, mock_store_cls, mock_pool_cls,
                             mock_settings_cls):
        from myswat.cli.main import status

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

        status(project="proj")


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
        mock_run_work.assert_called_once()

    @patch("myswat.cli.feed_cmd.run_feed")
    def test_feed_command(self, mock_run_feed):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["feed", "/tmp/doc.md", "--project", "proj"])
        mock_run_feed.assert_called_once()

    @patch("myswat.cli.learn_cmd.run_learn")
    def test_learn_command(self, mock_run_learn):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["learn", "--project", "proj"])
        mock_run_learn.assert_called_once()

    @patch("myswat.cli.init_cmd.run_init")
    def test_init_command(self, mock_run_init):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", "my-project"])
        mock_run_init.assert_called_once()
