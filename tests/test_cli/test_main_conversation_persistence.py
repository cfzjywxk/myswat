"""CLI tests for conversation persistence commands in myswat.cli.main."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner


class TestHistoryCommand:
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        result = CliRunner().invoke(app, ["history", "--project", "missing"])
        assert result.exit_code == 1

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_history_prints_chronological_turns(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.get_recent_turns_global.return_value = [
            {
                "agent_role": "developer",
                "role": "user",
                "created_at": datetime(2026, 3, 13, 10, 31, 12),
                "content": "implement it",
            },
            {
                "agent_role": "developer",
                "role": "assistant",
                "created_at": datetime(2026, 3, 13, 10, 31, 28),
                "content": "working on it",
            },
        ]
        mock_store_cls.return_value = mock_store

        result = CliRunner().invoke(app, ["history", "--project", "proj"])

        assert result.exit_code == 0
        assert "[developer] [2026-03-13 10:31:12] user: implement it" in result.output
        assert "[developer] [2026-03-13 10:31:28] assistant: working on it" in result.output
        mock_store.get_recent_turns_global.assert_called_once_with(1, limit=50, role=None)

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_history_applies_turn_limit_and_role_filter(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.get_recent_turns_global.return_value = []
        mock_store_cls.return_value = mock_store

        result = CliRunner().invoke(
            app,
            ["history", "--project", "proj", "--turns", "10", "--role", "developer"],
        )

        assert result.exit_code == 0
        mock_store.get_recent_turns_global.assert_called_once_with(1, limit=10, role="developer")


class TestGcCommand:
    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        result = CliRunner().invoke(app, ["gc", "--project", "missing"])
        assert result.exit_code == 1

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_gc_dry_run(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.gc_compacted_turns.return_value = {"turns_deleted": 7, "sessions_affected": 3}
        mock_store_cls.return_value = mock_store

        result = CliRunner().invoke(
            app,
            ["gc", "--project", "proj", "--grace-days", "9", "--keep-recent", "60", "--dry-run"],
        )

        assert result.exit_code == 0
        assert "Would delete 7 turns from 3 sessions" in result.output
        mock_store.gc_compacted_turns.assert_called_once_with(
            1,
            grace_days=9,
            keep_recent=60,
            dry_run=True,
        )

    @patch("myswat.config.settings.MySwatSettings")
    @patch("myswat.db.connection.TiDBPool")
    @patch("myswat.memory.store.MemoryStore")
    def test_gc_delete(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.main import app

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.gc_compacted_turns.return_value = {"turns_deleted": 4, "sessions_affected": 2}
        mock_store_cls.return_value = mock_store

        result = CliRunner().invoke(app, ["gc", "--project", "proj"])

        assert result.exit_code == 0
        assert "Deleted 4 turns from 2 sessions" in result.output
        mock_store.gc_compacted_turns.assert_called_once_with(
            1,
            grace_days=7,
            keep_recent=50,
            dry_run=False,
        )


def test_memory_purge_command_removed() -> None:
    from myswat.cli.main import app

    result = CliRunner().invoke(app, ["memory", "purge", "--project", "proj"])

    assert result.exit_code != 0
    assert "No such command 'purge'" in result.output
