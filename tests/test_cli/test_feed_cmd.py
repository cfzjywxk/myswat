"""Tests for myswat.cli.feed_cmd."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.cli.feed_cmd import _get_ingester, run_feed


# ---------------------------------------------------------------------------
# _get_ingester
# ---------------------------------------------------------------------------
class TestGetIngester:
    def test_no_ai_mode(self):
        store = MagicMock()
        proj = {"id": 1}
        settings = MagicMock()

        ingester = _get_ingester(store, proj, settings, no_ai=True)
        assert ingester is not None
        store.list_agents.assert_not_called()

    def test_with_ai_codex(self):
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

        ingester = _get_ingester(store, proj, settings, no_ai=False)
        assert ingester is not None

    def test_with_ai_kimi(self):
        store = MagicMock()
        store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "kimi", "cli_path": "kimi",
                "model_name": "k2", "cli_extra_args": None,
            }
        ]
        settings = MagicMock()
        settings.compaction.compaction_backend = "kimi"
        proj = {"id": 1}

        ingester = _get_ingester(store, proj, settings, no_ai=False)
        assert ingester is not None

    def test_no_matching_agent(self):
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

        ingester = _get_ingester(store, proj, settings, no_ai=False)
        assert ingester is not None  # still returns ingester, just without runner

    def test_no_agents_available(self):
        store = MagicMock()
        store.list_agents.return_value = []
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        proj = {"id": 1}

        ingester = _get_ingester(store, proj, settings, no_ai=False)
        assert ingester is not None

    def test_extra_args_parsed(self):
        import json
        store = MagicMock()
        store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "codex", "cli_path": "codex",
                "model_name": "gpt-5",
                "cli_extra_args": json.dumps(["--verbose"]),
            }
        ]
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        proj = {"id": 1}

        ingester = _get_ingester(store, proj, settings, no_ai=False)
        assert ingester is not None


# ---------------------------------------------------------------------------
# run_feed
# ---------------------------------------------------------------------------
class TestRunFeed:
    @patch("myswat.cli.feed_cmd.MySwatSettings")
    @patch("myswat.cli.feed_cmd.TiDBPool")
    @patch("myswat.cli.feed_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_feed("/tmp", "missing", "*.md", False)

    @patch("myswat.cli.feed_cmd._get_ingester")
    @patch("myswat.cli.feed_cmd.MySwatSettings")
    @patch("myswat.cli.feed_cmd.TiDBPool")
    @patch("myswat.cli.feed_cmd.MemoryStore")
    def test_ingest_single_file(self, mock_store_cls, mock_pool_cls,
                                 mock_settings_cls, mock_get_ing, tmp_path):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        ingester = MagicMock()
        ingester.ingest_file.return_value = [1, 2, 3]
        mock_get_ing.return_value = ingester

        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        run_feed(str(test_file), "proj", "*.md", False)
        ingester.ingest_file.assert_called_once()

    @patch("myswat.cli.feed_cmd._get_ingester")
    @patch("myswat.cli.feed_cmd.MySwatSettings")
    @patch("myswat.cli.feed_cmd.TiDBPool")
    @patch("myswat.cli.feed_cmd.MemoryStore")
    def test_ingest_directory(self, mock_store_cls, mock_pool_cls,
                               mock_settings_cls, mock_get_ing, tmp_path):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        ingester = MagicMock()
        ingester.ingest_file.return_value = [1]
        mock_get_ing.return_value = ingester

        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.md").write_text("# B")

        run_feed(str(tmp_path), "proj", "*.md", False)
        assert ingester.ingest_file.call_count == 2

    @patch("myswat.cli.feed_cmd._get_ingester")
    @patch("myswat.cli.feed_cmd.MySwatSettings")
    @patch("myswat.cli.feed_cmd.TiDBPool")
    @patch("myswat.cli.feed_cmd.MemoryStore")
    def test_directory_no_matches(self, mock_store_cls, mock_pool_cls,
                                   mock_settings_cls, mock_get_ing, tmp_path):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        # No .md files
        (tmp_path / "a.txt").write_text("text")

        run_feed(str(tmp_path), "proj", "*.md", False)
        mock_get_ing.assert_not_called()

    @patch("myswat.cli.feed_cmd.MySwatSettings")
    @patch("myswat.cli.feed_cmd.TiDBPool")
    @patch("myswat.cli.feed_cmd.MemoryStore")
    def test_path_not_found(self, mock_store_cls, mock_pool_cls,
                             mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_feed("/nonexistent/path/to/file.md", "proj", "*.md", False)

    @patch("myswat.cli.feed_cmd._get_ingester")
    @patch("myswat.cli.feed_cmd.MySwatSettings")
    @patch("myswat.cli.feed_cmd.TiDBPool")
    @patch("myswat.cli.feed_cmd.MemoryStore")
    def test_file_ingest_error_in_dir(self, mock_store_cls, mock_pool_cls,
                                       mock_settings_cls, mock_get_ing,
                                       tmp_path):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        ingester = MagicMock()
        ingester.ingest_file.side_effect = [Exception("fail"), [1]]
        mock_get_ing.return_value = ingester

        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.md").write_text("# B")

        # Should not raise, handles per-file errors
        run_feed(str(tmp_path), "proj", "*.md", False)
        assert ingester.ingest_file.call_count == 2
