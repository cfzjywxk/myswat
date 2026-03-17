"""Tests for myswat.cli.memory_cmd."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit


class TestSearch:
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            search("hello", project="missing")

    @patch("myswat.cli.memory_cmd.KnowledgeSearchEngine")
    @patch("myswat.cli.memory_cmd.SearchPlanBuilder")
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_no_results(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
        mock_plan_builder,
        mock_engine_cls,
    ):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        plan = MagicMock()
        plan.mode = "auto"
        plan.profile = "standard"
        plan.query = "hello"
        plan.limit = 10
        plan.use_vector = True
        mock_plan_builder.build.return_value = plan

        engine = MagicMock()
        engine.search_with_explanations.return_value = []
        mock_engine_cls.return_value = engine

        search("hello", project="proj")

        mock_plan_builder.build.assert_called_once()
        engine.search_with_explanations.assert_called_once_with(plan)

    @patch("myswat.cli.memory_cmd.console.print")
    @patch("myswat.cli.memory_cmd.KnowledgeSearchEngine")
    @patch("myswat.cli.memory_cmd.SearchPlanBuilder")
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_json_output_contains_results(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
        mock_plan_builder,
        mock_engine_cls,
        mock_console_print,
    ):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store_cls.return_value = mock_store

        plan = MagicMock()
        plan.mode = "exact"
        plan.profile = "quick"
        plan.query = "raft"
        plan.limit = 5
        plan.use_vector = False
        mock_plan_builder.build.return_value = plan

        engine = MagicMock()
        engine.search_with_explanations.return_value = [
            {
                "id": 7,
                "category": "architecture",
                "title": "RaftStore handles peer state",
                "content": "details",
                "search_score": 0.9,
                "confidence": 0.8,
                "why": ["title match"],
                "source_type": "session",
                "source_file": None,
                "tags": ["raft"],
                "search_metadata_json": {"learn_request_id": 3},
            }
        ]
        mock_engine_cls.return_value = engine

        search("raft", project="proj", json_output=True, no_vector=True)

        printed = mock_console_print.call_args.args[0]
        payload = json.loads(printed)
        assert payload["results"][0]["knowledge_id"] == 7
        assert payload["results"][0]["provenance"]["learn_request_id"] == 3


class TestListKnowledge:
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            list_knowledge(project="missing")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_no_entries(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = []
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")

        mock_store.search_knowledge.assert_called_once()

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_entries(self, mock_store_cls, mock_pool_cls, mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = [
            {
                "id": 1,
                "category": "decision",
                "title": "Use Rust",
                "relevance_score": 0.9,
                "tags": json.dumps(["rust", "lang"]),
            },
        ]
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")

        mock_store.search_knowledge.assert_called_once()


class TestRemovedCommands:
    def test_memory_add_command_removed(self):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["memory", "add", "Title", "Content", "--project", "proj"],
        )

        assert result.exit_code != 0
        assert "No such command 'add'" in result.output

    def test_memory_compact_command_removed(self):
        from typer.testing import CliRunner
        from myswat.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["memory", "compact", "--project", "proj"])

        assert result.exit_code != 0
        assert "No such command 'compact'" in result.output
