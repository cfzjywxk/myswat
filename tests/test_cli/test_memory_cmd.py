"""Tests for myswat.cli.memory_cmd."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
class TestSearch:
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            search("hello", project="missing")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_no_results(self, mock_store_cls, mock_pool_cls,
                         mock_settings_cls):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.side_effect = [[], []]
        mock_store.match_entities.return_value = []
        mock_store.get_related_entities.return_value = []
        mock_store_cls.return_value = mock_store

        search("hello", project="proj")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_results(self, mock_store_cls, mock_pool_cls,
                           mock_settings_cls):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.side_effect = [[
            {
                "id": 1, "category": "decision", "title": "Use Rust",
                "search_score": 0.95, "confidence": 0.9,
            },
        ], []]
        mock_store.match_entities.return_value = []
        mock_store.get_related_entities.return_value = []
        mock_store_cls.return_value = mock_store

        search("rust", project="proj", category=None, limit=10, no_vector=False)
        assert mock_store.search_knowledge.call_count == 2

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_no_vector_mode(self, mock_store_cls, mock_pool_cls,
                             mock_settings_cls):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = []
        mock_store.match_entities.return_value = []
        mock_store.get_related_entities.return_value = []
        mock_store_cls.return_value = mock_store

        search("hello", project="proj", no_vector=True)
        mock_store.search_knowledge.assert_called_once_with(
            project_id=1, query="hello", agent_id=None, category=None,
            source_type=None, limit=20, use_vector=False, use_fulltext=True,
        )

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_category_filter(self, mock_store_cls, mock_pool_cls,
                                   mock_settings_cls):
        from myswat.cli.memory_cmd import search

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.side_effect = [[], []]
        mock_store.match_entities.return_value = []
        mock_store.get_related_entities.return_value = []
        mock_store_cls.return_value = mock_store

        search("hello", project="proj", category="decision", limit=10, no_vector=False)
        assert mock_store.search_knowledge.call_count == 2


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------
class TestAdd:
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        from myswat.cli.memory_cmd import add

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            add("Title", "Content", project="missing")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_success_no_tags(self, mock_store_cls, mock_pool_cls,
                              mock_settings_cls):
        from myswat.cli.memory_cmd import add

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.upsert_knowledge.return_value = (42, "created")
        mock_store_cls.return_value = mock_store

        add("Title", "Content", project="proj", category="decision", tags=None)
        mock_store.upsert_knowledge.assert_called_once()

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_success_with_tags(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        from myswat.cli.memory_cmd import add

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.upsert_knowledge.return_value = (42, "created")
        mock_store_cls.return_value = mock_store

        add("Title", "Content", project="proj", tags="rust,perf")
        call_kwargs = mock_store.upsert_knowledge.call_args
        assert call_kwargs.kwargs.get("tags") == ["rust", "perf"] or \
               call_kwargs[1].get("tags") == ["rust", "perf"]


# ---------------------------------------------------------------------------
# list_knowledge
# ---------------------------------------------------------------------------
class TestListKnowledge:
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            list_knowledge(project="missing")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_no_entries(self, mock_store_cls, mock_pool_cls,
                         mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = []
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_entries(self, mock_store_cls, mock_pool_cls,
                           mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = [
            {
                "id": 1, "category": "decision", "title": "Use Rust",
                "relevance_score": 0.9, "tags": ["rust", "lang"],
            },
        ]
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_string_tags(self, mock_store_cls, mock_pool_cls,
                               mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = [
            {
                "id": 1, "category": "decision", "title": "Use Rust",
                "relevance_score": 0.9, "tags": json.dumps(["rust", "lang"]),
            },
        ]
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_invalid_tags_string(self, mock_store_cls, mock_pool_cls,
                                       mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = [
            {
                "id": 1, "category": "decision", "title": "Use Rust",
                "relevance_score": 0.9, "tags": "not-json",
            },
        ]
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")

    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_with_none_tags(self, mock_store_cls, mock_pool_cls,
                             mock_settings_cls):
        from myswat.cli.memory_cmd import list_knowledge

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.search_knowledge.return_value = [
            {
                "id": 1, "category": "decision", "title": "Use Rust",
                "relevance_score": 0.9, "tags": None,
            },
        ]
        mock_store_cls.return_value = mock_store

        list_knowledge(project="proj")


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------
class TestCompact:
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls):
        from myswat.cli.memory_cmd import compact

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            compact(project="missing")

    @patch("myswat.memory.compactor.KnowledgeCompactor")
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_no_agents(self, mock_store_cls, mock_pool_cls,
                        mock_settings_cls, mock_comp):
        from myswat.cli.memory_cmd import compact

        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1}
        mock_store.list_agents.return_value = []
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            compact(project="proj")

    @patch("myswat.memory.compactor.KnowledgeCompactor")
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_success_with_codex(self, mock_store_cls, mock_pool_cls,
                                 mock_settings_cls, mock_comp_cls):
        from myswat.cli.memory_cmd import compact

        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "name": "Proj"}
        mock_store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "codex", "cli_path": "codex",
                "model_name": "gpt-5", "cli_extra_args": None,
            }
        ]
        mock_store_cls.return_value = mock_store

        compactor = MagicMock()
        compactor.compact_all_pending.return_value = {
            "compacted": 3, "knowledge_created": 5, "skipped": 1,
        }
        mock_comp_cls.return_value = compactor

        compact(project="proj")
        compactor.compact_all_pending.assert_called_once()
        kwargs = mock_comp_cls.call_args.kwargs
        assert kwargs["threshold_turns"] == 200
        assert "threshold_tokens" not in kwargs

    @patch("myswat.memory.compactor.KnowledgeCompactor")
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_success_with_kimi(self, mock_store_cls, mock_pool_cls,
                                mock_settings_cls, mock_comp_cls):
        from myswat.cli.memory_cmd import compact

        settings = MagicMock()
        settings.compaction.compaction_backend = "kimi"
        settings.compaction.threshold_turns = 200
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "name": "Proj"}
        mock_store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "kimi", "cli_path": "kimi",
                "model_name": "k2", "cli_extra_args": None,
            }
        ]
        mock_store_cls.return_value = mock_store

        compactor = MagicMock()
        compactor.compact_all_pending.return_value = {
            "compacted": 1, "knowledge_created": 2, "skipped": 0,
        }
        mock_comp_cls.return_value = compactor

        compact(project="proj")

    @patch("myswat.memory.compactor.KnowledgeCompactor")
    @patch("myswat.cli.memory_cmd.MySwatSettings")
    @patch("myswat.cli.memory_cmd.TiDBPool")
    @patch("myswat.cli.memory_cmd.MemoryStore")
    def test_fallback_to_first_agent(self, mock_store_cls, mock_pool_cls,
                                      mock_settings_cls, mock_comp_cls):
        from myswat.cli.memory_cmd import compact

        settings = MagicMock()
        settings.compaction.compaction_backend = "claude"  # no match
        settings.compaction.threshold_turns = 200
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "name": "Proj"}
        mock_store.list_agents.return_value = [
            {
                "id": 1, "cli_backend": "codex", "cli_path": "codex",
                "model_name": "gpt-5", "cli_extra_args": None,
            }
        ]
        mock_store_cls.return_value = mock_store

        compactor = MagicMock()
        compactor.compact_all_pending.return_value = {
            "compacted": 0, "knowledge_created": 0, "skipped": 0,
        }
        mock_comp_cls.return_value = compactor

        compact(project="proj")
