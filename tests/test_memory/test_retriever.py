"""Tests for MemoryRetriever knowledge-first budget allocation."""

from unittest.mock import MagicMock

from myswat.memory.retriever import MemoryRetriever


def _make_store_mock(knowledge_results=None, history=None, artifacts=None, work_items=None):
    store = MagicMock()
    store.search_knowledge.return_value = knowledge_results or []
    store.get_recent_history_for_agent.return_value = history or []
    store.get_recent_artifacts_for_project.return_value = artifacts or []
    store.list_work_items.return_value = work_items or []
    store.list_knowledge.return_value = []  # no project_ops by default
    return store


class TestKnowledgeFirstRetriever:
    def test_no_raw_turns_when_knowledge_sufficient(self):
        """When knowledge has >= 3 results, raw turns should NOT be loaded."""
        knowledge = [
            {"category": "decision", "title": f"Item {i}", "content": f"Content {i}"}
            for i in range(5)
        ]
        store = _make_store_mock(knowledge_results=knowledge)
        retriever = MemoryRetriever(store)

        context = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="test task",
        )

        # Knowledge was loaded
        assert "Relevant Knowledge" in context
        # Raw turns were NOT loaded (get_recent_history_for_agent should not be called)
        store.get_recent_history_for_agent.assert_not_called()

    def test_raw_turns_fallback_when_knowledge_sparse(self):
        """When knowledge has < 3 results, raw turns should be loaded as fallback."""
        knowledge = [
            {"category": "decision", "title": "Item 1", "content": "Content 1"}
        ]
        history = [{
            "session_id": 1,
            "purpose": "test session",
            "turns": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        }]
        store = _make_store_mock(knowledge_results=knowledge, history=history)
        retriever = MemoryRetriever(store)

        context = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="test task",
        )

        assert "Previous Session Context" in context
        store.get_recent_history_for_agent.assert_called_once()

    def test_empty_context_when_nothing_available(self):
        store = _make_store_mock()
        retriever = MemoryRetriever(store)

        context = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="test",
        )

        assert "MySwat Project Access" in context

    def test_artifacts_included_in_context(self):
        artifacts = [{
            "artifact_type": "proposal",
            "work_item_title": "Feature X",
            "work_item_status": "in_progress",
            "content": "Proposed implementation...",
        }]
        store = _make_store_mock(artifacts=artifacts)
        retriever = MemoryRetriever(store)

        context = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="test",
        )

        assert "Recent Artifacts" in context
        assert "Feature X" in context
