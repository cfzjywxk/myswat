"""Tests for Phase 1c/3 search planning and graph expansion."""

from unittest.mock import MagicMock

from myswat.memory.search_engine import KnowledgeSearchEngine, SearchPlanBuilder


class TestSearchPlanBuilder:
    def test_build_defaults_to_hybrid(self):
        plan = SearchPlanBuilder.build(project_id=1, query="LeaseRead timeout")
        assert plan.use_fulltext is True
        assert plan.use_vector is True
        assert plan.mode == "auto"
        assert plan.profile == "standard"

    def test_concept_mode_disables_fulltext_bias(self):
        plan = SearchPlanBuilder.build(project_id=1, query="why lease read matters", mode="concept")
        assert plan.use_vector is True
        assert plan.use_fulltext is False

    def test_role_and_stage_add_category_preferences(self):
        plan = SearchPlanBuilder.build(
            project_id=1,
            query="review the bug fix",
            agent_role="qa_main",
            current_stage="review_loop_reviewing",
        )
        assert plan.preferred_categories is not None
        assert "bug_fix" in plan.preferred_categories
        assert "review_feedback" in plan.preferred_categories


class TestKnowledgeSearchEngine:
    def test_search_with_explanations_uses_semantic_fallback(self):
        store = MagicMock()
        store.search_knowledge.return_value = [{"id": 1, "title": "Result", "content": "Body"}]
        store._query_terms.return_value = []
        store.match_entities.return_value = []
        store.get_related_entities.return_value = []

        engine = KnowledgeSearchEngine(store)
        plan = SearchPlanBuilder.build(project_id=1, query="deployment strategy")
        results = engine.search_with_explanations(plan)

        assert results[0]["why"] == ["semantic match"]

    def test_precise_profile_expands_via_graph(self):
        store = MagicMock()
        store._query_terms.return_value = ["leaseread"]
        store.match_entities.return_value = ["LeaseRead"]
        store.get_related_entities.return_value = [
            {"source_entity": "LeaseRead", "related_entity": "ReadIndex", "relation": "depends_on"},
        ]
        store.search_knowledge.side_effect = [
            [{"id": 1, "title": "LeaseRead timeout", "content": "base", "category": "bug_fix"}],   # lexical
            [{"id": 1, "title": "LeaseRead timeout", "content": "base", "category": "bug_fix"}],   # vector
            [{"id": 2, "title": "ReadIndex flow", "content": "expanded", "category": "architecture"}],  # graph
        ]

        engine = KnowledgeSearchEngine(store)
        plan = SearchPlanBuilder.build(project_id=1, query="LeaseRead timeout", profile="precise")
        results = engine.search_with_explanations(plan)

        assert [row["id"] for row in results] == [1, 2]
        assert any("graph expansion" in reason for reason in results[1]["why"])

    def test_fusion_prefers_category_bias(self):
        store = MagicMock()
        store._query_terms.return_value = ["bug"]
        store.match_entities.return_value = []
        store.get_related_entities.return_value = []
        lexical_row = {"id": 1, "title": "Architecture note", "content": "x", "category": "architecture", "confidence": 0.5}
        vector_row = {"id": 2, "title": "Bug fix", "content": "y", "category": "bug_fix", "confidence": 0.9}
        store.search_knowledge.side_effect = [
            [lexical_row],
            [vector_row],
        ]

        engine = KnowledgeSearchEngine(store)
        plan = SearchPlanBuilder.build(
            project_id=1,
            query="review the bug",
            agent_role="qa_main",
            current_stage="review",
        )
        results = engine.search(plan)

        assert results[0]["id"] == 2

    def test_auto_mode_can_expand_graph_for_entity_rich_query(self):
        store = MagicMock()
        store._query_terms.return_value = ["leaseread"]
        store.match_entities.side_effect = [["LeaseRead"]]
        store.get_related_entities.return_value = [
            {"source_entity": "LeaseRead", "related_entity": "ReadIndex", "relation": "depends_on"},
        ]
        store.search_knowledge.side_effect = [
            [{"id": 1, "title": "LeaseRead timeout", "content": "base", "category": "bug_fix"}],
            [{"id": 1, "title": "LeaseRead timeout", "content": "base", "category": "bug_fix"}],
            [{"id": 2, "title": "ReadIndex flow", "content": "expanded", "category": "architecture"}],
        ]

        engine = KnowledgeSearchEngine(store)
        plan = SearchPlanBuilder.build(project_id=1, query="LeaseRead timeout", mode="auto")
        results = engine.search_with_explanations(plan)

        assert [row["id"] for row in results] == [1, 2]
        assert any("graph expansion" in reason for reason in results[1]["why"])

    def test_render_for_context_groups_by_category(self):
        rendered = KnowledgeSearchEngine.render_for_context(
            [
                {"category": "architecture", "title": "LeaseRead", "content": "Uses leader lease."},
                {"category": "bug_fix", "title": "Epoch bug", "content": "Fixed stale epoch handling."},
            ],
            budget_tokens=200,
        )
        assert "## Relevant Knowledge" in rendered
        assert "### [architecture]" in rendered
        assert "### [bug_fix]" in rendered
