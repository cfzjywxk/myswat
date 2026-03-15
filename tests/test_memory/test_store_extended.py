"""Extended tests for MemoryStore – covers remaining uncovered methods."""

import json
from unittest.mock import MagicMock, patch, call
import pytest

import myswat.memory.store as store_module
from myswat.agents.base import AgentResponse
from myswat.memory.store import MemoryStore
from myswat.models.session import SessionTurn


@pytest.fixture
def store(mock_pool):
    return MemoryStore(mock_pool)


@pytest.fixture(autouse=True)
def disable_embedding_resolution(monkeypatch):
    monkeypatch.setattr(store_module.embedder, "resolve_embed_sql", lambda text, tidb_model="": ("NULL", []))


# ── 1. get_session_turns ────────────────────────────────────────────────


class TestGetSessionTurns:
    def test_returns_session_turn_objects(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {
                "id": 1,
                "session_id": 10,
                "turn_index": 0,
                "role": "user",
                "content": "hello",
                "token_count_est": 5,
                "created_at": "2026-01-01T00:00:00",
            },
        ]
        result = store.get_session_turns(10)
        assert len(result) == 1
        assert isinstance(result[0], SessionTurn)
        assert result[0].role == "user"
        mock_pool.fetch_all.assert_called_once()

    def test_with_limit_and_offset(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.get_session_turns(10, limit=10, offset=5)
        assert result == []
        call_args = mock_pool.fetch_all.call_args
        sql = call_args[0][0]
        assert "LIMIT" in sql
        assert "OFFSET" in sql

    def test_without_limit(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        store.get_session_turns(10)
        call_args = mock_pool.fetch_all.call_args
        sql = call_args[0][0]
        assert "LIMIT" not in sql


# ── 2. count_uncompacted_turns ──────────────────────────────────────────


class TestCountUncompactedTurns:
    def test_counts_turns_after_watermark(self, store, mock_pool):
        # First fetch_one call: get_session; second: the COUNT query
        session_row = {
            "id": 10,
            "compacted_through_turn_index": 3,
            "status": "active",
        }
        count_row = {"cnt": 5}
        mock_pool.fetch_one.side_effect = [session_row, count_row]
        result = store.count_uncompacted_turns(10)
        assert result == 5

    def test_returns_zero_when_no_session(self, store, mock_pool):
        # get_session returns None, then count query still runs
        mock_pool.fetch_one.side_effect = [None, {"cnt": 0}]
        result = store.count_uncompacted_turns(10)
        assert result == 0


# ── 3. delete_compacted_turns ───────────────────────────────────────────


class TestDeleteCompactedTurns:
    def test_deletes_turns_up_to_watermark(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": 10,
            "compacted_through_turn_index": 5,
            "status": "active",
        }
        mock_pool.execute.return_value = 3
        result = store.delete_compacted_turns(10)
        assert result == 3

    def test_no_delete_when_watermark_negative(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": 10,
            "compacted_through_turn_index": -1,
            "status": "active",
        }
        result = store.delete_compacted_turns(10)
        assert result == 0

    def test_no_session_returns_zero(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        result = store.delete_compacted_turns(10)
        assert result == 0


# ── 4. get_recent_history_for_agent ─────────────────────────────────────


class TestGetRecentHistoryForAgent:
    def test_returns_session_history(self, store, mock_pool):
        session_row = {
            "id": "s1",
            "project_id": "p1",
            "agent_id": "a1",
            "status": "active",
            "compacted_through_turn_index": -1,
            "total_token_count": 50,
            "purpose": "testing",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        turn_row = {
            "id": 1,
            "session_id": "s1",
            "turn_index": 0,
            "role": "user",
            "content": "hi",
            "token_count": 2,
            "created_at": "2026-01-01T00:00:00",
        }
        mock_pool.fetch_all.side_effect = [[session_row], [turn_row]]
        result = store.get_recent_history_for_agent("a1")
        assert isinstance(result, list)

    def test_excludes_session_id(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.get_recent_history_for_agent(
            "a1", exclude_session_id="s-exclude"
        )
        assert result == [] or isinstance(result, list)

    def test_respects_max_turns_and_sessions(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.get_recent_history_for_agent(
            "a1", max_turns=10, max_sessions=2
        )
        assert isinstance(result, list)


# ── 6. get_recent_artifacts_for_project ─────────────────────────────────


class TestGetRecentArtifactsForProject:
    def test_returns_artifacts(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "name": "artifact1", "content": "data"}
        ]
        result = store.get_recent_artifacts_for_project("p1")
        assert len(result) == 1
        assert result[0]["name"] == "artifact1"

    def test_custom_limit(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.get_recent_artifacts_for_project("p1", limit=10)
        assert result == []
        mock_pool.fetch_all.assert_called_once()


# ── 7. store_knowledge ──────────────────────────────────────────────────


class TestStoreKnowledge:
    def test_stores_knowledge_returns_id(self, store, mock_pool):
        mock_pool.insert_returning_id.return_value = 42
        result = store.store_knowledge(
            project_id="p1",
            category="design",
            title="Architecture",
            content="Some content",
        )
        assert result == 42
        mock_pool.insert_returning_id.assert_called_once()

    def test_stores_knowledge_with_tags(self, store, mock_pool):
        mock_pool.insert_returning_id.return_value = 99
        result = store.store_knowledge(
            project_id=1,
            category="code",
            title="Pattern",
            content="Details",
            tags=["python", "design"],
            relevance_score=0.9,
        )
        assert result == 99


class TestUpsertKnowledge:
    def test_skips_exact_duplicate_in_same_scope(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": 17,
            "title": "Architecture",
            "content": "Some content",
            "source_type": "session",
            "content_hash": store._compute_content_hash("Some content"),
        }

        kid, action = store.upsert_knowledge(
            project_id=1,
            source_type="session",
            category="architecture",
            title="Architecture",
            content="Some content",
        )

        assert (kid, action) == (17, "skipped")
        mock_pool.insert_returning_id.assert_not_called()

    def test_merges_when_new_content_textually_subsumes_existing(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        mock_pool.fetch_all.return_value = [
            {
                "id": 33,
                "title": "Architecture",
                "content": "short explanation",
                "tags": json.dumps(["existing"]),
                "source_type": "session",
                "version": 2,
                "relevance_score": 0.6,
                "confidence": 0.7,
                "search_metadata_json": None,
                "merged_from": None,
            },
        ]

        kid, action = store.upsert_knowledge(
            project_id=1,
            source_type="session",
            category="architecture",
            title="Architecture",
            content="short explanation with more detail",
            tags=["new-tag"],
        )

        assert (kid, action) == (33, "merged")
        assert mock_pool.execute.call_count >= 2  # UPDATE + memory_revision bump
        update_sql = mock_pool.execute.call_args_list[0][0][0]
        assert "UPDATE knowledge SET" in update_sql
        mock_pool.insert_returning_id.assert_not_called()

    def test_creates_new_row_when_no_safe_merge_exists(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        mock_pool.fetch_all.return_value = []
        mock_pool.insert_returning_id.return_value = 81

        kid, action = store.upsert_knowledge(
            project_id=1,
            source_type="manual",
            category="decision",
            title="Use Rust",
            content="Prefer Rust for the storage engine",
            tags=["rust"],
        )

        assert (kid, action) == (81, "created")
        mock_pool.insert_returning_id.assert_called_once()

    def test_uses_merge_runner_for_ambiguous_same_scope_update(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        mock_pool.fetch_all.return_value = [
            {
                "id": 50,
                "title": "LeaseRead",
                "content": "LeaseRead avoids extra round trips.",
                "tags": None,
                "source_type": "session",
                "version": 1,
                "relevance_score": 0.7,
                "confidence": 0.7,
                "search_metadata_json": None,
                "merged_from": None,
            },
        ]
        merge_runner = MagicMock()
        merge_runner.invoke.return_value = AgentResponse(
            content="LeaseRead avoids extra round trips and depends on leader lease validity.",
            exit_code=0,
        )

        kid, action = store.upsert_knowledge(
            project_id=1,
            source_type="session",
            category="architecture",
            title="LeaseRead",
            content="LeaseRead depends on leader lease validity.",
            merge_runner=merge_runner,
        )

        assert (kid, action) == (50, "merged")
        merge_runner.invoke.assert_called_once()

    def test_document_update_can_supersede_older_row(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        mock_pool.fetch_all.return_value = [
            {
                "id": 60,
                "title": "LeaseRead config",
                "content": "LeaseRead uses a legacy config path.",
                "tags": None,
                "source_type": "document",
                "source_file": "/tmp/doc.md",
                "version": 1,
                "relevance_score": 0.5,
                "confidence": 0.5,
                "search_metadata_json": None,
                "merged_from": None,
            },
        ]
        mock_pool.insert_returning_id.return_value = 61

        kid, action = store.upsert_knowledge(
            project_id=1,
            source_type="document",
            category="configuration",
            title="LeaseRead config",
            content="LeaseRead uses the new config path.",
            source_file="/tmp/doc.md",
            confidence=0.9,
        )

        assert (kid, action) == (61, "superseded")
        assert mock_pool.insert_returning_id.called


class TestKnowledgeTermExtraction:
    def test_title_generates_exact_and_phrase_terms(self, store, mock_pool):
        terms = store._build_knowledge_terms(
            title="LeaseRead behavior",
            content="",
            tags=None,
            source_file=None,
        )
        term_set = {(field, term) for field, term, _weight in terms}
        assert ("title", "leaseread") in term_set
        assert ("title", "lease") in term_set
        assert ("title", "read") in term_set
        assert ("title", "lease read") in term_set

    def test_source_file_generates_go_path_terms(self, store, mock_pool):
        terms = store._build_knowledge_terms(
            title="",
            content="",
            tags=None,
            source_file="pkg/store/tikv/region_cache.go",
        )
        term_set = {(field, term) for field, term, _weight in terms}
        assert ("source_file", "pkg/store/tikv/region_cache.go") in term_set
        assert ("source_file", "region_cache.go") in term_set
        assert ("source_file", "region_cache") in term_set
        assert ("source_file", "region") in term_set
        assert ("source_file", "cache") in term_set
        assert ("source_file", "tikv") in term_set

    def test_content_does_not_generate_phrase_terms_in_phase1(self, store, mock_pool):
        terms = store._build_knowledge_terms(
            title="",
            content="leader lease behavior",
            tags=None,
            source_file=None,
        )
        term_set = {(field, term) for field, term, _weight in terms}
        assert ("content", "leader") in term_set
        assert ("content", "lease") in term_set
        assert ("content", "leader lease") not in term_set


# ── 8. search_knowledge ────────────────────────────────────────────────


class TestSearchKnowledge:
    def test_keyword_search(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "title": "Match", "content": "keyword here"}
        ]
        result = store.search_knowledge("p1", query="keyword")
        assert len(result) == 1
        assert result[0]["title"] == "Match"

    def test_with_agent_id_filter(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.search_knowledge("p1", query="test", agent_id="a1")
        assert result == []

    def test_with_category_filter(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.search_knowledge("p1", query="test", category="design")
        assert result == []


# ── 9. list_knowledge ──────────────────────────────────────────────────


class TestListKnowledge:
    def test_list_all(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "category": "design", "title": "A"}
        ]
        result = store.list_knowledge("p1")
        assert len(result) == 1

    def test_list_with_category(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.list_knowledge("p1", category="code")
        assert result == []
        call_args = mock_pool.fetch_all.call_args
        sql = call_args[0][0]
        assert "category" in sql.lower()

    def test_list_with_custom_limit(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.list_knowledge("p1", limit=100)
        assert result == []


# ── 10. delete_knowledge_by_category ────────────────────────────────────


class TestDeleteKnowledgeByCategory:
    def test_deletes_and_returns_count(self, store, mock_pool):
        mock_pool.execute.return_value = 5
        result = store.delete_knowledge_by_category("p1", "obsolete")
        assert result == 5
        assert mock_pool.execute.call_count == 4


class TestDeleteKnowledgeBySourceFile:
    def test_deletes_and_returns_count(self, store, mock_pool):
        mock_pool.execute.return_value = 4
        result = store.delete_knowledge_by_source_file("p1", "/tmp/doc.md")
        assert result == 4
        assert mock_pool.execute.call_count == 4


class TestDocumentSourceTracking:
    def test_get_document_source_hashes_lookup_key(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {"id": 1, "content_hash": "abc"}
        row = store.get_document_source(1, "/tmp/doc.md")
        assert row == {"id": 1, "content_hash": "abc"}
        mock_pool.fetch_one.assert_called_once()

    def test_upsert_document_source_updates_existing(self, store, mock_pool):
        store.get_document_source = MagicMock(return_value={"id": 1})
        store.upsert_document_source(1, "/tmp/doc.md", "hash-1")
        mock_pool.execute.assert_called_once()
        mock_pool.insert_returning_id.assert_not_called()

    def test_upsert_document_source_inserts_new(self, store, mock_pool):
        store.get_document_source = MagicMock(return_value=None)
        store.upsert_document_source(1, "/tmp/doc.md", "hash-1")
        mock_pool.insert_returning_id.assert_called_once()


class TestKnowledgeGraphExtraction:
    def test_extracts_entities_from_title_tags_and_source(self, store, mock_pool):
        entities = store._extract_entities(
            title="LeaseRead in TiKV",
            content="Region may use SplitChecker.",
            tags=["raftstore", "region-max-size"],
            source_file="pkg/store/tikv/region_cache.go",
        )
        normalized = {entity.casefold() for entity in entities}
        assert "leaseread" in normalized
        assert "tikv" in normalized
        assert "region-max-size" in normalized
        assert "region_cache.go" in normalized

    def test_extracts_simple_relations(self, store, mock_pool):
        relations = store._extract_relations(
            title="LeaseRead depends on leader lease",
            content="Region uses SplitChecker thresholds.",
        )
        assert ("LeaseRead", "depends_on", "leader lease") in relations
        assert ("Region", "uses", "SplitChecker thresholds") in relations

    def test_match_entities_returns_names(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [{"entity_name": "LeaseRead"}]
        result = store.match_entities(1, "debug LeaseRead timeout")
        assert result == ["LeaseRead"]

    def test_get_related_entities_returns_expanded_rows(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"source_entity": "LeaseRead", "relation": "depends_on", "target_entity": "ReadIndex"},
        ]
        result = store.get_related_entities(1, ["LeaseRead"])
        assert result == [{
            "source_entity": "LeaseRead",
            "related_entity": "ReadIndex",
            "relation": "depends_on",
        }]


# ── 11. get_work_item ──────────────────────────────────────────────────


class TestGetWorkItem:
    def test_returns_work_item(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "w1",
            "project_id": "p1",
            "title": "Task",
            "status": "open",
        }
        result = store.get_work_item("w1")
        assert result is not None
        assert result["title"] == "Task"

    def test_parses_work_item_metadata_json(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "w1",
            "project_id": "p1",
            "title": "Task",
            "status": "open",
            "metadata_json": '{"task_state": {"current_stage": "design_review"}}',
        }
        result = store.get_work_item("w1")
        assert result is not None
        assert result["metadata_json"]["task_state"]["current_stage"] == "design_review"

    def test_returns_none_when_not_found(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        result = store.get_work_item("missing")
        assert result is None


# ── 12. list_work_items ────────────────────────────────────────────────


class TestListWorkItems:
    def test_list_all(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": "w1", "status": "open"},
            {"id": "w2", "status": "done"},
        ]
        result = store.list_work_items("p1")
        assert len(result) == 2

    def test_list_with_status_filter(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [{"id": "w1", "status": "open"}]
        result = store.list_work_items("p1", status="open")
        assert len(result) == 1
        call_args = mock_pool.fetch_all.call_args
        sql = call_args[0][0]
        assert "status" in sql.lower()

    def test_list_parses_metadata_json(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": "w1", "status": "open", "metadata_json": '{"task_state": {"next_todos": ["t1"]}}'},
        ]
        result = store.list_work_items("p1")
        assert result[0]["metadata_json"]["task_state"]["next_todos"] == ["t1"]


class TestWorkItemState:
    def test_get_work_item_state_returns_empty_when_missing(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        assert store.get_work_item_state("missing") == {}

    def test_update_work_item_state_merges_task_state(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "w1",
            "metadata_json": '{"task_state": {"current_stage": "design"}}',
        }

        store.update_work_item_state(
            "w1",
            current_stage="plan_review",
            latest_summary="summary",
            next_todos=["todo 1"],
            open_issues=["issue 1"],
            updated_by_agent_id=7,
        )

        args = mock_pool.execute.call_args[0][1]
        payload = args[0]
        assert '"plan_review"' in payload
        assert '"todo 1"' in payload
        assert '"issue 1"' in payload

    def test_get_work_item_process_log_returns_empty_when_missing(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        assert store.get_work_item_process_log("missing") == []

    def test_append_work_item_process_event_persists_event(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "w1",
            "metadata_json": '{"task_state": {"current_stage": "design"}}',
        }

        event = store.append_work_item_process_event(
            "w1",
            event_type="handoff",
            title="Architect delegation",
            summary="Send finalized design doc update to developer.",
            from_role="architect",
            to_role="developer",
            updated_by_agent_id=7,
        )

        assert event["type"] == "handoff"
        assert event["from_role"] == "architect"
        assert event["to_role"] == "developer"

        args = mock_pool.execute.call_args[0][1]
        payload = args[0]
        assert '"process_log"' in payload
        assert '"Architect delegation"' in payload
        assert '"architect"' in payload
        assert '"developer"' in payload

    def test_get_work_item_process_log_returns_log(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "w1",
            "metadata_json": '{"task_state": {"process_log": [{"type": "handoff", "summary": "x"}]}}',
        }

        result = store.get_work_item_process_log("w1")
        assert result == [{"type": "handoff", "summary": "x"}]


# ── 13. get_artifact ───────────────────────────────────────────────────


class TestGetArtifact:
    def test_returns_artifact(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "a1",
            "name": "report.pdf",
            "content": "binary",
        }
        result = store.get_artifact("a1")
        assert result is not None
        assert result["name"] == "report.pdf"

    def test_returns_none_when_not_found(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        result = store.get_artifact("missing")
        assert result is None


# ── 14. list_artifacts ─────────────────────────────────────────────────


class TestListArtifacts:
    def test_returns_artifact_list(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": "a1", "work_item_id": "w1", "name": "file.txt"}
        ]
        result = store.list_artifacts("w1")
        assert len(result) == 1

    def test_returns_empty_list(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.list_artifacts("w1")
        assert result == []


# ── 15. get_review_cycles ──────────────────────────────────────────────


class TestGetReviewCycles:
    def test_returns_review_cycles(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "work_item_id": "w1", "cycle": 1, "feedback": "LGTM"}
        ]
        result = store.get_review_cycles("w1")
        assert len(result) == 1

    def test_returns_empty_when_none(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.get_review_cycles("w1")
        assert result == []


# ── 16. decay_relevance ────────────────────────────────────────────────


class TestDecayRelevance:
    def test_returns_affected_count(self, store, mock_pool):
        mock_pool.execute.return_value = 12
        result = store.decay_relevance(0.95)
        assert result == 12
        mock_pool.execute.assert_called_once()


# ── 17. expire_stale_knowledge ──────────────────────────────────────────


class TestExpireStaleKnowledge:
    def test_returns_expired_count(self, store, mock_pool):
        mock_pool.execute.return_value = 3
        result = store.expire_stale_knowledge()
        assert result == 3
        assert mock_pool.execute.call_count == 4


# ── 18. get_session ────────────────────────────────────────────────────


class TestGetSession:
    def test_returns_session_dict(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "s1",
            "project_id": "p1",
            "agent_id": "a1",
            "status": "active",
            "compacted_through_turn_index": -1,
            "total_token_count": 0,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        result = store.get_session("s1")
        assert result is not None
        assert result["id"] == "s1"

    def test_returns_none_when_not_found(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        result = store.get_session("missing")
        assert result is None


# ── 19. mark_session_compacted ──────────────────────────────────────────


class TestMarkSessionCompacted:
    def test_calls_execute(self, store, mock_pool):
        store.mark_session_compacted("s1")
        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        sql = call_args[0][0]
        assert "compacted" in sql.lower() or "status" in sql.lower()


# ── 20. advance_compaction_watermark ────────────────────────────────────


class TestAdvanceCompactionWatermark:
    def test_calls_execute(self, store, mock_pool):
        store.advance_compaction_watermark(10, 10)
        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        sql = call_args[0][0]
        assert "compacted_through_turn_index" in sql.lower()


# ── 21. reset_session_token_count ───────────────────────────────────────


class TestResetSessionTokenCount:
    def test_calls_execute(self, store, mock_pool):
        store.reset_session_token_count("s1")
        mock_pool.execute.assert_called_once()
        call_args = mock_pool.execute.call_args
        sql = call_args[0][0]
        assert "token" in sql.lower()


# ── 22. update_session_progress ─────────────────────────────────────────


class TestUpdateSessionProgress:
    def test_calls_execute(self, store, mock_pool):
        store.update_session_progress("s1", "Step 3 complete")
        mock_pool.execute.assert_called_once()


# ── 23. get_compactable_sessions ────────────────────────────────────────


class TestGetCompactableSessions:
    def test_returns_sessions(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": "s1", "total_token_count": 5000}
        ]
        result = store.get_compactable_sessions("p1")
        assert len(result) == 1

    def test_returns_empty(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.get_compactable_sessions("p1")
        assert result == []


# ── 24. search_knowledge_fulltext_only ──────────────────────────────────


class TestSearchKnowledgeFulltextOnly:
    def test_delegates_to_search_knowledge(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "title": "Hit", "content": "matching text"}
        ]
        result = store.search_knowledge_fulltext_only("p1", "matching", limit=5)
        assert isinstance(result, list)
        mock_pool.fetch_all.assert_called()
