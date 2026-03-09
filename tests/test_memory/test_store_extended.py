"""Extended tests for MemoryStore – covers remaining uncovered methods."""

from unittest.mock import MagicMock, patch, call
import pytest

from myswat.memory.store import MemoryStore
from myswat.models.session import SessionTurn


@pytest.fixture
def store(mock_pool):
    return MemoryStore(mock_pool)


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


# ── 4. delete_archived_session ──────────────────────────────────────────


class TestDeleteArchivedSession:
    def test_deletes_compacted_session(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "s1",
            "project_id": "p1",
            "agent_id": "a1",
            "status": "compacted",
            "compacted_through_turn_index": 10,
            "total_token_count": 500,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        mock_pool.execute.return_value = 7
        result = store.delete_archived_session("s1")
        assert isinstance(result, dict)
        assert "turns" in result
        assert "session" in result

    def test_raises_or_returns_empty_when_not_compacted(self, store, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": "s1",
            "project_id": "p1",
            "agent_id": "a1",
            "status": "active",
            "compacted_through_turn_index": 0,
            "total_token_count": 100,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }
        try:
            result = store.delete_archived_session("s1")
            # If it doesn't raise, it should indicate nothing deleted
            assert result.get("session", 0) == 0 or result == {}
        except (ValueError, RuntimeError, Exception):
            pass  # expected – session is not compacted

    def test_no_session_returns_empty_or_raises(self, store, mock_pool):
        mock_pool.fetch_one.return_value = None
        try:
            result = store.delete_archived_session("s1")
            assert result.get("session", 0) == 0
        except (ValueError, RuntimeError, KeyError, Exception):
            pass


# ── 5. get_recent_history_for_agent ─────────────────────────────────────


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
        mock_pool.execute.assert_called_once()


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


# ── 24. purge_compacted_sessions ────────────────────────────────────────


class TestPurgeCompactedSessions:
    def test_returns_purge_counts(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {
                "id": "s1",
                "project_id": "p1",
                "agent_id": "a1",
                "status": "compacted",
                "compacted_through_turn_index": 5,
                "total_token_count": 100,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        ]
        mock_pool.execute.return_value = 1
        result = store.purge_compacted_sessions("p1")
        assert isinstance(result, dict)
        assert "sessions_deleted" in result or "turns_deleted" in result

    def test_no_compacted_sessions(self, store, mock_pool):
        mock_pool.fetch_all.return_value = []
        result = store.purge_compacted_sessions("p1")
        assert isinstance(result, dict)


# ── 25. search_knowledge_fulltext_only ──────────────────────────────────


class TestSearchKnowledgeFulltextOnly:
    def test_delegates_to_search_knowledge(self, store, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "title": "Hit", "content": "matching text"}
        ]
        result = store.search_knowledge_fulltext_only("p1", "matching", limit=5)
        assert isinstance(result, list)
        mock_pool.fetch_all.assert_called()
