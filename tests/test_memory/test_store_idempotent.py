"""Tests for MemoryStore idempotent operations and error handling.

Covers:
- append_turn retry on duplicate turn_index (race condition)
- create_review_cycle returns existing on duplicate (artifact_id, reviewer_agent_id)
- create_artifact upserts on duplicate (work_item_id, agent_id, iteration, artifact_type)
- create_project returns existing on duplicate slug
- create_agent returns existing on duplicate (project_id, role)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pymysql.err
import pytest

from myswat.memory.store import MemoryStore


def _integrity_error(code=1062, msg="Duplicate entry"):
    return pymysql.err.IntegrityError(code, msg)


class TestAppendTurnIdempotent:
    def test_normal_append(self, mock_pool):
        """Normal case: no collision, insert succeeds first try."""
        mock_pool.fetch_one.return_value = {"next_idx": 0}
        mock_pool.insert_returning_id.return_value = 42

        store = MemoryStore(mock_pool)
        turn = store.append_turn(session_id=1, role="user", content="hello",
                                  token_count_est=5)

        assert turn.id == 42
        assert turn.turn_index == 0
        assert turn.role == "user"
        assert turn.content == "hello"
        # Token count update called
        mock_pool.execute.assert_called_once()

    def test_retry_on_duplicate_turn_index(self, mock_pool):
        """Race condition: first INSERT hits dup key, retry succeeds."""
        mock_pool.fetch_one.side_effect = [
            {"next_idx": 3},  # First attempt: thinks index 3 is next
            {"next_idx": 4},  # Retry: index 4 is now next
        ]
        mock_pool.insert_returning_id.side_effect = [
            _integrity_error(),  # First attempt fails
            99,                  # Second attempt succeeds
        ]

        store = MemoryStore(mock_pool)
        turn = store.append_turn(session_id=1, role="assistant", content="hi")

        assert turn.id == 99
        assert turn.turn_index == 4
        assert mock_pool.insert_returning_id.call_count == 2

    def test_max_retries_exceeded(self, mock_pool):
        """All 3 attempts fail with duplicate key → RuntimeError."""
        mock_pool.fetch_one.return_value = {"next_idx": 0}
        mock_pool.insert_returning_id.side_effect = _integrity_error()

        store = MemoryStore(mock_pool)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            store.append_turn(session_id=1, role="user", content="test")

    def test_non_duplicate_error_propagates(self, mock_pool):
        """IntegrityError with non-1062 code propagates immediately."""
        mock_pool.fetch_one.return_value = {"next_idx": 0}
        mock_pool.insert_returning_id.side_effect = pymysql.err.IntegrityError(
            1452, "Cannot add or update a child row: a foreign key constraint fails"
        )

        store = MemoryStore(mock_pool)
        with pytest.raises(pymysql.err.IntegrityError) as exc_info:
            store.append_turn(session_id=1, role="user", content="test")
        assert exc_info.value.args[0] == 1452

    def test_metadata_serialized(self, mock_pool):
        """Metadata dict is JSON-serialized before insert."""
        mock_pool.fetch_one.return_value = {"next_idx": 0}
        mock_pool.insert_returning_id.return_value = 1

        store = MemoryStore(mock_pool)
        turn = store.append_turn(
            session_id=1, role="assistant", content="hi",
            metadata={"elapsed_seconds": 1.5},
        )

        # Verify JSON was passed
        insert_call = mock_pool.insert_returning_id.call_args
        args = insert_call[0][1]  # positional args tuple
        assert '"elapsed_seconds"' in args[5]  # metadata_json position

    def test_token_count_updated(self, mock_pool):
        """Session token count is incremented after successful insert."""
        mock_pool.fetch_one.return_value = {"next_idx": 0}
        mock_pool.insert_returning_id.return_value = 1

        store = MemoryStore(mock_pool)
        store.append_turn(session_id=7, role="user", content="test",
                          token_count_est=100)

        mock_pool.execute.assert_called_once()
        execute_args = mock_pool.execute.call_args[0]
        assert "token_count_est" in execute_args[0]
        assert execute_args[1] == (100, 7)


class TestCreateReviewCycleIdempotent:
    def test_normal_create(self, mock_pool):
        """Normal case: no duplicate, returns new ID."""
        mock_pool.insert_returning_id.return_value = 42

        store = MemoryStore(mock_pool)
        cid = store.create_review_cycle(
            work_item_id=1, iteration=1,
            proposer_agent_id=1, reviewer_agent_id=2,
            artifact_id=10,
        )

        assert cid == 42

    def test_duplicate_returns_existing_id(self, mock_pool):
        """Duplicate (artifact_id, reviewer_agent_id) returns existing."""
        mock_pool.insert_returning_id.side_effect = _integrity_error()
        mock_pool.fetch_one.return_value = {"id": 77}

        store = MemoryStore(mock_pool)
        cid = store.create_review_cycle(
            work_item_id=1, iteration=1,
            proposer_agent_id=1, reviewer_agent_id=2,
            artifact_id=10,
        )

        assert cid == 77
        # Verify it queried for the existing cycle
        fetch_call = mock_pool.fetch_one.call_args
        assert "artifact_id" in fetch_call[0][0]
        assert "reviewer_agent_id" in fetch_call[0][0]

    def test_duplicate_but_not_found_raises(self, mock_pool):
        """Duplicate key but existing row not found (shouldn't happen) → re-raises."""
        mock_pool.insert_returning_id.side_effect = _integrity_error()
        mock_pool.fetch_one.return_value = None

        store = MemoryStore(mock_pool)
        with pytest.raises(pymysql.err.IntegrityError):
            store.create_review_cycle(
                work_item_id=1, iteration=1,
                proposer_agent_id=1, reviewer_agent_id=2,
                artifact_id=10,
            )

    def test_non_duplicate_error_propagates(self, mock_pool):
        """Non-1062 IntegrityError propagates."""
        mock_pool.insert_returning_id.side_effect = pymysql.err.IntegrityError(
            1452, "FK constraint"
        )

        store = MemoryStore(mock_pool)
        with pytest.raises(pymysql.err.IntegrityError) as exc_info:
            store.create_review_cycle(
                work_item_id=1, iteration=1,
                proposer_agent_id=1, reviewer_agent_id=2,
                artifact_id=10,
            )
        assert exc_info.value.args[0] == 1452

    def test_with_proposal_session_id(self, mock_pool):
        """proposal_session_id is passed through."""
        mock_pool.insert_returning_id.return_value = 1

        store = MemoryStore(mock_pool)
        store.create_review_cycle(
            work_item_id=1, iteration=2,
            proposer_agent_id=1, reviewer_agent_id=2,
            artifact_id=10, proposal_session_id=55,
        )

        args = mock_pool.insert_returning_id.call_args[0][1]
        assert 55 in args


class TestCreateArtifactIdempotent:
    def test_new_artifact_created(self, mock_pool):
        """No existing artifact → INSERT new row."""
        mock_pool.fetch_one.return_value = None  # No existing
        mock_pool.insert_returning_id.return_value = 42

        store = MemoryStore(mock_pool)
        aid = store.create_artifact(
            work_item_id=1, agent_id=1, iteration=1,
            artifact_type="proposal", content="my proposal",
        )

        assert aid == 42
        mock_pool.insert_returning_id.assert_called_once()

    def test_existing_artifact_updated(self, mock_pool):
        """Existing artifact with same signature → UPDATE content."""
        mock_pool.fetch_one.return_value = {"id": 99}

        store = MemoryStore(mock_pool)
        aid = store.create_artifact(
            work_item_id=1, agent_id=1, iteration=1,
            artifact_type="proposal", content="updated content",
            title="new title",
        )

        assert aid == 99
        mock_pool.insert_returning_id.assert_not_called()
        mock_pool.execute.assert_called_once()
        update_sql = mock_pool.execute.call_args[0][0]
        assert "UPDATE artifacts" in update_sql

    def test_metadata_json_serialized(self, mock_pool):
        """metadata_json dict is serialized to JSON string."""
        mock_pool.fetch_one.return_value = None
        mock_pool.insert_returning_id.return_value = 1

        store = MemoryStore(mock_pool)
        store.create_artifact(
            work_item_id=1, agent_id=1, iteration=1,
            artifact_type="diff", content="changes",
            metadata_json={"source": "review_loop"},
        )

        args = mock_pool.insert_returning_id.call_args[0][1]
        assert '"source"' in args[-1]  # Last arg is metadata_json


class TestCreateProjectIdempotent:
    def test_normal_create(self, mock_pool):
        mock_pool.insert_returning_id.return_value = 5

        store = MemoryStore(mock_pool)
        pid = store.create_project(slug="test", name="Test")

        assert pid == 5

    def test_duplicate_returns_existing(self, mock_pool):
        mock_pool.insert_returning_id.side_effect = _integrity_error()
        mock_pool.fetch_one.return_value = {"id": 3, "slug": "test"}

        store = MemoryStore(mock_pool)
        pid = store.create_project(slug="test", name="Test")

        assert pid == 3


class TestCreateAgentIdempotent:
    def test_normal_create(self, mock_pool):
        mock_pool.insert_returning_id.return_value = 10

        store = MemoryStore(mock_pool)
        aid = store.create_agent(
            project_id=1, role="developer", display_name="Dev",
            cli_backend="codex", model_name="gpt-5.4", cli_path="codex",
        )

        assert aid == 10

    def test_duplicate_returns_existing(self, mock_pool):
        mock_pool.insert_returning_id.side_effect = _integrity_error()
        mock_pool.fetch_one.return_value = {"id": 7, "role": "developer"}

        store = MemoryStore(mock_pool)
        aid = store.create_agent(
            project_id=1, role="developer", display_name="Dev",
            cli_backend="codex", model_name="gpt-5.4", cli_path="codex",
        )

        assert aid == 7


class TestStoreReadOperations:
    def test_get_project_by_slug(self, mock_pool):
        mock_pool.fetch_one.return_value = {"id": 1, "slug": "test"}

        store = MemoryStore(mock_pool)
        result = store.get_project_by_slug("test")

        assert result["id"] == 1

    def test_get_project_not_found(self, mock_pool):
        mock_pool.fetch_one.return_value = None

        store = MemoryStore(mock_pool)
        assert store.get_project_by_slug("nonexistent") is None

    def test_list_agents(self, mock_pool):
        mock_pool.fetch_all.return_value = [
            {"id": 1, "role": "developer"},
            {"id": 2, "role": "qa_main"},
        ]

        store = MemoryStore(mock_pool)
        agents = store.list_agents(1)

        assert len(agents) == 2

    def test_count_session_turns(self, mock_pool):
        mock_pool.fetch_one.return_value = {"cnt": 5}

        store = MemoryStore(mock_pool)
        assert store.count_session_turns(1) == 5

    def test_count_session_turns_empty(self, mock_pool):
        mock_pool.fetch_one.return_value = None

        store = MemoryStore(mock_pool)
        assert store.count_session_turns(1) == 0

    def test_close_session(self, mock_pool):
        store = MemoryStore(mock_pool)
        store.close_session(1)

        mock_pool.execute.assert_called_once()
        assert "completed" in mock_pool.execute.call_args[0][0]

    def test_delete_compacted_turns_no_session(self, mock_pool):
        mock_pool.fetch_one.return_value = None

        store = MemoryStore(mock_pool)
        assert store.delete_compacted_turns(1) == 0

    def test_delete_compacted_turns_no_watermark(self, mock_pool):
        mock_pool.fetch_one.return_value = {"compacted_through_turn_index": -1}

        store = MemoryStore(mock_pool)
        assert store.delete_compacted_turns(1) == 0

    def test_create_work_item(self, mock_pool):
        mock_pool.insert_returning_id.return_value = 42

        store = MemoryStore(mock_pool)
        wid = store.create_work_item(
            project_id=1, title="Task", item_type="code_change",
        )

        assert wid == 42

    def test_update_work_item_status(self, mock_pool):
        store = MemoryStore(mock_pool)
        store.update_work_item_status(1, "completed")

        mock_pool.execute.assert_called_once()

    def test_update_review_verdict(self, mock_pool):
        store = MemoryStore(mock_pool)
        store.update_review_verdict(
            cycle_id=1, verdict="lgtm",
            verdict_json={"verdict": "lgtm"}, review_session_id=5,
        )

        mock_pool.execute.assert_called_once()

    def test_list_knowledge(self, mock_pool):
        mock_pool.fetch_all.return_value = [{"id": 1, "title": "test"}]

        store = MemoryStore(mock_pool)
        results = store.list_knowledge(1, category="project_ops")

        assert len(results) == 1

    def test_delete_knowledge_by_category(self, mock_pool):
        mock_pool.execute.return_value = 3

        store = MemoryStore(mock_pool)
        deleted = store.delete_knowledge_by_category(1, "project_ops")

        assert deleted == 3

    def test_store_knowledge_without_embedding(self, mock_pool):
        mock_pool.insert_returning_id.return_value = 99

        store = MemoryStore(mock_pool)
        kid = store.store_knowledge(
            project_id=1, category="test", title="Test",
            content="content", compute_embedding=False,
        )

        assert kid == 99

    @patch("myswat.memory.embedder.embed", return_value=None)
    def test_store_knowledge_tidb_fallback(self, mock_embed, mock_pool):
        """When local embed is unavailable and tidb_model is set, uses EMBEDDING()."""
        mock_pool.insert_returning_id.return_value = 42

        store = MemoryStore(mock_pool, tidb_embedding_model="built-in")
        kid = store.store_knowledge(
            project_id=1, category="test", title="Test", content="content",
        )

        assert kid == 42
        sql = mock_pool.insert_returning_id.call_args[0][0]
        assert "EMBEDDING('built-in', %s)" in sql

    @patch("myswat.memory.embedder.embed", return_value=None)
    def test_store_knowledge_no_fallback_when_tidb_model_empty(self, mock_embed, mock_pool):
        """When local embed is unavailable and no tidb_model, stores NULL."""
        mock_pool.insert_returning_id.return_value = 42

        store = MemoryStore(mock_pool, tidb_embedding_model="")
        kid = store.store_knowledge(
            project_id=1, category="test", title="Test", content="content",
        )

        assert kid == 42
        sql = mock_pool.insert_returning_id.call_args[0][0]
        assert "NULL" in sql
        assert "EMBEDDING" not in sql

    @patch("myswat.memory.embedder.embed", return_value=None)
    def test_search_knowledge_tidb_fallback(self, mock_embed, mock_pool):
        """When local embed unavailable, search uses TiDB EMBEDDING() for vector."""
        mock_pool.fetch_all.return_value = []

        store = MemoryStore(mock_pool, tidb_embedding_model="built-in")
        store.search_knowledge(project_id=1, query="test query")

        sql = mock_pool.fetch_all.call_args[0][0]
        assert "EMBEDDING('built-in', %s)" in sql

    @patch("myswat.memory.embedder.embed", return_value=None)
    def test_search_knowledge_no_vector_without_fallback(self, mock_embed, mock_pool):
        """When no local embed and no tidb_model, vector search is skipped."""
        mock_pool.fetch_all.return_value = []

        store = MemoryStore(mock_pool, tidb_embedding_model="")
        store.search_knowledge(project_id=1, query="test query")

        sql = mock_pool.fetch_all.call_args[0][0]
        assert "VEC_COSINE_DISTANCE" not in sql
        assert "EMBEDDING" not in sql

    def test_create_session(self, mock_pool):
        mock_pool.insert_returning_id.return_value = 10

        store = MemoryStore(mock_pool)
        session = store.create_session(agent_id=1, purpose="test")

        assert session.id == 10
        assert session.agent_id == 1
        assert session.purpose == "test"

    def test_get_active_session(self, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": 5, "agent_id": 1, "session_uuid": "abc-123",
            "parent_session_id": None, "status": "active",
            "purpose": "test", "work_item_id": None,
            "token_count_est": 0, "compacted_through_turn_index": -1,
            "created_at": None, "updated_at": None,
        }

        store = MemoryStore(mock_pool)
        session = store.get_active_session(1)

        assert session is not None
        assert session.id == 5
        sql, args = mock_pool.fetch_one.call_args[0]
        assert "work_item_id IS NULL" in sql
        assert args == (1,)

    def test_get_active_session_none(self, mock_pool):
        mock_pool.fetch_one.return_value = None

        store = MemoryStore(mock_pool)
        assert store.get_active_session(1) is None

    def test_get_active_session_for_work_item(self, mock_pool):
        mock_pool.fetch_one.return_value = {
            "id": 6, "agent_id": 1, "session_uuid": "abc-456",
            "parent_session_id": None, "status": "active",
            "purpose": "test", "work_item_id": 42,
            "token_count_est": 0, "compacted_through_turn_index": -1,
            "created_at": None, "updated_at": None,
        }

        store = MemoryStore(mock_pool)
        session = store.get_active_session(1, work_item_id=42)

        assert session is not None
        assert session.work_item_id == 42
        sql, args = mock_pool.fetch_one.call_args[0]
        assert "work_item_id = %s" in sql
        assert args == (1, 42)
