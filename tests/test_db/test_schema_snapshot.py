"""Schema snapshot coverage tests."""

from myswat.db.schema import SCHEMA_STATEMENTS


SQL = "\n".join(SCHEMA_STATEMENTS)


def test_projects_snapshot_includes_memory_revision() -> None:
    assert "CREATE TABLE IF NOT EXISTS projects" in SQL
    assert "memory_revision BIGINT NOT NULL DEFAULT 0" in SQL


def test_sessions_snapshot_includes_conversation_fields() -> None:
    assert "compacted_through_turn_index" in SQL
    assert "compacted_at" in SQL
    assert "memory_revision_at_context_build" in SQL
    assert "idx_session_turns_recency" in SQL


def test_knowledge_snapshot_uses_final_vector_and_metadata_fields() -> None:
    assert "embedding                   VECTOR" in SQL
    assert "source_file                 VARCHAR(1024) DEFAULT NULL" in SQL
    assert "source_type                 VARCHAR(32) NOT NULL DEFAULT 'session'" in SQL
    assert "content_hash                CHAR(64) DEFAULT NULL" in SQL
    assert "search_metadata_json        JSON DEFAULT NULL" in SQL
    assert "merged_from                 JSON DEFAULT NULL" in SQL
    assert "idx_knowledge_source_scope" in SQL
    assert "idx_knowledge_content_hash" in SQL


def test_review_schema_uses_final_artifact_and_unique_key_shape() -> None:
    assert "ENUM('proposal', 'diff', 'patch', 'test_plan', 'design_doc', 'phase_result')" in SQL
    assert "updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP" in SQL
    assert "UNIQUE KEY uk_artifact_reviewer (artifact_id, reviewer_agent_id)" in SQL


def test_auxiliary_tables_exist_in_snapshot() -> None:
    assert "CREATE TABLE IF NOT EXISTS knowledge_terms" in SQL
    assert "CREATE TABLE IF NOT EXISTS document_sources" in SQL
    assert "CREATE TABLE IF NOT EXISTS knowledge_entities" in SQL
    assert "CREATE TABLE IF NOT EXISTS knowledge_relations" in SQL
    assert "CREATE TABLE IF NOT EXISTS learn_requests" in SQL
    assert "CREATE TABLE IF NOT EXISTS learn_runs" in SQL


def test_document_sources_does_not_include_removed_redundant_index() -> None:
    assert "idx_project_source_file" not in SQL
