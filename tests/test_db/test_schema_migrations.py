"""Schema migration registration tests."""

from myswat.db.schema import MIGRATION_MODULES
from myswat.db.migrations import v006_flexible_vector_dimension
from myswat.db.migrations import v005_review_cycles_artifact_unique_key
from myswat.db.migrations import v007_conversation_persistence
from myswat.db.migrations import v008_chat_workflow_agent_prompts
from myswat.db.migrations import v009_memory_phase1a
from myswat.db.migrations import v010_knowledge_terms
from myswat.db.migrations import v011_document_sources_and_session_revision
from myswat.db.migrations import v012_knowledge_graph
from myswat.db.migrations import v013_drop_redundant_document_sources_index
from myswat.db.migrations import v014_learn_requests_and_runs


def test_review_cycle_unique_key_migration_registered() -> None:
    assert "myswat.db.migrations.v005_review_cycles_artifact_unique_key" in MIGRATION_MODULES

    sql = "\n".join(
        statement for statement in v005_review_cycles_artifact_unique_key.STATEMENTS
        if isinstance(statement, str)
    )
    assert "DROP INDEX uk_work_iteration_reviewer" in sql
    assert "ADD UNIQUE KEY uk_artifact_reviewer (artifact_id, reviewer_agent_id)" in sql


def test_flexible_vector_dimension_migration_registered_before_v007() -> None:
    idx_v006 = MIGRATION_MODULES.index("myswat.db.migrations.v006_flexible_vector_dimension")
    idx_v007 = MIGRATION_MODULES.index("myswat.db.migrations.v007_conversation_persistence")
    idx_v008 = MIGRATION_MODULES.index("myswat.db.migrations.v008_chat_workflow_agent_prompts")
    idx_v009 = MIGRATION_MODULES.index("myswat.db.migrations.v009_memory_phase1a")
    assert idx_v006 < idx_v007 < idx_v008 < idx_v009
    assert v006_flexible_vector_dimension.VERSION == 6


def test_conversation_persistence_migration_defines_expected_statements() -> None:
    assert v007_conversation_persistence.VERSION == 7
    assert (
        v007_conversation_persistence.DESCRIPTION
        == "Add compacted_at timestamp and turn recency index for conversation persistence"
    )

    sql = "\n".join(v007_conversation_persistence.STATEMENTS)
    assert "ALTER TABLE sessions ADD COLUMN compacted_at TIMESTAMP NULL DEFAULT NULL" in sql
    assert "CREATE INDEX idx_session_turns_recency ON session_turns (created_at DESC, id DESC)" in sql


def test_chat_workflow_agent_prompt_migration_registered_before_phase1a() -> None:
    assert "myswat.db.migrations.v008_chat_workflow_agent_prompts" in MIGRATION_MODULES
    assert v008_chat_workflow_agent_prompts.VERSION == 8


def test_chat_workflow_agent_prompt_migration_defines_expected_updates() -> None:
    assert (
        v008_chat_workflow_agent_prompts.DESCRIPTION
        == "Backfill architect/developer/QA system prompts for chat-triggered workflow delegation"
    )
    statements = v008_chat_workflow_agent_prompts.STATEMENTS
    assert len(statements) == 5
    assert "role = 'architect'" in statements[0][0]
    assert "system_prompt IS NULL" in statements[0][0]
    assert "role = 'architect'" in statements[1][0]
    assert "system_prompt = %s" in statements[1][0]
    assert "role = 'developer'" in statements[2][0]
    assert "role = 'qa_main'" in statements[3][0]
    assert "role = 'qa_vice'" in statements[4][0]


def test_phase1a_memory_migration_registered_last() -> None:
    assert "myswat.db.migrations.v009_memory_phase1a" in MIGRATION_MODULES
    assert v009_memory_phase1a.VERSION == 9


def test_phase1a_memory_migration_defines_expected_schema_changes() -> None:
    sql = "\n".join(v009_memory_phase1a.STATEMENTS)
    assert "ALTER TABLE projects" in sql
    assert "ADD COLUMN memory_revision BIGINT NOT NULL DEFAULT 0" in sql
    assert "ADD COLUMN source_type VARCHAR(32)" in sql
    assert "ADD COLUMN content_hash CHAR(64)" in sql
    assert "ADD COLUMN version INT NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN search_metadata_json JSON DEFAULT NULL" in sql
    assert "CREATE INDEX idx_knowledge_source_scope" in sql
    assert "CREATE INDEX idx_knowledge_content_hash" in sql


def test_knowledge_terms_migration_registered_last() -> None:
    assert "myswat.db.migrations.v010_knowledge_terms" in MIGRATION_MODULES
    assert v010_knowledge_terms.VERSION == 10


def test_knowledge_terms_migration_defines_expected_schema() -> None:
    sql = "\n".join(v010_knowledge_terms.STATEMENTS)
    assert "CREATE TABLE IF NOT EXISTS knowledge_terms" in sql
    assert "term VARCHAR(255) NOT NULL" in sql
    assert "field VARCHAR(32) NOT NULL" in sql
    assert "weight FLOAT NOT NULL DEFAULT 1.0" in sql


def test_document_source_and_session_revision_migration_registered_before_graph() -> None:
    assert "myswat.db.migrations.v011_document_sources_and_session_revision" in MIGRATION_MODULES
    assert v011_document_sources_and_session_revision.VERSION == 11


def test_document_source_and_session_revision_migration_defines_expected_schema() -> None:
    sql = "\n".join(v011_document_sources_and_session_revision.STATEMENTS)
    assert "ALTER TABLE sessions" in sql
    assert "ADD COLUMN memory_revision_at_context_build BIGINT NULL DEFAULT NULL" in sql
    assert "CREATE TABLE IF NOT EXISTS document_sources" in sql
    assert "source_file_hash CHAR(64) NOT NULL" in sql
    assert "content_hash CHAR(64) NOT NULL" in sql


def test_graph_migration_registered_before_v013() -> None:
    assert "myswat.db.migrations.v012_knowledge_graph" in MIGRATION_MODULES
    assert v012_knowledge_graph.VERSION == 12


def test_graph_migration_defines_expected_schema() -> None:
    sql = "\n".join(v012_knowledge_graph.STATEMENTS)
    assert "CREATE TABLE IF NOT EXISTS knowledge_entities" in sql
    assert "CREATE TABLE IF NOT EXISTS knowledge_relations" in sql


def test_drop_redundant_document_sources_index_migration_registered_before_v014() -> None:
    idx_v013 = MIGRATION_MODULES.index(
        "myswat.db.migrations.v013_drop_redundant_document_sources_index"
    )
    idx_v014 = MIGRATION_MODULES.index("myswat.db.migrations.v014_learn_requests_and_runs")
    assert idx_v013 < idx_v014
    assert v013_drop_redundant_document_sources_index.VERSION == 13


def test_drop_redundant_document_sources_index_migration_defines_expected_sql() -> None:
    sql = "\n".join(v013_drop_redundant_document_sources_index.STATEMENTS)
    assert "DROP INDEX idx_project_source_file ON document_sources" in sql


def test_learn_request_run_migration_registered_last() -> None:
    assert MIGRATION_MODULES[-1] == "myswat.db.migrations.v014_learn_requests_and_runs"
    assert v014_learn_requests_and_runs.VERSION == 14


def test_learn_request_run_migration_defines_expected_schema() -> None:
    sql = "\n".join(v014_learn_requests_and_runs.STATEMENTS)
    assert "CREATE TABLE IF NOT EXISTS learn_requests" in sql
    assert "project_id BIGINT NOT NULL" in sql
    assert "trigger_kind VARCHAR(64) NOT NULL" in sql
    assert "payload_json JSON NOT NULL" in sql
    assert "CREATE TABLE IF NOT EXISTS learn_runs" in sql
    assert "learn_request_id BIGINT NOT NULL" in sql
    assert "output_envelope_json JSON NULL" in sql
