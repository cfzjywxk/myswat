"""Schema migration registration tests."""

from myswat.db.schema import MIGRATION_MODULES
from myswat.db.migrations import v006_flexible_vector_dimension
from myswat.db.migrations import v005_review_cycles_artifact_unique_key
from myswat.db.migrations import v007_conversation_persistence
from myswat.db.migrations import v008_chat_workflow_agent_prompts


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
    assert idx_v006 < idx_v007 < idx_v008
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


def test_chat_workflow_agent_prompt_migration_registered_after_v007() -> None:
    assert MIGRATION_MODULES[-1] == "myswat.db.migrations.v008_chat_workflow_agent_prompts"
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
