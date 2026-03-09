"""Schema migration registration tests."""

from myswat.db.schema import MIGRATION_MODULES
from myswat.db.migrations import v005_review_cycles_artifact_unique_key


def test_review_cycle_unique_key_migration_registered() -> None:
    assert MIGRATION_MODULES[-1] == "myswat.db.migrations.v005_review_cycles_artifact_unique_key"

    sql = "\n".join(
        statement for statement in v005_review_cycles_artifact_unique_key.STATEMENTS
        if isinstance(statement, str)
    )
    assert "DROP INDEX uk_work_iteration_reviewer" in sql
    assert "ADD UNIQUE KEY uk_artifact_reviewer (artifact_id, reviewer_agent_id)" in sql
