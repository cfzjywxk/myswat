"""v005: Make review cycle uniqueness follow the reviewed artifact."""

VERSION = 5
DESCRIPTION = "Scope review cycle uniqueness to artifact_id + reviewer_agent_id"

STATEMENTS = [
    """
    ALTER TABLE review_cycles
    DROP INDEX uk_work_iteration_reviewer,
    ADD UNIQUE KEY uk_artifact_reviewer (artifact_id, reviewer_agent_id)
    """,
]
