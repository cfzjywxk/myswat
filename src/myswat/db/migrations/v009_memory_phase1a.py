"""v009: Phase 1a memory schema foundations.

Adds source-aware knowledge metadata and project memory revision tracking for the
Phase 1a write-path redesign.
"""

VERSION = 9
DESCRIPTION = "Add knowledge source metadata, merge bookkeeping, and project memory revision"

STATEMENTS = [
    """
    ALTER TABLE projects
    ADD COLUMN memory_revision BIGINT NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE knowledge
    ADD COLUMN source_type VARCHAR(32) NOT NULL DEFAULT 'session'
    """,
    """
    ALTER TABLE knowledge
    ADD COLUMN content_hash CHAR(64) DEFAULT NULL
    """,
    """
    ALTER TABLE knowledge
    ADD COLUMN version INT NOT NULL DEFAULT 1
    """,
    """
    ALTER TABLE knowledge
    ADD COLUMN search_metadata_json JSON DEFAULT NULL
    """,
    """
    ALTER TABLE knowledge
    ADD COLUMN merged_from JSON DEFAULT NULL
    """,
    """
    CREATE INDEX idx_knowledge_source_scope
    ON knowledge (project_id, source_type, category, source_file(255))
    """,
    """
    CREATE INDEX idx_knowledge_content_hash
    ON knowledge (project_id, content_hash)
    """,
    """
    UPDATE knowledge
    SET source_type = 'document'
    WHERE source_file IS NOT NULL
    """,
    """
    UPDATE knowledge
    SET source_type = 'manual'
    WHERE category = 'project_ops'
    """,
]
