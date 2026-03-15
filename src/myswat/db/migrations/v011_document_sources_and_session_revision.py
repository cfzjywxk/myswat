"""v011: Document source tracking and session memory revision bookkeeping."""

VERSION = 11
DESCRIPTION = "Add document source content tracking and session context memory revision"

STATEMENTS = [
    """
    ALTER TABLE sessions
    ADD COLUMN memory_revision_at_context_build BIGINT NULL DEFAULT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS document_sources (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id BIGINT NOT NULL,
        source_file VARCHAR(1024) NOT NULL,
        source_file_hash CHAR(64) NOT NULL,
        content_hash CHAR(64) NOT NULL,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_project_source_hash (project_id, source_file_hash),
        INDEX idx_project_source_file (project_id, source_file_hash)
    )
    """,
]
