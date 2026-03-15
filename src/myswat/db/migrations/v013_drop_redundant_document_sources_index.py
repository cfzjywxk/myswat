"""v013: Drop redundant non-unique index on document_sources."""

VERSION = 13
DESCRIPTION = "Drop redundant idx_project_source_file from document_sources"

STATEMENTS = [
    """
    DROP INDEX idx_project_source_file ON document_sources
    """,
]
