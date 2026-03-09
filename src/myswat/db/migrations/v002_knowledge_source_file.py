"""v002: Add source_file column to knowledge table for document ingestion provenance."""

VERSION = 2
DESCRIPTION = "Add source_file column to knowledge table"

STATEMENTS = [
    """
    ALTER TABLE knowledge ADD COLUMN source_file VARCHAR(1024) DEFAULT NULL
    """,
]
