"""v010: Add lexical term index for knowledge search."""

VERSION = 10
DESCRIPTION = "Add knowledge_terms table for lexical technical search"

STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS knowledge_terms (
        project_id BIGINT NOT NULL,
        knowledge_id BIGINT NOT NULL,
        term VARCHAR(255) NOT NULL,
        field VARCHAR(32) NOT NULL,
        weight FLOAT NOT NULL DEFAULT 1.0,
        PRIMARY KEY (project_id, term, knowledge_id, field),
        INDEX idx_knowledge_terms_lookup (project_id, term),
        INDEX idx_knowledge_terms_knowledge (knowledge_id)
    )
    """,
]
