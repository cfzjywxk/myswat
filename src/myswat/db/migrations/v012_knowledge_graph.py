"""v012: Add lightweight knowledge entity and relation tables."""

VERSION = 12
DESCRIPTION = "Add knowledge_entities and knowledge_relations for graph-expanded retrieval"

STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS knowledge_entities (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id BIGINT NOT NULL,
        knowledge_id BIGINT NOT NULL,
        entity_name VARCHAR(255) NOT NULL,
        confidence FLOAT NOT NULL DEFAULT 1.0,
        INDEX idx_knowledge_entities_lookup (project_id, entity_name),
        INDEX idx_knowledge_entities_knowledge (knowledge_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_relations (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id BIGINT NOT NULL,
        knowledge_id BIGINT NOT NULL,
        source_entity VARCHAR(255) NOT NULL,
        relation VARCHAR(64) NOT NULL,
        target_entity VARCHAR(255) NOT NULL,
        confidence FLOAT NOT NULL DEFAULT 1.0,
        INDEX idx_knowledge_rel_source (project_id, source_entity),
        INDEX idx_knowledge_rel_target (project_id, target_entity),
        INDEX idx_knowledge_rel_knowledge (knowledge_id)
    )
    """,
]
