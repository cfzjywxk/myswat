"""Schema bootstrap for MySwat."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myswat.db.connection import TiDBPool

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS projects (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        slug            VARCHAR(128) NOT NULL UNIQUE,
        name            VARCHAR(256) NOT NULL,
        description     TEXT,
        repo_path       VARCHAR(512),
        config_json     JSON,
        memory_revision BIGINT NOT NULL DEFAULT 0,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agents (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id      BIGINT NOT NULL,
        role            VARCHAR(64) NOT NULL,
        display_name    VARCHAR(128) NOT NULL,
        cli_backend     VARCHAR(32) NOT NULL,
        model_name      VARCHAR(128) NOT NULL,
        cli_path        VARCHAR(512) NOT NULL,
        cli_extra_args  JSON,
        system_prompt   TEXT,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_project_role (project_id, role),
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id                                BIGINT AUTO_INCREMENT PRIMARY KEY,
        agent_id                          BIGINT NOT NULL,
        session_uuid                      CHAR(36) NOT NULL UNIQUE,
        parent_session_id                 BIGINT,
        status                            ENUM('active', 'completed', 'compacted', 'archived')
                                              NOT NULL DEFAULT 'active',
        purpose                           VARCHAR(512),
        work_item_id                      BIGINT,
        token_count_est                   INT DEFAULT 0,
        compacted_through_turn_index      INT DEFAULT -1,
        compacted_at                      TIMESTAMP NULL DEFAULT NULL,
        memory_revision_at_context_build  BIGINT NULL DEFAULT NULL,
        created_at                        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at                        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_agent_status (agent_id, status),
        INDEX idx_work_item (work_item_id),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_turns (
        id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id              BIGINT NOT NULL,
        turn_index              INT NOT NULL,
        role                    ENUM('system', 'user', 'assistant') NOT NULL,
        content                 LONGTEXT NOT NULL,
        token_count_est         INT DEFAULT 0,
        metadata_json           JSON,
        created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_session_turn (session_id, turn_index),
        INDEX idx_session_id (session_id),
        INDEX idx_session_turns_recency (created_at DESC, id DESC),
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id                          BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id                  BIGINT NOT NULL,
        agent_id                    BIGINT,
        source_session_id           BIGINT,
        source_turn_ids             JSON,
        category                    VARCHAR(64) NOT NULL,
        title                       VARCHAR(512) NOT NULL,
        content                     TEXT NOT NULL,
        embedding                   VECTOR,
        tags                        JSON,
        relevance_score             FLOAT DEFAULT 1.0,
        confidence                  FLOAT DEFAULT 1.0,
        ttl_days                    INT DEFAULT NULL,
        expires_at                  TIMESTAMP NULL,
        source_file                 VARCHAR(1024) DEFAULT NULL,
        source_type                 VARCHAR(32) NOT NULL DEFAULT 'session',
        content_hash                CHAR(64) DEFAULT NULL,
        version                     INT NOT NULL DEFAULT 1,
        search_metadata_json        JSON DEFAULT NULL,
        merged_from                 JSON DEFAULT NULL,
        created_at                  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at                  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_project_agent_category (project_id, agent_id, category),
        INDEX idx_expires (expires_at),
        INDEX idx_knowledge_source_scope (project_id, source_type, category, source_file(255)),
        INDEX idx_knowledge_content_hash (project_id, content_hash),
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (agent_id) REFERENCES agents(id),
        FOREIGN KEY (source_session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS work_items (
        id                BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id        BIGINT NOT NULL,
        title             VARCHAR(512) NOT NULL,
        description       TEXT,
        item_type         ENUM('task', 'design', 'code_change', 'review', 'benchmark') NOT NULL,
        status            ENUM('pending', 'in_progress', 'review', 'approved', 'completed', 'blocked')
                              NOT NULL DEFAULT 'pending',
        assigned_agent_id BIGINT,
        parent_item_id    BIGINT,
        priority          TINYINT DEFAULT 3,
        metadata_json     JSON,
        created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_project_status (project_id, status),
        INDEX idx_assigned (assigned_agent_id),
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (assigned_agent_id) REFERENCES agents(id),
        FOREIGN KEY (parent_item_id) REFERENCES work_items(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        work_item_id    BIGINT NOT NULL,
        agent_id        BIGINT NOT NULL,
        iteration       INT NOT NULL,
        artifact_type   ENUM('proposal', 'diff', 'patch', 'test_plan', 'design_doc', 'phase_result')
                            NOT NULL,
        title           VARCHAR(512),
        content         LONGTEXT NOT NULL,
        metadata_json   JSON,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_work_item_iteration (work_item_id, iteration),
        FOREIGN KEY (work_item_id) REFERENCES work_items(id),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_cycles (
        id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
        work_item_id        BIGINT NOT NULL,
        artifact_id         BIGINT,
        iteration           INT NOT NULL DEFAULT 1,
        proposer_agent_id   BIGINT NOT NULL,
        reviewer_agent_id   BIGINT NOT NULL,
        proposal_session_id BIGINT,
        review_session_id   BIGINT,
        verdict             ENUM('pending', 'changes_requested', 'lgtm')
                                NOT NULL DEFAULT 'pending',
        verdict_json        JSON,
        created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_artifact_reviewer (artifact_id, reviewer_agent_id),
        INDEX idx_work_item (work_item_id),
        FOREIGN KEY (work_item_id) REFERENCES work_items(id),
        FOREIGN KEY (artifact_id) REFERENCES artifacts(id),
        FOREIGN KEY (proposer_agent_id) REFERENCES agents(id),
        FOREIGN KEY (reviewer_agent_id) REFERENCES agents(id),
        FOREIGN KEY (proposal_session_id) REFERENCES sessions(id),
        FOREIGN KEY (review_session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_terms (
        project_id                  BIGINT NOT NULL,
        knowledge_id                BIGINT NOT NULL,
        term                        VARCHAR(255) NOT NULL,
        field                       VARCHAR(32) NOT NULL,
        weight                      FLOAT NOT NULL DEFAULT 1.0,
        PRIMARY KEY (project_id, term, knowledge_id, field),
        INDEX idx_knowledge_terms_lookup (project_id, term),
        INDEX idx_knowledge_terms_knowledge (knowledge_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_sources (
        id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id          BIGINT NOT NULL,
        source_file         VARCHAR(1024) NOT NULL,
        source_file_hash    CHAR(64) NOT NULL,
        content_hash        CHAR(64) NOT NULL,
        updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_project_source_hash (project_id, source_file_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_entities (
        id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id              BIGINT NOT NULL,
        knowledge_id            BIGINT NOT NULL,
        entity_name             VARCHAR(255) NOT NULL,
        confidence              FLOAT NOT NULL DEFAULT 1.0,
        INDEX idx_knowledge_entities_lookup (project_id, entity_name),
        INDEX idx_knowledge_entities_knowledge (knowledge_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_relations (
        id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id              BIGINT NOT NULL,
        knowledge_id            BIGINT NOT NULL,
        source_entity           VARCHAR(255) NOT NULL,
        relation                VARCHAR(64) NOT NULL,
        target_entity           VARCHAR(255) NOT NULL,
        confidence              FLOAT NOT NULL DEFAULT 1.0,
        INDEX idx_knowledge_rel_source (project_id, source_entity),
        INDEX idx_knowledge_rel_target (project_id, target_entity),
        INDEX idx_knowledge_rel_knowledge (knowledge_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learn_requests (
        id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id              BIGINT NOT NULL,
        source_kind             VARCHAR(64) NOT NULL,
        trigger_kind            VARCHAR(64) NOT NULL,
        source_session_id       BIGINT NULL,
        source_work_item_id     BIGINT NULL,
        payload_json            JSON NOT NULL,
        status                  VARCHAR(32) NOT NULL DEFAULT 'pending',
        created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_learn_requests_project_status (project_id, status, created_at),
        INDEX idx_learn_requests_session (source_session_id),
        INDEX idx_learn_requests_work_item (source_work_item_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learn_runs (
        id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
        learn_request_id        BIGINT NOT NULL,
        worker_backend          VARCHAR(32) NOT NULL,
        worker_model            VARCHAR(128) NOT NULL,
        input_context_json      JSON NOT NULL,
        output_envelope_json    JSON NULL,
        status                  VARCHAR(32) NOT NULL DEFAULT 'started',
        error_text              TEXT NULL,
        created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_learn_runs_request_status (learn_request_id, status, created_at)
    )
    """,
]


def ensure_database(pool: TiDBPool) -> None:
    """Create the myswat database if it doesn't exist."""
    db_name = pool._settings.database
    # Connect without specifying a database
    import pymysql
    conn = pymysql.connect(
        host=pool._settings.host,
        port=pool._settings.port,
        user=pool._settings.user,
        password=pool._settings.password,
        ssl={"ca": pool._settings.ssl_ca} if pool._settings.ssl_ca else None,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
            )
    finally:
        conn.close()


def ensure_schema(pool: TiDBPool) -> None:
    """Create the current MySwat schema if it does not already exist."""
    ensure_database(pool)
    pool.execute_many(SCHEMA_STATEMENTS)
