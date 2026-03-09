"""v001: Initial schema — all 8 tables for MySwat."""

VERSION = 1
DESCRIPTION = "Initial schema with projects, agents, sessions, knowledge (vector), work_items, artifacts, review_cycles"

STATEMENTS = [
    # -- projects --
    """
    CREATE TABLE IF NOT EXISTS projects (
        id          BIGINT AUTO_INCREMENT PRIMARY KEY,
        slug        VARCHAR(128) NOT NULL UNIQUE,
        name        VARCHAR(256) NOT NULL,
        description TEXT,
        repo_path   VARCHAR(512),
        config_json JSON,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """,

    # -- agents --
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

    # -- sessions --
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id                BIGINT AUTO_INCREMENT PRIMARY KEY,
        agent_id          BIGINT NOT NULL,
        session_uuid      CHAR(36) NOT NULL UNIQUE,
        parent_session_id BIGINT,
        status            ENUM('active', 'completed', 'compacted', 'archived')
                              NOT NULL DEFAULT 'active',
        purpose           VARCHAR(512),
        work_item_id      BIGINT,
        token_count_est   INT DEFAULT 0,
        created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_agent_status (agent_id, status),
        INDEX idx_work_item (work_item_id),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    )
    """,

    # -- session_turns --
    """
    CREATE TABLE IF NOT EXISTS session_turns (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id      BIGINT NOT NULL,
        turn_index      INT NOT NULL,
        role            ENUM('system', 'user', 'assistant') NOT NULL,
        content         LONGTEXT NOT NULL,
        token_count_est INT DEFAULT 0,
        metadata_json   JSON,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_session_turn (session_id, turn_index),
        INDEX idx_session_id (session_id),
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,

    # -- knowledge (with vector embedding) --
    """
    CREATE TABLE IF NOT EXISTS knowledge (
        id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id          BIGINT NOT NULL,
        agent_id            BIGINT,
        source_session_id   BIGINT,
        source_turn_ids     JSON,
        category            VARCHAR(64) NOT NULL,
        title               VARCHAR(512) NOT NULL,
        content             TEXT NOT NULL,
        embedding           VECTOR(1024),
        tags                JSON,
        relevance_score     FLOAT DEFAULT 1.0,
        confidence          FLOAT DEFAULT 1.0,
        ttl_days            INT DEFAULT NULL,
        expires_at          TIMESTAMP NULL,
        created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_project_agent_category (project_id, agent_id, category),
        INDEX idx_expires (expires_at),
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (agent_id) REFERENCES agents(id),
        FOREIGN KEY (source_session_id) REFERENCES sessions(id)
    )
    """,

    # -- work_items --
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

    # -- artifacts (must be before review_cycles due to FK) --
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        work_item_id    BIGINT NOT NULL,
        agent_id        BIGINT NOT NULL,
        iteration       INT NOT NULL,
        artifact_type   ENUM('proposal', 'diff', 'patch', 'test_plan', 'design_doc') NOT NULL,
        title           VARCHAR(512),
        content         LONGTEXT NOT NULL,
        metadata_json   JSON,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_work_item_iteration (work_item_id, iteration),
        FOREIGN KEY (work_item_id) REFERENCES work_items(id),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    )
    """,

    # -- review_cycles --
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
        UNIQUE KEY uk_work_iteration_reviewer (work_item_id, iteration, reviewer_agent_id),
        INDEX idx_work_item (work_item_id),
        FOREIGN KEY (work_item_id) REFERENCES work_items(id),
        FOREIGN KEY (artifact_id) REFERENCES artifacts(id),
        FOREIGN KEY (proposer_agent_id) REFERENCES agents(id),
        FOREIGN KEY (reviewer_agent_id) REFERENCES agents(id),
        FOREIGN KEY (proposal_session_id) REFERENCES sessions(id),
        FOREIGN KEY (review_session_id) REFERENCES sessions(id)
    )
    """,
]
