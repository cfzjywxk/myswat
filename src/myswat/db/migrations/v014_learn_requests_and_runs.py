"""v014: Add unified learn request and worker run audit tables."""

VERSION = 14
DESCRIPTION = "Add learn_requests and learn_runs audit tables for the unified learn pipeline"

STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS learn_requests (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id BIGINT NOT NULL,
        source_kind VARCHAR(64) NOT NULL,
        trigger_kind VARCHAR(64) NOT NULL,
        source_session_id BIGINT NULL,
        source_work_item_id BIGINT NULL,
        payload_json JSON NOT NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_learn_requests_project_status (project_id, status, created_at),
        INDEX idx_learn_requests_session (source_session_id),
        INDEX idx_learn_requests_work_item (source_work_item_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learn_runs (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        learn_request_id BIGINT NOT NULL,
        worker_backend VARCHAR(32) NOT NULL,
        worker_model VARCHAR(128) NOT NULL,
        input_context_json JSON NOT NULL,
        output_envelope_json JSON NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'started',
        error_text TEXT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_learn_runs_request_status (learn_request_id, status, created_at)
    )
    """,
]
