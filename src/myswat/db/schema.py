"""Schema migration runner for MySwat."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myswat.db.connection import TiDBPool

MIGRATION_MODULES = [
    "myswat.db.migrations.v001_initial",
    "myswat.db.migrations.v002_knowledge_source_file",
    "myswat.db.migrations.v003_compaction_watermark",
    "myswat.db.migrations.v004_architect_system_prompt",
    "myswat.db.migrations.v005_review_cycles_artifact_unique_key",
    "myswat.db.migrations.v006_flexible_vector_dimension",
    "myswat.db.migrations.v007_conversation_persistence",
    "myswat.db.migrations.v008_chat_workflow_agent_prompts",
    "myswat.db.migrations.v009_memory_phase1a",
    "myswat.db.migrations.v010_knowledge_terms",
    "myswat.db.migrations.v011_document_sources_and_session_revision",
    "myswat.db.migrations.v012_knowledge_graph",
    "myswat.db.migrations.v013_drop_redundant_document_sources_index",
]


def ensure_schema_version_table(pool: TiDBPool) -> None:
    """Create the schema_version tracking table if it doesn't exist."""
    pool.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INT NOT NULL PRIMARY KEY,
            description VARCHAR(512),
            applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def get_current_version(pool: TiDBPool) -> int:
    """Return the highest applied migration version, or 0 if none."""
    row = pool.fetch_one("SELECT MAX(version) AS v FROM schema_version")
    return row["v"] or 0 if row else 0


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


def run_migrations(pool: TiDBPool) -> list[int]:
    """Run all pending migrations in order. Returns list of applied versions."""
    ensure_database(pool)
    ensure_schema_version_table(pool)
    current = get_current_version(pool)
    applied = []

    for module_path in MIGRATION_MODULES:
        mod = importlib.import_module(module_path)
        if mod.VERSION <= current:
            continue

        # Execute all statements in this migration
        for stmt in mod.STATEMENTS:
            if isinstance(stmt, tuple):
                # Parameterized: (sql, params)
                pool.execute(stmt[0], stmt[1])
            else:
                stmt = stmt.strip()
                if stmt:
                    pool.execute(stmt)

        # Record the migration
        pool.execute(
            "INSERT INTO schema_version (version, description) VALUES (%s, %s)",
            (mod.VERSION, mod.DESCRIPTION),
        )
        applied.append(mod.VERSION)

    return applied
