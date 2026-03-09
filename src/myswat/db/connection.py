"""TiDB connection pool manager."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pymysql
import pymysql.cursors

from myswat.config.settings import TiDBSettings


class TiDBPool:
    """Lightweight connection manager for TiDB Cloud."""

    def __init__(self, settings: TiDBSettings) -> None:
        self._settings = settings

    def _connect(self, database: str | None = None) -> pymysql.Connection:
        ssl_opts = {"ca": self._settings.ssl_ca} if self._settings.ssl_ca else None
        return pymysql.connect(
            host=self._settings.host,
            port=self._settings.port,
            user=self._settings.user,
            password=self._settings.password,
            database=database or self._settings.database,
            ssl=ssl_opts,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    @contextmanager
    def connection(self, database: str | None = None) -> Generator[pymysql.Connection, None, None]:
        conn = self._connect(database)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def cursor(self, database: str | None = None) -> Generator[pymysql.cursors.DictCursor, None, None]:
        with self.connection(database) as conn:
            with conn.cursor() as cur:
                yield cur

    def health_check(self) -> bool:
        try:
            # Use no specific database for health check (DB may not exist yet)
            conn = pymysql.connect(
                host=self._settings.host,
                port=self._settings.port,
                user=self._settings.user,
                password=self._settings.password,
                ssl={"ca": self._settings.ssl_ca} if self._settings.ssl_ca else None,
                charset="utf8mb4",
            )
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
            finally:
                conn.close()
        except Exception:
            return False

    def execute(self, sql: str, args: tuple | None = None, database: str | None = None) -> int:
        """Execute a single statement, return affected row count."""
        with self.cursor(database) as cur:
            cur.execute(sql, args)
            return cur.rowcount

    def execute_many(self, statements: list[str], database: str | None = None) -> None:
        """Execute multiple DDL/DML statements in sequence."""
        with self.cursor(database) as cur:
            for stmt in statements:
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)

    def fetch_one(self, sql: str, args: tuple | None = None, database: str | None = None) -> dict | None:
        with self.cursor(database) as cur:
            cur.execute(sql, args)
            return cur.fetchone()

    def fetch_all(self, sql: str, args: tuple | None = None, database: str | None = None) -> list[dict]:
        with self.cursor(database) as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def insert_returning_id(self, sql: str, args: tuple | None = None, database: str | None = None) -> int:
        """Execute an INSERT and return the auto-generated ID."""
        with self.cursor(database) as cur:
            cur.execute(sql, args)
            cur.execute("SELECT LAST_INSERT_ID() AS id")
            row = cur.fetchone()
            return row["id"]
