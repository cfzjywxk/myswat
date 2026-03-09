"""TiDB connection pool manager with retry for transient errors."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import pymysql
import pymysql.cursors
import pymysql.err

from myswat.config.settings import TiDBSettings

# PyMySQL error codes for transient connection issues
_TRANSIENT_ERRORS = frozenset({
    2003,  # Can't connect to MySQL server
    2006,  # MySQL server has gone away
    2013,  # Lost connection to MySQL server during query
    4031,  # TiDB server timeout
})

_MAX_RETRIES = 2
_RETRY_DELAY = 0.5  # seconds, multiplied by attempt number


class TiDBPool:
    """Lightweight connection manager for TiDB Cloud."""

    def __init__(self, settings: TiDBSettings) -> None:
        self._settings = settings

    def _with_retry(self, fn):
        """Execute fn(), retrying on transient connection errors."""
        last_err = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return fn()
            except pymysql.err.OperationalError as e:
                if e.args and e.args[0] in _TRANSIENT_ERRORS and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * (attempt + 1))
                    last_err = e
                    continue
                raise
        raise last_err

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
        def _op():
            with self.cursor(database) as cur:
                cur.execute(sql, args)
                return cur.rowcount
        return self._with_retry(_op)

    def execute_many(self, statements: list[str], database: str | None = None) -> None:
        """Execute multiple DDL/DML statements in sequence."""
        def _op():
            with self.cursor(database) as cur:
                for stmt in statements:
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
        return self._with_retry(_op)

    def fetch_one(self, sql: str, args: tuple | None = None, database: str | None = None) -> dict | None:
        def _op():
            with self.cursor(database) as cur:
                cur.execute(sql, args)
                return cur.fetchone()
        return self._with_retry(_op)

    def fetch_all(self, sql: str, args: tuple | None = None, database: str | None = None) -> list[dict]:
        def _op():
            with self.cursor(database) as cur:
                cur.execute(sql, args)
                return cur.fetchall()
        return self._with_retry(_op)

    def insert_returning_id(self, sql: str, args: tuple | None = None, database: str | None = None) -> int:
        """Execute an INSERT and return the auto-generated ID."""
        def _op():
            with self.cursor(database) as cur:
                cur.execute(sql, args)
                cur.execute("SELECT LAST_INSERT_ID() AS id")
                row = cur.fetchone()
                return row["id"]
        return self._with_retry(_op)
