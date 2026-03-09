"""Tests for myswat.db.schema module."""

from unittest.mock import MagicMock, patch, call
import pytest

from myswat.db.schema import (
    ensure_schema_version_table,
    get_current_version,
    ensure_database,
    run_migrations,
    MIGRATION_MODULES,
)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.fetch_one.return_value = None
    pool.fetch_all.return_value = []
    pool.execute.return_value = 0
    pool._settings = MagicMock()
    pool._settings.database = "test_db"
    pool._settings.host = "localhost"
    pool._settings.port = 4000
    pool._settings.user = "root"
    pool._settings.password = "pass"
    pool._settings.ssl_ca = None
    return pool


# ── ensure_schema_version_table ──


class TestEnsureSchemaVersionTable:
    def test_creates_table(self, mock_pool):
        ensure_schema_version_table(mock_pool)
        mock_pool.execute.assert_called_once()
        sql = mock_pool.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS schema_version" in sql

    def test_table_has_version_column(self, mock_pool):
        ensure_schema_version_table(mock_pool)
        sql = mock_pool.execute.call_args[0][0]
        assert "version" in sql.lower()


# ── get_current_version ──


class TestGetCurrentVersion:
    def test_returns_max_version(self, mock_pool):
        mock_pool.fetch_one.return_value = {"v": 5}
        result = get_current_version(mock_pool)
        assert result == 5

    def test_returns_zero_when_no_rows(self, mock_pool):
        mock_pool.fetch_one.return_value = None
        result = get_current_version(mock_pool)
        assert result == 0

    def test_returns_zero_when_null_max(self, mock_pool):
        mock_pool.fetch_one.return_value = {"v": None}
        result = get_current_version(mock_pool)
        assert result == 0


# ── ensure_database ──


class TestEnsureDatabase:
    @patch("pymysql.connect")
    def test_creates_database(self, mock_connect, mock_pool):
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        ensure_database(mock_pool)

        mock_connect.assert_called_once()
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "CREATE DATABASE IF NOT EXISTS" in sql
        assert "test_db" in sql
        mock_conn.close.assert_called_once()

    @patch("pymysql.connect")
    def test_uses_ssl_when_configured(self, mock_connect, mock_pool):
        mock_pool._settings.ssl_ca = "/path/to/ca.pem"
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        ensure_database(mock_pool)

        call_kwargs = mock_connect.call_args
        assert call_kwargs[1].get("ssl") == {"ca": "/path/to/ca.pem"} or \
               (len(call_kwargs[0]) == 0 and "ssl" in call_kwargs[1])

    @patch("pymysql.connect")
    def test_no_ssl_when_not_configured(self, mock_connect, mock_pool):
        mock_pool._settings.ssl_ca = None
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        ensure_database(mock_pool)

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs.get("ssl") is None


# ── run_migrations ──


class TestRunMigrations:
    @patch("myswat.db.schema.get_current_version")
    @patch("myswat.db.schema.ensure_schema_version_table")
    @patch("myswat.db.schema.ensure_database")
    def test_skips_already_applied(self, mock_ensure_db, mock_ensure_table, mock_version, mock_pool):
        mock_version.return_value = 999  # all migrations are below this
        result = run_migrations(mock_pool)
        assert result == []

    @patch("myswat.db.schema.importlib.import_module")
    @patch("myswat.db.schema.get_current_version")
    @patch("myswat.db.schema.ensure_schema_version_table")
    @patch("myswat.db.schema.ensure_database")
    def test_applies_pending_migration(self, mock_ensure_db, mock_ensure_table, mock_version, mock_import, mock_pool):
        mock_version.return_value = 0
        fake_mod = MagicMock()
        fake_mod.VERSION = 1
        fake_mod.DESCRIPTION = "Initial schema"
        fake_mod.STATEMENTS = ["CREATE TABLE foo (id INT);"]
        mock_import.return_value = fake_mod

        result = run_migrations(mock_pool)

        assert len(result) >= 1
        assert mock_pool.execute.call_count >= 2  # at least 1 statement + 1 version record

    @patch("myswat.db.schema.importlib.import_module")
    @patch("myswat.db.schema.get_current_version")
    @patch("myswat.db.schema.ensure_schema_version_table")
    @patch("myswat.db.schema.ensure_database")
    def test_handles_parameterized_statements(self, mock_ensure_db, mock_ensure_table, mock_version, mock_import, mock_pool):
        mock_version.return_value = 0
        fake_mod = MagicMock()
        fake_mod.VERSION = 1
        fake_mod.DESCRIPTION = "With params"
        fake_mod.STATEMENTS = [("INSERT INTO foo VALUES (%s)", (42,))]
        mock_import.return_value = fake_mod

        result = run_migrations(mock_pool)

        calls = mock_pool.execute.call_args_list
        param_calls = [c for c in calls if len(c[0]) >= 2 and c[0][1] is not None]
        assert len(param_calls) >= 1

    @patch("myswat.db.schema.importlib.import_module")
    @patch("myswat.db.schema.get_current_version")
    @patch("myswat.db.schema.ensure_schema_version_table")
    @patch("myswat.db.schema.ensure_database")
    def test_skips_empty_statements(self, mock_ensure_db, mock_ensure_table, mock_version, mock_import, mock_pool):
        mock_version.return_value = 0
        fake_mod = MagicMock()
        fake_mod.VERSION = 1
        fake_mod.DESCRIPTION = "Has blanks"
        fake_mod.STATEMENTS = ["CREATE TABLE t (id INT);", "  ", ""]
        mock_import.return_value = fake_mod

        run_migrations(mock_pool)

        execute_calls = mock_pool.execute.call_args_list
        sqls = [c[0][0].strip() for c in execute_calls]
        assert all(s for s in sqls)  # no empty SQL strings

    @patch("myswat.db.schema.importlib.import_module")
    @patch("myswat.db.schema.get_current_version")
    @patch("myswat.db.schema.ensure_schema_version_table")
    @patch("myswat.db.schema.ensure_database")
    def test_records_version_after_migration(self, mock_ensure_db, mock_ensure_table, mock_version, mock_import, mock_pool):
        mock_version.return_value = 0
        fake_mod = MagicMock()
        fake_mod.VERSION = 1
        fake_mod.DESCRIPTION = "Test migration"
        fake_mod.STATEMENTS = ["CREATE TABLE t (id INT);"]
        mock_import.return_value = fake_mod

        run_migrations(mock_pool)

        last_call = mock_pool.execute.call_args_list[-1]
        sql = last_call[0][0]
        assert "INSERT INTO schema_version" in sql

    def test_migration_modules_list_not_empty(self):
        assert len(MIGRATION_MODULES) > 0
