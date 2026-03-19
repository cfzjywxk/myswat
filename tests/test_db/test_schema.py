"""Tests for myswat.db.schema."""

from unittest.mock import MagicMock, patch

import pytest

from myswat.db.schema import SCHEMA_STATEMENTS, ensure_database, ensure_schema


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.execute.return_value = 0
    pool.execute_many.return_value = None
    pool._settings = MagicMock()
    pool._settings.database = "test_db"
    pool._settings.host = "localhost"
    pool._settings.port = 4000
    pool._settings.user = "root"
    pool._settings.password = "pass"
    pool._settings.ssl_ca = None
    return pool


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

        assert mock_connect.call_args.kwargs.get("ssl") == {"ca": "/path/to/ca.pem"}

    @patch("pymysql.connect")
    def test_no_ssl_when_not_configured(self, mock_connect, mock_pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        ensure_database(mock_pool)

        assert mock_connect.call_args.kwargs.get("ssl") is None


class TestEnsureSchema:
    @patch("myswat.db.schema.ensure_database")
    def test_bootstraps_schema_snapshot(self, mock_ensure_database, mock_pool):
        ensure_schema(mock_pool)

        mock_ensure_database.assert_called_once_with(mock_pool)
        mock_pool.execute_many.assert_called_once_with(SCHEMA_STATEMENTS)

    def test_schema_statements_not_empty(self):
        assert SCHEMA_STATEMENTS

    def test_schema_snapshot_contains_current_tables(self):
        sql = "\n".join(SCHEMA_STATEMENTS)

        assert "CREATE TABLE IF NOT EXISTS projects" in sql
        assert "CREATE TABLE IF NOT EXISTS artifacts" in sql
        assert "CREATE TABLE IF NOT EXISTS learn_requests" in sql
