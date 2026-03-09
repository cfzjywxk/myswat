"""Tests for TiDBPool retry logic on transient errors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pymysql.err
import pytest

from myswat.db.connection import TiDBPool, _TRANSIENT_ERRORS, _MAX_RETRIES


@pytest.fixture
def pool():
    """TiDBPool with mocked settings (never actually connects)."""
    settings = MagicMock()
    settings.host = "localhost"
    settings.port = 4000
    settings.user = "test"
    settings.password = "test"
    settings.ssl_ca = None
    settings.database = "myswat"
    return TiDBPool(settings)


def _transient_error(code=2006):
    return pymysql.err.OperationalError(code, "MySQL server has gone away")


def _non_transient_error():
    return pymysql.err.OperationalError(1045, "Access denied")


class TestWithRetry:
    def test_success_no_retry(self, pool):
        """Successful call returns immediately, no retry."""
        fn = MagicMock(return_value=42)
        result = pool._with_retry(fn)

        assert result == 42
        assert fn.call_count == 1

    def test_retry_on_transient_error(self, pool):
        """Transient error triggers retry, succeeds on second attempt."""
        fn = MagicMock(side_effect=[_transient_error(2006), 42])
        result = pool._with_retry(fn)

        assert result == 42
        assert fn.call_count == 2

    def test_retry_on_connection_refused(self, pool):
        """Error 2003 (can't connect) triggers retry."""
        fn = MagicMock(side_effect=[_transient_error(2003), "ok"])
        result = pool._with_retry(fn)

        assert result == "ok"
        assert fn.call_count == 2

    def test_retry_on_lost_connection(self, pool):
        """Error 2013 (lost connection during query) triggers retry."""
        fn = MagicMock(side_effect=[_transient_error(2013), "ok"])
        result = pool._with_retry(fn)

        assert result == "ok"

    def test_no_retry_on_non_transient_error(self, pool):
        """Non-transient error propagates immediately, no retry."""
        fn = MagicMock(side_effect=_non_transient_error())

        with pytest.raises(pymysql.err.OperationalError) as exc_info:
            pool._with_retry(fn)

        assert exc_info.value.args[0] == 1045
        assert fn.call_count == 1

    def test_max_retries_exceeded(self, pool):
        """Transient error persists beyond max retries → raises."""
        fn = MagicMock(side_effect=_transient_error())

        with pytest.raises(pymysql.err.OperationalError):
            pool._with_retry(fn)

        assert fn.call_count == _MAX_RETRIES + 1

    def test_non_operational_error_not_retried(self, pool):
        """Non-OperationalError exceptions propagate immediately."""
        fn = MagicMock(side_effect=ValueError("bad"))

        with pytest.raises(ValueError):
            pool._with_retry(fn)

        assert fn.call_count == 1

    def test_all_transient_codes_covered(self):
        """Verify the known transient error codes."""
        assert 2003 in _TRANSIENT_ERRORS  # Can't connect
        assert 2006 in _TRANSIENT_ERRORS  # Server gone away
        assert 2013 in _TRANSIENT_ERRORS  # Lost connection


class TestPoolMethodsUseRetry:
    """Verify that execute/fetch_one/fetch_all/insert_returning_id use _with_retry."""

    @patch.object(TiDBPool, "_with_retry")
    def test_execute_uses_retry(self, mock_retry, pool):
        mock_retry.return_value = 1
        pool.execute("SELECT 1")
        mock_retry.assert_called_once()

    @patch.object(TiDBPool, "_with_retry")
    def test_fetch_one_uses_retry(self, mock_retry, pool):
        mock_retry.return_value = {"id": 1}
        pool.fetch_one("SELECT 1")
        mock_retry.assert_called_once()

    @patch.object(TiDBPool, "_with_retry")
    def test_fetch_all_uses_retry(self, mock_retry, pool):
        mock_retry.return_value = []
        pool.fetch_all("SELECT 1")
        mock_retry.assert_called_once()

    @patch.object(TiDBPool, "_with_retry")
    def test_insert_returning_id_uses_retry(self, mock_retry, pool):
        mock_retry.return_value = 42
        pool.insert_returning_id("INSERT INTO t VALUES (1)")
        mock_retry.assert_called_once()

    @patch.object(TiDBPool, "_with_retry")
    def test_execute_many_uses_retry(self, mock_retry, pool):
        mock_retry.return_value = None
        pool.execute_many(["SELECT 1", "SELECT 2"])
        mock_retry.assert_called_once()


class TestHealthCheck:
    @patch("pymysql.connect")
    def test_health_check_success(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        assert pool.health_check() is True

    @patch("pymysql.connect", side_effect=Exception("connection failed"))
    def test_health_check_failure(self, mock_connect, pool):
        assert pool.health_check() is False
