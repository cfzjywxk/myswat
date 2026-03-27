"""Comprehensive tests for myswat.db.connection (TiDBPool)."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pymysql.cursors
import pymysql.err
import pytest

from myswat.db.connection import (
    TiDBPool,
    _MAX_RETRIES,
    _RETRY_DELAY,
    _TRANSIENT_ERRORS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    """A MagicMock standing in for TiDBSettings."""
    s = MagicMock()
    s.host = "tidb.example.com"
    s.port = 4000
    s.user = "root"
    s.password = "secret"
    s.ssl_ca = "/path/to/ca.pem"
    s.database = "myswat"
    s.connect_timeout_seconds = 180
    return s


@pytest.fixture
def settings_no_ssl():
    """TiDBSettings mock with ssl_ca empty (falsy)."""
    s = MagicMock()
    s.host = "tidb.example.com"
    s.port = 4000
    s.user = "root"
    s.password = "secret"
    s.ssl_ca = ""
    s.database = "myswat"
    s.connect_timeout_seconds = 180
    return s


@pytest.fixture
def pool(settings):
    return TiDBPool(settings)


@pytest.fixture
def pool_no_ssl(settings_no_ssl):
    return TiDBPool(settings_no_ssl)


def _transient_error(code: int = 2006) -> pymysql.err.OperationalError:
    return pymysql.err.OperationalError(code, "transient failure")


def _non_transient_error(code: int = 1045) -> pymysql.err.OperationalError:
    return pymysql.err.OperationalError(code, "Access denied")


# ===================================================================
# _with_retry
# ===================================================================


class TestWithRetry:
    """Tests for the retry wrapper around transient OperationalError codes."""

    @patch("myswat.db.connection.time.sleep")
    def test_success_on_first_try(self, mock_sleep, pool):
        fn = MagicMock(return_value="ok")
        assert pool._with_retry(fn) == "ok"
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("myswat.db.connection.time.sleep")
    def test_success_on_second_try_after_transient(self, mock_sleep, pool):
        fn = MagicMock(side_effect=[_transient_error(2006), "recovered"])
        assert pool._with_retry(fn) == "recovered"
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(_RETRY_DELAY * 1)

    @patch("myswat.db.connection.time.sleep")
    def test_success_on_third_try(self, mock_sleep, pool):
        """Two transient failures then success (max retries == 2)."""
        fn = MagicMock(
            side_effect=[_transient_error(2003), _transient_error(2013), "done"]
        )
        assert pool._with_retry(fn) == "done"
        assert fn.call_count == 3
        mock_sleep.assert_has_calls(
            [call(_RETRY_DELAY * 1), call(_RETRY_DELAY * 2)]
        )

    @patch("myswat.db.connection.time.sleep")
    @pytest.mark.parametrize("code", sorted(_TRANSIENT_ERRORS))
    def test_each_transient_code_retries(self, mock_sleep, pool, code):
        """Every code in _TRANSIENT_ERRORS triggers a retry."""
        fn = MagicMock(side_effect=[_transient_error(code), "ok"])
        assert pool._with_retry(fn) == "ok"
        assert fn.call_count == 2

    @patch("myswat.db.connection.time.sleep")
    def test_exhausted_retries_raises_last_error(self, mock_sleep, pool):
        """When all retries are exhausted the last transient error propagates."""
        errors = [_transient_error(2006) for _ in range(_MAX_RETRIES + 1)]
        fn = MagicMock(side_effect=errors)

        with pytest.raises(pymysql.err.OperationalError) as exc_info:
            pool._with_retry(fn)

        assert exc_info.value.args[0] == 2006
        assert fn.call_count == _MAX_RETRIES + 1
        # sleep should have been called _MAX_RETRIES times (not on the last attempt)
        assert mock_sleep.call_count == _MAX_RETRIES

    @patch("myswat.db.connection.time.sleep")
    def test_non_transient_operational_error_raises_immediately(
        self, mock_sleep, pool
    ):
        fn = MagicMock(side_effect=_non_transient_error(1045))

        with pytest.raises(pymysql.err.OperationalError) as exc_info:
            pool._with_retry(fn)

        assert exc_info.value.args[0] == 1045
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("myswat.db.connection.time.sleep")
    def test_non_operational_error_propagates(self, mock_sleep, pool):
        """Exceptions that are not OperationalError are never retried."""
        fn = MagicMock(side_effect=ValueError("bad data"))

        with pytest.raises(ValueError, match="bad data"):
            pool._with_retry(fn)

        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("myswat.db.connection.time.sleep")
    def test_operational_error_with_empty_args_raises(self, mock_sleep, pool):
        """OperationalError with no args is not considered transient."""
        err = pymysql.err.OperationalError()
        fn = MagicMock(side_effect=err)

        with pytest.raises(pymysql.err.OperationalError):
            pool._with_retry(fn)

        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("myswat.db.connection.time.sleep")
    def test_sleep_delay_increases_linearly(self, mock_sleep, pool):
        """Delay is _RETRY_DELAY * (attempt + 1): 0.5, 1.0 for default values."""
        fn = MagicMock(
            side_effect=[
                _transient_error(2006),
                _transient_error(2006),
                "ok",
            ]
        )
        pool._with_retry(fn)
        mock_sleep.assert_has_calls(
            [call(_RETRY_DELAY * 1), call(_RETRY_DELAY * 2)]
        )

    @patch("myswat.db.connection.time.sleep")
    def test_transient_on_final_attempt_raises_directly(self, mock_sleep, pool):
        """On the last attempt (attempt == _MAX_RETRIES), even a transient code
        triggers the `raise` rather than `continue`."""
        fn = MagicMock(
            side_effect=[
                _transient_error(2006),
                _transient_error(2006),
                _transient_error(2006),
            ]
        )
        with pytest.raises(pymysql.err.OperationalError):
            pool._with_retry(fn)

        # The third call (attempt==2 == _MAX_RETRIES) hits `raise` directly
        assert fn.call_count == _MAX_RETRIES + 1


# ===================================================================
# _connect
# ===================================================================


class TestConnect:
    @patch("myswat.db.connection.pymysql.connect")
    def test_connect_with_ssl(self, mock_connect, pool, settings):
        mock_connect.return_value = MagicMock()
        pool._connect()

        mock_connect.assert_called_once_with(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            ssl={"ca": settings.ssl_ca},
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=settings.connect_timeout_seconds,
        )

    @patch("myswat.db.connection.pymysql.connect")
    def test_connect_without_ssl(self, mock_connect, pool_no_ssl, settings_no_ssl):
        mock_connect.return_value = MagicMock()
        pool_no_ssl._connect()

        mock_connect.assert_called_once_with(
            host=settings_no_ssl.host,
            port=settings_no_ssl.port,
            user=settings_no_ssl.user,
            password=settings_no_ssl.password,
            database=settings_no_ssl.database,
            ssl=None,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=settings_no_ssl.connect_timeout_seconds,
        )

    @patch("myswat.db.connection.pymysql.connect")
    def test_connect_timeout_is_clamped_to_one_second(self, mock_connect, pool, settings):
        settings.connect_timeout_seconds = 0
        mock_connect.return_value = MagicMock()

        pool._connect()

        _, kwargs = mock_connect.call_args
        assert kwargs["connect_timeout"] == 1

    @patch("myswat.db.connection.pymysql.connect")
    def test_connect_custom_database(self, mock_connect, pool, settings):
        mock_connect.return_value = MagicMock()
        pool._connect(database="other_db")

        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "other_db"

    @patch("myswat.db.connection.pymysql.connect")
    def test_connect_defaults_to_settings_database(self, mock_connect, pool, settings):
        mock_connect.return_value = MagicMock()
        pool._connect()

        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == settings.database

    @patch("myswat.db.connection.pymysql.connect")
    def test_connect_none_database_uses_settings(self, mock_connect, pool, settings):
        """Passing database=None explicitly falls back to settings.database."""
        mock_connect.return_value = MagicMock()
        pool._connect(database=None)

        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == settings.database


# ===================================================================
# connection context manager
# ===================================================================


class TestConnectionContextManager:
    @patch("myswat.db.connection.pymysql.connect")
    def test_yields_connection(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with pool.connection() as conn:
            assert conn is mock_conn

    @patch("myswat.db.connection.pymysql.connect")
    def test_closes_on_normal_exit(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with pool.connection():
            pass

        mock_conn.close.assert_called_once()

    @patch("myswat.db.connection.pymysql.connect")
    def test_closes_on_exception(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with pytest.raises(RuntimeError):
            with pool.connection():
                raise RuntimeError("boom")

        mock_conn.close.assert_called_once()

    @patch("myswat.db.connection.pymysql.connect")
    def test_passes_database_to_connect(self, mock_connect, pool):
        mock_connect.return_value = MagicMock()
        with pool.connection(database="custom_db"):
            pass

        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "custom_db"


# ===================================================================
# cursor context manager
# ===================================================================


class TestCursorContextManager:
    @patch("myswat.db.connection.pymysql.connect")
    def test_yields_cursor(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        with pool.cursor() as cur:
            assert cur is mock_cursor

    @patch("myswat.db.connection.pymysql.connect")
    def test_closes_connection_after_cursor(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        with pool.cursor():
            pass

        mock_conn.close.assert_called_once()

    @patch("myswat.db.connection.pymysql.connect")
    def test_cursor_passes_database(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        with pool.cursor(database="other"):
            pass

        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "other"


# ===================================================================
# health_check
# ===================================================================


class TestHealthCheck:
    @patch("myswat.db.connection.pymysql.connect")
    def test_success_returns_true(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        assert pool.health_check() is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @patch("myswat.db.connection.pymysql.connect")
    def test_closes_connection_on_success(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.health_check()
        mock_conn.close.assert_called_once()

    @patch("myswat.db.connection.pymysql.connect", side_effect=Exception("refused"))
    def test_connection_failure_returns_false(self, mock_connect, pool):
        assert pool.health_check() is False

    @patch("myswat.db.connection.pymysql.connect")
    def test_query_failure_returns_false(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("query fail")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        assert pool.health_check() is False

    @patch("myswat.db.connection.pymysql.connect")
    def test_closes_connection_even_on_query_failure(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("query fail")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.health_check()
        mock_conn.close.assert_called_once()

    @patch("myswat.db.connection.pymysql.connect")
    def test_health_check_uses_ssl_when_set(self, mock_connect, pool, settings):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.health_check()
        _, kwargs = mock_connect.call_args
        assert kwargs["ssl"] == {"ca": settings.ssl_ca}

    @patch("myswat.db.connection.pymysql.connect")
    def test_health_check_no_ssl_when_unset(self, mock_connect, pool_no_ssl):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool_no_ssl.health_check()
        _, kwargs = mock_connect.call_args
        assert kwargs["ssl"] is None

    @patch("myswat.db.connection.pymysql.connect")
    def test_health_check_does_not_specify_database(self, mock_connect, pool):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.health_check()
        _, kwargs = mock_connect.call_args
        assert "database" not in kwargs

    @patch("myswat.db.connection.pymysql.connect")
    def test_health_check_timeout_is_clamped_to_one_second(self, mock_connect, pool, settings):
        settings.connect_timeout_seconds = -5
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.health_check()
        _, kwargs = mock_connect.call_args
        assert kwargs["connect_timeout"] == 1


# ===================================================================
# execute
# ===================================================================


class TestExecute:
    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_returns_rowcount(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = pool.execute("UPDATE t SET x=1")
        assert result == 3

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_passes_sql_and_args(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.execute("INSERT INTO t(a) VALUES(%s)", (42,))
        mock_cursor.execute.assert_called_once_with("INSERT INTO t(a) VALUES(%s)", (42,))

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_passes_custom_database(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.execute("SELECT 1", database="other_db")
        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "other_db"

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_retries_on_transient_then_succeeds(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # First connect raises transient, second succeeds
        mock_connect.side_effect = [
            pymysql.err.OperationalError(2006, "gone away"),
            mock_conn,
        ]

        result = pool.execute("SELECT 1")
        assert result == 1
        assert mock_connect.call_count == 2
        mock_sleep.assert_called_once()


# ===================================================================
# execute_many
# ===================================================================


class TestExecuteMany:
    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_executes_all_statements(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.execute_many(["CREATE TABLE t(id INT)", "INSERT INTO t VALUES(1)"])
        assert mock_cursor.execute.call_count == 2
        mock_cursor.execute.assert_any_call("CREATE TABLE t(id INT)")
        mock_cursor.execute.assert_any_call("INSERT INTO t VALUES(1)")

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_skips_empty_statements(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.execute_many(["SELECT 1", "", "  ", "SELECT 2"])
        assert mock_cursor.execute.call_count == 2

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_strips_whitespace(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.execute_many(["  SELECT 1  "])
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_empty_list_does_nothing(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.execute_many([])
        mock_cursor.execute.assert_not_called()


# ===================================================================
# fetch_one
# ===================================================================


class TestFetchOne:
    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_returns_single_row(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1, "name": "alice"}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = pool.fetch_one("SELECT * FROM t WHERE id=%s", (1,))
        assert result == {"id": 1, "name": "alice"}
        mock_cursor.execute.assert_called_once_with(
            "SELECT * FROM t WHERE id=%s", (1,)
        )

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_returns_none_when_no_rows(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = pool.fetch_one("SELECT * FROM t WHERE id=999")
        assert result is None

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_passes_database(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.fetch_one("SELECT 1", database="alt_db")
        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "alt_db"


# ===================================================================
# fetch_all
# ===================================================================


class TestFetchAll:
    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_returns_list_of_rows(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        rows = [{"id": 1}, {"id": 2}]
        mock_cursor.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = pool.fetch_all("SELECT id FROM t")
        assert result == rows
        mock_cursor.execute.assert_called_once_with("SELECT id FROM t", None)

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_returns_empty_list(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = pool.fetch_all("SELECT id FROM t WHERE 1=0")
        assert result == []

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_passes_args_and_database(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.fetch_all("SELECT * FROM t WHERE x=%s", (5,), database="db2")
        mock_cursor.execute.assert_called_once_with("SELECT * FROM t WHERE x=%s", (5,))
        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "db2"


# ===================================================================
# insert_returning_id
# ===================================================================


class TestInsertReturningId:
    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_returns_last_insert_id(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 42}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = pool.insert_returning_id(
            "INSERT INTO t(name) VALUES(%s)", ("alice",)
        )
        assert result == 42

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_executes_insert_then_last_insert_id(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 7}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.insert_returning_id("INSERT INTO t(v) VALUES(%s)", ("x",))

        calls = mock_cursor.execute.call_args_list
        assert len(calls) == 2
        assert calls[0] == call("INSERT INTO t(v) VALUES(%s)", ("x",))
        assert calls[1] == call("SELECT LAST_INSERT_ID() AS id")

    @patch("myswat.db.connection.time.sleep")
    @patch("myswat.db.connection.pymysql.connect")
    def test_passes_database(self, mock_connect, mock_sleep, pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        pool.insert_returning_id("INSERT INTO t VALUES(1)", database="db3")
        _, kwargs = mock_connect.call_args
        assert kwargs["database"] == "db3"


# ===================================================================
# Module-level constants
# ===================================================================


class TestConstants:
    def test_transient_errors_contains_expected_codes(self):
        assert _TRANSIENT_ERRORS == frozenset({2003, 2006, 2013, 4031})

    def test_max_retries_is_positive(self):
        assert _MAX_RETRIES >= 1

    def test_retry_delay_is_positive(self):
        assert _RETRY_DELAY > 0
