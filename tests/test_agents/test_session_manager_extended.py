"""Extended tests for SessionManager covering _update_progress, _check_mid_session_compaction, close, and _compact."""

import io
import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from myswat.agents.base import AgentResponse
from myswat.agents.session_manager import SessionManager
from myswat.models.session import Session


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sm(store=None, runner=None, agent_row=None, compactor=None):
    store = store or MagicMock()
    runner = runner or MagicMock()
    agent_row = agent_row or {"id": 1, "role": "developer", "system_prompt": "Be helpful."}
    type(runner).is_session_started = PropertyMock(return_value=False)
    runner.workdir = "/tmp/test"
    runner.cli_session_id = None
    sm = SessionManager(store=store, runner=runner, agent_row=agent_row, project_id=1, compactor=compactor)
    return sm, store, runner


def _attach_session(sm, session_id=42):
    """Attach a fake Session object to the SessionManager."""
    session = MagicMock(spec=Session)
    session.id = session_id
    session.session_uuid = "abcdef1234567890"
    session.status = "open"
    sm._session = session
    return session


# ===========================================================================
# _update_progress
# ===========================================================================


class TestUpdateProgress:
    """Tests for SessionManager._update_progress."""

    def test_returns_immediately_when_session_is_none(self):
        sm, store, _ = _make_sm()
        # _session is None by default after construction
        response = MagicMock(spec=AgentResponse)
        sm._update_progress("do something", response, 1.5)
        store.update_session_progress.assert_not_called()

    def test_cancelled_response_builds_cancelled_note(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=10)
        store.count_session_turns.return_value = 5

        response = MagicMock(spec=AgentResponse)
        response.cancelled = True
        response.success = False
        response.output = "user aborted"

        sm._update_progress("fix the bug", response, 2.0)

        store.update_session_progress.assert_called_once()
        args = store.update_session_progress.call_args
        assert args[0][0] == 10  # session_id
        note = args[0][1]
        assert "CANCELLED" in note
        assert "turn 5" in note
        assert "2s" in note  # elapsed seconds

    def test_success_response_builds_success_note_with_agent_summary(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=7)
        store.count_session_turns.return_value = 2

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        # content lines: first short ones are skipped (<=10 chars), then a real line
        response.content = "ok\nThis is a meaningful response line from the agent"

        sm._update_progress("write tests", response, 3.5)

        store.update_session_progress.assert_called_once()
        args = store.update_session_progress.call_args
        assert args[0][0] == 7
        note = args[0][1]
        assert "turn 2" in note
        assert "3s" in note  # elapsed
        assert "write tests" in note
        # The agent_summary is extracted from response.content (first line > 10 chars)
        assert "This is a meaningful response line" in note
        assert "CANCELLED" not in note
        assert "ERROR" not in note

    def test_success_response_no_summary_when_content_lines_short(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=7)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "ok\nfine\nyes"  # all lines <= 10 chars

        sm._update_progress("do stuff", response, 1.0)

        store.update_session_progress.assert_called_once()
        note = store.update_session_progress.call_args[0][1]
        # No arrow/summary since no content line exceeds 10 chars
        assert "→" not in note
        assert "do stuff" in note

    def test_success_response_uses_first_line_of_prompt(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "A long enough response line for summary"

        sm._update_progress("first line\nsecond line\nthird line", response, 0.5)

        note = store.update_session_progress.call_args[0][1]
        assert "first line" in note
        assert "second line" not in note

    def test_error_response_builds_error_note(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=99)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = False
        response.output = "something went wrong"

        sm._update_progress("deploy", response, 0.8)

        store.update_session_progress.assert_called_once()
        args = store.update_session_progress.call_args
        assert args[0][0] == 99
        note = args[0][1]
        assert "ERROR" in note
        assert "turn 1" in note
        assert "deploy" in note

    def test_note_is_truncated_to_512_chars(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "A" * 1000  # long enough (> 10 chars) to be used as summary

        sm._update_progress("B" * 1000, response, 1.0)

        store.update_session_progress.assert_called_once()
        note = store.update_session_progress.call_args[0][1]
        assert len(note) <= 512

    def test_catches_exceptions_silently(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)

        store.count_session_turns.side_effect = RuntimeError("db down")

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = True
        response.content = "ok"

        # Should not raise
        sm._update_progress("prompt", response, 1.0)

    def test_duration_formatting_minutes(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = False

        sm._update_progress("task", response, 125.0)  # 2m05s

        note = store.update_session_progress.call_args[0][1]
        assert "2m05s" in note

    def test_duration_formatting_hours(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=1)
        store.count_session_turns.return_value = 1

        response = MagicMock(spec=AgentResponse)
        response.cancelled = False
        response.success = False

        sm._update_progress("task", response, 3661.0)  # 1h01m01s

        note = store.update_session_progress.call_args[0][1]
        assert "1h01m01s" in note


# ===========================================================================
# _check_mid_session_compaction
# ===========================================================================


class TestCheckMidSessionCompaction:
    """Tests for SessionManager._check_mid_session_compaction."""

    def test_returns_when_session_is_none(self):
        sm, store, _ = _make_sm()
        # no session attached, no compactor
        sm._check_mid_session_compaction()
        store.delete_compacted_turns.assert_not_called()

    def test_returns_when_compactor_is_none(self):
        sm, store, _ = _make_sm()
        _attach_session(sm)
        # compactor is None by default
        sm._check_mid_session_compaction()
        store.delete_compacted_turns.assert_not_called()

    def test_returns_when_should_compact_is_false(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = False
        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)

        sm._check_mid_session_compaction()

        compactor.should_compact.assert_called_once()
        compactor.compact_session.assert_not_called()

    def test_runs_compaction_when_should_compact_is_true(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = [10, 11, 12]

        sm, store, _ = _make_sm(compactor=compactor)
        session = _attach_session(sm, session_id=5)

        sm._check_mid_session_compaction()

        compactor.compact_session.assert_called_once_with(
            session_id=5,
            project_id=1,
            agent_id=1,
            mark_compacted=False,
        )
        store.delete_compacted_turns.assert_called_once_with(5)
        store.reset_session_token_count.assert_called_once_with(5)

    def test_prints_to_stderr_when_ids_and_deleted(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = [1, 2, 3]

        sm, store, _ = _make_sm(compactor=compactor)
        session = _attach_session(sm)
        store.delete_compacted_turns.return_value = 5

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._check_mid_session_compaction()

        output = captured.getvalue()
        assert "mid-session compaction" in output
        assert "3 knowledge entries created" in output
        assert "5 old turns deleted" in output

    def test_no_stderr_when_ids_empty_and_deleted_zero(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = []

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)
        store.delete_compacted_turns.return_value = 0

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._check_mid_session_compaction()

        output = captured.getvalue()
        # ids=[] and deleted=0 are both falsy, so no print
        assert output == ""

    def test_catches_exceptions_during_compaction(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.side_effect = RuntimeError("compaction failed")

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._check_mid_session_compaction()

        output = captured.getvalue()
        assert "Failed" in output
        assert "compaction failed" in output

    def test_catches_exceptions_from_delete_compacted_turns(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = [1]
        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)
        store.delete_compacted_turns.side_effect = RuntimeError("delete failed")

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._check_mid_session_compaction()

        output = captured.getvalue()
        assert "Failed" in output


# ===========================================================================
# close
# ===========================================================================


class TestClose:
    """Tests for SessionManager.close."""

    def test_returns_when_session_is_none(self):
        sm, store, _ = _make_sm()
        sm.close()
        store.close_session.assert_not_called()

    def test_calls_close_session_on_store(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=20)
        store.get_session.return_value = None

        sm.close()

        store.close_session.assert_called_once_with(20)

    def test_runs_compaction_on_close_when_compactor_says_yes(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = [1, 2]

        sm, store, _ = _make_sm(compactor=compactor)
        session = _attach_session(sm, session_id=30)
        store.get_session.return_value = None

        sm.close()

        store.close_session.assert_called_once_with(30)
        compactor.compact_session.assert_called_once()

    def test_no_compaction_when_compactor_says_no(self):
        compactor = MagicMock()
        compactor.should_compact.return_value = False

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm, session_id=30)
        store.get_session.return_value = None

        sm.close()

        store.close_session.assert_called_once_with(30)
        compactor.compact_session.assert_not_called()

    def test_no_compaction_when_compactor_is_none(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=30)
        store.get_session.return_value = None

        sm.close()

        store.close_session.assert_called_once_with(30)

    def test_deletes_archived_session_when_status_compacted(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=50)
        # store.get_session returns a dict with "status" key
        store.get_session.return_value = {"status": "compacted"}

        sm.close()

        store.delete_archived_session.assert_called_once_with(50)

    def test_no_delete_archived_when_status_not_compacted(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=50)
        store.get_session.return_value = {"status": "closed"}

        sm.close()

        store.delete_archived_session.assert_not_called()

    def test_no_delete_archived_when_get_session_returns_none(self):
        sm, store, _ = _make_sm()
        _attach_session(sm, session_id=50)
        store.get_session.return_value = None

        sm.close()

        store.delete_archived_session.assert_not_called()

    def test_close_with_compaction_and_compacted_status(self):
        """Full close flow: compaction runs AND session ends up compacted."""
        compactor = MagicMock()
        compactor.should_compact.return_value = True
        compactor.compact_session.return_value = [1]

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm, session_id=60)
        store.get_session.return_value = {"status": "compacted"}

        sm.close()

        store.close_session.assert_called_once_with(60)
        compactor.compact_session.assert_called_once()
        store.delete_archived_session.assert_called_once_with(60)


# ===========================================================================
# _compact
# ===========================================================================


class TestCompact:
    """Tests for SessionManager._compact."""

    def test_returns_when_session_is_none(self):
        compactor = MagicMock()
        sm, store, _ = _make_sm(compactor=compactor)
        # no session
        sm._compact()
        compactor.compact_session.assert_not_called()

    def test_returns_when_compactor_is_none(self):
        sm, store, _ = _make_sm()
        _attach_session(sm)
        sm._compact()
        # Nothing should happen, no error

    def test_calls_compact_session_with_mark_compacted_true(self):
        compactor = MagicMock()
        compactor.compact_session.return_value = [5, 6, 7]

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm, session_id=15)

        sm._compact()

        compactor.compact_session.assert_called_once_with(
            session_id=15,
            project_id=1,
            agent_id=1,
            mark_compacted=True,
        )

    def test_prints_to_stderr_when_ids_returned(self):
        compactor = MagicMock()
        compactor.compact_session.return_value = [10, 11]

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._compact()

        output = captured.getvalue()
        assert "[compaction]" in output
        assert "2 knowledge entries" in output
        assert "session archived" in output

    def test_no_stderr_when_ids_empty(self):
        compactor = MagicMock()
        compactor.compact_session.return_value = []

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._compact()

        output = captured.getvalue()
        assert output == ""

    def test_catches_exceptions_and_prints_failure(self):
        compactor = MagicMock()
        compactor.compact_session.side_effect = Exception("compact error")

        sm, store, _ = _make_sm(compactor=compactor)
        _attach_session(sm)

        captured = io.StringIO()
        with patch("sys.stderr", captured):
            sm._compact()

        output = captured.getvalue()
        assert "[compaction] Failed" in output
        assert "compact error" in output
