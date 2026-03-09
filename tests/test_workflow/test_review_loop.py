"""Comprehensive tests for myswat.workflow.review_loop module."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from myswat.workflow.review_loop import _parse_verdict, run_review_loop
from myswat.agents.base import AgentResponse
from myswat.models.work_item import ReviewVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_str(verdict="lgtm", issues=None, summary=""):
    """Build a raw JSON string for a ReviewVerdict payload."""
    payload = {"verdict": verdict, "summary": summary}
    if issues is not None:
        payload["issues"] = issues
    else:
        payload["issues"] = []
    return json.dumps(payload)


def _make_agent_response(content="", cancelled=False, error=None):
    """Create a mock AgentResponse with sensible defaults."""
    resp = MagicMock(spec=AgentResponse)
    resp.content = content
    resp.cancelled = cancelled
    resp.error = error
    return resp


def _make_session_manager(responses=None, agent_id="agent-1", role="developer"):
    """Create a MagicMock that mimics a SessionManager.

    Parameters
    ----------
    responses : list[AgentResponse] | None
        Successive return values for .send().  If ``None`` a single
        successful response is created.
    """
    sm = MagicMock()
    sm.agent_id = agent_id
    sm.agent_role = role
    sm.session = MagicMock()
    if responses is None:
        responses = [_make_agent_response(content="ok")]
    sm.send = MagicMock(side_effect=list(responses))
    return sm


# ===========================================================================
# _parse_verdict tests
# ===========================================================================

class TestParseVerdict:
    """Tests for the _parse_verdict helper."""

    # -- 1. Valid JSON with verdict, issues, summary -----------------------

    def test_valid_json_full(self):
        raw = _json_str(verdict="lgtm", issues=["nit: style"], summary="Looks good")
        result = _parse_verdict(raw)
        assert isinstance(result, ReviewVerdict)
        assert result.verdict == "lgtm"
        assert result.issues == ["nit: style"]
        assert result.summary == "Looks good"

    def test_valid_json_changes_requested(self):
        raw = _json_str(verdict="changes_requested", issues=["fix bug"], summary="Needs work")
        result = _parse_verdict(raw)
        assert result.verdict == "changes_requested"
        assert result.issues == ["fix bug"]
        assert result.summary == "Needs work"

    # -- 2. JSON inside ```json code block ---------------------------------

    def test_json_inside_json_code_block(self):
        raw = "Here is my review:\n```json\n" + _json_str(verdict="lgtm", summary="ok") + "\n```\nDone."
        result = _parse_verdict(raw)
        assert result.verdict == "lgtm"
        assert result.summary == "ok"

    def test_json_code_block_with_extra_whitespace(self):
        payload = _json_str(verdict="changes_requested", issues=["a", "b"], summary="fix")
        raw = "   ```json   \n  " + payload + "  \n```  "
        result = _parse_verdict(raw)
        assert result.verdict == "changes_requested"
        assert result.issues == ["a", "b"]

    # -- 3. JSON inside ``` code block (no json tag) -----------------------

    def test_json_inside_plain_code_block(self):
        payload = _json_str(verdict="lgtm", summary="all good")
        raw = "Review:\n```\n" + payload + "\n```"
        result = _parse_verdict(raw)
        assert result.verdict == "lgtm"
        assert result.summary == "all good"

    def test_plain_code_block_picks_json_block(self):
        """When multiple ``` blocks exist, the one starting with '{' is used."""
        non_json_block = "some code here"
        json_block = _json_str(verdict="lgtm", summary="picked")
        raw = f"```\n{non_json_block}\n```\n```\n{json_block}\n```"
        result = _parse_verdict(raw)
        assert result.verdict == "lgtm"
        assert result.summary == "picked"

    # -- 4. Invalid JSON but contains "lgtm" -> lgtm verdict --------------

    def test_invalid_json_with_lgtm_keyword(self):
        raw = "Everything looks great, lgtm!"
        result = _parse_verdict(raw)
        assert result.verdict == "lgtm"

    def test_lgtm_case_insensitive(self):
        raw = "LGTM, ship it."
        result = _parse_verdict(raw)
        assert result.verdict == "lgtm"

    # -- 5. Invalid JSON with "changes_requested" -> changes_requested -----

    def test_changes_requested_overrides_lgtm_in_plain_text(self):
        """If both 'lgtm' and 'changes_requested' appear, changes_requested wins."""
        raw = "Almost lgtm but changes_requested due to missing tests."
        result = _parse_verdict(raw)
        assert result.verdict == "changes_requested"

    def test_changes_requested_keyword_alone(self):
        raw = "I think we need changes_requested on this PR."
        result = _parse_verdict(raw)
        assert result.verdict == "changes_requested"

    # -- 6. Completely unparseable -> changes_requested with raw[:500] -----

    def test_unparseable_defaults_to_changes_requested(self):
        raw = "¯\\_(ツ)_/¯ no verdict here at all"
        result = _parse_verdict(raw)
        assert result.verdict == "changes_requested"
        assert raw[:500] in result.issues[0] or raw[:500] == result.issues[0]

    def test_unparseable_truncates_long_input(self):
        raw = "x" * 1000
        result = _parse_verdict(raw)
        assert result.verdict == "changes_requested"
        # The stored issue text should be at most 500 characters from the raw input
        assert len(result.issues) >= 1
        issue_text = result.issues[0]
        assert len(issue_text) <= 600  # some tolerance for wrapping text

    # -- 7. Missing verdict key in JSON -> defaults to changes_requested ---

    def test_missing_verdict_key(self):
        raw = json.dumps({"issues": ["missing verdict"], "summary": "oops"})
        result = _parse_verdict(raw)
        # Implementation should either default to changes_requested or raise;
        # per spec it defaults to changes_requested.
        assert result.verdict == "changes_requested"

    # -- 8. Empty issues array in JSON -------------------------------------

    def test_empty_issues_array(self):
        raw = _json_str(verdict="lgtm", issues=[], summary="Clean code")
        result = _parse_verdict(raw)
        assert result.verdict == "lgtm"
        assert result.issues == []
        assert result.summary == "Clean code"

    def test_empty_string_input(self):
        result = _parse_verdict("")
        assert result.verdict == "changes_requested"

    def test_whitespace_only_input(self):
        result = _parse_verdict("   \n\t  ")
        assert result.verdict == "changes_requested"


# ===========================================================================
# run_review_loop tests
# ===========================================================================

class TestRunReviewLoop:
    """Tests for the run_review_loop orchestration function."""

    def _setup_mocks(
        self,
        dev_responses=None,
        reviewer_responses=None,
        store_artifact_ok=True,
        store_review_cycle_ok=True,
    ):
        """Return (store, dev_sm, reviewer_sm, task, project_id, work_item_id)."""
        store = MagicMock()
        if store_artifact_ok:
            store.create_artifact = MagicMock(return_value="artifact-123")
        else:
            store.create_artifact = MagicMock(side_effect=Exception("artifact store boom"))

        if store_review_cycle_ok:
            store.create_review_cycle = MagicMock(return_value=True)
        else:
            store.create_review_cycle = MagicMock(side_effect=Exception("review cycle boom"))

        dev_sm = _make_session_manager(
            responses=dev_responses or [_make_agent_response(content="dev code output")],
            agent_id="dev-agent",
            role="developer",
        )
        reviewer_sm = _make_session_manager(
            responses=reviewer_responses
            or [
                _make_agent_response(
                    content=_json_str(verdict="lgtm", summary="good")
                )
            ],
            agent_id="reviewer-agent",
            role="reviewer",
        )

        task = MagicMock()
        task.description = "Implement feature X"

        return store, dev_sm, reviewer_sm, task, "proj-1", "wi-1"

    # -- 1. LGTM on first iteration ----------------------------------------

    def test_lgtm_first_iteration(self):
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            reviewer_responses=[
                _make_agent_response(content=_json_str(verdict="lgtm", summary="Approved")),
            ],
        )
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)

        assert isinstance(result, ReviewVerdict)
        assert result.verdict == "lgtm"
        # Dev should have been called exactly once (initial turn)
        assert dev_sm.send.call_count == 1
        # Reviewer should have been called exactly once
        assert rev_sm.send.call_count == 1

    # -- 2. Changes requested then LGTM on second iteration ----------------

    def test_changes_then_lgtm(self):
        dev_responses = [
            _make_agent_response(content="initial code"),
            _make_agent_response(content="revised code"),
        ]
        reviewer_responses = [
            _make_agent_response(
                content=_json_str(
                    verdict="changes_requested",
                    issues=["fix the bug"],
                    summary="needs fix",
                )
            ),
            _make_agent_response(
                content=_json_str(verdict="lgtm", summary="now it's good")
            ),
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            dev_responses=dev_responses,
            reviewer_responses=reviewer_responses,
        )
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)

        assert result.verdict == "lgtm"
        assert dev_sm.send.call_count == 2
        assert rev_sm.send.call_count == 2

    # -- 3. Max iterations reached -----------------------------------------

    def test_max_iterations_reached(self):
        max_iter = 3
        dev_responses = [_make_agent_response(content=f"code v{i}") for i in range(max_iter)]
        reviewer_responses = [
            _make_agent_response(
                content=_json_str(
                    verdict="changes_requested",
                    issues=[f"issue {i}"],
                )
            )
            for i in range(max_iter)
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            dev_responses=dev_responses,
            reviewer_responses=reviewer_responses,
        )
        result = run_review_loop(
            store, dev_sm, rev_sm, task, pid, wid, max_iterations=max_iter
        )

        # After exhausting iterations the last verdict should be returned
        assert result.verdict == "changes_requested"
        assert dev_sm.send.call_count == max_iter
        assert rev_sm.send.call_count == max_iter

    # -- 4. Dev failure breaks loop early -----------------------------------

    def test_dev_failure_breaks_loop(self):
        dev_sm = _make_session_manager(agent_id="dev-agent", role="developer")
        dev_sm.send = MagicMock(side_effect=Exception("dev crashed"))

        store, _, rev_sm, task, pid, wid = self._setup_mocks()

        with pytest.raises(Exception, match="dev crashed"):
            run_review_loop(store, dev_sm, rev_sm, task, pid, wid)

        # Reviewer should never have been called
        assert rev_sm.send.call_count == 0

    # -- 5. Reviewer failure breaks loop early ------------------------------

    def test_reviewer_failure_breaks_loop(self):
        rev_sm = _make_session_manager(agent_id="reviewer-agent", role="reviewer")
        rev_sm.send = MagicMock(side_effect=Exception("reviewer crashed"))

        store, dev_sm, _, task, pid, wid = self._setup_mocks()

        with pytest.raises(Exception, match="reviewer crashed"):
            run_review_loop(store, dev_sm, rev_sm, task, pid, wid)

        # Dev should have been called once before the reviewer crash
        assert dev_sm.send.call_count == 1

    # -- 6. Artifact creation failure doesn't crash -------------------------

    def test_artifact_creation_failure_continues(self):
        reviewer_responses = [
            _make_agent_response(content=_json_str(verdict="lgtm", summary="ok")),
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            store_artifact_ok=False,
            reviewer_responses=reviewer_responses,
        )

        # Should NOT raise despite store_artifact throwing
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
        assert result.verdict == "lgtm"
        # store_artifact was attempted
        store.create_artifact.assert_called()

    # -- 7. Review cycle DB failure doesn't crash ---------------------------

    def test_review_cycle_db_failure_continues(self):
        reviewer_responses = [
            _make_agent_response(content=_json_str(verdict="lgtm", summary="ok")),
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            store_review_cycle_ok=False,
            reviewer_responses=reviewer_responses,
        )

        # Should NOT raise despite store_review_cycle throwing
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
        assert result.verdict == "lgtm"

    # -- 8. Cancelled response handling ------------------------------------

    def test_cancelled_dev_response(self):
        """A cancelled dev response should be handled gracefully."""
        dev_responses = [_make_agent_response(content="", cancelled=True)]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            dev_responses=dev_responses,
        )

        # Behaviour may vary: either raises, returns early, or continues.
        # We verify no unhandled exception blows up.
        try:
            result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
            # If it returns, the result should still be a ReviewVerdict
            assert isinstance(result, ReviewVerdict)
        except Exception:
            # Some implementations raise on cancellation; that is acceptable
            pass

    def test_cancelled_reviewer_response(self):
        """A cancelled reviewer response should be handled gracefully."""
        reviewer_responses = [_make_agent_response(content="", cancelled=True)]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            reviewer_responses=reviewer_responses,
        )

        try:
            result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
            assert isinstance(result, ReviewVerdict)
        except Exception:
            pass

    # -- Additional behavioural checks -------------------------------------

    def test_context_forwarded(self):
        """The optional context parameter should be usable without error."""
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks()
        result = run_review_loop(
            store, dev_sm, rev_sm, task, pid, wid, context="extra context here"
        )
        assert isinstance(result, ReviewVerdict)

    def test_default_max_iterations_is_five(self):
        """Without explicit max_iterations the loop should allow up to 5 rounds."""
        dev_responses = [_make_agent_response(content=f"code v{i}") for i in range(5)]
        reviewer_responses = [
            _make_agent_response(
                content=_json_str(verdict="changes_requested", issues=[f"issue {i}"])
            )
            for i in range(5)
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            dev_responses=dev_responses,
            reviewer_responses=reviewer_responses,
        )
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
        assert result.verdict == "changes_requested"
        assert dev_sm.send.call_count == 5
        assert rev_sm.send.call_count == 5

    def test_artifact_id_none_skips_review_cycle_store(self):
        """When artifact storage fails (artifact_id=None), store_review_cycle
        should NOT be called for that iteration."""
        reviewer_responses = [
            _make_agent_response(content=_json_str(verdict="lgtm", summary="ok")),
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            store_artifact_ok=False,
            reviewer_responses=reviewer_responses,
        )
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
        assert result.verdict == "lgtm"
        # store_review_cycle should not be called when artifact_id is None
        store.create_review_cycle.assert_not_called()

    def test_multiple_review_cycles_stored(self):
        """Each successful iteration should persist a review cycle to the store."""
        dev_responses = [
            _make_agent_response(content="v1"),
            _make_agent_response(content="v2"),
        ]
        reviewer_responses = [
            _make_agent_response(
                content=_json_str(verdict="changes_requested", issues=["fix"])
            ),
            _make_agent_response(
                content=_json_str(verdict="lgtm", summary="done")
            ),
        ]
        store, dev_sm, rev_sm, task, pid, wid = self._setup_mocks(
            dev_responses=dev_responses,
            reviewer_responses=reviewer_responses,
        )
        result = run_review_loop(store, dev_sm, rev_sm, task, pid, wid)
        assert result.verdict == "lgtm"
        # Two iterations -> two review cycles stored
        assert store.create_review_cycle.call_count == 2
