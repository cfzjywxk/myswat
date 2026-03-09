"""Extended tests for WorkflowEngine — covers uncovered lines."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from myswat.agents.base import AgentResponse
from myswat.models.work_item import ReviewVerdict
from myswat.workflow.engine import (
    WorkflowEngine,
    _default_ask,
    PhaseResult,
    BugFixResult,
    GATestResult,
    WorkflowResult,
    MAX_GA_BUGS,
)

# Re-use the conftest helper
from tests.conftest import make_fake_session_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(content: str = "ok") -> AgentResponse:
    return AgentResponse(content=content, exit_code=0)


def _fail(content: str = "error") -> AgentResponse:
    return AgentResponse(content=content, exit_code=1)


def _lgtm_json() -> str:
    return json.dumps({"verdict": "lgtm", "issues": [], "summary": "Looks good"})


def _changes_json(issues: list[str] | None = None) -> str:
    return json.dumps({
        "verdict": "changes_requested",
        "issues": issues or ["fix this"],
        "summary": "Needs work",
    })


def _make_engine(
    *,
    dev_responses=None,
    qa_responses=None,
    qa_count: int = 1,
    ask_return="y",
    work_item_id=1,
    max_review: int = 5,
    store=None,
):
    """Build a WorkflowEngine with fake session managers."""
    dev_sm = make_fake_session_manager(
        agent_id=10, agent_role="developer", responses=dev_responses,
    )
    qa_sms = [
        make_fake_session_manager(
            agent_id=20 + i,
            agent_role=f"qa-{i}",
            responses=qa_responses,
            session_id=100 + i,
        )
        for i in range(qa_count)
    ]
    if store is None:
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
    ask = MagicMock(return_value=ask_return) if isinstance(ask_return, str) else ask_return
    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=qa_sms,
        project_id=1,
        work_item_id=work_item_id,
        max_review_iterations=max_review,
        ask_user=ask,
    )
    return engine, dev_sm, qa_sms


# ===================================================================
# 1. _default_ask  (lines 116-119)
# ===================================================================

class TestDefaultAsk:
    """Tests for the module-level _default_ask helper."""

    def test_returns_stripped_input(self):
        with patch("builtins.input", return_value="  hello  "):
            assert _default_ask("prompt") == "hello"

    def test_eof_error_returns_n(self):
        with patch("builtins.input", side_effect=EOFError):
            assert _default_ask("prompt") == "n"

    def test_keyboard_interrupt_returns_n(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _default_ask("prompt") == "n"


# ===================================================================
# 2. run() — main orchestrator  (lines 180-289)
# ===================================================================

class TestRun:
    """Tests for WorkflowEngine.run, mocking all sub-methods."""

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Report")
    @patch.object(WorkflowEngine, "_run_ga_test_phase")
    @patch.object(WorkflowEngine, "_run_phase")
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["phase-A"])
    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="the plan")
    @patch.object(WorkflowEngine, "_run_design", return_value="the design")
    def test_happy_path(
        self, m_design, m_plan, m_review, m_checkpoint, m_phases,
        m_phase, m_ga, m_report,
    ):
        # Setup return values
        m_review.side_effect = [
            ("reviewed design", 1),  # design review
            ("reviewed plan", 1),    # plan review
        ]
        m_checkpoint.side_effect = [
            "approved design",  # after design review
            "approved plan",    # after plan review
        ]
        m_phase.return_value = PhaseResult(
            name="phase-A", summary="done", review_iterations=1, committed=True,
        )
        m_ga.return_value = GATestResult(passed=True)

        engine, dev, qas = _make_engine()
        result = engine.run("build a widget")

        assert result.requirement == "build a widget"
        assert result.design == "approved design"
        assert result.plan == "approved plan"
        assert len(result.phases) == 1
        assert result.phases[0].committed is True
        assert result.ga_test.passed is True
        assert result.final_report == "# Report"
        assert result.success is True

    @patch.object(WorkflowEngine, "_run_design", return_value="")
    def test_design_fails_aborts(self, m_design):
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.design == ""
        assert result.success is False

    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("design", 1))
    @patch.object(WorkflowEngine, "_run_design", return_value="the design")
    def test_user_rejects_design_aborts(self, m_design, m_review):
        engine, _, _ = _make_engine(ask_return="n")
        result = engine.run("req")
        # Should abort after user rejects
        assert result.plan == ""
        assert result.success is False

    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="")
    @patch.object(WorkflowEngine, "_run_design", return_value="design")
    def test_planning_fails_aborts(self, m_design, m_plan, m_review, m_cp):
        m_review.return_value = ("design", 1)
        m_cp.return_value = "design"
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.plan == ""
        assert result.success is False

    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="plan")
    @patch.object(WorkflowEngine, "_run_design", return_value="design")
    def test_user_rejects_plan_aborts(self, m_design, m_plan, m_review, m_cp):
        m_review.side_effect = [("design", 1), ("plan", 1)]
        m_cp.side_effect = ["design", None]  # accept design, reject plan
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.success is False

    @patch.object(WorkflowEngine, "_generate_report", return_value="report")
    @patch.object(WorkflowEngine, "_run_ga_test_phase")
    @patch.object(WorkflowEngine, "_run_phase")
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["p1"])
    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="plan")
    @patch.object(WorkflowEngine, "_run_design", return_value="design")
    def test_ga_test_aborted(
        self, m_design, m_plan, m_review, m_cp, m_phases, m_phase, m_ga, m_report,
    ):
        m_review.side_effect = [("design", 1), ("plan", 1)]
        m_cp.side_effect = ["design", "plan"]
        m_phase.return_value = PhaseResult(
            name="p1", summary="s", review_iterations=1, committed=True,
        )
        m_ga.return_value = GATestResult(aborted=True, bugs_found=7, bugs_fixed=2)
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.ga_test.aborted is True
        # report is still generated even when aborted

    @patch.object(WorkflowEngine, "_generate_report", return_value="report")
    @patch.object(WorkflowEngine, "_run_ga_test_phase")
    @patch.object(WorkflowEngine, "_run_phase")
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["p1"])
    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="plan")
    @patch.object(WorkflowEngine, "_run_design", return_value="design")
    def test_not_committed_means_not_success(
        self, m_design, m_plan, m_review, m_cp, m_phases, m_phase, m_ga, m_report,
    ):
        m_review.side_effect = [("design", 1), ("plan", 1)]
        m_cp.side_effect = ["design", "plan"]
        m_phase.return_value = PhaseResult(
            name="p1", summary="s", review_iterations=1, committed=False,
        )
        m_ga.return_value = GATestResult(passed=True)
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.success is False


# ===================================================================
# 3. _run_design  (lines 296-303)
# ===================================================================

class TestRunDesign:

    def test_success(self):
        engine, dev, _ = _make_engine(dev_responses=["the design"])
        result = engine._run_design("build X")
        assert result == "the design"
        dev.send.assert_called_once()
        assert "build X" in dev.send.call_args[0][0]

    def test_failure_returns_empty(self):
        engine, dev, _ = _make_engine(dev_responses=[_fail()])
        result = engine._run_design("req")
        assert result == ""


# ===================================================================
# 4. _run_planning  (lines 306-316)
# ===================================================================

class TestRunPlanning:

    def test_success(self):
        engine, dev, _ = _make_engine(dev_responses=["the plan"])
        result = engine._run_planning("design", "requirement")
        assert result == "the plan"

    def test_failure_returns_empty(self):
        engine, dev, _ = _make_engine(dev_responses=[_fail()])
        result = engine._run_planning("design", "req")
        assert result == ""


# ===================================================================
# 5. _run_phase  (lines 328-378)
# ===================================================================

class TestRunPhase:

    def test_implementation_fails(self):
        """When dev fails to implement, return failed PhaseResult."""
        engine, dev, _ = _make_engine(
            dev_responses=[_fail()],  # implement fails
            qa_responses=[_lgtm_json()],
        )
        result = engine._run_phase(
            phase_name="setup",
            phase_index=1,
            total_phases=2,
            requirement="req",
            design="design",
            plan="plan",
            completed_summaries=[],
        )
        assert result.name == "setup"
        assert result.summary == "Implementation failed."
        assert result.review_iterations == 0

    def test_summary_success_and_commit_success(self):
        """Full happy path for a phase."""
        engine, dev, qas = _make_engine(
            dev_responses=[
                "impl done",       # implement
                "summary text",    # summarize
                "committed",       # commit
            ],
            qa_responses=[_lgtm_json()],
            max_review=1,
        )
        # Override _run_review_loop to avoid complexity
        with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1)):
            result = engine._run_phase(
                phase_name="core",
                phase_index=1,
                total_phases=1,
                requirement="req",
                design="design",
                plan="plan",
                completed_summaries=["prev phase"],
            )
        assert result.committed is True
        assert result.review_iterations == 1

    def test_summary_fails_uses_impl_content(self):
        """When summarization fails, it falls back to implementation content."""
        engine, dev, _ = _make_engine(
            dev_responses=[
                "impl content",   # implement
                _fail(),          # summarize fails
                "committed",      # commit
            ],
            qa_responses=[_lgtm_json()],
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("impl content", 1)):
            result = engine._run_phase(
                phase_name="core",
                phase_index=1,
                total_phases=1,
                requirement="req",
                design="design",
                plan="plan",
                completed_summaries=[],
            )
        assert result.committed is True

    def test_commit_fails(self):
        """When commit fails, committed should be False."""
        engine, dev, _ = _make_engine(
            dev_responses=[
                "impl done",      # implement
                "summary",        # summarize
                _fail(),          # commit fails
            ],
            qa_responses=[_lgtm_json()],
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1)):
            result = engine._run_phase(
                phase_name="core",
                phase_index=1,
                total_phases=1,
                requirement="req",
                design="design",
                plan="plan",
                completed_summaries=[],
            )
        assert result.committed is False


# ===================================================================
# 6. _run_ga_test_phase  (lines 396-529)
# ===================================================================

class TestRunGATestPhase:

    def test_qa_fails_test_plan(self):
        """When QA fails to generate test plan, return early."""
        engine, dev, qas = _make_engine(qa_responses=[_fail()])
        result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.test_plan == ""
        assert not result.passed

    def test_user_rejects_test_plan(self):
        """When user rejects the test plan, aborted should be True."""
        engine, dev, qas = _make_engine(
            qa_responses=["test plan content"],
            dev_responses=[_lgtm_json()],
            ask_return="n",
            max_review=1,
        )
        # _run_review_loop returns the reviewed plan
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.aborted is True

    def test_qa_fails_test_execution(self):
        """When QA fails to execute tests, return early."""
        engine, dev, qas = _make_engine(
            qa_responses=["test plan", _fail()],  # plan ok, exec fails
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert not result.passed

    def test_all_tests_pass_immediately(self):
        """When tests pass immediately, result.passed is True."""
        pass_output = json.dumps({"status": "pass"})
        engine, dev, qas = _make_engine(
            qa_responses=["test plan", pass_output],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.passed is True
        assert result.bugs_found == 0

    def test_bug_fix_loop_single_round(self):
        """Bugs found, fixed, re-test passes."""
        bugs_output = json.dumps({
            "bugs": [{"title": "bug1", "severity": "major", "description": "d"}]
        })
        pass_output = json.dumps({"status": "pass"})
        engine, dev, qas = _make_engine(
            qa_responses=[
                "test plan",     # generate test plan
                bugs_output,     # execute tests - bugs found
                pass_output,     # re-test after fixes - pass
                "test report",   # generate test report
            ],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            with patch.object(engine, "_run_bug_fix", return_value=BugFixResult(
                title="bug1", fixed=True, summary="fixed it",
            )):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")

        assert result.passed is True
        assert result.bugs_found == 1
        assert result.bugs_fixed == 1
        assert len(result.bug_fixes) == 1

    def test_bug_fix_fails(self):
        """Bug fix fails, but we still continue to re-test."""
        bugs_output = json.dumps({
            "bugs": [{"title": "bug1", "severity": "major"}]
        })
        pass_output = json.dumps({"status": "pass"})
        engine, dev, qas = _make_engine(
            qa_responses=[
                "test plan",
                bugs_output,     # first run - bugs
                pass_output,     # re-test - pass
                "report",
            ],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            with patch.object(engine, "_run_bug_fix", return_value=BugFixResult(
                title="bug1", fixed=False, summary="couldn't fix",
            )):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")

        assert result.passed is True  # re-test passed
        assert result.bugs_fixed == 0

    def test_too_many_bugs_aborts(self):
        """More than MAX_GA_BUGS total bugs aborts the loop."""
        many_bugs = [{"title": f"bug{i}", "severity": "major"} for i in range(MAX_GA_BUGS + 1)]
        bugs_output = json.dumps({"bugs": many_bugs})
        engine, dev, qas = _make_engine(
            qa_responses=[
                "test plan",
                bugs_output,   # too many bugs at once
                "report",
            ],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            with patch.object(engine, "_run_bug_fix", return_value=BugFixResult(
                title="x", fixed=True, summary="f",
            )):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")

        assert result.aborted is True

    def test_qa_fails_retest(self):
        """QA fails during re-test, loop breaks."""
        bugs_output = json.dumps({
            "bugs": [{"title": "bug1", "severity": "major"}]
        })
        engine, dev, qas = _make_engine(
            qa_responses=[
                "test plan",
                bugs_output,   # first run - bugs
                _fail(),       # re-test fails
                "report",
            ],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            with patch.object(engine, "_run_bug_fix", return_value=BugFixResult(
                title="bug1", fixed=True, summary="fixed",
            )):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")

        # Not passed because re-test failed and broke the loop
        # bugs list stays truthy so not result.passed
        assert not result.passed

    def test_two_rounds_of_bugs_then_pass(self):
        """Bugs found, fixed, re-test finds more bugs, fixed, re-test passes."""
        bugs1 = json.dumps({"bugs": [{"title": "b1", "severity": "high"}]})
        bugs2 = json.dumps({"bugs": [{"title": "b2", "severity": "low"}]})
        pass_out = json.dumps({"status": "pass"})

        engine, dev, qas = _make_engine(
            qa_responses=[
                "test plan",
                bugs1,          # first run
                bugs2,          # re-test after first fix
                pass_out,       # re-test after second fix
                "final report",
            ],
            ask_return="y",
            max_review=1,
        )
        fix_results = [
            BugFixResult(title="b1", fixed=True, summary="f1"),
            BugFixResult(title="b2", fixed=True, summary="f2"),
        ]
        fix_iter = iter(fix_results)
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            with patch.object(engine, "_run_bug_fix", side_effect=lambda *a, **k: next(fix_iter)):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")

        assert result.passed is True
        assert result.bugs_found == 2
        assert result.bugs_fixed == 2

    def test_report_generated_on_success(self):
        """QA test report is generated after all tests pass."""
        pass_output = json.dumps({"status": "pass"})
        engine, dev, qas = _make_engine(
            qa_responses=["test plan", pass_output],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        # For immediate pass, test_report is the exec output
        assert result.test_report == pass_output

    def test_report_failure_graceful(self):
        """If QA fails to generate report, test_report stays empty."""
        bugs_output = json.dumps({"bugs": [{"title": "b1", "severity": "low"}]})
        pass_output = json.dumps({"status": "pass"})
        engine, dev, qas = _make_engine(
            qa_responses=[
                "test plan",
                bugs_output,     # bugs found
                pass_output,     # re-test passes
                _fail(),         # report generation fails
            ],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1)):
            with patch.object(engine, "_run_bug_fix", return_value=BugFixResult(
                title="b1", fixed=True, summary="f",
            )):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.test_report == ""


# ===================================================================
# 7. _run_bug_fix  (lines 554-602)
# ===================================================================

class TestRunBugFix:

    def test_estimation_fails(self):
        """If dev fails to estimate, returns unfixed result."""
        engine, dev, _ = _make_engine(dev_responses=[_fail()])
        bug = {"title": "crash", "description": "app crashes"}
        result = engine._run_bug_fix(bug, "req", "design")
        assert result.title == "crash"
        assert not result.fixed

    def test_simple_fix_success(self):
        """Simple fix path: estimate -> fix -> summarize."""
        engine, dev, _ = _make_engine(dev_responses=[
            json.dumps({"assessment": "simple_fix"}),   # estimate
            "fix applied",                                # fix
            "summary of fix",                             # summarize
        ])
        bug = {"title": "typo", "description": "typo in code", "repro_steps": "run it"}
        result = engine._run_bug_fix(bug, "req", "design")
        assert result.fixed is True
        assert result.summary == "summary of fix"
        assert not result.arch_change

    def test_simple_fix_fails(self):
        """Dev fails to fix the bug."""
        engine, dev, _ = _make_engine(dev_responses=[
            json.dumps({"assessment": "simple_fix"}),   # estimate
            _fail(),                                      # fix fails
        ])
        bug = {"title": "bug1"}
        result = engine._run_bug_fix(bug, "req", "design")
        assert not result.fixed

    def test_simple_fix_summarize_fails_uses_fix_content(self):
        """When summarization fails, summary comes from fix response."""
        engine, dev, _ = _make_engine(dev_responses=[
            json.dumps({"assessment": "simple_fix"}),
            "fix content here",
            _fail(),  # summarize fails
        ])
        bug = {"title": "bug1"}
        result = engine._run_bug_fix(bug, "req", "design")
        assert result.fixed is True
        assert result.summary == "fix content here"

    def test_arch_change_triggers_sub_workflow(self):
        """Arch change estimation triggers _run_bug_fix_arch_change."""
        engine, dev, _ = _make_engine(dev_responses=[
            json.dumps({"assessment": "arch_change"}),
        ])
        sub_result = WorkflowResult(
            requirement="bug", success=True, final_report="sub report done",
        )
        with patch.object(engine, "_run_bug_fix_arch_change", return_value=sub_result):
            bug = {"title": "big bug", "description": "needs redesign"}
            result = engine._run_bug_fix(bug, "req", "design")
        assert result.arch_change is True
        assert result.fixed is True
        assert "sub report" in result.summary

    def test_arch_change_sub_workflow_fails(self):
        """Arch change sub-workflow fails."""
        engine, dev, _ = _make_engine(dev_responses=[
            json.dumps({"assessment": "arch_change"}),
        ])
        sub_result = WorkflowResult(requirement="bug", success=False, final_report="")
        with patch.object(engine, "_run_bug_fix_arch_change", return_value=sub_result):
            bug = {"title": "big bug"}
            result = engine._run_bug_fix(bug, "req", "design")
        assert result.arch_change is True
        assert not result.fixed
        assert result.summary == "Sub-workflow completed"

    def test_bug_with_missing_fields(self):
        """Bug dict with no title/description fields still works."""
        engine, dev, _ = _make_engine(dev_responses=[
            json.dumps({"assessment": "simple_fix"}),
            "fixed",
            "summary",
        ])
        bug = {}
        result = engine._run_bug_fix(bug, "req", "design")
        assert result.title == "Unknown bug"
        assert result.fixed is True


# ===================================================================
# 8. _run_bug_fix_arch_change  (lines 606-692)
# ===================================================================

class TestRunBugFixArchChange:

    def test_design_fails_early_return(self):
        """If design fails, return early with empty sub-result."""
        engine, _, _ = _make_engine()
        with patch.object(engine, "_run_design", return_value=""):
            result = engine._run_bug_fix_arch_change(
                {"title": "bug"}, "req", "design",
            )
        assert result.design == ""
        assert not result.success

    def test_user_rejects_design(self):
        """User rejects bug fix design."""
        engine, _, _ = _make_engine()
        with patch.object(engine, "_run_design", return_value="new design"):
            with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1)):
                with patch.object(engine, "_user_checkpoint", return_value=None):
                    result = engine._run_bug_fix_arch_change(
                        {"title": "bug"}, "req", "design",
                    )
        assert not result.success

    def test_planning_fails(self):
        """Planning fails after design approval."""
        engine, _, _ = _make_engine()
        with patch.object(engine, "_run_design", return_value="design"):
            with patch.object(engine, "_run_review_loop", return_value=("design", 1)):
                with patch.object(engine, "_user_checkpoint", return_value="design"):
                    with patch.object(engine, "_run_planning", return_value=""):
                        result = engine._run_bug_fix_arch_change(
                            {"title": "bug"}, "req", "design",
                        )
        assert result.plan == ""
        assert not result.success

    def test_user_rejects_plan(self):
        """User rejects bug fix plan."""
        engine, _, _ = _make_engine()
        with patch.object(engine, "_run_design", return_value="design"):
            with patch.object(engine, "_run_review_loop", side_effect=[
                ("design", 1), ("plan", 1),
            ]):
                with patch.object(engine, "_user_checkpoint", side_effect=[
                    "design", None,  # accept design, reject plan
                ]):
                    with patch.object(engine, "_run_planning", return_value="plan"):
                        result = engine._run_bug_fix_arch_change(
                            {"title": "bug"}, "req", "design",
                        )
        assert not result.success

    def test_full_success(self):
        """Full arch change sub-workflow succeeds."""
        engine, _, _ = _make_engine()
        phase_result = PhaseResult(name="fix-phase", summary="done", review_iterations=1, committed=True)
        with patch.object(engine, "_run_design", return_value="new design"):
            with patch.object(engine, "_run_review_loop", side_effect=[
                ("reviewed design", 1), ("reviewed plan", 1),
            ]):
                with patch.object(engine, "_user_checkpoint", side_effect=[
                    "approved design", "approved plan",
                ]):
                    with patch.object(engine, "_run_planning", return_value="new plan"):
                        with patch.object(engine, "_parse_phases", return_value=["fix-phase"]):
                            with patch.object(engine, "_run_phase", return_value=phase_result):
                                result = engine._run_bug_fix_arch_change(
                                    {"title": "critical bug", "description": "desc"},
                                    "req", "design",
                                )
        assert result.success is True
        assert len(result.phases) == 1
        assert result.phases[0].committed is True
        assert "critical bug" in result.final_report

    def test_phase_not_committed(self):
        """Sub-workflow with uncommitted phase = not successful."""
        engine, _, _ = _make_engine()
        phase_result = PhaseResult(name="p", summary="s", review_iterations=1, committed=False)
        with patch.object(engine, "_run_design", return_value="d"):
            with patch.object(engine, "_run_review_loop", side_effect=[("d", 1), ("p", 1)]):
                with patch.object(engine, "_user_checkpoint", side_effect=["d", "p"]):
                    with patch.object(engine, "_run_planning", return_value="p"):
                        with patch.object(engine, "_parse_phases", return_value=["p"]):
                            with patch.object(engine, "_run_phase", return_value=phase_result):
                                result = engine._run_bug_fix_arch_change(
                                    {"title": "b"}, "req", "design",
                                )
        assert not result.success

    def test_multiple_phases(self):
        """Arch change with multiple phases."""
        engine, _, _ = _make_engine()
        pr1 = PhaseResult(name="p1", summary="s1", review_iterations=1, committed=True)
        pr2 = PhaseResult(name="p2", summary="s2", review_iterations=1, committed=True)
        with patch.object(engine, "_run_design", return_value="d"):
            with patch.object(engine, "_run_review_loop", side_effect=[("d", 1), ("p", 1)]):
                with patch.object(engine, "_user_checkpoint", side_effect=["d", "p"]):
                    with patch.object(engine, "_run_planning", return_value="p"):
                        with patch.object(engine, "_parse_phases", return_value=["p1", "p2"]):
                            with patch.object(engine, "_run_phase", side_effect=[pr1, pr2]):
                                result = engine._run_bug_fix_arch_change(
                                    {"title": "b"}, "req", "design",
                                )
        assert result.success is True
        assert len(result.phases) == 2
        assert "2" in result.final_report  # "Phases: 2"


# ===================================================================
# 9. _run_review_loop edge cases  (lines 746-747, 759-760, 771,
#    792-793, 807-808, 814-832)
# ===================================================================

class TestRunReviewLoopEdgeCases:

    def test_artifact_persistence_error(self):
        """Exception during create_artifact is caught gracefully."""
        store = MagicMock()
        store.create_artifact.side_effect = Exception("DB down")
        engine, dev, qas = _make_engine(store=store, max_review=1)
        # QA returns LGTM
        qas[0].send.return_value = _ok(_lgtm_json())
        result, iters = engine._run_review_loop(
            artifact="artifact text",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 1
        assert result == "artifact text"

    def test_reviewer_failure_continues(self):
        """Reviewer failure doesn't crash the loop; other reviewers proceed."""
        engine, dev, qas = _make_engine(qa_count=2, max_review=1)
        # First QA fails, second QA gives LGTM
        qas[0].send.return_value = _fail()
        qas[1].send.return_value = _ok(_lgtm_json())
        result, iters = engine._run_review_loop(
            artifact="artifact",
            artifact_type="code",
            context="ctx",
        )
        # Because first reviewer failed (continues), second said LGTM,
        # but all_lgtm stays True only if all reviewers who responded said LGTM.
        # A failed reviewer is skipped (continue), so all_lgtm stays True.
        assert iters == 1

    def test_reviewer_verdict_with_summary_no_issues(self):
        """Non-lgtm verdict with summary but no issues uses summary as issue."""
        engine, dev, qas = _make_engine(max_review=2)
        # iter 1: changes_requested with summary only, no issues array
        verdict_json = json.dumps({
            "verdict": "changes_requested",
            "issues": [],
            "summary": "Needs more detail",
        })
        qas[0].send.return_value = _ok(verdict_json)
        # dev addresses and re-submits
        dev.send.return_value = _ok("addressed version")
        # iter 2: lgtm
        qas[0].send.side_effect = [_ok(verdict_json), _ok(_lgtm_json())]
        dev.send.return_value = _ok("addressed version")
        result, iters = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 2

    def test_proposer_fails_to_address_breaks(self):
        """When proposer fails to address comments, loop breaks."""
        engine, dev, qas = _make_engine(max_review=3)
        changes = _changes_json(["issue 1"])
        qas[0].send.return_value = _ok(changes)
        dev.send.return_value = _fail()  # proposer fails to address
        result, iters = engine._run_review_loop(
            artifact="original",
            artifact_type="code",
            context="ctx",
        )
        assert iters == 3  # returns max_review when loop ends early via break

    def test_max_iterations_unreviewed_artifact_persisted(self):
        """At max iterations, the final unreviewed artifact is persisted."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        engine, dev, qas = _make_engine(store=store, max_review=1)
        changes = _changes_json(["issue"])
        qas[0].send.return_value = _ok(changes)
        dev.send.return_value = _ok("revised")

        result, iters = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 1
        assert result == "revised"
        # The last create_artifact call should be the unreviewed final revision
        last_create_call = store.create_artifact.call_args_list[-1]
        assert last_create_call.kwargs.get("iteration") == 2 or last_create_call[1].get("iteration") == 2

    def test_max_iterations_unreviewed_artifact_persistence_error(self):
        """Exception persisting final unreviewed artifact is caught."""
        store = MagicMock()
        # First create_artifact succeeds, second raises
        store.create_artifact.side_effect = [42, Exception("DB error")]
        store.create_review_cycle.return_value = 99
        engine, dev, qas = _make_engine(store=store, max_review=1)
        changes = _changes_json(["issue"])
        qas[0].send.return_value = _ok(changes)
        dev.send.return_value = _ok("revised")

        # Should not raise
        result, iters = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
        )
        assert result == "revised"

    def test_review_cycle_persistence_error(self):
        """Exception during create_review_cycle is caught."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.side_effect = Exception("DB error")
        engine, dev, qas = _make_engine(store=store, max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())
        result, iters = engine._run_review_loop(
            artifact="artifact",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 1

    def test_lgtm_verdict_with_summary_printed(self):
        """LGTM verdict with summary is accepted."""
        engine, dev, qas = _make_engine(max_review=1)
        verdict = json.dumps({
            "verdict": "lgtm",
            "issues": [],
            "summary": "Great work!",
        })
        qas[0].send.return_value = _ok(verdict)
        result, iters = engine._run_review_loop(
            artifact="good artifact",
            artifact_type="code",
            context="ctx",
        )
        assert iters == 1
        assert result == "good artifact"

    def test_no_work_item_id_skips_persistence(self):
        """When work_item_id is None, no artifact/review persistence."""
        store = MagicMock()
        engine, dev, qas = _make_engine(store=store, work_item_id=None, max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())
        result, iters = engine._run_review_loop(
            artifact="artifact",
            artifact_type="design",
            context="ctx",
        )
        store.create_artifact.assert_not_called()
        store.create_review_cycle.assert_not_called()

    def test_custom_proposer_and_reviewers(self):
        """Test with swapped proposer/reviewers (like test_plan review)."""
        engine, dev, qas = _make_engine(max_review=1)
        # Use QA as proposer, dev as reviewer
        dev.send.return_value = _ok(_lgtm_json())
        result, iters = engine._run_review_loop(
            artifact="test plan",
            artifact_type="test_plan",
            context="ctx",
            proposer=qas[0],
            reviewers=[dev],
        )
        assert iters == 1
        # dev was used as reviewer
        dev.send.assert_called_once()


# ===================================================================
# 10. _build_address_prompt  (lines 861, 863)
# ===================================================================

class TestBuildAddressPrompt:

    def test_plan_type(self):
        engine, _, _ = _make_engine()
        result = engine._build_address_prompt("plan", "the plan", "fix things")
        assert isinstance(result, str)
        assert "fix things" in result

    def test_test_plan_type(self):
        engine, _, _ = _make_engine()
        result = engine._build_address_prompt("test_plan", "the test plan", "add cases")
        assert isinstance(result, str)
        assert "add cases" in result

    def test_design_type(self):
        engine, _, _ = _make_engine()
        result = engine._build_address_prompt("design", "the design", "reconsider")
        assert isinstance(result, str)
        assert "reconsider" in result

    def test_code_type(self):
        engine, _, _ = _make_engine()
        result = engine._build_address_prompt("code", "the code", "refactor")
        assert isinstance(result, str)
        assert "refactor" in result

    def test_unknown_falls_through_to_code(self):
        engine, _, _ = _make_engine()
        result = engine._build_address_prompt("random_type", "artifact", "feedback")
        assert isinstance(result, str)
        assert "feedback" in result


# ===================================================================
# 11. _user_checkpoint  (lines 883-902)
# ===================================================================

class TestUserCheckpoint:

    def test_approve_with_empty_string(self):
        engine, _, _ = _make_engine(ask_return="")
        result = engine._user_checkpoint("artifact", "design", "Approve?")
        assert result == "artifact"

    def test_approve_with_y(self):
        engine, _, _ = _make_engine(ask_return="y")
        result = engine._user_checkpoint("artifact", "design", "Approve?")
        assert result == "artifact"

    def test_approve_with_yes(self):
        engine, _, _ = _make_engine(ask_return="yes")
        result = engine._user_checkpoint("artifact", "design", "Approve?")
        assert result == "artifact"

    def test_reject_with_n(self):
        engine, _, _ = _make_engine(ask_return="n")
        result = engine._user_checkpoint("artifact", "design", "Approve?")
        assert result is None

    def test_reject_with_no(self):
        engine, _, _ = _make_engine(ask_return="no")
        result = engine._user_checkpoint("artifact", "design", "Approve?")
        assert result is None

    def test_feedback_then_approve(self):
        """User gives feedback, agent addresses it, then user approves."""
        ask_mock = MagicMock(side_effect=["make it better", "y"])
        engine, dev, _ = _make_engine(ask_return=ask_mock)
        dev.send.return_value = _ok("improved artifact")
        result = engine._user_checkpoint("original", "design", "Approve?")
        assert result == "improved artifact"
        assert dev.send.call_count == 1

    def test_feedback_agent_fails(self):
        """User gives feedback, agent fails to address, loop continues, user approves."""
        ask_mock = MagicMock(side_effect=["add tests", "y"])
        engine, dev, _ = _make_engine(ask_return=ask_mock)
        dev.send.return_value = _fail()
        result = engine._user_checkpoint("original", "design", "Approve?")
        # After failure, artifact is unchanged; next loop user approves
        assert result == "original"

    def test_custom_proposer(self):
        """Feedback goes to custom proposer (e.g., QA for test plan)."""
        ask_mock = MagicMock(side_effect=["feedback text", "y"])
        engine, dev, qas = _make_engine(ask_return=ask_mock)
        qas[0].send.return_value = _ok("qa revised")
        result = engine._user_checkpoint(
            "original", "test_plan", "Approve?", proposer=qas[0],
        )
        assert result == "qa revised"
        qas[0].send.assert_called_once()
        dev.send.assert_not_called()

    def test_multiple_feedback_rounds(self):
        """Multiple rounds of feedback before approval."""
        ask_mock = MagicMock(side_effect=["fix A", "fix B", "y"])
        engine, dev, _ = _make_engine(ask_return=ask_mock)
        dev.send.side_effect = [_ok("v2"), _ok("v3")]
        result = engine._user_checkpoint("v1", "code", "Approve?")
        assert result == "v3"
        assert dev.send.call_count == 2


# ===================================================================
# 12. _parse_phases — additional edge cases  (lines 926, 935, 941)
# ===================================================================

class TestParsePhasesExtended:

    def test_phase_without_separator(self):
        """Phase with number but no colon/dot separator triggers else branch."""
        engine, _, _ = _make_engine()
        plan = "Phase 1 Setup\nPhase 2 Build"
        result = engine._parse_phases(plan)
        assert len(result) == 2

    def test_markdown_header_without_colon(self):
        """## Phase N without colon -> strips # and uses whole text."""
        engine, _, _ = _make_engine()
        plan = "## Phase 1 Setup Everything"
        result = engine._parse_phases(plan)
        assert len(result) >= 1

    def test_blank_lines_skipped(self):
        """Blank lines in plan are properly skipped."""
        engine, _, _ = _make_engine()
        plan = "\n\nPhase 1: Do stuff\n\n\nPhase 2: More stuff\n\n"
        result = engine._parse_phases(plan)
        assert len(result) == 2

    def test_numbered_list_with_dashes(self):
        """Numbered items with dash separator."""
        engine, _, _ = _make_engine()
        plan = "1- Setup the environment\n2- Write the code\n3- Test everything"
        result = engine._parse_phases(plan)
        assert len(result) == 3

    def test_numbered_list_with_parens(self):
        """Numbered items with parenthesis separator."""
        engine, _, _ = _make_engine()
        plan = "1) Configure database\n2) Build API layer\n3) Create frontend"
        result = engine._parse_phases(plan)
        assert len(result) == 3

    def test_numbered_list_short_items_skipped(self):
        """Numbered items with very short names (<=3 chars) are skipped."""
        engine, _, _ = _make_engine()
        plan = "1. ab\n2. cd\n3. This is a real phase name"
        result = engine._parse_phases(plan)
        # only the last item with len > 3 should be picked
        assert len(result) == 1
        assert "This is a real phase name" in result[0]

    def test_step_format_with_period(self):
        """Step N. format parsing."""
        engine, _, _ = _make_engine()
        plan = "Step 1. Initialize\nStep 2. Implement\nStep 3. Deploy"
        result = engine._parse_phases(plan)
        assert len(result) == 3

    def test_step_format_with_paren(self):
        """Step N) format parsing."""
        engine, _, _ = _make_engine()
        plan = "Step 1) First step\nStep 2) Second step"
        result = engine._parse_phases(plan)
        assert len(result) == 2

    def test_triple_hash_header_phase(self):
        """### Phase N: name format."""
        engine, _, _ = _make_engine()
        plan = "### Phase 1: Setup\n### Phase 2: Build"
        result = engine._parse_phases(plan)
        assert "Setup" in result
        assert "Build" in result

    def test_fallback_single_digit_list(self):
        """No Phase/Step keywords, fall back to numbered list."""
        engine, _, _ = _make_engine()
        plan = "Here is the plan:\n1. Design the system\n2. Build the core\n3. Deploy to prod"
        result = engine._parse_phases(plan)
        assert len(result) == 3

    def test_phase_prefix_with_empty_rest(self):
        """Phase N: with nothing after colon -> uses the number part."""
        engine, _, _ = _make_engine()
        plan = "Phase 1:"
        result = engine._parse_phases(plan)
        assert len(result) >= 1


# ===================================================================
# 13. _generate_report  (lines 960-1036)
# ===================================================================

class TestGenerateReport:

    def test_basic_report_structure(self):
        """Report includes all expected sections."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("Developer's narrative report")
        result = WorkflowResult(
            requirement="build a widget",
            design="design doc",
            design_review_iterations=2,
            plan="plan doc",
            plan_review_iterations=1,
            phases=[
                PhaseResult(name="Setup", summary="set up infra", review_iterations=1, committed=True),
                PhaseResult(name="Build", summary="built core", review_iterations=2, committed=True),
            ],
            ga_test=GATestResult(passed=True, test_plan_review_iterations=1),
        )
        report = engine._generate_report(result, ["Phase 1: set up infra", "Phase 2: built core"])
        assert "# Workflow Report" in report
        assert "build a widget" in report
        assert "2 review iteration" in report
        assert "Setup" in report
        assert "Build" in report
        assert "**PASSED**" in report
        assert "Developer's narrative report" in report

    def test_report_with_aborted_ga(self):
        """Report includes aborted GA test details."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("report")
        result = WorkflowResult(
            requirement="req",
            design_review_iterations=1,
            plan_review_iterations=1,
            phases=[PhaseResult(name="p", summary="s", review_iterations=1, committed=True)],
            ga_test=GATestResult(
                aborted=True,
                bugs_found=7,
                bugs_fixed=3,
                bug_fixes=[
                    BugFixResult(title="crash", fixed=True, summary="fixed crash", arch_change=False),
                    BugFixResult(title="perf", fixed=False, summary="", arch_change=True),
                ],
                test_report="detailed test report",
            ),
        )
        report = engine._generate_report(result, ["Phase 1: s"])
        assert "**ABORTED**" in report
        assert "7 bugs" in report
        assert "crash" in report
        assert "simple fix" in report
        assert "arch change" in report
        assert "Not fixed" in report
        assert "detailed test report" in report

    def test_report_with_incomplete_ga(self):
        """GA not aborted but not passed -> INCOMPLETE."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("report")
        result = WorkflowResult(
            requirement="req",
            phases=[],
            ga_test=GATestResult(passed=False, aborted=False, bugs_found=2, bugs_fixed=1),
        )
        report = engine._generate_report(result, [])
        assert "**INCOMPLETE**" in report

    def test_report_no_ga_test(self):
        """Report without GA test."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("report")
        result = WorkflowResult(requirement="req", phases=[], ga_test=None)
        report = engine._generate_report(result, [])
        assert "Skipped" in report

    def test_report_dev_fails(self):
        """Dev fails to generate narrative; report still built."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _fail()
        result = WorkflowResult(
            requirement="req",
            phases=[PhaseResult(name="p", summary="s", review_iterations=1, committed=False)],
            ga_test=GATestResult(passed=True),
        )
        report = engine._generate_report(result, ["Phase 1: s"])
        assert "# Workflow Report" in report
        assert "Not committed" in report
        assert "Developer's Final Summary" not in report

    def test_report_no_completed_summaries(self):
        """Empty completed_summaries means dev is not asked for narrative."""
        engine, dev, _ = _make_engine()
        result = WorkflowResult(requirement="req", phases=[], ga_test=None)
        report = engine._generate_report(result, [])
        dev.send.assert_not_called()
        assert "# Workflow Report" in report

    def test_report_totals_calculation(self):
        """Verify total review cycles are calculated correctly."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("narrative")
        result = WorkflowResult(
            requirement="req",
            design_review_iterations=3,
            plan_review_iterations=2,
            phases=[
                PhaseResult(name="p1", summary="s", review_iterations=4, committed=True),
                PhaseResult(name="p2", summary="s", review_iterations=1, committed=True),
            ],
            ga_test=GATestResult(
                passed=True,
                test_plan_review_iterations=2,
            ),
        )
        report = engine._generate_report(result, ["p1", "p2"])
        # Total: 3+2+4+1+2 = 12
        assert "Total review cycles: 12" in report

    def test_report_committed_count(self):
        """Verify committed phase count is correct."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("narrative")
        result = WorkflowResult(
            requirement="req",
            phases=[
                PhaseResult(name="p1", summary="s", review_iterations=1, committed=True),
                PhaseResult(name="p2", summary="s", review_iterations=1, committed=False),
                PhaseResult(name="p3", summary="s", review_iterations=1, committed=True),
            ],
            ga_test=GATestResult(passed=False),
        )
        report = engine._generate_report(result, ["p1", "p2", "p3"])
        assert "3 (2 committed)" in report

    def test_report_bugs_found_line(self):
        """Bugs found/fixed line appears when ga_test is present."""
        engine, dev, _ = _make_engine()
        dev.send.return_value = _ok("narrative")
        result = WorkflowResult(
            requirement="req",
            phases=[],
            ga_test=GATestResult(bugs_found=5, bugs_fixed=3, passed=False),
        )
        report = engine._generate_report(result, [])
        assert "Bugs found: 5, fixed: 3" in report


# ===================================================================
# 14. Data classes
# ===================================================================

class TestDataClasses:

    def test_phase_result_defaults(self):
        pr = PhaseResult(name="p", summary="s", review_iterations=1)
        assert pr.committed is False

    def test_bug_fix_result_defaults(self):
        bf = BugFixResult(title="bug")
        assert bf.arch_change is False
        assert bf.fixed is False
        assert bf.summary == ""

    def test_ga_test_result_defaults(self):
        ga = GATestResult()
        assert ga.test_plan == ""
        assert ga.bugs_found == 0
        assert ga.bug_fixes == []
        assert ga.passed is False
        assert ga.aborted is False

    def test_workflow_result_defaults(self):
        wr = WorkflowResult(requirement="req")
        assert wr.design == ""
        assert wr.plan == ""
        assert wr.phases == []
        assert wr.ga_test is None
        assert wr.final_report == ""
        assert wr.success is False


# ===================================================================
# 15. Additional coverage for remaining lines (849, 926, 941)
# ===================================================================

class TestBuildReviewPromptPlanBranch:
    """Cover line 849: _build_review_prompt with artifact_type='plan'."""

    def test_build_review_prompt_plan(self):
        engine, _, _ = _make_engine()
        result = engine._build_review_prompt(
            artifact_type="plan",
            context="Build a REST API",
            artifact="Plan document content",
            iteration=2,
        )
        assert isinstance(result, str)
        assert "Plan document content" in result


class TestParsePhasesNoSeparator:
    """Cover line 926: Phase prefix where rest has no separator chars at all."""

    def test_phase_single_char_no_separator(self):
        """'Phase X' where X has no : . ) or space after position 0."""
        engine, _, _ = _make_engine()
        # "Phase A" -> rest is "A", no separator found, hits the else on line 926
        plan = "Phase A"
        result = engine._parse_phases(plan)
        assert len(result) >= 1
        assert result[0] == "A"

    def test_step_single_char_no_separator(self):
        """'Step X' where X is a single char."""
        engine, _, _ = _make_engine()
        plan = "Step X"
        result = engine._parse_phases(plan)
        assert len(result) >= 1
        assert result[0] == "X"


class TestParsePhasesBlankLinesInFallback:
    """Cover line 941: blank lines in the fallback numbered-list parser."""

    def test_numbered_list_with_blank_lines(self):
        """Numbered list with blank lines in between — no Phase/Step keywords."""
        engine, _, _ = _make_engine()
        plan = "1. Design the system\n\n2. Build the core\n\n3. Deploy to production"
        result = engine._parse_phases(plan)
        assert len(result) == 3


class TestReviewLoopPlanReviewPrompt:
    """Exercise _run_review_loop with artifact_type='plan' to hit line 849."""

    def test_review_loop_with_plan_artifact_type(self):
        engine, dev, qas = _make_engine(max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())
        result, iters = engine._run_review_loop(
            artifact="plan content",
            artifact_type="plan",
            context="ctx",
        )
        assert iters == 1
        # Verify the QA reviewer was called with a prompt containing the plan
        call_args = qas[0].send.call_args[0][0]
        assert "plan content" in call_args
