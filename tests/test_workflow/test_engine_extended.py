"""Extended tests for WorkflowEngine — covers uncovered lines."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from myswat.agents.base import AgentResponse
from myswat.models.work_item import ReviewVerdict
from myswat.workflow.engine import (
    WorkMode,
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
    mode: WorkMode = WorkMode.full,
    resume_stage: str | None = None,
    arch_sm=None,
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
        mode=mode,
        ask_user=ask,
        resume_stage=resume_stage,
        arch_sm=arch_sm,
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
            ("reviewed design", 1, True),  # design review
            ("reviewed plan", 1, True),    # plan review
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
        assert result.design_review_passed is True
        assert result.plan_review_passed is True

    @patch.object(WorkflowEngine, "_run_design", return_value="")
    def test_design_fails_aborts(self, m_design):
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.design == ""
        assert result.success is False

    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("design", 1, True))
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
        m_review.return_value = ("design", 1, True)
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
        m_review.side_effect = [("design", 1, True), ("plan", 1, True)]
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
        m_review.side_effect = [("design", 1, True), ("plan", 1, True)]
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
        m_review.side_effect = [("design", 1, True), ("plan", 1, True)]
        m_cp.side_effect = ["design", "plan"]
        m_phase.return_value = PhaseResult(
            name="p1", summary="s", review_iterations=1, committed=False,
        )
        m_ga.return_value = GATestResult(passed=True)
        engine, _, _ = _make_engine()
        result = engine.run("req")
        assert result.success is False


class TestRunDesignMode:
    """Tests for design-mode orchestration added in phase 3."""

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Design Report")
    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="the plan")
    @patch.object(WorkflowEngine, "_run_design", return_value="the design")
    def test_happy_path(
        self, m_design, m_plan, m_review, m_checkpoint, m_report,
    ):
        m_review.side_effect = [
            ("reviewed design", 1, True),
            ("reviewed plan", 2, True),
        ]
        m_checkpoint.side_effect = ["approved design", "approved plan"]
        engine, _, _ = _make_engine(mode=WorkMode.design)

        result = engine.run("draft api")

        assert result.design == "approved design"
        assert result.plan == "approved plan"
        assert result.design_review_passed is True
        assert result.plan_review_passed is True
        assert result.ga_test is None
        assert result.phases == []
        assert result.final_report == "# Design Report"
        assert result.success is True
        assert m_checkpoint.call_count == 2

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Design Report")
    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="the plan")
    @patch.object(WorkflowEngine, "_run_design", return_value="the design")
    def test_review_failure_keeps_success_false(
        self, m_design, m_plan, m_review, m_checkpoint, m_report,
    ):
        m_review.side_effect = [
            ("reviewed design", 1, True),
            ("reviewed plan", 2, False),
        ]
        m_checkpoint.side_effect = ["approved design", "approved plan"]
        engine, _, _ = _make_engine(mode=WorkMode.design)

        result = engine.run("draft api")

        assert result.design_review_passed is True
        assert result.plan_review_passed is False
        assert result.success is False
        assert result.final_report == "# Design Report"

    @patch.object(WorkflowEngine, "_run_planning")
    @patch.object(WorkflowEngine, "_user_checkpoint", return_value=None)
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed design", 1, True))
    @patch.object(WorkflowEngine, "_run_design", return_value="the design")
    def test_user_stop_after_design_sets_final_report(
        self, m_design, m_review, m_checkpoint, m_planning,
    ):
        engine, _, _ = _make_engine(mode=WorkMode.design)

        result = engine.run("draft api")

        assert result.success is False
        assert result.final_report == "Design workflow stopped by user after design review."
        m_planning.assert_not_called()

    def test_cancellation_after_design_review_sets_final_report(self):
        engine, _, _ = _make_engine(mode=WorkMode.design)
        result = WorkflowResult(requirement="req")

        with patch.object(engine, "_run_design", return_value="the design"):
            with patch.object(engine, "_run_review_loop", return_value=("reviewed design", 1, True)):
                with patch.object(engine, "_cancelled", side_effect=[False, True]):
                    result = engine._run_design_mode("req", result)

        assert result.success is False
        assert result.final_report == "Design workflow cancelled during design review."



class TestRunArchitectDesignMode:
    @patch.object(WorkflowEngine, "_generate_report", return_value="# Architect Report")
    @patch.object(WorkflowEngine, "_user_checkpoint", return_value="approved design")
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed design", 2, True))
    def test_happy_path(self, m_review, m_checkpoint, m_report):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=["the design"], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.architect_design,
        )

        result = engine.run("design a cache")

        assert result.success is True
        assert result.design == "approved design"
        assert result.design_review_iterations == 2
        assert result.design_review_passed is True
        assert result.final_report == "# Architect Report"
        m_review.assert_called_once()
        assert m_review.call_args.kwargs["proposer"] is arch
        assert m_review.call_args.kwargs["reviewers"] == [dev, qa]
        assert m_review.call_args.kwargs["artifact_type"] == "arch_design"
        assert m_review.call_args.kwargs["abort_on_agent_failure"] is True
        m_checkpoint.assert_called_once()
        assert m_checkpoint.call_args.kwargs["proposer"] is arch
        draft_events = [
            call.kwargs for call in store.append_work_item_process_event.call_args_list
            if call.kwargs.get("event_type") == "design_draft"
        ]
        assert draft_events
        assert draft_events[-1]["from_role"] == "architect"
        assert draft_events[-1]["to_role"] == "developer"

    def test_draft_failure_marks_blocked(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[_fail()], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.architect_design,
        )

        result = engine.run("design a cache")

        assert result.success is False
        assert result.blocked is True
        assert "design draft failed" in result.failure_summary
        assert result.final_report

    def test_direct_architect_mode_draft_failure_syncs_result_state(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[_fail()], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.architect_design,
        )
        result = WorkflowResult(requirement="design a cache")

        result = engine._run_architect_design_mode("design a cache", result)

        assert result.success is False
        assert result.blocked is True
        assert "design draft failed" in result.failure_summary
        assert result.final_report

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Architect Report")
    @patch.object(WorkflowEngine, "_user_checkpoint", return_value=None)
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed design", 1, True))
    def test_user_stop_after_review(self, m_review, m_checkpoint, m_report):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=["the design"], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.architect_design,
        )

        result = engine.run("design a cache")

        assert result.success is False
        assert result.final_report == "Architect-design workflow stopped by user after design review."

    def test_cancellation_after_review_sets_final_report(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.architect_design,
        )
        result = WorkflowResult(requirement="req")

        with patch.object(arch, "send", return_value=_ok("the design")):
            with patch.object(engine, "_run_review_loop", return_value=("reviewed design", 1, True)):
                with patch.object(engine, "_cancelled", side_effect=[False, True]):
                    result = engine._run_architect_design_mode("req", result)

        assert result.success is False
        assert result.final_report == "Architect-design workflow cancelled during design review."


class TestRunFullArchitectLed:
    """Tests for _run_full when arch_sm is provided (architect-led design stages)."""

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Full Report")
    @patch.object(WorkflowEngine, "_run_ga_test_phase")
    @patch.object(WorkflowEngine, "_run_phase")
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["phase-A"])
    @patch.object(WorkflowEngine, "_user_checkpoint")
    @patch.object(WorkflowEngine, "_run_review_loop")
    @patch.object(WorkflowEngine, "_run_planning", return_value="the plan")
    def test_happy_path_uses_architect_for_design(
        self, m_plan, m_review, m_checkpoint, m_phases,
        m_phase, m_ga, m_report,
    ):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=["the design"], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        engine = WorkflowEngine(
            store=store, dev_sm=dev, qa_sms=[qa], arch_sm=arch,
            project_id=1, work_item_id=1, mode=WorkMode.full,
            ask_user=MagicMock(return_value="y"),
        )
        m_review.side_effect = [
            ("reviewed design", 2, True),  # design review
            ("reviewed plan", 1, True),    # plan review
        ]
        m_checkpoint.side_effect = ["approved design", "approved plan"]
        m_phase.return_value = PhaseResult(
            name="phase-A", summary="done", review_iterations=1, committed=True,
        )
        m_ga.return_value = GATestResult(passed=True)

        result = engine.run("build a widget")

        assert result.success is True
        assert result.design == "approved design"
        # Architect should have been called for the design draft
        arch.send.assert_called_once()
        # Design review should use arch as proposer, dev+qa as reviewers
        design_review_call = m_review.call_args_list[0]
        assert design_review_call.kwargs["proposer"] is arch
        assert design_review_call.kwargs["reviewers"] == [dev, qa]
        assert design_review_call.kwargs["artifact_type"] == "arch_design"
        # Plan review should use default (no proposer/reviewers override)
        plan_review_call = m_review.call_args_list[1]
        assert "proposer" not in plan_review_call.kwargs
        # User checkpoint should pass proposer=arch for design
        design_cp_call = m_checkpoint.call_args_list[0]
        assert design_cp_call.kwargs.get("proposer") is arch
        assert design_cp_call.args[1] == "arch_design"

    def test_architect_design_draft_failure_blocks(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[_fail()], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store, dev_sm=dev, qa_sms=[qa], arch_sm=arch,
            project_id=1, work_item_id=1, mode=WorkMode.full,
        )

        result = engine.run("build a widget")

        assert result.success is False
        assert result.blocked is True
        assert "design draft failed" in result.failure_summary

    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed design", 1, False))
    def test_design_review_failure_persists_state_and_report_says_not_approved(self, m_review):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=["the design"], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store, dev_sm=dev, qa_sms=[qa], arch_sm=arch,
            project_id=1, work_item_id=1, mode=WorkMode.full,
        )

        result = engine.run("build a widget")

        assert result.success is False
        assert result.design_review_passed is False
        # Report must not claim approval
        assert "Not approved" in result.final_report
        assert "Approved after" not in result.final_report
        # Should persist design_review_failed state
        state_calls = [
            c for c in store.update_work_item_state.call_args_list
            if c.kwargs.get("current_stage") == "design_review_failed"
        ]
        assert state_calls

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Report")
    @patch.object(WorkflowEngine, "_user_checkpoint", return_value=None)
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed design", 1, True))
    def test_user_rejects_design_persists_rejection(self, m_review, m_checkpoint, m_report):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=["the design"], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store, dev_sm=dev, qa_sms=[qa], arch_sm=arch,
            project_id=1, work_item_id=1, mode=WorkMode.full,
        )

        result = engine.run("build a widget")

        assert result.success is False
        assert result.final_report == "Workflow stopped by user after design review."
        # Should persist design_rejected_by_user state
        state_calls = [
            c for c in store.update_work_item_state.call_args_list
            if c.kwargs.get("current_stage") == "design_rejected_by_user"
        ]
        assert state_calls

    @patch.object(WorkflowEngine, "_user_checkpoint", return_value=None)
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed design", 1, True))
    @patch.object(WorkflowEngine, "_run_design", return_value="the design")
    def test_user_rejects_design_without_arch_also_persists_rejection(
        self, m_design, m_review, m_checkpoint,
    ):
        """Dev-led full mode (no arch_sm) should also persist rejection state."""
        engine, _, _ = _make_engine()

        result = engine.run("build a widget")

        assert result.success is False
        assert result.final_report == "Workflow stopped by user after design review."
        state_calls = [
            c for c in engine._store.update_work_item_state.call_args_list
            if c.kwargs.get("current_stage") == "design_rejected_by_user"
        ]
        assert state_calls


class TestRunTestplanDesignMode:
    @patch.object(WorkflowEngine, "_generate_report", return_value="# Testplan Report")
    @patch.object(WorkflowEngine, "_user_checkpoint", return_value="approved plan")
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed plan", 2, True))
    def test_happy_path(self, m_review, m_checkpoint, m_report):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=["the plan"], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )

        result = engine.run("test the cache")

        assert result.success is True
        assert result.ga_test is not None
        assert result.ga_test.test_plan == "approved plan"
        assert result.ga_test.test_plan_review_iterations == 2
        assert result.ga_test.test_plan_review_passed is True
        assert result.final_report == "# Testplan Report"
        m_review.assert_called_once()
        assert m_review.call_args.kwargs["proposer"] is qa
        assert m_review.call_args.kwargs["reviewers"] == [arch, dev]
        assert m_review.call_args.kwargs["artifact_type"] == "test_plan"
        assert m_review.call_args.kwargs["abort_on_agent_failure"] is True
        m_checkpoint.assert_called_once()
        assert m_checkpoint.call_args.kwargs["proposer"] is qa
        draft_events = [
            call.kwargs for call in store.append_work_item_process_event.call_args_list
            if call.kwargs.get("event_type") == "testplan_draft"
        ]
        assert draft_events
        assert draft_events[-1]["from_role"] == "qa_main"
        assert draft_events[-1]["to_role"] == "architect"

    def test_draft_event_targets_architect_first(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=["the plan"], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )

        with patch.object(engine, "_run_review_loop", return_value=("reviewed plan", 1, True)):
            with patch.object(engine, "_user_checkpoint", return_value="approved plan"):
                with patch.object(engine, "_generate_report", return_value="# Testplan Report"):
                    result = engine.run("test the cache")

        assert result.success is True
        draft_events = [
            call.kwargs for call in store.append_work_item_process_event.call_args_list
            if call.kwargs.get("event_type") == "testplan_draft"
        ]
        assert draft_events
        assert draft_events[-1]["to_role"] == "architect"

    def test_draft_failure_marks_blocked(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=[_fail()], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )

        result = engine.run("test the cache")

        assert result.success is False
        assert result.blocked is True
        assert "test plan draft failed" in result.failure_summary
        assert result.final_report

    def test_direct_testplan_mode_draft_failure_syncs_result_state(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=[_fail()], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )
        result = WorkflowResult(requirement="test the cache")

        result = engine._run_testplan_design_mode("test the cache", result)

        assert result.success is False
        assert result.blocked is True
        assert "test plan draft failed" in result.failure_summary
        assert result.final_report

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Testplan Report")
    @patch.object(WorkflowEngine, "_user_checkpoint", return_value=None)
    @patch.object(WorkflowEngine, "_run_review_loop", return_value=("reviewed plan", 1, True))
    def test_user_stop_after_review(self, m_review, m_checkpoint, m_report):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=["the plan"], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )

        result = engine.run("test the cache")

        assert result.success is False
        assert result.final_report == "Testplan-design workflow stopped by user after test plan review."

    def test_cancellation_after_review_sets_final_report(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        dev = make_fake_session_manager(agent_id=10, agent_role="developer", responses=[], session_id=100)
        qa = make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=[], session_id=101)
        store = MagicMock()
        engine = WorkflowEngine(
            store=store,
            dev_sm=dev,
            qa_sms=[qa],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )
        result = WorkflowResult(requirement="req")

        with patch.object(qa, "send", return_value=_ok("the plan")):
            with patch.object(engine, "_run_review_loop", return_value=("reviewed plan", 1, True)):
                with patch.object(engine, "_cancelled", side_effect=[False, True]):
                    result = engine._run_testplan_design_mode("req", result)

        assert result.success is False
        assert result.final_report == "Testplan-design workflow cancelled during test plan review."


class TestRunDevelopMode:
    """Tests for develop-mode orchestration added in phase 4."""

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Development Report")
    @patch.object(WorkflowEngine, "_run_phase")
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["phase-A"])
    def test_happy_path(self, m_phases, m_run_phase, m_report):
        m_run_phase.return_value = PhaseResult(
            name="phase-A", summary="done", review_iterations=1, review_passed=True, committed=True,
        )
        engine, _, _ = _make_engine(mode=WorkMode.develop)

        result = engine.run("implement feature")

        assert result.success is True
        assert result.final_report == "# Development Report"
        assert len(result.phases) == 1
        assert result.phases[0].committed is True
        kwargs = m_run_phase.call_args.kwargs
        assert kwargs["design"] == "implement feature"
        assert kwargs["plan"] == "implement feature"

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Development Report")
    @patch.object(WorkflowEngine, "_run_phase")
    def test_phase_parsing_fallback_from_requirement(self, m_run_phase, m_report):
        m_run_phase.return_value = PhaseResult(
            name="Full implementation", summary="done", review_iterations=1, review_passed=True, committed=True,
        )
        engine, _, _ = _make_engine(mode=WorkMode.develop)

        result = engine.run("Just do everything at once, no structure here.")

        assert result.success is True
        assert m_run_phase.call_args.kwargs["phase_name"] == "Full implementation"

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Development Report")
    @patch.object(WorkflowEngine, "_run_phase")
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["phase-A"])
    def test_uncommitted_phase_keeps_success_false(self, m_phases, m_run_phase, m_report):
        m_run_phase.return_value = PhaseResult(
            name="phase-A", summary="failed commit", review_iterations=1, review_passed=True, committed=False,
        )
        engine, _, _ = _make_engine(mode=WorkMode.develop)

        result = engine.run("implement feature")

        assert result.success is False
        assert result.final_report == "# Development Report"


class TestRunTestMode:
    """Tests for test-mode orchestration added in phase 4."""

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Test Report")
    @patch.object(WorkflowEngine, "_run_ga_test_phase")
    def test_happy_path(self, m_ga, m_report):
        m_ga.return_value = GATestResult(passed=True)
        engine, _, _ = _make_engine(mode=WorkMode.test)

        result = engine.run("validate release")

        assert result.success is True
        assert result.final_report == "# Test Report"
        assert result.ga_test.passed is True
        assert m_ga.call_args.kwargs["allow_arch_fix"] is False
        assert m_ga.call_args.args == ("validate release", "validate release", "", "validate release")

    @patch.object(WorkflowEngine, "_generate_report", return_value="# Test Report")
    @patch.object(WorkflowEngine, "_run_ga_test_phase")
    def test_failed_ga_keeps_success_false(self, m_ga, m_report):
        m_ga.return_value = GATestResult(passed=False)
        engine, _, _ = _make_engine(mode=WorkMode.test)

        result = engine.run("validate release")

        assert result.success is False
        assert result.final_report == "# Test Report"



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

    def test_failure_with_abort_records_blocked_state(self):
        engine, dev, _ = _make_engine(dev_responses=[_fail()])
        result = engine._run_design("req", abort_on_failure=True)
        assert result == ""
        assert engine._blocked is True
        assert "design draft failed" in engine._failure_summary
        engine._store.update_work_item_state.assert_called()
        assert any(
            call.kwargs.get("event_type") == "proposal_failure"
            for call in engine._store.append_work_item_process_event.call_args_list
        )


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

    def test_failure_with_abort_records_blocked_state(self):
        engine, dev, _ = _make_engine(dev_responses=[_fail()])
        result = engine._run_planning("design", "req", abort_on_failure=True)
        assert result == ""
        assert engine._blocked is True
        assert "implementation plan failed" in engine._failure_summary
        engine._store.update_work_item_state.assert_called()
        assert any(
            call.kwargs.get("event_type") == "proposal_failure"
            for call in engine._store.append_work_item_process_event.call_args_list
        )


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
        with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1, True)):
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
        assert result.review_passed is True
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
        with patch.object(engine, "_run_review_loop", return_value=("impl content", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1, True)):
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
        assert result.review_passed is True


    def test_review_not_passed_is_recorded_but_commit_still_attempted(self):
        """Phase review remains informational in phase 2."""
        engine, dev, _ = _make_engine(
            dev_responses=[
                "impl done",
                "summary",
                "committed",
            ],
            qa_responses=[_changes_json(["follow-up"])],
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1, False)):
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
        assert result.review_passed is False



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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.aborted is True
        assert result.test_plan_review_passed is True

    def test_qa_fails_test_execution(self):
        """When QA fails to execute tests, return early."""
        engine, dev, qas = _make_engine(
            qa_responses=["test plan", _fail()],  # plan ok, exec fails
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.passed is True
        assert result.test_plan_review_passed is True
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
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
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
            with patch.object(engine, "_run_bug_fix", return_value=BugFixResult(
                title="b1", fixed=True, summary="f",
            )):
                result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.test_report == ""


    def test_test_plan_review_flag_can_be_false_while_ga_passes(self):
        """Review outcome is recorded even when the later GA run passes."""
        pass_output = json.dumps({"status": "pass"})
        engine, dev, qas = _make_engine(
            qa_responses=["test plan", pass_output],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, False)):
            result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        assert result.test_plan_review_passed is False
        assert result.passed is True



    def test_allow_arch_fix_false_skips_arch_change_subworkflow(self):
        """Test-only GA runs record arch-change findings without escalation."""
        bugs_output = json.dumps({
            "bugs": [{"title": "big bug", "severity": "major", "description": "needs redesign"}]
        })
        engine, dev, qas = _make_engine(
            dev_responses=[json.dumps({"assessment": "arch_change"})],
            qa_responses=["test plan", bugs_output, "final test report"],
            ask_return="y",
            max_review=1,
        )
        with patch.object(engine, "_run_review_loop", return_value=("test plan", 1, True)):
            with patch.object(engine, "_run_bug_fix_arch_change") as mock_arch_fix:
                result = engine._run_ga_test_phase(
                    "req", "design", "plan", "summary", allow_arch_fix=False,
                )

        mock_arch_fix.assert_not_called()
        assert result.passed is False
        assert result.aborted is False
        assert result.bug_fixes[0].arch_change is True
        assert result.bug_fixes[0].fixed is False
        assert "follow-up" in result.bug_fixes[0].summary.lower()
        assert result.test_report == "final test report"



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
            with patch.object(engine, "_run_review_loop", return_value=("reviewed", 1, True)):
                with patch.object(engine, "_user_checkpoint", return_value=None):
                    result = engine._run_bug_fix_arch_change(
                        {"title": "bug"}, "req", "design",
                    )
        assert not result.success

    def test_planning_fails(self):
        """Planning fails after design approval."""
        engine, _, _ = _make_engine()
        with patch.object(engine, "_run_design", return_value="design"):
            with patch.object(engine, "_run_review_loop", return_value=("design", 1, True)):
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
                ("design", 1, True), ("plan", 1, True),
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
                ("reviewed design", 1, True), ("reviewed plan", 1, True),
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
        assert result.design_review_passed is True
        assert result.plan_review_passed is True
        assert len(result.phases) == 1
        assert result.phases[0].committed is True
        assert "critical bug" in result.final_report

    def test_phase_not_committed(self):
        """Sub-workflow with uncommitted phase = not successful."""
        engine, _, _ = _make_engine()
        phase_result = PhaseResult(name="p", summary="s", review_iterations=1, committed=False)
        with patch.object(engine, "_run_design", return_value="d"):
            with patch.object(engine, "_run_review_loop", side_effect=[("d", 1, True), ("p", 1, True)]):
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
            with patch.object(engine, "_run_review_loop", side_effect=[("d", 1, True), ("p", 1, True)]):
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

    def test_review_loops_disable_artifact_persistence(self):
        """Bug-fix review loops should not overwrite top-level review artifacts."""
        engine, _, _ = _make_engine()
        phase_result = PhaseResult(name="fix-phase", summary="done", review_iterations=1, committed=True)
        with patch.object(engine, "_run_design", return_value="new design"):
            with patch.object(engine, "_run_review_loop", side_effect=[
                ("reviewed design", 1, True), ("reviewed plan", 1, True),
            ]) as m_review:
                with patch.object(engine, "_user_checkpoint", side_effect=[
                    "approved design", "approved plan",
                ]):
                    with patch.object(engine, "_run_planning", return_value="new plan"):
                        with patch.object(engine, "_parse_phases", return_value=["fix-phase"]):
                            with patch.object(engine, "_run_phase", return_value=phase_result):
                                engine._run_bug_fix_arch_change(
                                    {"title": "critical bug", "description": "desc"},
                                    "req", "design",
                                )

        assert m_review.call_count == 2
        for call in m_review.call_args_list:
            assert call.kwargs["persist_artifacts"] is False

    def test_checkpoints_disable_artifact_persistence(self):
        """Bug-fix checkpoints should not overwrite top-level checkpoint artifacts."""
        engine, _, _ = _make_engine()
        phase_result = PhaseResult(name="fix-phase", summary="done", review_iterations=1, committed=True)
        with patch.object(engine, "_run_design", return_value="new design"):
            with patch.object(engine, "_run_review_loop", side_effect=[
                ("reviewed design", 1, True), ("reviewed plan", 1, True),
            ]):
                with patch.object(engine, "_user_checkpoint", side_effect=[
                    "approved design", "approved plan",
                ]) as m_checkpoint:
                    with patch.object(engine, "_run_planning", return_value="new plan"):
                        with patch.object(engine, "_parse_phases", return_value=["fix-phase"]):
                            with patch.object(engine, "_run_phase", return_value=phase_result):
                                engine._run_bug_fix_arch_change(
                                    {"title": "critical bug", "description": "desc"},
                                    "req", "design",
                                )

        assert m_checkpoint.call_count == 2
        for call in m_checkpoint.call_args_list:
            assert call.kwargs["persist_artifact"] is False


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
        result, iters, passed = engine._run_review_loop(
            artifact="artifact text",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 1
        assert passed is True
        assert result == "artifact text"

    def test_reviewer_failure_continues(self):
        """Reviewer failure doesn't crash the loop; other reviewers proceed."""
        engine, dev, qas = _make_engine(qa_count=2, max_review=1)
        # First QA fails, second QA gives LGTM
        qas[0].send.return_value = _fail()
        qas[1].send.return_value = _ok(_lgtm_json())
        result, iters, passed = engine._run_review_loop(
            artifact="artifact",
            artifact_type="code",
            context="ctx",
        )
        assert iters == 1
        assert passed is False

    def test_reviewer_failure_with_abort_marks_blocked(self):
        engine, dev, qas = _make_engine(qa_count=2, max_review=3)
        qas[0].send.return_value = _fail()
        qas[1].send.return_value = _ok(_lgtm_json())
        result, iters, passed = engine._run_review_loop(
            artifact="artifact",
            artifact_type="code",
            context="ctx",
            abort_on_agent_failure=True,
        )
        assert result == "artifact"
        assert iters == 1
        assert passed is False
        assert engine._blocked is True
        assert engine._failure_summary == "[qa-0] review failed (exit=1)"
        dev.send.assert_not_called()
        # With concurrent reviews, all reviewers are dispatched simultaneously,
        # so qas[1] will have been called even though qas[0] failed.
        qas[1].send.assert_called_once()
        engine._store.update_work_item_state.assert_called()
        # All responses are still processed before aborting: both the failure
        # event and the successful LGTM review_response should be logged.
        event_types = [
            call.kwargs.get("event_type")
            for call in engine._store.append_work_item_process_event.call_args_list
        ]
        assert "review_failure" in event_types
        assert "review_response" in event_types

    def test_empty_reviewers_auto_approves(self):
        """With no reviewers, the loop auto-approves with proper state persistence."""
        engine, dev, _qas = _make_engine(qa_count=0, max_review=3)
        result, iters, passed = engine._run_review_loop(
            artifact="artifact",
            artifact_type="design",
            context="ctx",
        )
        assert result == "artifact"
        assert passed is True
        dev.send.assert_not_called()
        # Verify work item state was persisted as approved.
        engine._store.update_work_item_state.assert_called()
        state_call = engine._store.update_work_item_state.call_args
        assert state_call.kwargs["current_stage"] == "design_approved"
        # Verify a reaction event was logged.
        assert any(
            call.kwargs.get("event_type") == "reaction"
            for call in engine._store.append_work_item_process_event.call_args_list
        )

    def test_second_round_prompt_includes_change_summary(self):
        """On iteration 2+, the review prompt includes what changed since last round."""
        engine, dev, qas = _make_engine(qa_count=1, max_review=3)
        # iter 1: changes_requested with a specific issue
        changes = _changes_json(["Missing error handling"])
        qas[0].send.side_effect = [
            _ok(changes),       # iter 1: request changes
            _ok(_lgtm_json()),  # iter 2: approve
        ]
        dev.send.return_value = _ok("Revised: added error handling for edge cases")
        result, iters, passed = engine._run_review_loop(
            artifact="original design",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 2
        assert passed is True
        # The second call to qas[0].send should contain the change summary.
        second_call_prompt = qas[0].send.call_args_list[1][0][0]
        assert "Changes Since Last Review" in second_call_prompt
        assert "Missing error handling" in second_call_prompt
        assert "Revised: added error handling for edge cases" in second_call_prompt

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
        result, iters, passed = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 2
        assert passed is True

    def test_proposer_fails_to_address_breaks(self):
        """When proposer fails to address comments, loop breaks."""
        engine, dev, qas = _make_engine(max_review=3)
        changes = _changes_json(["issue 1"])
        qas[0].send.return_value = _ok(changes)
        dev.send.return_value = _fail()  # proposer fails to address
        result, iters, passed = engine._run_review_loop(
            artifact="original",
            artifact_type="code",
            context="ctx",
        )
        assert iters == 3  # returns max_review when loop ends early via break
        assert passed is False

    def test_proposer_fails_to_address_with_abort_marks_blocked(self):
        engine, dev, qas = _make_engine(max_review=3)
        changes = _changes_json(["issue 1"])
        qas[0].send.return_value = _ok(changes)
        dev.send.return_value = _fail()
        result, iters, passed = engine._run_review_loop(
            artifact="original",
            artifact_type="code",
            context="ctx",
            abort_on_agent_failure=True,
        )
        assert result == "original"
        assert iters == 1
        assert passed is False
        assert engine._blocked is True
        assert "failed to address code comments" in engine._failure_summary
        engine._store.update_work_item_state.assert_called()
        assert any(
            call.kwargs.get("event_type") == "revision_failure"
            for call in engine._store.append_work_item_process_event.call_args_list
        )

    def test_max_iterations_unreviewed_artifact_persisted(self):
        """At max iterations, the final unreviewed artifact is persisted."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        engine, dev, qas = _make_engine(store=store, max_review=1)
        changes = _changes_json(["issue"])
        qas[0].send.return_value = _ok(changes)
        dev.send.return_value = _ok("revised")

        result, iters, passed = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 1
        assert passed is False
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
        result, iters, passed = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
        )
        assert passed is False
        assert result == "revised"

    def test_persist_artifacts_false_skips_all_persistence(self):
        """Opting out skips both per-iteration and final draft artifact writes."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        engine, dev, qas = _make_engine(store=store, max_review=1)
        qas[0].send.return_value = _ok(_changes_json(["issue"]))
        dev.send.return_value = _ok("revised")

        result, iters, passed = engine._run_review_loop(
            artifact="original",
            artifact_type="design",
            context="ctx",
            persist_artifacts=False,
        )

        assert result == "revised"
        assert iters == 1
        assert passed is False
        store.create_artifact.assert_not_called()
        store.create_review_cycle.assert_not_called()

    def test_review_cycle_persistence_error(self):
        """Exception during create_review_cycle is caught."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.side_effect = Exception("DB error")
        engine, dev, qas = _make_engine(store=store, max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())
        result, iters, passed = engine._run_review_loop(
            artifact="artifact",
            artifact_type="design",
            context="ctx",
        )
        assert iters == 1
        assert passed is True

    def test_lgtm_verdict_with_summary_printed(self):
        """LGTM verdict with summary is accepted."""
        engine, dev, qas = _make_engine(max_review=1)
        verdict = json.dumps({
            "verdict": "lgtm",
            "issues": [],
            "summary": "Great work!",
        })
        qas[0].send.return_value = _ok(verdict)
        result, iters, passed = engine._run_review_loop(
            artifact="good artifact",
            artifact_type="code",
            context="ctx",
        )
        assert iters == 1
        assert passed is True
        assert result == "good artifact"

    def test_no_work_item_id_skips_persistence(self):
        """When work_item_id is None, no artifact/review persistence."""
        store = MagicMock()
        engine, dev, qas = _make_engine(store=store, work_item_id=None, max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())
        result, iters, passed = engine._run_review_loop(
            artifact="artifact",
            artifact_type="design",
            context="ctx",
        )
        assert passed is True
        store.create_artifact.assert_not_called()
        store.create_review_cycle.assert_not_called()

    def test_custom_proposer_and_reviewers(self):
        """Test with swapped proposer/reviewers (like test_plan review)."""
        engine, dev, qas = _make_engine(max_review=1)
        # Use QA as proposer, dev as reviewer
        dev.send.return_value = _ok(_lgtm_json())
        result, iters, passed = engine._run_review_loop(
            artifact="test plan",
            artifact_type="test_plan",
            context="ctx",
            proposer=qas[0],
            reviewers=[dev],
        )
        assert iters == 1
        assert passed is True
        # dev was used as reviewer
        dev.send.assert_called_once()

    def test_review_loop_passes_reviewer_to_prompt_builder(self):
        engine, dev, qas = _make_engine(qa_count=1, max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())
        with patch.object(engine, "_build_review_prompt", return_value="prompt") as mock_prompt:
            result, iters, passed = engine._run_review_loop(
                artifact="artifact",
                artifact_type="arch_design",
                context="ctx",
                proposer=dev,
                reviewers=[qas[0]],
            )
        assert passed is True
        assert result == "artifact"
        mock_prompt.assert_called_once()
        assert mock_prompt.call_args.kwargs["reviewer"] is qas[0]


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

    def test_auto_approve_skips_prompt(self):
        engine, _, _ = _make_engine()
        engine._auto_approve = True
        engine._ask.reset_mock()

        result = engine._user_checkpoint("artifact", "design", "Approve?")

        assert result == "artifact"
        engine._ask.assert_not_called()

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

    def test_feedback_persists_revised_artifact(self):
        """Post-feedback revision is persisted so resume loads the right version."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        ask_mock = MagicMock(side_effect=["make it better", "y"])
        engine, dev, _ = _make_engine(store=store, ask_return=ask_mock)
        dev.send.return_value = _ok("improved design")

        result = engine._user_checkpoint("original", "design", "Approve?")

        assert result == "improved design"
        store.create_artifact.assert_called_once()
        call_kwargs = store.create_artifact.call_args[1]
        assert call_kwargs["artifact_type"] == "design_doc"
        assert call_kwargs["content"] == "improved design"
        assert call_kwargs["metadata_json"]["source"] == "user_checkpoint"

    def test_feedback_persists_plan_as_proposal(self):
        """Plan revisions are persisted with artifact_type 'proposal'."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        ask_mock = MagicMock(side_effect=["add phase 3", "y"])
        engine, dev, _ = _make_engine(store=store, ask_return=ask_mock)
        dev.send.return_value = _ok("revised plan")

        result = engine._user_checkpoint("original plan", "plan", "Approve?")

        assert result == "revised plan"
        call_kwargs = store.create_artifact.call_args[1]
        assert call_kwargs["artifact_type"] == "proposal"

    def test_feedback_each_round_persisted(self):
        """Multiple feedback rounds each persist the revised artifact."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        ask_mock = MagicMock(side_effect=["fix A", "fix B", "y"])
        engine, dev, _ = _make_engine(store=store, ask_return=ask_mock)
        dev.send.side_effect = [_ok("v2"), _ok("v3")]

        result = engine._user_checkpoint("v1", "design", "Approve?")

        assert result == "v3"
        assert store.create_artifact.call_count == 2
        # Last persisted version should be v3
        last_call_kwargs = store.create_artifact.call_args_list[-1][1]
        assert last_call_kwargs["content"] == "v3"

    def test_feedback_persist_failure_does_not_crash(self):
        """If artifact persistence fails, checkpoint still returns the revision."""
        store = MagicMock()
        store.create_artifact.side_effect = RuntimeError("DB down")
        store.create_review_cycle.return_value = 99
        ask_mock = MagicMock(side_effect=["feedback", "y"])
        engine, dev, _ = _make_engine(store=store, ask_return=ask_mock)
        dev.send.return_value = _ok("revised")

        result = engine._user_checkpoint("original", "design", "Approve?")

        assert result == "revised"

    def test_no_persist_when_no_work_item(self):
        """Without a work_item_id, skip artifact persistence."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        ask_mock = MagicMock(side_effect=["feedback", "y"])
        engine, dev, _ = _make_engine(store=store, work_item_id=0, ask_return=ask_mock)
        dev.send.return_value = _ok("revised")

        result = engine._user_checkpoint("original", "design", "Approve?")

        assert result == "revised"
        store.create_artifact.assert_not_called()

    def test_persist_artifact_false_skips_persist(self):
        """Callers can opt out of checkpoint artifact persistence."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        ask_mock = MagicMock(side_effect=["feedback", "y"])
        engine, dev, _ = _make_engine(store=store, ask_return=ask_mock)
        dev.send.return_value = _ok("revised")

        result = engine._user_checkpoint(
            "original", "design", "Approve?", persist_artifact=False,
        )

        assert result == "revised"
        store.create_artifact.assert_not_called()

    def test_resume_loads_post_feedback_version(self):
        """End-to-end: review-loop artifact at iteration 3, checkpoint at iteration 0.

        Simulates the DB with updated_at tracking to confirm _load_latest_artifact
        returns the checkpoint revision (latest updated_at) not the review-loop
        version (higher iteration or id).
        """
        artifacts_db, store = _make_fake_artifact_store()

        # 1) Simulate review loop persisting at iterations 1-3
        for i in range(1, 4):
            store.create_artifact(
                work_item_id=1, agent_id=10, iteration=i,
                artifact_type="design_doc", title=f"review v{i}",
                content=f"review version {i}", metadata_json={},
            )

        # 2) Now user gives feedback at checkpoint
        ask_mock = MagicMock(side_effect=["please revise", "y"])
        engine, dev, _ = _make_engine(store=store, ask_return=ask_mock)
        dev.send.return_value = _ok("user-approved final design")
        engine._user_checkpoint("review version 3", "design", "Approve?")

        # 3) Verify resume would load the post-feedback version
        loaded = engine._load_latest_artifact("design_doc")
        assert loaded is not None
        assert loaded["content"] == "user-approved final design"

    def test_resume_after_upsert_loads_updated_row(self):
        """Upsert scenario: review-loop updates an older row in place.

        Run 1: review loop creates iter 1-3, user gives feedback → checkpoint iter 0.
        Run 2 (resume design_review): review loop upserts iter 1-3. User approves
        with no feedback (no new checkpoint row).
        Resume from plan: must load the upserted iter 3 (latest updated_at),
        NOT the stale checkpoint at iter 0 (higher id but older updated_at).
        """
        artifacts_db, store = _make_fake_artifact_store()

        # Run 1: review loop creates iterations 1-3
        for i in range(1, 4):
            store.create_artifact(
                work_item_id=1, agent_id=10, iteration=i,
                artifact_type="design_doc", title=f"review v{i}",
                content=f"run1 review {i}", metadata_json={},
            )
        # Run 1: user checkpoint creates iteration 0 (gets highest id so far)
        store.create_artifact(
            work_item_id=1, agent_id=10, iteration=0,
            artifact_type="design_doc", title="checkpoint",
            content="run1 user-approved", metadata_json={},
        )

        # Run 2: review loop upserts iterations 1-3 (updates existing rows)
        for i in range(1, 4):
            store.create_artifact(
                work_item_id=1, agent_id=10, iteration=i,
                artifact_type="design_doc", title=f"review v{i} run2",
                content=f"run2 review {i}", metadata_json={},
            )
        # Run 2: user approves with "y" — NO checkpoint write

        # Resume from plan: must load the latest-modified artifact
        engine, _, _ = _make_engine(store=store)
        loaded = engine._load_latest_artifact("design_doc")
        assert loaded is not None
        # Must be the upserted run2 content, not the stale run1 checkpoint
        assert loaded["content"] == "run2 review 3"

    @pytest.mark.parametrize(
        ("workflow_artifact_type", "db_artifact_type", "main_content", "sub_content"),
        [
            ("design", "design_doc", "main approved design", "bug-fix reviewed design"),
            ("plan", "proposal", "main approved plan", "bug-fix reviewed plan"),
        ],
    )
    def test_subworkflow_review_loop_does_not_clobber_main_artifacts(
        self,
        workflow_artifact_type,
        db_artifact_type,
        main_content,
        sub_content,
    ):
        """Bug-fix review loops and checkpoints leave the main artifact as latest."""
        artifacts_db, store = _make_fake_artifact_store()

        for i in range(1, 4):
            store.create_artifact(
                work_item_id=1,
                agent_id=10,
                iteration=i,
                artifact_type=db_artifact_type,
                title=f"main review v{i}",
                content=f"main review {i}",
                metadata_json={"source": "review_loop"},
            )
        store.create_artifact(
            work_item_id=1,
            agent_id=10,
            iteration=0,
            artifact_type=db_artifact_type,
            title="main checkpoint",
            content=main_content,
            metadata_json={"source": "user_checkpoint"},
        )

        engine, _, qas = _make_engine(store=store, ask_return="y", max_review=1)
        qas[0].send.return_value = _ok(_lgtm_json())

        reviewed, iters, passed = engine._run_review_loop(
            artifact=sub_content,
            artifact_type=workflow_artifact_type,
            context="Bug fix context",
            persist_artifacts=False,
        )
        approved = engine._user_checkpoint(
            reviewed,
            workflow_artifact_type,
            "Approve?",
            persist_artifact=False,
        )

        assert approved == sub_content
        assert iters == 1
        assert passed is True
        loaded = engine._load_latest_artifact(db_artifact_type)
        assert loaded is not None
        assert loaded["content"] == main_content
        assert len(artifacts_db) == 4


def _make_fake_artifact_store():
    """Build a fake store that simulates real DB upsert + updated_at ordering.

    Returns (artifacts_db list, configured MagicMock store).
    """
    from datetime import datetime, timedelta

    artifacts_db: list[dict] = []
    next_id = [0]
    clock = [datetime(2026, 1, 1)]

    def _tick():
        clock[0] += timedelta(seconds=1)
        return clock[0]

    def fake_create_artifact(**kwargs):
        wi = kwargs.get("work_item_id")
        ai = kwargs.get("agent_id")
        it = kwargs.get("iteration")
        at = kwargs.get("artifact_type")
        # Upsert: match on (work_item_id, agent_id, iteration, artifact_type)
        for row in artifacts_db:
            if (row.get("work_item_id") == wi and row.get("agent_id") == ai
                    and row.get("iteration") == it and row.get("artifact_type") == at):
                row["content"] = kwargs.get("content")
                row["title"] = kwargs.get("title")
                row["metadata_json"] = kwargs.get("metadata_json")
                row["updated_at"] = _tick()
                return row["id"]
        next_id[0] += 1
        now = _tick()
        artifacts_db.append({
            "id": next_id[0],
            **kwargs,
            "created_at": now,
            "updated_at": now,
        })
        return next_id[0]

    def fake_get_latest(work_item_id, artifact_type):
        matches = [
            a for a in artifacts_db
            if a.get("work_item_id") == work_item_id
            and a.get("artifact_type") == artifact_type
        ]
        if not matches:
            return None
        # Mirrors the fixed store: ORDER BY updated_at DESC, id DESC LIMIT 1
        return max(matches, key=lambda a: (a["updated_at"], a["id"]))

    store = MagicMock()
    store.create_artifact.side_effect = fake_create_artifact
    store.create_review_cycle.return_value = 99
    store.get_latest_artifact_by_type.side_effect = fake_get_latest
    store.list_artifacts.return_value = []
    return artifacts_db, store


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

    def test_design_mode_report_omits_development_and_ga_sections(self):
        """Design-mode report only includes design/planning sections."""
        engine, dev, _ = _make_engine(mode=WorkMode.design)
        result = WorkflowResult(
            requirement="req",
            design_review_iterations=2,
            design_review_passed=True,
            plan_review_iterations=1,
            plan_review_passed=False,
        )

        report = engine._generate_report(result, [])

        dev.send.assert_not_called()
        assert "## Design Review" in report
        assert "## Plan Review" in report
        assert "## Development Phases" not in report
        assert "## GA Test" not in report
        assert "Plan review: Not passed" in report

    def test_architect_design_mode_report_includes_failure_and_final_design(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        engine = WorkflowEngine(
            store=MagicMock(),
            dev_sm=make_fake_session_manager(agent_id=10, agent_role="developer", responses=[]),
            qa_sms=[make_fake_session_manager(agent_id=20, agent_role="qa-0", responses=[])],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.architect_design,
        )
        result = WorkflowResult(
            requirement="req",
            design="final design text",
            design_review_iterations=2,
            success=False,
            failure_summary="[qa_main] review failed (exit=1)",
        )

        report = engine._generate_report(result, [])

        assert "## Design Review" in report
        assert "## Failure" in report
        assert "final design text" in report

    def test_testplan_design_mode_report_includes_failure_and_final_plan(self):
        arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[], session_id=300)
        engine = WorkflowEngine(
            store=MagicMock(),
            dev_sm=make_fake_session_manager(agent_id=10, agent_role="developer", responses=[]),
            qa_sms=[make_fake_session_manager(agent_id=20, agent_role="qa_main", responses=[])],
            arch_sm=arch,
            project_id=1,
            work_item_id=1,
            mode=WorkMode.testplan_design,
        )
        result = WorkflowResult(
            requirement="req",
            ga_test=GATestResult(
                test_plan="final plan text",
                test_plan_review_iterations=2,
                test_plan_review_passed=False,
            ),
            success=False,
            failure_summary="[qa_main] review failed (exit=1)",
        )

        report = engine._generate_report(result, [])

        assert "## Test Plan Review" in report
        assert "## Failure" in report
        assert "final plan text" in report

    def test_develop_mode_report_omits_design_and_ga_sections(self):
        """Develop-mode report only includes development content."""
        engine, dev, _ = _make_engine(mode=WorkMode.develop)
        dev.send.return_value = _ok("narrative")
        result = WorkflowResult(
            requirement="req",
            phases=[
                PhaseResult(name="p1", summary="s", review_iterations=1, review_passed=False, committed=True),
            ],
        )

        report = engine._generate_report(result, ["Phase 1: s"])

        assert "## Development Phases" in report
        assert "## Design Review" not in report
        assert "## Plan Review" not in report
        assert "## GA Test" not in report
        assert "Review: Not passed" in report

    def test_test_mode_report_omits_design_and_development_sections(self):
        """Test-mode report only includes test-related sections."""
        engine, dev, _ = _make_engine(mode=WorkMode.test)
        result = WorkflowResult(
            requirement="req",
            ga_test=GATestResult(
                passed=False,
                test_plan_review_iterations=1,
                test_plan_review_passed=False,
                bugs_found=2,
                bugs_fixed=1,
            ),
        )

        report = engine._generate_report(result, [])

        dev.send.assert_not_called()
        assert "## Test Plan Review" in report
        assert "## GA Test" in report
        assert "## Design Review" not in report
        assert "## Plan Review" not in report
        assert "## Development Phases" not in report
        assert "Test plan review: Not passed" in report

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
        assert pr.review_passed is False
        assert pr.committed is False

    def test_bug_fix_result_defaults(self):
        bf = BugFixResult(title="bug")
        assert bf.arch_change is False
        assert bf.fixed is False
        assert bf.summary == ""

    def test_ga_test_result_defaults(self):
        ga = GATestResult()
        assert ga.test_plan == ""
        assert ga.test_plan_review_passed is False
        assert ga.bugs_found == 0
        assert ga.bug_fixes == []
        assert ga.passed is False
        assert ga.aborted is False

    def test_workflow_result_defaults(self):
        wr = WorkflowResult(requirement="req")
        assert wr.design == ""
        assert wr.design_review_passed is False
        assert wr.plan == ""
        assert wr.plan_review_passed is False
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
        result, iters, passed = engine._run_review_loop(
            artifact="plan content",
            artifact_type="plan",
            context="ctx",
        )
        assert iters == 1
        assert passed is True
        # Verify the QA reviewer was called with a prompt containing the plan
        call_args = qas[0].send.call_args[0][0]
        assert "plan content" in call_args


# ===================================================================
# Resume tests
# ===================================================================


class TestResumeCompletedWorkflow:
    """Issue 1: resuming a completed workflow must not downgrade status."""

    @pytest.mark.parametrize("mode", [WorkMode.full, WorkMode.design, WorkMode.develop, WorkMode.test])
    def test_resume_completed_returns_success(self, mode):
        engine, dev, qas = _make_engine(
            mode=mode, resume_stage="workflow_completed",
        )
        result = engine.run("some req")
        assert result.success is True
        assert "already completed" in result.final_report.lower()

    def test_resume_finished_with_issues_reruns_from_start(self):
        """workflow_finished_with_issues is NOT treated as done — it re-runs."""
        engine, dev, qas = _make_engine(
            mode=WorkMode.full,
            resume_stage="workflow_finished_with_issues",
            dev_responses=[
                "design",           # Stage 1: _run_design
                "plan",             # Stage 3: _run_planning
                "impl",             # Stage 5: _run_phase implement
                "summary",          # Stage 5: _run_phase summarize
                "committed",        # Stage 5: _run_phase commit
                _lgtm_json(),       # Stage 6: test plan review (dev is reviewer)
                "report",           # Stage 7: _generate_report
            ],
            qa_responses=[
                _ok(_lgtm_json()),  # Stage 2: design review
                _ok(_lgtm_json()),  # Stage 4: plan review
                _ok(_lgtm_json()),  # Stage 5: code review
                _ok("test plan"),   # Stage 6: GA test plan generation
                _ok("PASS"),        # Stage 6: GA test execute
            ],
        )
        result = engine.run("some req")
        # Should have actually run the workflow, not returned early
        assert result.design is not None
        assert dev.send.call_count > 0


class TestResumeDoesNotClobberStage:
    """Issue 1: run() must not overwrite current_stage to workflow_started on resume."""

    def test_resume_skips_startup_persist(self):
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        engine, dev, qas = _make_engine(
            store=store,
            mode=WorkMode.full,
            resume_stage="workflow_completed",
        )
        engine.run("req")

        # _persist_task_state should NOT have been called with "workflow_started"
        for c in store.update_work_item.call_args_list:
            args, kwargs = c
            if len(args) > 1 and isinstance(args[1], dict):
                meta = args[1]
            elif kwargs:
                meta = kwargs
            else:
                continue
            task_state = meta.get("metadata_json", {})
            if isinstance(task_state, str):
                task_state = json.loads(task_state)
            task_state = task_state.get("task_state", {})
            assert task_state.get("current_stage") != "workflow_started"


class TestResumeDesignModeSkipsCompletedStages:
    """Issue 2: design mode must respect resume_stage."""

    def test_resume_from_plan_approved_skips_review(self):
        """resume_stage=plan_approved should skip design + plan and finalize."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.get_latest_artifact_by_type.side_effect = lambda wi, at: (
            {"content": "loaded design"} if at == "design_doc"
            else {"content": "loaded plan"} if at == "proposal"
            else None
        )
        engine, dev, qas = _make_engine(
            store=store,
            mode=WorkMode.design,
            resume_stage="plan_approved",
        )
        result = engine.run("req")

        # Dev and QA should NOT have been called (all stages skipped)
        dev.send.assert_not_called()
        qas[0].send.assert_not_called()
        # Result should show success with both reviews marked passed
        assert result.success is True
        assert result.design_review_passed is True
        assert result.plan_review_passed is True

    def test_resume_from_plan_draft_skips_design(self):
        """resume_stage=plan_draft should skip design, re-run planning + plan review."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.get_latest_artifact_by_type.return_value = {"content": "loaded design"}

        engine, dev, qas = _make_engine(
            store=store,
            mode=WorkMode.design,
            resume_stage="plan_draft",
            dev_responses=["new plan"],
            qa_responses=[_ok(_lgtm_json())],
        )
        result = engine.run("req")

        # Dev should produce the plan (not design)
        assert dev.send.call_count == 1
        assert result.plan is not None
        assert result.design_review_passed is True  # skipped = assumed passed


class TestResumeDevelopMode:
    """Issue 2: develop mode must respect resume_stage for phase skipping."""

    def test_resume_from_phase_2_skips_phase_1(self):
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        # Phase 1 artifact for reconstruction
        store.list_artifacts.return_value = [
            {
                "artifact_type": "phase_result",
                "iteration": 1,
                "content": "Phase 1 done",
                "metadata_json": json.dumps({
                    "name": "Phase 1",
                    "review_iterations": 1,
                    "review_passed": True,
                    "committed": True,
                }),
            },
        ]

        # Dev: implement phase 2, summarize, commit, final report
        engine, dev, qas = _make_engine(
            store=store,
            mode=WorkMode.develop,
            resume_stage="phase_1_committed",
            # The requirement is the plan in develop mode — two phases
            dev_responses=["implemented", "summary: done", "committed", "final report"],
            qa_responses=[_ok(_lgtm_json())],
        )
        result = engine.run("Phase 1: setup\nPhase 2: implement")

        # Phase 1 was loaded from artifacts, phase 2 was run
        assert len(result.phases) == 2
        assert result.phases[0].name == "Phase 1"  # loaded
        assert result.phases[0].committed is True


class TestResumeTestMode:
    """Issue 2: test mode must respect resume_stage for done state."""

    def test_resume_completed_test_returns_success(self):
        engine, dev, qas = _make_engine(
            mode=WorkMode.test,
            resume_stage="workflow_completed",
        )
        result = engine.run("test req")
        assert result.success is True
        dev.send.assert_not_called()
        qas[0].send.assert_not_called()


class TestResumePhaseCodeReviewPreservesIndex:
    """Phase code review stages must include the phase index."""

    def test_phase_code_review_stage_includes_phase_index(self):
        """_run_phase should persist phase_N_code_review, not code_review."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99

        engine, dev, qas = _make_engine(
            store=store,
            mode=WorkMode.develop,
            dev_responses=["implemented", "summary: done", "committed"],
            qa_responses=[_ok(_lgtm_json())],
        )
        # Run just one phase
        engine._run_phase(
            phase_name="setup",
            phase_index=3,
            total_phases=5,
            requirement="req",
            design="design",
            plan="plan",
            completed_summaries=[],
        )

        # Check that _persist_task_state was called with phase_3_code_approved
        stage_names = []
        for c in store.update_work_item.call_args_list:
            args = c[0] if c[0] else ()
            kwargs = c[1] if len(c) > 1 else c.kwargs
            meta = kwargs.get("metadata_json") or (args[1] if len(args) > 1 else {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            ts = meta.get("task_state", {})
            if ts.get("current_stage"):
                stage_names.append(ts["current_stage"])

        # Should contain phase_3_code_approved, not code_approved
        code_stages = [s for s in stage_names if "code" in s]
        for s in code_stages:
            assert "phase_3" in s, f"Stage {s!r} should include phase index"

    def test_resume_phase_index_parses_code_review_stage(self):
        """_resume_phase_index should return N for phase_N_code_review."""
        engine, _, _ = _make_engine(
            mode=WorkMode.develop,
            resume_stage="phase_3_code_review",
        )
        assert engine._resume_phase_index() == 3

    def test_resume_phase_index_parses_code_approved_stage(self):
        """phase_N_code_approved means phase N is done → return N+1."""
        engine, _, _ = _make_engine(
            mode=WorkMode.develop,
            resume_stage="phase_2_code_approved",
        )
        # code_approved != "committed", so returns N (re-run commit step)
        assert engine._resume_phase_index() == 2


class TestResumeStubPhases:
    """Pre-migration items without phase_result artifacts get stub phases."""

    def test_stub_phases_filled_when_artifacts_missing(self):
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.list_artifacts.return_value = []  # no phase_result artifacts

        engine, _, _ = _make_engine(store=store, mode=WorkMode.develop)
        phases = engine._load_completed_phases(before_phase=4)

        assert len(phases) == 3  # stubs for phases 1, 2, 3
        for p in phases:
            assert p.committed is True
            assert "prior run" in p.summary

    def test_partial_artifacts_filled_with_stubs(self):
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.list_artifacts.return_value = [
            {
                "artifact_type": "phase_result",
                "iteration": 1,
                "content": "Phase 1 done",
                "metadata_json": json.dumps({
                    "name": "Setup",
                    "review_iterations": 1,
                    "review_passed": True,
                    "committed": True,
                }),
            },
        ]

        engine, _, _ = _make_engine(store=store, mode=WorkMode.develop)
        phases = engine._load_completed_phases(before_phase=4)

        assert len(phases) == 3
        assert phases[0].name == "Setup"  # real artifact
        assert "prior run" in phases[1].summary  # stub
        assert "prior run" in phases[2].summary  # stub

    def test_no_before_phase_uses_expected_count(self):
        """Resuming from ga_test (no before_phase) with expected_count fills stubs."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.list_artifacts.return_value = []  # pre-migration, no artifacts

        engine, _, _ = _make_engine(store=store, mode=WorkMode.develop)
        phases = engine._load_completed_phases(expected_count=3)

        assert len(phases) == 3
        for i, p in enumerate(phases, 1):
            assert p.name == f"Phase {i}"
            assert p.committed is True
            assert "prior run" in p.summary

    def test_no_before_phase_uses_max_artifact_index(self):
        """Without before_phase or expected_count, max artifact index determines count."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.list_artifacts.return_value = [
            {
                "artifact_type": "phase_result",
                "iteration": 3,
                "content": "Phase 3 done",
                "metadata_json": json.dumps({
                    "name": "Final",
                    "review_iterations": 1,
                    "review_passed": True,
                    "committed": True,
                }),
            },
        ]

        engine, _, _ = _make_engine(store=store, mode=WorkMode.develop)
        phases = engine._load_completed_phases()  # no before_phase, no expected_count

        assert len(phases) == 3
        assert "prior run" in phases[0].summary  # stub for phase 1
        assert "prior run" in phases[1].summary  # stub for phase 2
        assert phases[2].name == "Final"  # real artifact at phase 3

    def test_non_contiguous_artifacts_slotted_correctly(self):
        """Phase 2 artifact with before_phase=4 puts it at position 2, not 1."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.list_artifacts.return_value = [
            {
                "artifact_type": "phase_result",
                "iteration": 2,
                "content": "Phase 2 done",
                "metadata_json": json.dumps({
                    "name": "Two",
                    "review_iterations": 1,
                    "review_passed": True,
                    "committed": True,
                }),
            },
        ]

        engine, _, _ = _make_engine(store=store, mode=WorkMode.develop)
        phases = engine._load_completed_phases(before_phase=4)

        assert len(phases) == 3
        assert "prior run" in phases[0].summary  # stub for missing phase 1
        assert phases[1].name == "Two"  # real artifact at correct slot
        assert "prior run" in phases[2].summary  # stub for missing phase 3

    def test_summaries_non_contiguous_numbered_correctly(self):
        """Phase summaries use correct numbering even with non-contiguous artifacts."""
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.list_artifacts.return_value = [
            {
                "artifact_type": "phase_result",
                "iteration": 2,
                "content": "Phase 2 done",
                "metadata_json": json.dumps({
                    "name": "Two",
                    "review_iterations": 1,
                    "review_passed": True,
                    "committed": True,
                }),
            },
        ]

        engine, _, _ = _make_engine(store=store, mode=WorkMode.develop)
        summaries = engine._load_completed_phase_summaries(before_phase=4)

        assert len(summaries) == 3
        assert summaries[0].startswith("Phase 1 (Phase 1)")  # stub
        assert summaries[1].startswith("Phase 2 (Two)")  # real, correct number
        assert summaries[2].startswith("Phase 3 (Phase 3)")  # stub


class TestWorkflowEngineEvents:
    """Regression tests for event-driven workflow output."""

    @patch("myswat.workflow.engine.console", new_callable=MagicMock)
    @patch.object(WorkflowEngine, "_generate_report", return_value="# Development Report")
    @patch.object(
        WorkflowEngine,
        "_run_phase",
        return_value=PhaseResult(
            name="phase-A",
            summary="done",
            review_iterations=1,
            review_passed=True,
            committed=True,
        ),
    )
    @patch.object(WorkflowEngine, "_parse_phases", return_value=["phase-A"])
    def test_develop_mode_emits_stage_complete_for_final_report(
        self,
        _mock_parse_phases,
        _mock_run_phase,
        _mock_generate_report,
        _mock_console,
    ):
        store = MagicMock()
        events = []
        dev_sm = make_fake_session_manager(agent_id=10, agent_role="developer")
        qa_sm = make_fake_session_manager(agent_id=20, agent_role="qa-0")

        engine = WorkflowEngine(
            store=store,
            dev_sm=dev_sm,
            qa_sms=[qa_sm],
            project_id=1,
            work_item_id=1,
            mode=WorkMode.develop,
            on_event=events.append,
        )

        result = engine.run("Ship it")

        assert result.success is True
        assert any(
            event.event_type == "stage_complete"
            and event.stage == "report"
            and event.message == "Report generated"
            for event in events
        )

    @patch("myswat.workflow.engine.console", new_callable=MagicMock)
    def test_review_loop_emits_failed_verdict_and_stage_failure_for_reviewer_crash(self, _mock_console):
        store = MagicMock()
        store.create_artifact.return_value = 42
        store.create_review_cycle.return_value = 99
        store.create_work_item_process_event.return_value = 1
        events = []
        dev_sm = make_fake_session_manager(agent_id=10, agent_role="developer")
        qa_sm = make_fake_session_manager(
            agent_id=20,
            agent_role="qa-0",
            responses=[_fail("qa crashed hard")],
        )

        engine = WorkflowEngine(
            store=store,
            dev_sm=dev_sm,
            qa_sms=[qa_sm],
            project_id=1,
            work_item_id=1,
            on_event=events.append,
        )

        artifact, iteration, passed = engine._run_review_loop(
            artifact="draft design",
            artifact_type="design",
            context="Requirement",
            abort_on_agent_failure=True,
        )

        assert artifact == "draft design"
        assert iteration == 1
        assert passed is False
        assert any(
            event.event_type == "review_verdict"
            and event.metadata.get("verdict") == "failed"
            and event.agent_role == "qa-0"
            for event in events
        )
        assert any(
            event.event_type == "stage_complete"
            and event.metadata.get("failed") is True
            and "review failed" in event.message
            for event in events
        )
