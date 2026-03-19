"""Additional coverage-focused tests for workflow.engine helpers and branches."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from myswat.workflow.engine import (
    GATestResult,
    PhaseResult,
    WorkflowEngine,
    WorkflowResult,
    WorkMode,
)
from tests.conftest import make_fake_session_manager


def _arch_design_doc() -> str:
    return """
    # Technical Design

    ## Problem statement and goals
    Build a reviewed technical design artifact that is concrete enough for developer and QA review.

    ## Architecture overview and approach
    The architect provides a written design document with explicit sections and a stable structure for later implementation.

    ## Key decisions and trade-offs
    The design favors clarity and reviewability over terse notes so reviewers can reason about correctness and testability.

    ## Component interfaces and data flow
    The proposal defines the public API, the internal responsibilities, and how inputs move through the main components.

    ## Dependencies and risks
    The design documents important constraints, risks, and review concerns before implementation starts.
    """


def _make_engine(
    *,
    mode: WorkMode = WorkMode.full,
    resume_stage: str | None = None,
    work_item_id: int | None = 1,
    store: MagicMock | None = None,
    arch_sm=None,
    qa_count: int = 1,
    max_review_iterations: int = 5,
):
    if store is None:
        store = MagicMock()
    dev = make_fake_session_manager(agent_id=10, agent_role="developer", session_id=100)
    qas = [
        make_fake_session_manager(agent_id=20 + i, agent_role=f"qa-{i}", session_id=200 + i)
        for i in range(qa_count)
    ]
    if arch_sm is None and mode in {WorkMode.architect_design, WorkMode.testplan_design}:
        arch_sm = make_fake_session_manager(agent_id=30, agent_role="architect", session_id=300)
    engine = WorkflowEngine(
        store=store,
        dev_sm=dev,
        qa_sms=qas,
        project_id=1,
        work_item_id=work_item_id,
        mode=mode,
        arch_sm=arch_sm,
        resume_stage=resume_stage,
        max_review_iterations=max_review_iterations,
        ask_user=lambda _prompt: "y",
    )
    return engine, store, dev, qas


@pytest.mark.parametrize(
    ("resume_stage", "expected"),
    [
        ("ga_test_executing", "ga_test"),
        ("proposal_review", "plan_review"),
        ("arch_design_approved", "design_checkpoint"),
        ("plan_approved", "plan_checkpoint"),
        ("arch_design_review", "design_review"),
        ("design_draft_blocked", "design"),
        ("code_review", "phases"),
        ("phase_2_committed", "phases"),
        ("mystery_stage", "start"),
    ],
)
def test_resume_entry_point_maps_stage_groups(resume_stage, expected):
    engine, _, _, _ = _make_engine(resume_stage=resume_stage)

    assert engine._resume_entry_point() == expected


def test_resume_phase_index_handles_default_invalid_and_committed():
    engine, _, _, _ = _make_engine(resume_stage=None)
    assert engine._resume_phase_index() == 1

    engine, _, _, _ = _make_engine(resume_stage="phase_x_review")
    assert engine._resume_phase_index() == 1

    engine, _, _, _ = _make_engine(resume_stage="phase_2_committed")
    assert engine._resume_phase_index() == 3


def test_parse_artifact_meta_handles_none_dict_invalid_json_and_other():
    engine, _, _, _ = _make_engine()

    assert engine._parse_artifact_meta({"metadata_json": None}) == {}
    assert engine._parse_artifact_meta({"metadata_json": {"name": "P1"}}) == {"name": "P1"}
    assert engine._parse_artifact_meta({"metadata_json": "{"}) == {}
    assert engine._parse_artifact_meta({"metadata_json": 7}) == {}


def test_load_latest_artifact_and_completed_phases_return_empty_without_work_item():
    engine, store, _, _ = _make_engine(work_item_id=None)

    assert engine._load_latest_artifact("design_doc") is None
    assert engine._load_completed_phases() == []
    store.get_latest_artifact_by_type.assert_not_called()


def test_load_completed_phases_skips_non_phase_rows_and_filtered_rows():
    store = MagicMock()
    store.list_artifacts.return_value = [
        {"artifact_type": "proposal", "iteration": 1},
        {
            "artifact_type": "phase_result",
            "iteration": 4,
            "content": "late phase",
            "metadata_json": json.dumps({"name": "Late"}),
        },
    ]
    engine, _, _, _ = _make_engine(store=store)

    assert engine._load_completed_phases(before_phase=4) == []


def test_load_completed_phases_returns_empty_when_no_expected_phases_exist():
    store = MagicMock()
    store.list_artifacts.return_value = [{"artifact_type": "proposal", "iteration": 1}]
    engine, _, _, _ = _make_engine(store=store)

    assert engine._load_completed_phases() == []


def test_run_returns_cancelled_before_start():
    engine, store, _, _ = _make_engine()
    engine._should_cancel = lambda: True

    result = engine.run("ship feature")

    assert result.final_report == "Workflow cancelled before start."
    store.update_work_item_state.assert_not_called()


def test_dispatch_mode_rejects_unknown_mode():
    engine, _, _, _ = _make_engine()
    engine._mode = SimpleNamespace(value="unknown-mode")

    with pytest.raises(NotImplementedError, match="unknown-mode"):
        engine._dispatch_mode("req", WorkflowResult(requirement="req"))


def test_architect_design_mode_cancelled_during_technical_design():
    arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=[_arch_design_doc()], session_id=300)
    engine, store, _, _ = _make_engine(mode=WorkMode.architect_design, arch_sm=arch)
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_cancelled", return_value=True):
        result = engine._run_architect_design_mode("req", result)

    assert result.final_report == "Architect-design workflow cancelled during technical design."


def test_architect_design_mode_review_failure_generates_report():
    arch = make_fake_session_manager(agent_id=30, agent_role="architect", responses=["design"], session_id=300)
    engine, store, _, _ = _make_engine(mode=WorkMode.architect_design, arch_sm=arch)
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_cancelled", return_value=False):
        with patch.object(engine, "_run_review_loop", return_value=("reviewed design", 2, False)):
            with patch.object(engine, "_generate_report", return_value="# report"):
                result = engine._run_architect_design_mode("req", result)

    assert result.final_report == "# report"
    assert result.design_review_passed is False


def test_testplan_design_mode_cancelled_during_drafting():
    arch = make_fake_session_manager(agent_id=30, agent_role="architect", session_id=300)
    engine, _, _, qas = _make_engine(mode=WorkMode.testplan_design, arch_sm=arch)
    qas[0].send.return_value = make_fake_session_manager(responses=["plan"]).send()
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_cancelled", return_value=True):
        result = engine._run_testplan_design_mode("req", result)

    assert result.final_report == "Testplan-design workflow cancelled during test plan drafting."
    assert result.ga_test is not None


def test_testplan_design_mode_review_failure_generates_report():
    arch = make_fake_session_manager(agent_id=30, agent_role="architect", session_id=300)
    engine, _, _, qas = _make_engine(mode=WorkMode.testplan_design, arch_sm=arch)
    qas[0].send.return_value = make_fake_session_manager(responses=["plan"]).send()
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_cancelled", return_value=False):
        with patch.object(engine, "_run_review_loop", return_value=("reviewed plan", 2, False)):
            with patch.object(engine, "_generate_report", return_value="# report"):
                result = engine._run_testplan_design_mode("req", result)

    assert result.final_report == "# report"
    assert result.ga_test is not None
    assert result.ga_test.test_plan_review_passed is False


def test_full_mode_resume_plan_loads_design_artifact_and_skips_design_review():
    engine, _, _, _ = _make_engine(mode=WorkMode.full, resume_stage="design_approved")
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_load_latest_artifact", return_value={"content": "saved design"}):
        with patch.object(engine, "_run_planning", return_value=""):
            result = engine._run_full("req", result)

    assert result.design == "saved design"
    assert result.design_review_passed is True


def test_full_mode_resume_phases_loads_plan_artifact_and_cancels_before_phase():
    engine, _, _, _ = _make_engine(mode=WorkMode.full, resume_stage="phase_2_review")
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_load_latest_artifact", return_value={"content": "saved plan"}):
            with patch.object(engine, "_parse_phases", return_value=["Phase 1", "Phase 2"]):
                with patch.object(engine, "_resume_phase_index", return_value=2):
                    with patch.object(engine, "_load_completed_phase_summaries", return_value=["done"]):
                        with patch.object(
                            engine,
                            "_load_completed_phases",
                            return_value=[
                                PhaseResult(
                                    name="Phase 1",
                                    summary="done",
                                    review_iterations=1,
                                    committed=True,
                                )
                            ],
                        ):
                            with patch.object(engine, "_cancelled", return_value=True):
                                result = engine._run_full("req", result)

    assert result.plan == "saved plan"
    assert result.plan_review_passed is True
    assert result.final_report == "Workflow cancelled before phase 2."


def test_full_mode_resume_ga_test_reconstructs_completed_phases():
    phase_result = PhaseResult(name="Phase 1", summary="done", review_iterations=1, committed=True)
    engine, _, _, _ = _make_engine(mode=WorkMode.full, resume_stage="ga_test_executing")
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_load_latest_artifact", side_effect=[{"content": "design"}, {"content": "plan"}]):
        with patch.object(engine, "_parse_phases", return_value=["Phase 1"]):
            with patch.object(engine, "_load_completed_phase_summaries", return_value=["Phase 1: done"]):
                with patch.object(engine, "_load_completed_phases", return_value=[phase_result]):
                    with patch.object(engine, "_run_ga_test_phase", return_value=GATestResult(passed=False)):
                        with patch.object(engine, "_generate_report", return_value="# report"):
                            with patch.object(engine, "_cancelled", return_value=False):
                                result = engine._run_full("req", result)

    assert result.design == "design"
    assert result.plan == "plan"
    assert result.phases == [phase_result]
    assert result.final_report == "# report"


def test_full_mode_cancellation_branches_across_stages():
    engine, _, _, _ = _make_engine(mode=WorkMode.full)

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_cancelled", return_value=True):
            result = engine._run_full("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Workflow cancelled during technical design."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", return_value=("design", 1, True)):
            with patch.object(engine, "_cancelled", side_effect=[False, True]):
                result = engine._run_full("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Workflow cancelled during design review."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", side_effect=[("design", 1, True)]):
            with patch.object(engine, "_user_checkpoint", return_value="design"):
                with patch.object(engine, "_run_planning", return_value="plan"):
                    with patch.object(engine, "_cancelled", side_effect=[False, False, True]):
                        result = engine._run_full("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Workflow cancelled during planning."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", side_effect=[("design", 1, True), ("plan", 1, True)]):
            with patch.object(engine, "_user_checkpoint", side_effect=["design", "plan"]):
                with patch.object(engine, "_run_planning", return_value="plan"):
                    with patch.object(engine, "_cancelled", side_effect=[False, False, False, True]):
                        result = engine._run_full("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Workflow cancelled during plan review."


def test_full_mode_cancellation_during_phase_and_ga():
    engine, _, _, _ = _make_engine(mode=WorkMode.full, resume_stage="phase_1_review")
    result = WorkflowResult(requirement="req")

    with patch.object(engine, "_load_latest_artifact", return_value={"content": "plan"}):
        with patch.object(engine, "_parse_phases", return_value=["Phase 1"]):
            with patch.object(engine, "_resume_phase_index", return_value=1):
                with patch.object(engine, "_run_phase", return_value=PhaseResult(name="Phase 1", summary="done", review_iterations=1, committed=True)):
                    with patch.object(engine, "_cancelled", side_effect=[False, True]):
                        result = engine._run_full("req", result)
    assert result.final_report == "Workflow cancelled during phase 1."

    engine, _, _, _ = _make_engine(mode=WorkMode.full, resume_stage="ga_test_executing")
    result = WorkflowResult(requirement="req")
    with patch.object(engine, "_load_latest_artifact", side_effect=[{"content": "design"}, {"content": "plan"}]):
        with patch.object(engine, "_parse_phases", return_value=[]):
            with patch.object(engine, "_load_completed_phase_summaries", return_value=[]):
                with patch.object(engine, "_load_completed_phases", return_value=[]):
                    with patch.object(engine, "_run_ga_test_phase", return_value=GATestResult(passed=False)):
                        with patch.object(engine, "_cancelled", return_value=True):
                            result = engine._run_full("req", result)
    assert result.final_report == "Workflow cancelled during GA testing."


def test_design_mode_failure_and_cancellation_branches():
    engine, _, _, _ = _make_engine(mode=WorkMode.design)

    with patch.object(engine, "_run_design", return_value=""):
        result = engine._run_design_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Design workflow failed: developer did not produce a technical design."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_cancelled", return_value=True):
            result = engine._run_design_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Design workflow cancelled during technical design."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", return_value=("design", 1, True)):
            with patch.object(engine, "_user_checkpoint", return_value="design"):
                with patch.object(engine, "_run_planning", return_value=""):
                    result = engine._run_design_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Design workflow failed: developer did not produce an implementation plan."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", return_value=("design", 1, True)):
            with patch.object(engine, "_user_checkpoint", return_value="design"):
                with patch.object(engine, "_run_planning", return_value="plan"):
                    with patch.object(engine, "_cancelled", side_effect=[False, False, True]):
                        result = engine._run_design_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Design workflow cancelled during planning."

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", side_effect=[("design", 1, True), ("plan", 1, True)]):
            with patch.object(engine, "_user_checkpoint", side_effect=["design", None]):
                with patch.object(engine, "_run_planning", return_value="plan"):
                    with patch.object(engine, "_cancelled", side_effect=[False, False, False, False]):
                        result = engine._run_design_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Design workflow stopped by user after plan review."


def test_design_mode_cancellation_during_plan_review():
    engine, _, _, _ = _make_engine(mode=WorkMode.design)

    with patch.object(engine, "_run_design", return_value="design"):
        with patch.object(engine, "_run_review_loop", side_effect=[("design", 1, True), ("plan", 1, True)]):
            with patch.object(engine, "_user_checkpoint", return_value="design"):
                with patch.object(engine, "_run_planning", return_value="plan"):
                    with patch.object(engine, "_cancelled", side_effect=[False, False, False, True]):
                        result = engine._run_design_mode("req", WorkflowResult(requirement="req"))

    assert result.final_report == "Design workflow cancelled during plan review."


def test_develop_and_test_mode_cancellation_paths():
    engine, _, _, _ = _make_engine(mode=WorkMode.develop, resume_stage="phase_2_review")
    with patch.object(engine, "_parse_phases", return_value=["Phase 1", "Phase 2"]):
        with patch.object(engine, "_resume_phase_index", return_value=2):
            with patch.object(engine, "_load_completed_phase_summaries", return_value=["done"]):
                with patch.object(engine, "_load_completed_phases", return_value=[PhaseResult(name="Phase 1", summary="done", review_iterations=1, committed=True)]):
                    with patch.object(engine, "_cancelled", return_value=True):
                        result = engine._run_develop_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Development workflow cancelled before phase 2."

    engine, _, _, _ = _make_engine(mode=WorkMode.develop)
    with patch.object(engine, "_parse_phases", return_value=["Phase 1"]):
        with patch.object(engine, "_run_phase", return_value=PhaseResult(name="Phase 1", summary="done", review_iterations=1, committed=True)):
            with patch.object(engine, "_cancelled", side_effect=[False, True]):
                result = engine._run_develop_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Development workflow cancelled during phase 1."

    engine, _, _, _ = _make_engine(mode=WorkMode.test, resume_stage="ga_test_executing")
    with patch.object(engine, "_run_ga_test_phase", return_value=GATestResult(passed=False)):
        with patch.object(engine, "_cancelled", return_value=True):
            result = engine._run_test_mode("req", WorkflowResult(requirement="req"))
    assert result.final_report == "Test workflow cancelled during GA testing."


def test_run_design_planning_phase_and_ga_helpers_cancel_or_warn():
    engine, store, dev, qas = _make_engine()

    with patch.object(engine, "_cancelled", return_value=True):
        assert engine._run_design("req") == ""
        assert engine._run_planning("design", "req") == ""
        phase = engine._run_phase("Phase 1", 1, 1, "req", "design", "plan", [])
    assert phase.summary == "Cancelled by user."

    with patch.object(engine, "_cancelled", side_effect=[False, True]):
        dev.send.side_effect = [dev.send.return_value, dev.send.return_value]
        phase = engine._run_phase("Phase 1", 1, 1, "req", "design", "plan", [])
    assert phase.summary == "Cancelled by user."

    with patch.object(engine, "_cancelled", return_value=True):
        ga_result = engine._run_ga_test_phase("req", "design", "plan", "summary")
        bug_fix = engine._run_bug_fix({"title": "Bug"}, "req", "design")
        arch_fix = engine._run_bug_fix_arch_change({"title": "Bug"}, "req", "design")
    assert ga_result.aborted is True
    assert bug_fix.fixed is False
    assert arch_fix.requirement == "cancelled"


def test_run_phase_cancel_after_review_loop_and_warn_on_artifact_persist_failure():
    store = MagicMock()
    store.create_artifact.side_effect = Exception("persist failed")
    engine, _, _, _ = _make_engine(store=store)

    with patch.object(engine, "_cancelled", side_effect=[False, False, True]):
        with patch.object(engine, "_run_review_loop", return_value=("summary", 2, True)):
            result = engine._run_phase("Phase 1", 1, 1, "req", "design", "plan", [])

    assert result.summary == "Cancelled by user."
    assert result.review_iterations == 2

    with patch.object(engine, "_cancelled", return_value=False):
        with patch.object(engine, "_run_review_loop", return_value=("summary", 2, True)):
            with patch("myswat.workflow.engine.console.print") as mock_print:
                engine._run_phase("Phase 1", 1, 1, "req", "design", "plan", [])

    assert any("Failed to persist phase result" in str(call) for call in mock_print.call_args_list)


def test_ga_test_phase_cancel_points_after_plan_execute_and_retest():
    engine, _, _, qas = _make_engine()
    qa = qas[0]

    qa.send.return_value = SimpleNamespace(success=True, content="plan")
    with patch.object(engine, "_cancelled", side_effect=[False, True]):
        result = engine._run_ga_test_phase("req", "design", "plan", "summary")
    assert result.aborted is True

    qa.send.side_effect = [SimpleNamespace(success=True, content="plan"), SimpleNamespace(success=True, content="exec")]
    with patch.object(engine, "_run_review_loop", return_value=("plan", 1, True)):
        with patch.object(engine, "_user_checkpoint", return_value="plan"):
            with patch.object(engine, "_parse_test_results", return_value=[]):
                with patch.object(engine, "_cancelled", side_effect=[False, False, True]):
                    result = engine._run_ga_test_phase("req", "design", "plan", "summary")
    assert result.aborted is True

    qa.send.side_effect = [
        SimpleNamespace(success=True, content="plan"),
        SimpleNamespace(success=True, content='{"status":"fail","bugs":[{"title":"Bug 1"}]}'),
        SimpleNamespace(success=True, content="report"),
    ]
    with patch.object(engine, "_run_review_loop", return_value=("plan", 1, True)):
        with patch.object(engine, "_user_checkpoint", return_value="plan"):
            with patch.object(engine, "_parse_test_results", return_value=[{"title": "Bug 1"}]):
                with patch.object(engine, "_cancelled", side_effect=[False, False, False, True]):
                    result = engine._run_ga_test_phase("req", "design", "plan", "summary")
    assert result.aborted is True

    qa.send.side_effect = [
        SimpleNamespace(success=True, content="plan"),
        SimpleNamespace(success=True, content='{"status":"fail","bugs":[{"title":"Bug 1"}]}'),
        SimpleNamespace(success=True, content="retest output"),
        SimpleNamespace(success=True, content="report"),
    ]
    with patch.object(engine, "_run_review_loop", return_value=("plan", 1, True)):
        with patch.object(engine, "_user_checkpoint", return_value="plan"):
            with patch.object(engine, "_parse_test_results", side_effect=[[{"title": "Bug 1"}], [{"title": "Bug 2"}]]):
                with patch.object(engine, "_run_bug_fix", return_value=SimpleNamespace(fixed=True, title="Bug 1", summary="fixed", arch_change=False)):
                    with patch.object(engine, "_cancelled", side_effect=[False, False, False, False, True]):
                        result = engine._run_ga_test_phase("req", "design", "plan", "summary")
    assert result.aborted is True


def test_bug_fix_cancel_after_estimate_and_after_fix():
    engine, _, dev, _ = _make_engine()
    bug = {"title": "Race", "description": "details"}

    dev.send.side_effect = [SimpleNamespace(success=True, content='{"assessment":"simple_fix"}')]
    with patch.object(engine, "_cancelled", side_effect=[False, True]):
        result = engine._run_bug_fix(bug, "req", "design")
    assert result.fixed is False

    dev.send.side_effect = [
        SimpleNamespace(success=True, content='{"assessment":"simple_fix"}'),
        SimpleNamespace(success=True, content="fixed"),
    ]
    with patch.object(engine, "_cancelled", side_effect=[False, False, True]):
        result = engine._run_bug_fix(bug, "req", "design")
    assert result.fixed is False


def test_run_review_loop_cancel_and_low_info_verdict_paths():
    engine, _, dev, qas = _make_engine(max_review_iterations=1)
    qa = qas[0]

    with patch.object(engine, "_cancelled", return_value=True):
        artifact, iters, passed = engine._run_review_loop("artifact", "design")
    assert passed is False

    qa.send.return_value = SimpleNamespace(
        success=False,
        content="boom",
        exit_code=1,
        raw_stderr="stderr text",
    )
    with patch("myswat.workflow.engine.console.print") as mock_print:
        artifact, iters, passed = engine._run_review_loop(
            "artifact",
            "design",
            reviewers=[qa],
            abort_on_agent_failure=False,
        )
    assert any("stderr text" in str(call) for call in mock_print.call_args_list)

    qa.send.return_value = SimpleNamespace(
        success=True,
        content=json.dumps({"verdict": "lgtm", "issues": [], "summary": ""}),
        exit_code=0,
        raw_stderr="",
    )
    artifact, iters, passed = engine._run_review_loop("artifact", "design", reviewers=[qa])
    assert passed is True

    qa.send.return_value = SimpleNamespace(
        success=True,
        content=json.dumps({"verdict": "changes_requested", "issues": ["fix"]}),
        exit_code=0,
        raw_stderr="",
    )
    dev.send.return_value = SimpleNamespace(success=True, content="revised artifact")
    with patch.object(engine, "_cancelled", side_effect=[False, False, True]):
        artifact, iters, passed = engine._run_review_loop("artifact", "design", reviewers=[qa])
    assert passed is False

    qa2 = make_fake_session_manager(agent_id=99, agent_role="qa-2")
    qa.send.return_value = SimpleNamespace(success=False, content="boom1", exit_code=1, raw_stderr="")
    qa2.send.return_value = SimpleNamespace(success=False, content="boom2", exit_code=1, raw_stderr="")
    artifact, iters, passed = engine._run_review_loop(
        "artifact",
        "design",
        reviewers=[qa, qa2],
        abort_on_agent_failure=True,
    )
    assert passed is False

    qa.send.return_value = SimpleNamespace(
        success=True,
        content=json.dumps({"verdict": "changes_requested", "issues": ["fix"]}),
        exit_code=0,
        raw_stderr="",
    )
    dev.send.return_value = SimpleNamespace(success=True, content="revised artifact")
    with patch.object(engine, "_cancelled", side_effect=[False, True]):
        artifact, iters, passed = engine._run_review_loop("artifact", "design", reviewers=[qa])
    assert passed is False


def test_generate_report_includes_ga_outcomes_bug_fixes_and_test_report():
    engine, _, _, _ = _make_engine(mode=WorkMode.full)

    aborted = WorkflowResult(
        requirement="req",
        ga_test=GATestResult(aborted=True, bugs_found=6, bugs_fixed=2),
    )
    report = engine._generate_report(aborted, [])
    assert "**ABORTED**" in report

    passed = WorkflowResult(
        requirement="req",
        ga_test=GATestResult(
            passed=True,
            bug_fixes=[SimpleNamespace(title="Bug 1", arch_change=False, fixed=True, summary="fixed it")],
            test_report="all green",
        ),
    )
    report = engine._generate_report(passed, [])
    assert "**PASSED**" in report
    assert "Bug Fixes" in report
    assert "all green" in report

    test_engine, _, _, _ = _make_engine(mode=WorkMode.test)
    aborted = WorkflowResult(
        requirement="req",
        ga_test=GATestResult(aborted=True, bugs_found=6, bugs_fixed=2),
    )
    report = test_engine._generate_report(aborted, [])
    assert "**ABORTED**" in report

    passed = WorkflowResult(
        requirement="req",
        ga_test=GATestResult(
            passed=True,
            bug_fixes=[SimpleNamespace(title="Bug 1", arch_change=False, fixed=True, summary="fixed it")],
            test_report="all green",
        ),
    )
    report = test_engine._generate_report(passed, [])
    assert "**PASSED**" in report
    assert "Bug Fixes" in report
    assert "all green" in report


def test_testplan_design_mode_rehydrates_ga_result_when_patched_none():
    arch = make_fake_session_manager(agent_id=30, agent_role="architect", session_id=300)
    engine, _, _, qas = _make_engine(mode=WorkMode.testplan_design, arch_sm=arch)
    qas[0].send.return_value = SimpleNamespace(success=True, content="plan")
    result = WorkflowResult(requirement="req")
    hydrated = SimpleNamespace(test_plan="", test_plan_review_iterations=0, test_plan_review_passed=False)

    with patch.object(engine, "_cancelled", return_value=False):
        with patch.object(engine, "_run_review_loop", return_value=("reviewed plan", 2, True)):
            with patch.object(engine, "_user_checkpoint", return_value="approved plan"):
                with patch("myswat.workflow.engine.GATestResult", side_effect=[None, hydrated]):
                    with patch.object(engine, "_generate_report", return_value="# report"):
                        result = engine._run_testplan_design_mode("req", result)

    assert result.ga_test is hydrated
    assert result.ga_test.test_plan == "approved plan"
