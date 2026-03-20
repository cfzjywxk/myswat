"""Tests for the queued, MCP-oriented workflow kernel."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from myswat.server.contracts import ReviewVerdictEnvelope, StageRunCompletion
from myswat.workflow.kernel import PhaseResult, WorkflowKernel, _extract_json_block
from myswat.workflow.modes import WorkMode
from myswat.workflow.runtime import WorkflowRuntime


def _long_text(marker: str, *, repeats: int = 1600) -> str:
    return ("0123456789" * repeats) + marker


def _participant(agent_id: int, role: str) -> WorkflowRuntime:
    return WorkflowRuntime(
        agent_row={
            "id": agent_id,
            "role": role,
            "display_name": role,
        }
    )


def _service():
    service = Mock()
    stage_ids = iter(range(100, 200))
    cycle_ids = iter(range(2000, 2100))
    service.start_stage_run.side_effect = lambda request: SimpleNamespace(stage_run_id=next(stage_ids))
    service.request_review.side_effect = lambda request: SimpleNamespace(cycle_id=next(cycle_ids))
    service.report_status.return_value = {}
    return service


def _store():
    store = Mock()
    store.get_latest_artifact_by_type.return_value = None
    store.list_artifacts.return_value = []
    store.get_latest_stage_run.return_value = None
    return store


def test_design_mode_queues_design_and_plan_and_waits_for_reviews():
    store = _store()
    service = _service()
    arch = _participant(30, "architect")
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=7,
            stage_name="design",
            status="completed",
            summary="Design drafted.",
            artifact_id=1000,
            artifact_content="# Design\nUse iterative review.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=7,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1001,
            artifact_content="Phase 1: Initial implementation\nDo the work.\n",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="developer",
                verdict="lgtm",
                summary="Looks implementable.",
            ),
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Design is testable.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2002,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan is scoped correctly.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        arch=arch,
        project_id=1,
        work_item_id=7,
        mode=WorkMode.design,
        auto_approve=True,
    )

    result = kernel.run("Implement staged coordination")

    assert result.success is True
    assert result.design.startswith("# Design")
    assert "Phase 1" in result.plan
    assert service.start_stage_run.call_count == 2
    assert service.request_review.call_count == 3
    assert service.wait_for_review_verdicts.call_count == 2


def test_develop_mode_queues_plan_phase_and_final_report():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=8,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1000,
            artifact_content="Phase 1: Implement kernel\nDo the change.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=8,
            stage_name="phase_1",
            status="completed",
            summary="Phase implementation complete.",
            artifact_id=1001,
            artifact_content="Summary of phase 1",
        ),
        StageRunCompletion(
            stage_run_id=102,
            work_item_id=8,
            stage_name="report",
            status="completed",
            summary="Final report generated.",
            artifact_id=1002,
            artifact_content="Final report text",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan approved.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Code approved.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=8,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    result = kernel.run("Implement the new workflow kernel")

    assert result.success is True
    assert len(result.phases) == 1
    assert result.phases[0].committed is True
    assert result.final_report == "Final report text"
    assert service.start_stage_run.call_count == 3
    assert service.request_review.call_count == 2


def test_kernel_keeps_split_review_limits_separate():
    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        arch=_participant(30, "architect"),
        project_id=1,
        work_item_id=7,
        design_plan_review_limit=1,
        dev_plan_review_limit=2,
        dev_code_review_limit=3,
        ga_plan_review_limit=4,
        ga_test_review_limit=5,
    )

    assert kernel._review_limit_for("design_review") == 1
    assert kernel._review_limit_for("plan_review") == 2
    assert kernel._review_limit_for("phase_1_review") == 3
    assert kernel._review_limit_for("test_plan_review") == 4
    assert kernel._review_limit_for("ga_test_review") == 5
    assert kernel._review_limit_for("misc_review") == 5
    assert kernel._max_review == 5


def test_design_stage_and_review_prompts_include_requirement_and_design_context():
    store = _store()
    service = _service()
    arch = _participant(30, "architect")
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=9,
            stage_name="design",
            status="completed",
            summary="Design drafted.",
            artifact_id=1000,
            artifact_content="# Design\nUse iterative fibonacci.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=9,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1001,
            artifact_content="Phase 1: Implement fibonacci\nDo the work.\n",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="developer",
                verdict="lgtm",
                summary="Looks implementable.",
            ),
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Design is testable.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2002,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan is scoped correctly.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        arch=arch,
        project_id=1,
        work_item_id=9,
        mode=WorkMode.design,
        auto_approve=True,
    )

    kernel.run("design and implement a fibonacci sequence generator in rust")

    design_stage = service.start_stage_run.call_args_list[0].args[0]
    assert design_stage.stage_name == "design"
    assert "fibonacci sequence generator in rust" in design_stage.task_prompt
    assert "Do NOT delegate this task" in design_stage.task_prompt
    assert design_stage.artifact_type == "design_doc"

    design_review_requests = [call.args[0] for call in service.request_review.call_args_list[:2]]
    assert all(request.stage == "design_review" for request in design_review_requests)
    assert all("Requirement:" in request.task_prompt for request in design_review_requests)
    assert all("# Design\nUse iterative fibonacci." in request.task_prompt for request in design_review_requests)
    assert all("Output ONLY a JSON object" in request.task_prompt for request in design_review_requests)

    plan_stage = service.start_stage_run.call_args_list[1].args[0]
    assert plan_stage.stage_name == "plan"
    assert "The following design has been approved" in plan_stage.task_prompt
    assert "# Design\nUse iterative fibonacci." in plan_stage.task_prompt


def test_plan_review_feedback_builds_revision_prompt_with_collected_issues():
    store = _store()
    store.get_latest_artifact_by_type.side_effect = lambda work_item_id, artifact_type: (
        {"id": 950, "content": "# Design\nUse iterative fibonacci with Rust tests.\n"}
        if artifact_type == "design_doc"
        else None
    )
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=10,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1000,
            artifact_content="Phase 1: Implement fibonacci\nAdd tests.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=10,
            stage_name="plan",
            status="completed",
            summary="Plan revised.",
            artifact_id=1001,
            artifact_content="Phase 1: Implement fibonacci\nAdd unit and edge-case tests.\n",
        ),
        StageRunCompletion(
            stage_run_id=102,
            work_item_id=10,
            stage_name="report",
            status="completed",
            summary="Final report generated.",
            artifact_id=1002,
            artifact_content="Done.",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="qa_main",
                verdict="changes_requested",
                issues=["Split implementation and test work more clearly."],
                summary="Needs finer phase boundaries.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan is scoped correctly.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=10,
        mode=WorkMode.design,
        auto_approve=True,
    )

    kernel.run("implement a fibonacci sequence generator in rust")

    revised_plan_stage = service.start_stage_run.call_args_list[1].args[0]
    assert revised_plan_stage.stage_name == "plan"
    assert "Address the following review comments on your implementation plan." in revised_plan_stage.task_prompt
    assert "Split implementation and test work more clearly." in revised_plan_stage.task_prompt
    assert "Phase 1: Implement fibonacci" in revised_plan_stage.task_prompt


def test_review_loop_reaching_max_iterations_records_skip_without_blocking():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_review_verdicts.return_value = [
        ReviewVerdictEnvelope(
            cycle_id=2000,
            reviewer_role="qa_main",
            verdict="changes_requested",
            summary="Need logging coverage.",
            issues=["Add coverage for log_filter."],
        ),
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=10,
        mode=WorkMode.develop,
        auto_approve=True,
        max_review_iterations=1,
    )

    reviewed, iterations, passed = kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="Phase 1: Ship it",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )

    assert reviewed == "Phase 1: Ship it"
    assert iterations == 1
    assert passed is False
    assert kernel._blocked is False
    assert kernel._last_review_limit_reached is True
    assert kernel._last_review_limit_stage == "plan_review"
    assert "Max review iterations reached for Implementation plan" in kernel._last_review_limit_summary
    assert any(
        call.kwargs.get("current_stage") == "plan_review_skipped"
        for call in store.update_work_item_state.call_args_list
    )
    assert any(
        call.kwargs.get("event_type") == "review_skipped"
        for call in store.append_work_item_process_event.call_args_list
    )


def test_phase_and_code_review_prompts_include_requirement_design_plan_and_summary():
    store = _store()
    store.get_latest_artifact_by_type.side_effect = lambda work_item_id, artifact_type: (
        {"id": 900, "content": "# Design\nUse iterative fibonacci with Rust tests.\n"}
        if artifact_type == "design_doc"
        else None
    )
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=11,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1000,
            artifact_content="Phase 1: Implement fibonacci\nCode and unit tests.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=11,
            stage_name="phase_1",
            status="completed",
            summary="Phase implementation complete.",
            artifact_id=1001,
            artifact_content="Implemented fibonacci generator and unit tests.",
        ),
        StageRunCompletion(
            stage_run_id=102,
            work_item_id=11,
            stage_name="report",
            status="completed",
            summary="Final report generated.",
            artifact_id=1002,
            artifact_content="Final report text",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan approved.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Code approved.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=11,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    kernel.run("implement a fibonacci sequence generator in rust")

    phase_stage = service.start_stage_run.call_args_list[1].args[0]
    assert phase_stage.stage_name == "phase_1"
    assert "implementing phase 1 of 1" in phase_stage.task_prompt.lower()
    assert "Approved Design (brief)" in phase_stage.task_prompt
    assert "# Design\nUse iterative fibonacci with Rust tests." in phase_stage.task_prompt
    assert "Phase 1: Implement fibonacci" in phase_stage.task_prompt
    assert "Implement ONLY this phase" in phase_stage.task_prompt

    code_review_request = service.request_review.call_args_list[1].args[0]
    assert code_review_request.stage == "phase_1_review"
    assert "You are a QA engineer reviewing a development phase." in code_review_request.task_prompt
    assert "Developer's Summary (iteration 1)" in code_review_request.task_prompt
    assert "Implemented fibonacci generator and unit tests." in code_review_request.task_prompt
    assert "Approved Design" in code_review_request.task_prompt
    assert "Plan:" in code_review_request.task_prompt


def test_large_design_body_survives_design_review_and_plan_prompt_building():
    store = _store()
    service = _service()
    arch = _participant(30, "architect")
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    large_design = "# Design\n" + _long_text("DESIGN-TAIL-MARKER")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=13,
            stage_name="design",
            status="completed",
            summary="Design drafted.",
            artifact_id=1000,
            artifact_content=large_design,
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=13,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1001,
            artifact_content="Phase 1: Implement fibonacci\nDo the work.\n",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="developer",
                verdict="lgtm",
                summary="Looks implementable.",
            ),
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Design is testable.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2002,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan is scoped correctly.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        arch=arch,
        project_id=1,
        work_item_id=13,
        mode=WorkMode.design,
        auto_approve=True,
    )

    kernel.run("design and implement a fibonacci sequence generator in rust")

    design_review_request = service.request_review.call_args_list[0].args[0]
    plan_stage = service.start_stage_run.call_args_list[1].args[0]
    assert "DESIGN-TAIL-MARKER" in design_review_request.task_prompt
    assert "DESIGN-TAIL-MARKER" in plan_stage.task_prompt


def test_large_plan_and_phase_summary_survive_phase_and_code_review_prompt_building():
    store = _store()
    large_design = "# Design\n" + _long_text("DESIGN-TAIL-MARKER")
    store.get_latest_artifact_by_type.side_effect = lambda work_item_id, artifact_type: (
        {"id": 900, "content": large_design}
        if artifact_type == "design_doc"
        else None
    )
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    large_plan = "Phase 1: Implement fibonacci\n" + _long_text("PLAN-TAIL-MARKER")
    large_phase_summary = "Implemented fibonacci generator.\n" + _long_text("PHASE-SUMMARY-TAIL-MARKER")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=14,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1000,
            artifact_content=large_plan,
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=14,
            stage_name="phase_1",
            status="completed",
            summary="Phase implementation complete.",
            artifact_id=1001,
            artifact_content=large_phase_summary,
        ),
        StageRunCompletion(
            stage_run_id=102,
            work_item_id=14,
            stage_name="report",
            status="completed",
            summary="Final report generated.",
            artifact_id=1002,
            artifact_content="Final report text",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan approved.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Code approved.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=14,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    kernel.run("implement a fibonacci sequence generator in rust")

    phase_stage = service.start_stage_run.call_args_list[1].args[0]
    code_review_request = service.request_review.call_args_list[1].args[0]
    assert "DESIGN-TAIL-MARKER" in phase_stage.task_prompt
    assert "PLAN-TAIL-MARKER" in phase_stage.task_prompt
    assert "PLAN-TAIL-MARKER" in code_review_request.task_prompt
    assert "PHASE-SUMMARY-TAIL-MARKER" in code_review_request.task_prompt


def test_test_mode_continues_after_review_limit_and_can_still_finish():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=15,
            stage_name="test_plan",
            status="completed",
            summary="Test plan drafted.",
            artifact_id=1000,
            artifact_content="# GA Plan\nRun tests.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=15,
            stage_name="ga_test",
            status="completed",
            summary="GA tests passed.",
            artifact_id=1001,
            artifact_content='{"status":"pass","summary":"All tests passed."}',
        ),
        StageRunCompletion(
            stage_run_id=102,
            work_item_id=15,
            stage_name="report",
            status="completed",
            summary="Final report generated.",
            artifact_id=1002,
            artifact_content="Final report text",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="developer",
                verdict="changes_requested",
                summary="Need log_filter coverage.",
                issues=["Cover the implemented log_filter setting."],
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=15,
        mode=WorkMode.test,
        auto_approve=True,
        max_review_iterations=1,
    )

    result = kernel.run("implement an echo server")

    assert result.success is True
    assert result.blocked is False
    assert result.failure_summary == ""
    assert result.ga_test is not None
    assert result.ga_test.passed is True
    assert result.final_report == "Final report text"
    assert service.start_stage_run.call_count == 3
    assert service.request_review.call_count == 1
    assert any(
        call.kwargs.get("event_type") == "review_skipped"
        for call in store.append_work_item_process_event.call_args_list
    )


def test_full_mode_can_skip_ga_test_by_request():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    arch = _participant(30, "architect")

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        arch=arch,
        project_id=1,
        work_item_id=16,
        mode=WorkMode.full,
        skip_ga_test=True,
        auto_approve=True,
    )

    with patch.object(kernel, "_run_design", return_value=("design", True)):
        with patch.object(kernel, "_run_plan", return_value=("plan", True)):
            with patch.object(kernel, "_parse_phases", return_value=["Implementation"]):
                with patch.object(
                    kernel,
                    "_run_phase",
                    return_value=PhaseResult(name="Implementation", summary="done", committed=True),
                ):
                    with patch.object(kernel, "_run_test") as mock_run_test:
                        with patch.object(kernel, "_generate_final_report", return_value="report") as mock_report:
                            result = kernel.run("ship it")

    mock_run_test.assert_not_called()
    assert "GA test skipped by request." in mock_report.call_args.args[0]
    store.update_work_item_state.assert_any_call(
        16,
        current_stage="ga_test_skipped",
        latest_summary="GA test skipped by request.",
        next_todos=["Generate final report"],
    )
    assert any(
        call.kwargs.get("event_type") == "ga_test_skipped"
        for call in store.append_work_item_process_event.call_args_list
    )
    assert result.success is True
    assert result.ga_test is None
    assert result.final_report == "report"


def test_code_review_feedback_builds_phase_revision_prompt():
    store = _store()
    store.get_latest_artifact_by_type.side_effect = lambda work_item_id, artifact_type: (
        {"id": 901, "content": "# Design\nUse iterative fibonacci with Rust tests.\n"}
        if artifact_type == "design_doc"
        else None
    )
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    service.wait_for_stage_run_completion.side_effect = [
        StageRunCompletion(
            stage_run_id=100,
            work_item_id=12,
            stage_name="plan",
            status="completed",
            summary="Plan drafted.",
            artifact_id=1000,
            artifact_content="Phase 1: Implement fibonacci\nCode and unit tests.\n",
        ),
        StageRunCompletion(
            stage_run_id=101,
            work_item_id=12,
            stage_name="phase_1",
            status="completed",
            summary="Phase implementation complete.",
            artifact_id=1001,
            artifact_content="Implemented fibonacci generator and basic tests.",
        ),
        StageRunCompletion(
            stage_run_id=102,
            work_item_id=12,
            stage_name="phase_1",
            status="completed",
            summary="Phase revised after review.",
            artifact_id=1002,
            artifact_content="Implemented fibonacci generator and expanded edge-case tests.",
        ),
        StageRunCompletion(
            stage_run_id=103,
            work_item_id=12,
            stage_name="report",
            status="completed",
            summary="Final report generated.",
            artifact_id=1003,
            artifact_content="Final report text",
        ),
    ]
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Plan approved.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="qa_main",
                verdict="changes_requested",
                issues=["Add edge-case coverage for n=0 and n=1."],
                summary="Need better boundary tests.",
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2002,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Code approved.",
            ),
        ],
    ]

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    kernel.run("implement a fibonacci sequence generator in rust")

    revised_phase_stage = service.start_stage_run.call_args_list[2].args[0]
    assert revised_phase_stage.stage_name == "phase_1"
    assert "Address the following code review comments." in revised_phase_stage.task_prompt
    assert "Implemented fibonacci generator and basic tests." in revised_phase_stage.task_prompt
    assert "Add edge-case coverage for n=0 and n=1." in revised_phase_stage.task_prompt
    assert "Fix each issue raised in the codebase." in revised_phase_stage.task_prompt


def test_extract_json_block_resolves_externalized_json_fields():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as summary_handle:
        summary_handle.write("All fibonacci checks passed.\n")
        summary_path = summary_handle.name
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as bug_handle:
        bug_handle.write("No bugs.\n")
        bug_path = bug_handle.name

    payload = _extract_json_block(
        "{"
        '"status":"pass",'
        f'"summary":"See `{summary_path}`",'
        f'"bugs":["See `{bug_path}`"]'
        "}"
    )

    assert payload == {
        "status": "pass",
        "summary": "All fibonacci checks passed.\n",
        "bugs": ["No bugs.\n"],
    }
    Path(summary_path).unlink(missing_ok=True)
    Path(bug_path).unlink(missing_ok=True)


def test_extract_json_block_handles_plain_code_fence_and_externalized_json_text():
    assert _extract_json_block("Review:\n```\n{\"ok\": true}\n```") == {"ok": True}
    assert _extract_json_block("```json\n{\"ok\": true}\n```") == {"ok": True}
    assert _extract_json_block("{broken}") is None

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write('{"status":"pass"}\n')
        path = handle.name
    try:
        assert _extract_json_block(f"The detailed response is in `{path}`.") == {"status": "pass"}
    finally:
        Path(path).unlink(missing_ok=True)


def test_kernel_requires_coordinator_emits_events_and_uses_stage_index_fallbacks():
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    try:
        WorkflowKernel(
            store=_store(),
            dev=dev,
            qas=[qa],
            project_id=1,
            work_item_id=1,
            mode=WorkMode.full,
        )
    except ValueError as exc:
        assert "coordinator is required" in str(exc)
    else:
        raise AssertionError("Expected missing coordinator to raise ValueError")

    events = []
    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=1,
        mode=WorkMode.full,
        on_event=events.append,
    )
    kernel._emit("warning", "hello", stage="phase_x", agent_role="developer", detail="detail", count=2)

    assert events[0].event_type == "warning"
    assert events[0].message == "hello"
    assert kernel._stage_index("phase_bad") == 700
    assert kernel._stage_index("other") == 1000


def test_wait_for_stage_result_failure_marks_kernel_blocked():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=7,
        stage_name="plan",
        status="failed",
        summary="plan failed",
        artifact_id=None,
        artifact_content="",
    )
    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=7,
        mode=WorkMode.full,
        auto_approve=True,
    )

    result = kernel._wait_for_stage_result(stage_run_id=100, stage_name="plan", owner=dev)

    assert result.status == "failed"
    assert kernel._blocked is True
    assert kernel._failure_summary == "plan failed"


def test_checkpoint_handles_user_rejection_manual_changes_and_revision_outcomes():
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    reject_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=8,
        mode=WorkMode.full,
        auto_approve=False,
        ask_user=lambda _prompt: "n",
    )
    artifact, ok = reject_kernel._checkpoint(
        "draft",
        prompt="continue?",
        stage="design",
        owner=dev,
        focus="ctx",
        artifact_type="design_doc",
        artifact_title="Technical design",
    )
    assert artifact == "draft"
    assert ok is False
    assert reject_kernel._failure_summary == "User rejected design checkpoint."

    no_builder_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=8,
        mode=WorkMode.full,
        auto_approve=False,
        ask_user=lambda _prompt: "please revise",
    )
    artifact, ok = no_builder_kernel._checkpoint(
        "draft",
        prompt="continue?",
        stage="plan",
        owner=dev,
        focus="ctx",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
    )
    assert artifact == "draft"
    assert ok is False
    assert "User requested changes during plan" in no_builder_kernel._failure_summary

    fail_store = _store()
    fail_store.get_latest_stage_run.return_value = SimpleNamespace(iteration=2)
    fail_service = _service()
    fail_service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=8,
        stage_name="plan",
        status="failed",
        summary="revision failed",
        artifact_id=None,
        artifact_content="",
    )
    fail_kernel = WorkflowKernel(
        store=fail_store,
        service=fail_service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=8,
        mode=WorkMode.full,
        auto_approve=False,
        ask_user=lambda _prompt: "please revise",
    )
    artifact, ok = fail_kernel._checkpoint(
        "draft",
        prompt="continue?",
        stage="plan",
        owner=dev,
        focus="ctx",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        revision_prompt_builder=lambda artifact, feedback: f"{artifact}\n{feedback}",
    )
    assert artifact == "draft"
    assert ok is False
    revision_request = fail_service.start_stage_run.call_args.args[0]
    assert revision_request.iteration == 3

    success_store = _store()
    success_store.get_latest_stage_run.return_value = SimpleNamespace(iteration=1)
    success_service = _service()
    success_service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=8,
        stage_name="design",
        status="completed",
        summary="revision done",
        artifact_id=1000,
        artifact_content="revised artifact",
    )
    success_kernel = WorkflowKernel(
        store=success_store,
        service=success_service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=8,
        mode=WorkMode.full,
        auto_approve=False,
        ask_user=lambda _prompt: "please revise",
    )
    artifact, ok = success_kernel._checkpoint(
        "draft",
        prompt="continue?",
        stage="design",
        owner=dev,
        focus="ctx",
        artifact_type="design_doc",
        artifact_title="Technical design",
        revision_prompt_builder=lambda artifact, feedback: f"{artifact}\n{feedback}",
    )
    assert artifact == "revised artifact"
    assert ok is True

    approve_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=8,
        mode=WorkMode.full,
        auto_approve=False,
        ask_user=lambda _prompt: "y",
    )
    artifact, ok = approve_kernel._checkpoint(
        "draft",
        prompt="continue?",
        stage="design",
        owner=dev,
        focus="ctx",
        artifact_type="design_doc",
        artifact_title="Technical design",
    )
    assert artifact == "draft"
    assert ok is True


def test_review_loop_handles_cancel_missing_artifact_revision_failure_and_zero_limit():
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    cancel_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=9,
        mode=WorkMode.full,
        auto_approve=True,
        should_cancel=lambda: True,
    )
    artifact, iterations, passed = cancel_kernel._run_review_loop(
        owner_stage_name="design",
        review_stage_name="design_review",
        artifact_type="design_doc",
        artifact_title="Technical design",
        initial_artifact="draft",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )
    assert (artifact, iterations, passed) == ("draft", 1, False)
    assert cancel_kernel._failure_summary == "Workflow cancelled."

    missing_store = _store()
    missing_store.get_latest_artifact_by_type.return_value = None
    missing_kernel = WorkflowKernel(
        store=missing_store,
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=9,
        mode=WorkMode.full,
        auto_approve=True,
    )
    artifact, iterations, passed = missing_kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="draft",
        initial_artifact_id=None,
        owner=dev,
        reviewers=[qa],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )
    assert (artifact, iterations, passed) == ("draft", 1, False)
    assert missing_kernel._failure_summary == "Missing artifact for plan_review."

    revision_service = _service()
    revision_service.wait_for_review_verdicts.return_value = [
        ReviewVerdictEnvelope(
            cycle_id=2000,
            reviewer_role="qa_main",
            verdict="changes_requested",
            summary="Need more tests.",
            issues=["Add tests."],
        ),
    ]
    revision_service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=9,
        stage_name="plan",
        status="failed",
        summary="revision failed",
        artifact_id=None,
        artifact_content="",
    )
    revision_kernel = WorkflowKernel(
        store=_store(),
        service=revision_service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=9,
        mode=WorkMode.full,
        auto_approve=True,
    )
    artifact, iterations, passed = revision_kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="draft",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )
    assert (artifact, iterations, passed) == ("draft", 1, False)

    exhausted_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=9,
        mode=WorkMode.full,
        auto_approve=True,
        max_review_iterations=0,
    )
    artifact, iterations, passed = exhausted_kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="draft",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )
    assert (artifact, iterations, passed) == ("draft", 0, False)
    assert exhausted_kernel._failure_summary == "Implementation plan review loop exhausted."


def test_kernel_parses_completed_phase_rows_and_phase_names():
    store = _store()
    store.list_artifacts.return_value = [
        {
            "artifact_type": "phase_result",
            "iteration": 2,
            "title": "ignored",
            "content": "beta summary",
            "metadata_json": '{"phase_index":2,"phase_name":"Beta","review_iterations":3,"review_passed":false,"committed":false}',
        },
        {
            "artifact_type": "phase_result",
            "iteration": 1,
            "title": "Fallback",
            "content": "alpha summary",
            "metadata_json": "{",
        },
    ]
    kernel = WorkflowKernel(
        store=store,
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=10,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    results = kernel._load_completed_phase_results()

    assert [result.name for result in results] == ["Fallback", "Beta"]
    assert results[1].review_iterations == 3
    assert results[1].review_passed is False
    assert results[1].committed is False
    assert kernel._parse_phases("\nPhase Alpha\nStep 2) Ship\n") == ["Alpha", "Ship"]

    store.list_artifacts.return_value = [
        {
            "artifact_type": "phase_result",
            "iteration": 1,
            "title": "DictMeta",
            "content": "summary",
            "metadata_json": {
                "phase_index": 1,
                "phase_name": "Dict Name",
                "review_iterations": 4,
                "review_passed": True,
                "committed": True,
            },
        },
        {
            "artifact_type": "phase_result",
            "iteration": 2,
            "title": "OtherMeta",
            "content": "summary",
            "metadata_json": 7,
        },
    ]
    results = kernel._load_completed_phase_results()
    assert [result.name for result in results] == ["Dict Name", "OtherMeta"]


def test_kernel_generates_report_fallback_and_run_failure_states():
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    report_service = _service()
    report_service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=11,
        stage_name="report",
        status="failed",
        summary="report failed",
        artifact_id=None,
        artifact_content="",
    )
    report_kernel = WorkflowKernel(
        store=_store(),
        service=report_service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=11,
        mode=WorkMode.develop,
        auto_approve=True,
    )
    assert report_kernel._generate_final_report(["Phase 1: done"]) == "Phase 1: done"

    with patch.object(WorkflowKernel, "_run_design", return_value=("", False)):
        design_result = WorkflowKernel(
            store=_store(),
            service=_service(),
            dev=dev,
            qas=[qa],
            project_id=1,
            work_item_id=11,
            mode=WorkMode.full,
            auto_approve=True,
        ).run("req")
    assert design_result.blocked is True
    assert design_result.failure_summary == "Design stage failed."

    with patch.object(WorkflowKernel, "_run_design", return_value=("design", True)):
        with patch.object(WorkflowKernel, "_run_plan", return_value=("", False)):
            plan_result = WorkflowKernel(
                store=_store(),
                service=_service(),
                dev=dev,
                qas=[qa],
                project_id=1,
                work_item_id=11,
                mode=WorkMode.full,
                auto_approve=True,
            ).run("req")
    assert plan_result.blocked is True
    assert plan_result.failure_summary == "Planning stage failed."

    with patch.object(WorkflowKernel, "_run_phase", return_value=SimpleNamespace(summary="phase failed", committed=False)):
        phase_kernel = WorkflowKernel(
            store=_store(),
            service=_service(),
            dev=dev,
            qas=[qa],
            project_id=1,
            work_item_id=11,
            mode=WorkMode.develop,
            auto_approve=True,
        )
        with patch.object(phase_kernel, "_load_latest_artifact", return_value="plan"):
            with patch.object(phase_kernel, "_parse_phases", return_value=["Implementation"]):
                phase_result = phase_kernel.run("req")
    assert phase_result.blocked is True
    assert phase_result.failure_summary == "Phase 1 failed."

    ga_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=11,
        mode=WorkMode.test,
        auto_approve=True,
    )
    with patch.object(ga_kernel, "_run_test", side_effect=lambda requirement, design, completed: setattr(ga_kernel, "_blocked", True) or setattr(ga_kernel, "_failure_summary", "ga failed") or SimpleNamespace(passed=False)):
        ga_result = ga_kernel.run("req")
    assert ga_result.blocked is True
    assert ga_result.failure_summary == "ga failed"

    cancelled_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=11,
        mode=WorkMode.full,
        auto_approve=True,
        should_cancel=lambda: True,
    )
    cancelled_result = cancelled_kernel.run("req")
    assert cancelled_result.blocked is True
    assert cancelled_result.final_report == "Workflow cancelled before start."


def test_run_design_plan_phase_and_test_branch_failures():
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")

    design_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.full,
        auto_approve=True,
    )
    with patch.object(design_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="failed")):
        assert design_kernel._run_design("req") == ("", False)
    with patch.object(design_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="completed", artifact_content="draft", artifact_id=1000)):
        with patch.object(design_kernel, "_run_review_loop", return_value=("draft", 1, False)):
            design_kernel._last_review_limit_reached = False
            assert design_kernel._run_design("req") == ("draft", False)

    plan_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.full,
        auto_approve=True,
    )
    with patch.object(plan_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="failed")):
        assert plan_kernel._run_plan("req", "design") == ("", False)
    with patch.object(plan_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="completed", artifact_content="plan", artifact_id=1000)):
        with patch.object(plan_kernel, "_run_review_loop", return_value=("plan", 1, False)):
            plan_kernel._last_review_limit_reached = False
            assert plan_kernel._run_plan("req", "design") == ("plan", False)

    phase_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.develop,
        auto_approve=True,
    )
    with patch.object(phase_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="failed")):
        phase_result = phase_kernel._run_phase(
            requirement="req",
            design="design",
            plan="plan",
            phase_name="Implementation",
            phase_index=1,
            total_phases=1,
            completed_summaries=[],
        )
    assert phase_result.summary == "Phase 1 failed."
    with patch.object(phase_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="completed", artifact_content="summary", artifact_id=1000)):
        with patch.object(phase_kernel, "_run_review_loop", return_value=("summary", 1, False)):
            phase_kernel._last_review_limit_reached = False
            phase_result = phase_kernel._run_phase(
                requirement="req",
                design="design",
                plan="plan",
                phase_name="Implementation",
                phase_index=1,
                total_phases=1,
                completed_summaries=[],
            )
    assert phase_result.committed is False
    assert phase_result.review_passed is False

    test_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.test,
        auto_approve=True,
    )
    with patch.object(test_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="failed")):
        ga_result = test_kernel._run_test("req", "design", [])
    assert ga_result.test_plan == ""
    assert ga_result.test_report == ""
    assert ga_result.passed is False

    review_fail_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.test,
        auto_approve=True,
    )
    with patch.object(review_fail_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="completed", artifact_content="plan", artifact_id=1000)):
        with patch.object(review_fail_kernel, "_run_review_loop", return_value=("plan", 1, False)):
            review_fail_kernel._last_review_limit_reached = False
            ga_result = review_fail_kernel._run_test("req", "design", [])
    assert ga_result.test_plan == "plan"
    assert ga_result.passed is False

    checkpoint_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.test,
        auto_approve=True,
    )
    with patch.object(checkpoint_kernel, "_wait_for_stage_result", return_value=SimpleNamespace(status="completed", artifact_content="plan", artifact_id=1000)):
        with patch.object(checkpoint_kernel, "_run_review_loop", return_value=("plan", 1, True)):
            with patch.object(checkpoint_kernel, "_checkpoint", return_value=("plan", False)):
                ga_result = checkpoint_kernel._run_test("req", "design", [])
    assert ga_result.test_plan == "plan"
    assert ga_result.passed is False

    execute_fail_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.test,
        auto_approve=True,
    )
    execute_fail_kernel._last_review_limit_reached = False
    with patch.object(
        execute_fail_kernel,
        "_wait_for_stage_result",
        side_effect=[
            SimpleNamespace(status="completed", artifact_content="plan", artifact_id=1000),
            SimpleNamespace(status="failed", artifact_content="", artifact_id=None),
        ],
    ):
        with patch.object(execute_fail_kernel, "_run_review_loop", return_value=("plan", 1, True)):
            with patch.object(execute_fail_kernel, "_checkpoint", return_value=("plan", True)):
                ga_result = execute_fail_kernel._run_test("req", "design", [])
    assert ga_result.test_plan == "plan"
    assert ga_result.passed is False

    payload_fail_kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=12,
        mode=WorkMode.test,
        auto_approve=True,
    )
    payload_fail_kernel._last_review_limit_reached = False
    with patch.object(
        payload_fail_kernel,
        "_wait_for_stage_result",
        side_effect=[
            SimpleNamespace(status="completed", artifact_content="plan", artifact_id=1000),
            SimpleNamespace(status="completed", artifact_content="not-json", artifact_id=1001, summary="ga failed"),
        ],
    ):
        with patch.object(payload_fail_kernel, "_run_review_loop", return_value=("plan", 1, True)):
            with patch.object(payload_fail_kernel, "_checkpoint", return_value=("plan", True)):
                ga_result = payload_fail_kernel._run_test("req", "design", [])
    assert ga_result.passed is False
    assert payload_fail_kernel._blocked is True
    assert payload_fail_kernel._failure_summary == "ga failed"
