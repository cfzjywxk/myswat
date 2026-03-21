"""Tests for the queued, MCP-oriented workflow kernel."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from myswat.repo_ops import GitCommitResult, GitProbeResult, GitPushResult
from myswat.server.contracts import ReviewCycleCancellationRequest, ReviewVerdictEnvelope, StageRunCompletion
from myswat.workflow.kernel import GATestResult, PhaseResult, WorkflowKernel, _extract_json_block
from myswat.workflow.modes import WorkMode
from myswat.workflow.runtime import WorkflowRuntime


def _long_text(marker: str, *, repeats: int = 1600) -> str:
    return ("0123456789" * repeats) + marker


def _participant(
    agent_id: int,
    role: str,
    *,
    backend: str | None = None,
    model_name: str | None = None,
) -> WorkflowRuntime:
    resolved_backend = backend or ("claude" if role.startswith("qa") else "codex")
    resolved_model = model_name or ("claude-opus-4-6" if role.startswith("qa") else "gpt-5.4")
    return WorkflowRuntime(
        agent_row={
            "id": agent_id,
            "role": role,
            "display_name": role,
            "cli_backend": resolved_backend,
            "model_name": resolved_model,
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


def test_run_exports_design_plan_to_docs_and_commits_when_plan_finalized(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=81,
        mode=WorkMode.design,
        auto_approve=True,
        repo_path=str(repo_path),
    )

    with patch("myswat.workflow.kernel.probe_git_repository", return_value=GitProbeResult(True, True, True, "")):
        with patch.object(kernel, "_run_design", return_value=("approved design", True)):
            with patch.object(kernel, "_run_plan", return_value=("approved plan", True)):
                with patch.object(kernel, "_export_final_report_to_docs", return_value=True):
                    with patch(
                        "myswat.workflow.kernel.write_design_plan_doc",
                        return_value=repo_path / "myswat-design-plan.md",
                    ) as mock_write:
                        with patch(
                            "myswat.workflow.kernel.commit_repo_changes",
                            return_value=GitCommitResult(True, True, "Committed local changes."),
                        ) as mock_commit:
                            result = kernel.run("ship it")

    assert result.success is True
    mock_write.assert_called_once_with(
        repo_path.resolve(),
        requirement="ship it",
        design="approved design",
        plan="approved plan",
    )
    mock_commit.assert_called_once_with(
        repo_path.resolve(),
        message="docs: sync myswat design plan",
        paths=[repo_path / "myswat-design-plan.md"],
        trailers=["Co-Authored-By: MySwat Dev (GPT-5.4) <noreply@myswat.invalid>"],
    )


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
    assert "Default to exactly 1 phase." in plan_stage.task_prompt
    assert "Use multiple phases ONLY when the work is genuinely large" in plan_stage.task_prompt
    assert "Add Phase 2+ only if the work genuinely requires additional sequential milestones." in plan_stage.task_prompt


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


def test_review_loop_only_requeues_reviewers_without_lgtm():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa_main = _participant(20, "qa_main")
    security = _participant(21, "security")
    prompt_calls: list[tuple[int, str, str]] = []

    # _service() allocates review cycle IDs sequentially starting at 2000.
    # These verdict envelopes intentionally mirror that request order.
    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=2000,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Looks good.",
            ),
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="security",
                verdict="changes_requested",
                summary="Need rollback notes.",
                issues=["Add rollback notes."],
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2002,
                reviewer_role="security",
                verdict="lgtm",
                summary="Rollback notes added.",
            ),
        ],
    ]
    service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=10,
        stage_name="plan",
        status="completed",
        summary="Plan revised.",
        artifact_id=1001,
        artifact_content="Phase 1: Ship it with rollback notes.",
    )

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa_main, security],
        project_id=1,
        work_item_id=10,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    reviewed, iterations, passed = kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="Phase 1: Ship it",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa_main, security],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: (
            prompt_calls.append((iteration, reviewer_role, artifact)) or artifact
        ),
        revision_prompt_builder=lambda artifact, feedback: artifact + "\n" + feedback,
    )

    assert reviewed == "Phase 1: Ship it with rollback notes."
    assert iterations == 2
    assert passed is True
    assert service.request_review.call_count == 3
    queued_reviewers = [
        (request.iteration, request.reviewer_role)
        for request in (call.args[0] for call in service.request_review.call_args_list)
    ]
    assert queued_reviewers == [
        (1, "qa_main"),
        (1, "security"),
        (2, "security"),
    ]
    assert prompt_calls == [
        (1, "qa_main", "Phase 1: Ship it"),
        (1, "security", "Phase 1: Ship it"),
        (2, "security", "Phase 1: Ship it with rollback notes."),
    ]


def test_review_loop_recovers_missing_cycle_mapping_via_reviewer_role():
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa_main = _participant(20, "qa_main")
    security = _participant(21, "security")
    events = []

    service.wait_for_review_verdicts.side_effect = [
        [
            ReviewVerdictEnvelope(
                cycle_id=9999,
                reviewer_role="qa_main",
                verdict="lgtm",
                summary="Looks good.",
            ),
            ReviewVerdictEnvelope(
                cycle_id=2001,
                reviewer_role="security",
                verdict="changes_requested",
                summary="Need rollback notes.",
                issues=["Add rollback notes."],
            ),
        ],
        [
            ReviewVerdictEnvelope(
                cycle_id=2002,
                reviewer_role="security",
                verdict="lgtm",
                summary="Rollback notes added.",
            ),
        ],
    ]
    service.wait_for_stage_run_completion.return_value = StageRunCompletion(
        stage_run_id=100,
        work_item_id=10,
        stage_name="plan",
        status="completed",
        summary="Plan revised.",
        artifact_id=1001,
        artifact_content="Phase 1: Ship it with rollback notes.",
    )

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa_main, security],
        project_id=1,
        work_item_id=10,
        mode=WorkMode.develop,
        auto_approve=True,
        on_event=events.append,
    )

    reviewed, iterations, passed = kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="Phase 1: Ship it",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa_main, security],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + "\n" + feedback,
    )

    assert reviewed == "Phase 1: Ship it with rollback notes."
    assert iterations == 2
    assert passed is True
    queued_reviewers = [
        (request.iteration, request.reviewer_role)
        for request in (call.args[0] for call in service.request_review.call_args_list)
    ]
    assert queued_reviewers == [
        (1, "qa_main"),
        (1, "security"),
        (2, "security"),
    ]
    assert any(
        event.event_type == "warning"
        and "Recovered reviewer mapping" in event.message
        and event.metadata.get("cycle_id") == 9999
        for event in events
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


def test_full_mode_skips_ga_test_by_default():
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
                        with patch.object(kernel, "_generate_final_report", return_value="report"):
                            result = kernel.run("ship it")

    mock_run_test.assert_not_called()
    assert result.success is True
    assert result.ga_test is None
    assert result.final_report == "report"
    assert not any(
        call.kwargs.get("event_type") == "ga_test_skipped"
        for call in store.append_work_item_process_event.call_args_list
    )


def test_full_mode_runs_ga_test_when_requested():
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
        work_item_id=17,
        mode=WorkMode.full,
        with_ga_test=True,
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
                    with patch.object(kernel, "_run_test", return_value=GATestResult(passed=True)) as mock_run_test:
                        with patch.object(kernel, "_generate_final_report", return_value="report"):
                            result = kernel.run("ship it")

    mock_run_test.assert_called_once()
    assert result.success is True
    assert result.ga_test is not None
    assert result.ga_test.passed is True


def test_run_phase_commits_local_changes_after_lgtm(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=82,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch.object(kernel, "_queue_stage_task", return_value=100):
        with patch.object(
            kernel,
            "_wait_for_stage_result",
            return_value=StageRunCompletion(
                stage_run_id=100,
                work_item_id=82,
                stage_name="phase_1",
                status="completed",
                summary="Phase done.",
                artifact_id=1001,
                artifact_content="Updated `src/lib.rs:10` and `src/service.rs:20`",
            ),
        ):
            with patch.object(kernel, "_run_review_loop", return_value=("Phase summary", 1, True)):
                with patch.object(
                    kernel,
                    "_current_workflow_repo_paths",
                    return_value=[repo_path / "src/lib.rs", repo_path / "src/service.rs"],
                ):
                    with patch(
                        "myswat.workflow.kernel.commit_repo_changes",
                        return_value=GitCommitResult(True, True, "Committed local changes."),
                    ) as mock_commit:
                        result = kernel._run_phase(
                            requirement="ship it",
                            design="design",
                            plan="Phase 1: ship it",
                            phase_name="Ship it",
                            phase_index=1,
                            total_phases=1,
                            completed_summaries=[],
                        )

    assert result.committed is True
    mock_commit.assert_called_once_with(
        repo_path.resolve(),
        message="phase 1: Ship it",
        paths=[repo_path / "src/lib.rs", repo_path / "src/service.rs"],
        trailers=["Co-Authored-By: MySwat Dev (GPT-5.4) <noreply@myswat.invalid>"],
    )


def test_run_phase_blocks_when_local_commit_fails(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=83,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch.object(kernel, "_queue_stage_task", return_value=100):
        with patch.object(
            kernel,
            "_wait_for_stage_result",
            return_value=StageRunCompletion(
                stage_run_id=100,
                work_item_id=83,
                stage_name="phase_1",
                status="completed",
                summary="Phase done.",
                artifact_id=1001,
                artifact_content="Phase summary",
            ),
        ):
            with patch.object(kernel, "_run_review_loop", return_value=("Phase summary", 1, True)):
                with patch.object(kernel, "_current_workflow_repo_paths", return_value=[repo_path / "src/lib.rs"]):
                    with patch(
                        "myswat.workflow.kernel.commit_repo_changes",
                        return_value=GitCommitResult(False, False, "git commit failed."),
                    ):
                        result = kernel._run_phase(
                            requirement="ship it",
                            design="design",
                            plan="Phase 1: ship it",
                            phase_name="Ship it",
                            phase_index=1,
                            total_phases=1,
                            completed_summaries=[],
                        )

    assert result.committed is False
    assert kernel._blocked is True
    assert "Failed to commit phase 1" in kernel._failure_summary


def test_run_test_commits_local_changes_after_pass(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=84,
        mode=WorkMode.test,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch.object(
        kernel,
        "_wait_for_stage_result",
        side_effect=[
            StageRunCompletion(
                stage_run_id=100,
                work_item_id=84,
                stage_name="test_plan",
                status="completed",
                summary="Test plan ready.",
                artifact_id=1000,
                artifact_content="Test plan",
            ),
            StageRunCompletion(
                stage_run_id=101,
                work_item_id=84,
                stage_name="ga_test",
                status="completed",
                summary="All tests passed.",
                artifact_id=1001,
                artifact_content='{"status":"pass","summary":"All tests passed.","tests_failed":0}',
            ),
        ],
    ):
        with patch.object(kernel, "_run_review_loop", return_value=("Test plan", 1, True)):
            with patch.object(kernel, "_checkpoint", return_value=("Test plan", True)):
                with patch.object(kernel, "_current_workflow_repo_paths", return_value=[repo_path / "tests/test_api.rs"]):
                    with patch(
                        "myswat.workflow.kernel.commit_repo_changes",
                        return_value=GitCommitResult(True, True, "Committed test changes."),
                    ) as mock_commit:
                        result = kernel._run_test("ship it", "design", [])

    assert result.passed is True
    mock_commit.assert_called_once_with(
        repo_path.resolve(),
        message="test: sync approved test changes",
        paths=[repo_path / "tests/test_api.rs"],
        trailers=["Co-Authored-By: MySwat Dev (Opus 4.6) <noreply@myswat.invalid>"],
    )


def test_run_test_preserves_passed_status_when_post_test_commit_fails(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=86,
        mode=WorkMode.test,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch.object(
        kernel,
        "_wait_for_stage_result",
        side_effect=[
            StageRunCompletion(
                stage_run_id=100,
                work_item_id=86,
                stage_name="test_plan",
                status="completed",
                summary="Test plan ready.",
                artifact_id=1000,
                artifact_content="Test plan",
            ),
            StageRunCompletion(
                stage_run_id=101,
                work_item_id=86,
                stage_name="ga_test",
                status="completed",
                summary="All tests passed.",
                artifact_id=1001,
                artifact_content='{"status":"pass","summary":"All tests passed.","tests_failed":0}',
            ),
        ],
    ):
        with patch.object(kernel, "_run_review_loop", return_value=("Test plan", 1, True)):
            with patch.object(kernel, "_checkpoint", return_value=("Test plan", True)):
                with patch.object(kernel, "_current_workflow_repo_paths", return_value=[repo_path / "tests/test_api.rs"]):
                    with patch(
                        "myswat.workflow.kernel.commit_repo_changes",
                        return_value=GitCommitResult(False, False, "git commit failed."),
                    ):
                        result = kernel._run_test("ship it", "design", [])

    assert result.passed is True
    assert kernel._blocked is True
    assert "Failed to commit approved test changes." in kernel._failure_summary


def test_develop_mode_pushes_repo_after_successful_workflow(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa = _participant(20, "qa_main")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa],
        project_id=1,
        work_item_id=85,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )

    with patch(
        "myswat.workflow.kernel.probe_git_repository",
        return_value=GitProbeResult(True, True, True, ""),
    ):
        with patch.object(
            kernel,
            "_load_latest_artifact",
            side_effect=lambda artifact_type: {
                "design_doc": "Approved design",
                "implementation_plan": "Phase 1: Ship it",
            }.get(artifact_type, ""),
        ):
            with patch.object(kernel, "_export_design_plan_to_docs", return_value=True):
                with patch.object(kernel, "_parse_phases", return_value=["Ship it"]):
                    with patch.object(
                        kernel,
                        "_run_phase",
                        return_value=PhaseResult(name="Ship it", summary="done", committed=True),
                    ):
                        with patch.object(kernel, "_generate_final_report", return_value="report"):
                            with patch.object(kernel, "_export_final_report_to_docs", return_value=True):
                                with patch.object(kernel, "_current_workflow_repo_paths", return_value=[repo_path / "src/lib.rs"]):
                                    with patch(
                                        "myswat.workflow.kernel.commit_repo_changes",
                                        return_value=GitCommitResult(True, True, "Committed final workflow changes."),
                                    ) as mock_commit:
                                        with patch(
                                            "myswat.workflow.kernel.push_repo_changes",
                                            return_value=GitPushResult(True, True, "Pushed local workflow commits."),
                                        ) as mock_push:
                                            result = kernel.run("ship it")

    assert result.success is True
    assert result.final_report == "report"
    mock_commit.assert_called_once_with(
        repo_path.resolve(),
        message="workflow: finalize develop",
        paths=[repo_path / "src/lib.rs"],
        trailers=["Co-Authored-By: MySwat Dev (GPT-5.4) <noreply@myswat.invalid>"],
    )
    mock_push.assert_called_once_with(repo_path.resolve())


def test_finalize_workflow_repo_sync_skips_push_when_no_workflow_commit_was_created(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=903,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch.object(kernel, "_current_workflow_repo_paths", return_value=[]):
        with patch("myswat.workflow.kernel.push_repo_changes") as mock_push:
            ok = kernel._finalize_workflow_repo_sync()

    assert ok is True
    mock_push.assert_not_called()


def test_dirty_repo_at_start_uses_scoped_commit_strategy(tmp_path):
    store = _store()
    kernel = WorkflowKernel(
        store=store,
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=90,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(tmp_path / "repo"),
    )

    with patch(
        "myswat.workflow.kernel.probe_git_repository",
        return_value=GitProbeResult(True, True, False, ""),
    ):
        with patch(
            "myswat.workflow.kernel.list_changed_repo_paths",
            return_value=SimpleNamespace(ok=True, paths={"notes.txt"}),
        ):
            kernel._ensure_repo_commit_ready()

    assert kernel._repo_commit_ready is True
    assert kernel._repo_initial_dirty_paths == {"notes.txt"}
    assert any(
        "workflow-owned paths" in call.kwargs.get("summary", "")
        for call in store.append_work_item_process_event.call_args_list
    )


def test_current_workflow_repo_paths_excludes_initial_dirty_paths_even_when_preferred(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=901,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_initial_dirty_paths = {"notes.txt"}
    kernel._repo_managed_paths = {"docs/implementation-plan.md"}

    with patch(
        "myswat.workflow.kernel.list_changed_repo_paths",
        return_value=SimpleNamespace(
            ok=True,
            paths={
                "notes.txt",
                "src/lib.rs",
                "docs/implementation-plan.md",
            },
        ),
    ):
        result = kernel._current_workflow_repo_paths("notes.txt", "src/lib.rs")

    assert result == [
        repo_path / "docs/implementation-plan.md",
        repo_path / "src/lib.rs",
    ]


def test_export_final_report_to_docs_keeps_workflow_report_local_only(tmp_path):
    store = _store()
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    kernel = WorkflowKernel(
        store=store,
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=91,
        mode=WorkMode.design,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch(
        "myswat.workflow.kernel.write_workflow_report_doc",
        return_value=repo_path / ".myswat" / "workflow-reports" / "myswat-design-workflow-report-20260321-214832.md",
    ):
        with patch("myswat.workflow.kernel.commit_repo_changes") as mock_commit:
            ok = kernel._export_final_report_to_docs("# Workflow Report")

    assert ok is True
    assert kernel._repo_managed_paths == set()
    mock_commit.assert_not_called()
    assert any(
        "workflow report locally" in call.kwargs.get("summary", "").lower()
        for call in store.append_work_item_process_event.call_args_list
    )


def test_commit_trailers_use_humanized_runtime_model_names():
    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=_participant(10, "developer", model_name="gpt-5.4"),
        qas=[_participant(20, "qa_main", model_name="claude-opus-4-6")],
        project_id=1,
        work_item_id=87,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    assert kernel._commit_trailers_for(kernel._dev) == [
        "Co-Authored-By: MySwat Dev (GPT-5.4) <noreply@myswat.invalid>",
    ]
    assert kernel._commit_trailers_for(kernel._qas[0]) == [
        "Co-Authored-By: MySwat Dev (Opus 4.6) <noreply@myswat.invalid>",
    ]


def test_commit_trailers_humanize_dated_and_legacy_claude_model_names():
    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=88,
        mode=WorkMode.develop,
        auto_approve=True,
    )

    dated_claude = _participant(30, "developer", backend="claude", model_name="claude-opus-4-6-20251001")
    legacy_claude = _participant(31, "developer", backend="claude", model_name="claude-3-5-sonnet-20241022")

    assert kernel._commit_trailers_for(dated_claude) == [
        "Co-Authored-By: MySwat Dev (Opus 4.6) <noreply@myswat.invalid>",
    ]
    assert kernel._commit_trailers_for(legacy_claude) == [
        "Co-Authored-By: MySwat Dev (Sonnet 3.5) <noreply@myswat.invalid>",
    ]


def test_finalize_workflow_repo_sync_warns_but_does_not_block_on_push_failure(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    store = _store()
    kernel = WorkflowKernel(
        store=store,
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=890,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True
    kernel._repo_commits_created = True

    with patch.object(kernel, "_current_workflow_repo_paths", return_value=[]):
        with patch(
            "myswat.workflow.kernel.push_repo_changes",
            return_value=GitPushResult(False, False, "fatal: could not read from remote repository"),
        ):
            ok = kernel._finalize_workflow_repo_sync()

    assert ok is True
    assert kernel._blocked is False
    assert kernel._failure_summary == ""
    assert any(
        call.kwargs.get("event_type") == "repo_push_failed"
        for call in store.append_work_item_process_event.call_args_list
    )


def test_test_commit_trailer_uses_qa_lead_when_multiple_qas(tmp_path):
    store = _store()
    service = _service()
    dev = _participant(10, "developer")
    qa_lead = _participant(20, "qa_main", model_name="claude-opus-4-6")
    qa_vice = _participant(21, "qa_vice", backend="kimi", model_name="kimi-code/kimi-for-coding")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=dev,
        qas=[qa_lead, qa_vice],
        project_id=1,
        work_item_id=89,
        mode=WorkMode.test,
        auto_approve=True,
        repo_path=str(repo_path),
    )
    kernel._repo_commit_checked = True
    kernel._repo_commit_ready = True

    with patch(
        "myswat.workflow.kernel.commit_repo_changes",
        return_value=GitCommitResult(True, True, "Committed test changes."),
    ) as mock_commit:
        with patch.object(kernel, "_current_workflow_repo_paths", return_value=[repo_path / "tests/test_api.rs"]):
            ok = kernel._commit_test_changes()

    assert ok is True
    mock_commit.assert_called_once_with(
        repo_path.resolve(),
        message="test: sync approved test changes",
        paths=[repo_path / "tests/test_api.rs"],
        trailers=["Co-Authored-By: MySwat Dev (Opus 4.6) <noreply@myswat.invalid>"],
    )


def test_extract_repo_paths_from_text_handles_root_level_and_absolute_repo_paths(tmp_path):
    repo_path = tmp_path / "repo"
    (repo_path / "src").mkdir(parents=True)
    (repo_path / "src/lib.rs").write_text("", encoding="utf-8")
    (repo_path / "src/service.rs").write_text("", encoding="utf-8")
    (repo_path / "Cargo.toml").write_text("[package]\nname = 'demo'\n", encoding="utf-8")

    kernel = WorkflowKernel(
        store=_store(),
        service=_service(),
        dev=_participant(10, "developer"),
        qas=[_participant(20, "qa_main")],
        project_id=1,
        work_item_id=902,
        mode=WorkMode.develop,
        auto_approve=True,
        repo_path=str(repo_path),
    )

    extracted = kernel._extract_repo_paths_from_text(
        (
            "Updated `src/lib.rs:10`, "
            f"`{repo_path / 'src/service.rs'}:20:3`, "
            "`Cargo.toml`, "
            "`StatementCtx`, "
            "`../outside.rs`."
        )
    )

    assert extracted == {"Cargo.toml", "src/lib.rs", "src/service.rs"}


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


def test_review_loop_blocks_immediately_on_failed_verdict_and_cancels_siblings():
    dev = _participant(10, "developer")
    qa_main = _participant(20, "qa_main")
    security = _participant(21, "security")
    service = _service()
    service.wait_for_review_verdicts.return_value = [
        ReviewVerdictEnvelope(
            cycle_id=2000,
            reviewer_role="qa_main",
            verdict="failed",
            summary="review failed after retry exhaustion.",
        ),
    ]
    service.cancel_review_cycles.return_value = {"ok": True}

    kernel = WorkflowKernel(
        store=_store(),
        service=service,
        dev=dev,
        qas=[qa_main, security],
        project_id=1,
        work_item_id=9,
        mode=WorkMode.full,
        auto_approve=True,
    )

    artifact, iterations, passed = kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="draft",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa_main, security],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )

    assert (artifact, iterations, passed) == ("draft", 1, False)
    assert kernel._blocked is True
    assert kernel._failure_summary == "[qa_main] review failed after retry exhaustion."
    wait_request = service.wait_for_review_verdicts.call_args.args[0]
    assert wait_request.return_on_failed is True
    cancel_request = service.cancel_review_cycles.call_args.args[0]
    assert cancel_request == ReviewCycleCancellationRequest(
        cycle_ids=[2001],
        summary="Cancelled remaining plan_review cycles after a sibling review failed in iteration 1.",
    )
    assert service.start_stage_run.call_count == 0


def test_review_failure_takes_precedence_over_changes_requested():
    dev = _participant(10, "developer")
    qa_main = _participant(20, "qa_main")
    security = _participant(21, "security")
    service = _service()
    service.wait_for_review_verdicts.return_value = [
        ReviewVerdictEnvelope(
            cycle_id=2000,
            reviewer_role="qa_main",
            verdict="failed",
            summary="review failed after retry exhaustion.",
        ),
        ReviewVerdictEnvelope(
            cycle_id=2001,
            reviewer_role="security",
            verdict="changes_requested",
            summary="Need a rollback plan.",
            issues=["Add rollback steps."],
        ),
    ]

    kernel = WorkflowKernel(
        store=_store(),
        service=service,
        dev=dev,
        qas=[qa_main, security],
        project_id=1,
        work_item_id=9,
        mode=WorkMode.full,
        auto_approve=True,
    )

    artifact, iterations, passed = kernel._run_review_loop(
        owner_stage_name="plan",
        review_stage_name="plan_review",
        artifact_type="implementation_plan",
        artifact_title="Implementation plan",
        initial_artifact="draft",
        initial_artifact_id=1000,
        owner=dev,
        reviewers=[qa_main, security],
        focus="ctx",
        review_prompt_builder=lambda artifact, iteration, reviewer_role: artifact,
        revision_prompt_builder=lambda artifact, feedback: artifact + feedback,
    )

    assert (artifact, iterations, passed) == ("draft", 1, False)
    assert kernel._blocked is True
    assert kernel._failure_summary == "[qa_main] review failed after retry exhaustion."
    assert service.start_stage_run.call_count == 0


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

    test_service = _service()
    test_kernel = WorkflowKernel(
        store=_store(),
        service=test_service,
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
    test_plan_stage = test_service.start_stage_run.call_args.args[0]
    assert test_plan_stage.stage_name == "test_plan"
    assert "Right-size the test plan to the scope:" in test_plan_stage.task_prompt
    assert "Do NOT turn a simple test plan into artificial phases" in test_plan_stage.task_prompt

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
