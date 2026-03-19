"""Stage-oriented workflow kernel backed by queued MCP assignments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from myswat.large_payloads import resolve_externalized_text, resolve_externalized_value
from myswat.server import (
    ReviewRequest,
    ReviewWaitRequest,
    StageRunStart,
    StageRunWaitRequest,
    StatusReport,
)
from myswat.server.service import MySwatToolService
from myswat.workflow.events import WorkflowEvent
from myswat.workflow.modes import WorkMode
from myswat.workflow.prompts import (
    ARCH_ADDRESS_DESIGN_COMMENTS,
    ARCH_TECH_DESIGN,
    DEV_ADDRESS_CODE_COMMENTS,
    DEV_ADDRESS_DESIGN_COMMENTS,
    DEV_ADDRESS_PLAN_COMMENTS,
    DEV_FINAL_REPORT,
    DEV_IMPLEMENTATION_PLAN,
    DEV_IMPLEMENT_PHASE,
    DEV_REVIEW_TEST_PLAN,
    DEV_TECH_DESIGN,
    QA_ADDRESS_TEST_PLAN_COMMENTS,
    QA_CODE_REVIEW,
    QA_DESIGN_REVIEW,
    QA_EXECUTE_GA_TEST,
    QA_GA_TEST_PLAN,
    QA_PLAN_REVIEW,
)
from myswat.workflow.runtime import WorkflowRuntime


@dataclass
class PhaseResult:
    name: str
    summary: str
    review_iterations: int = 0
    review_passed: bool = False
    committed: bool = False


@dataclass
class GATestResult:
    test_plan: str = ""
    test_report: str = ""
    passed: bool = False
    bugs_found: int = 0


@dataclass
class WorkflowResult:
    requirement: str
    design: str = ""
    plan: str = ""
    phases: list[PhaseResult] = field(default_factory=list)
    ga_test: GATestResult | None = None
    final_report: str = ""
    success: bool = False
    blocked: bool = False
    failure_summary: str = ""


def _extract_json_block(text: str) -> dict | list | None:
    def _strip_code_fences(value: str) -> str:
        if "```json" in value:
            return value.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in value:
            parts = value.split("```")
            for part in parts[1::2]:
                part = part.strip()
                if part.startswith("{") or part.startswith("["):
                    return part
        return value.strip()

    def _parse_json(value: str) -> dict | list | None:
        for start_ch, end_ch in [("{", "}"), ("[", "]")]:
            start = value.find(start_ch)
            end = value.rfind(end_ch)
            if start != -1 and end > start:
                try:
                    return resolve_externalized_value(json.loads(value[start:end + 1]))
                except json.JSONDecodeError:
                    continue
        try:
            return resolve_externalized_value(json.loads(value))
        except json.JSONDecodeError:
            return None

    stripped = _strip_code_fences(text.strip())
    parsed = _parse_json(stripped)
    if parsed is not None:
        return parsed

    stripped = _strip_code_fences(resolve_externalized_text(stripped).strip())
    return _parse_json(stripped)


class WorkflowKernel:
    """Deterministic workflow orchestrator over queued MCP tasks."""

    def __init__(
        self,
        *,
        store,
        service: MySwatToolService,
        dev: WorkflowRuntime,
        qas: list[WorkflowRuntime],
        project_id: int,
        work_item_id: int,
        mode: WorkMode = WorkMode.full,
        max_review_iterations: int = 5,
        ask_user: Callable[[str], str] | None = None,
        auto_approve: bool = True,
        should_cancel: Callable[[], bool] | None = None,
        arch: WorkflowRuntime | None = None,
        resume_stage: str | None = None,
        on_event: Callable[[WorkflowEvent], None] | None = None,
        assignment_poll_interval_seconds: float = 1.0,
        assignment_timeout_seconds: float | None = None,
    ) -> None:
        self._store = store
        self._service = service
        self._dev = dev
        self._qas = qas
        self._arch = arch
        self._project_id = project_id
        self._work_item_id = work_item_id
        self._mode = WorkMode(mode)
        self._max_review = max_review_iterations
        self._ask = ask_user or (lambda prompt: input(f"\n{prompt}").strip())
        self._auto_approve = auto_approve
        self._should_cancel = should_cancel
        self._resume_stage = resume_stage
        self._on_event = on_event
        self._blocked = False
        self._failure_summary = ""
        self._assignment_poll_interval_seconds = assignment_poll_interval_seconds
        self._assignment_timeout_seconds = assignment_timeout_seconds

    def _emit(
        self,
        event_type: str,
        message: str,
        *,
        stage: str = "",
        agent_role: str | None = None,
        detail: str | None = None,
        **metadata: object,
    ) -> None:
        if self._on_event is not None:
            self._on_event(
                WorkflowEvent(
                    event_type=event_type,
                    message=message,
                    stage=stage,
                    agent_role=agent_role,
                    detail=detail,
                    metadata=metadata,
                )
            )

    def _cancelled(self) -> bool:
        return bool(self._should_cancel and self._should_cancel())

    def _stage_index(self, stage_name: str) -> int:
        if stage_name == "design":
            return 10
        if stage_name == "plan":
            return 30
        if stage_name == "test_plan":
            return 500
        if stage_name == "ga_test":
            return 520
        if stage_name == "report":
            return 900
        if stage_name.startswith("phase_"):
            try:
                phase_index = int(stage_name.split("_")[1])
            except (IndexError, ValueError):
                return 700
            return 100 + phase_index * 10
        return 1000

    def _record_status(
        self,
        runtime: WorkflowRuntime,
        *,
        stage: str,
        summary: str,
        next_todos: list[str] | None = None,
        open_issues: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        self._service.report_status(
            StatusReport(
                work_item_id=self._work_item_id,
                agent_id=runtime.agent_id,
                agent_role=runtime.agent_role,
                stage=stage,
                summary=summary,
                next_todos=next_todos or [],
                open_issues=open_issues or [],
                title=title,
            )
        )

    def _queue_stage_task(
        self,
        *,
        owner: WorkflowRuntime,
        stage_name: str,
        prompt: str,
        focus: str,
        artifact_type: str,
        artifact_title: str,
        summary: str,
        iteration: int = 1,
        metadata_json: dict | None = None,
    ) -> int:
        self._emit(
            "stage_start",
            f"Queued stage: {stage_name}",
            stage=stage_name,
            agent_role=owner.agent_role,
        )
        stage_run = self._service.start_stage_run(
            StageRunStart(
                work_item_id=self._work_item_id,
                stage_name=stage_name,
                stage_index=self._stage_index(stage_name),
                iteration=iteration,
                owner_agent_id=owner.agent_id,
                owner_role=owner.agent_role,
                status="pending",
                summary=summary,
                task_prompt=prompt,
                task_focus=focus,
                artifact_type=artifact_type,
                artifact_title=artifact_title,
                metadata_json=metadata_json,
            )
        )
        return stage_run.stage_run_id

    def _wait_for_stage_result(
        self,
        *,
        stage_run_id: int,
        stage_name: str,
        owner: WorkflowRuntime,
    ):
        self._emit(
            "agent_waiting",
            f"Waiting for {owner.agent_role} runtime to complete {stage_name}",
            stage=stage_name,
            agent_role=owner.agent_role,
        )
        result = self._service.wait_for_stage_run_completion(
            StageRunWaitRequest(
                stage_run_id=stage_run_id,
                poll_interval_seconds=self._assignment_poll_interval_seconds,
                timeout_seconds=self._assignment_timeout_seconds,
            )
        )
        if result.status == "completed":
            self._emit(
                "stage_complete",
                result.summary or f"{stage_name} completed",
                stage=stage_name,
                agent_role=owner.agent_role,
            )
        else:
            self._emit(
                "error",
                result.summary or f"{stage_name} failed",
                stage=stage_name,
                agent_role=owner.agent_role,
            )
            self._blocked = True
            self._failure_summary = result.summary or f"{stage_name} failed"
        return result

    def _checkpoint(
        self,
        artifact: str,
        *,
        prompt: str,
        stage: str,
        owner: WorkflowRuntime,
        focus: str,
        artifact_type: str,
        artifact_title: str,
        revision_prompt_builder: Callable[[str, str], str] | None = None,
    ) -> tuple[str, bool]:
        if self._auto_approve:
            return artifact, True
        answer = self._ask(prompt).strip()
        if answer.lower() in {"", "y", "yes"}:
            return artifact, True
        if answer.lower() in {"n", "no"}:
            self._blocked = True
            self._failure_summary = f"User rejected {stage} checkpoint."
            return artifact, False
        if revision_prompt_builder is None:
            self._blocked = True
            self._failure_summary = f"User requested changes during {stage}: {answer}"
            return artifact, False

        latest_stage_run = self._store.get_latest_stage_run(self._work_item_id, stage)
        iteration = int(getattr(latest_stage_run, "iteration", 1) or 1) + 1
        stage_run_id = self._queue_stage_task(
            owner=owner,
            stage_name=stage,
            prompt=revision_prompt_builder(artifact, answer),
            focus=focus,
            artifact_type=artifact_type,
            artifact_title=artifact_title,
            summary=f"Address user feedback for {artifact_title}",
            iteration=iteration,
        )
        completion = self._wait_for_stage_result(
            stage_run_id=stage_run_id,
            stage_name=stage,
            owner=owner,
        )
        if completion.status != "completed":
            return artifact, False
        return completion.artifact_content, True

    def _wait_for_review_verdicts(self, cycle_ids: list[int]) -> list:
        return self._service.wait_for_review_verdicts(
            ReviewWaitRequest(
                cycle_ids=cycle_ids,
                poll_interval_seconds=self._assignment_poll_interval_seconds,
                timeout_seconds=self._assignment_timeout_seconds,
            )
        )

    def _run_review_loop(
        self,
        *,
        owner_stage_name: str,
        review_stage_name: str,
        artifact_type: str,
        artifact_title: str,
        initial_artifact: str,
        initial_artifact_id: int | None,
        owner: WorkflowRuntime,
        reviewers: list[WorkflowRuntime],
        focus: str,
        review_prompt_builder: Callable[[str, int, str], str],
        revision_prompt_builder: Callable[[str, str], str],
    ) -> tuple[str, int, bool]:
        artifact = initial_artifact
        artifact_id = initial_artifact_id
        for iteration in range(1, self._max_review + 1):
            if self._cancelled():
                self._blocked = True
                self._failure_summary = "Workflow cancelled."
                return artifact, iteration, False

            if artifact_id is None:
                latest_artifact = self._store.get_latest_artifact_by_type(self._work_item_id, artifact_type)
                artifact_id = int(latest_artifact["id"]) if latest_artifact else None
            if artifact_id is None:
                self._blocked = True
                self._failure_summary = f"Missing artifact for {review_stage_name}."
                return artifact, iteration, False

            self._emit(
                "review_start",
                f"Queued review iteration {iteration}",
                stage=review_stage_name,
                agent_role=owner.agent_role,
            )
            cycle_ids: list[int] = []
            for reviewer in reviewers:
                request = self._service.request_review(
                    ReviewRequest(
                        work_item_id=self._work_item_id,
                        artifact_id=artifact_id,
                        iteration=iteration,
                        proposer_agent_id=owner.agent_id,
                        proposer_role=owner.agent_role,
                        reviewer_agent_id=reviewer.agent_id,
                        reviewer_role=reviewer.agent_role,
                        stage=review_stage_name,
                        summary=f"Review {artifact_title} iteration {iteration}",
                        task_prompt=review_prompt_builder(artifact, iteration, reviewer.agent_role),
                        task_focus=focus,
                        task_json={
                            "artifact_type": artifact_type,
                            "artifact_title": artifact_title,
                        },
                    )
                )
                cycle_ids.append(request.cycle_id)

            verdicts = self._wait_for_review_verdicts(cycle_ids)
            all_lgtm = True
            collected_feedback: list[str] = []
            for verdict in verdicts:
                self._emit(
                    "review_verdict",
                    f"{verdict.reviewer_role}: {verdict.verdict.upper()}",
                    stage=review_stage_name,
                    agent_role=verdict.reviewer_role,
                    detail=verdict.summary,
                    verdict=verdict.verdict,
                )
                if verdict.verdict != "lgtm":
                    all_lgtm = False
                    issues = verdict.issues or [verdict.summary]
                    collected_feedback.extend(
                        [f"[{verdict.reviewer_role}] {issue}" for issue in issues if issue]
                    )

            if all_lgtm:
                return artifact, iteration, True

            if iteration >= self._max_review:
                self._blocked = True
                self._failure_summary = f"{artifact_title} did not reach LGTM."
                return artifact, iteration, False

            feedback = "\n".join(f"- {item}" for item in collected_feedback if item)
            self._record_status(
                owner,
                stage=review_stage_name,
                summary=f"Addressing review feedback for {artifact_title}.",
                open_issues=collected_feedback[:20],
            )
            stage_run_id = self._queue_stage_task(
                owner=owner,
                stage_name=owner_stage_name,
                prompt=revision_prompt_builder(artifact, feedback),
                focus=focus + "\n\n" + feedback,
                artifact_type=artifact_type,
                artifact_title=artifact_title,
                summary=f"Revise {artifact_title} after review iteration {iteration}",
                iteration=iteration + 1,
            )
            completion = self._wait_for_stage_result(
                stage_run_id=stage_run_id,
                stage_name=owner_stage_name,
                owner=owner,
            )
            if completion.status != "completed":
                return artifact, iteration, False
            artifact = completion.artifact_content
            artifact_id = completion.artifact_id

        self._blocked = True
        self._failure_summary = f"{artifact_title} review loop exhausted."
        return artifact, self._max_review, False

    def _load_latest_artifact(self, artifact_type: str) -> str:
        artifact = self._store.get_latest_artifact_by_type(self._work_item_id, artifact_type)
        return str(artifact.get("content") or "") if artifact else ""

    def _load_completed_phase_results(self) -> list[PhaseResult]:
        artifacts = self._store.list_artifacts(self._work_item_id)
        phase_rows = [row for row in artifacts if row.get("artifact_type") == "phase_result"]

        def _metadata(row: dict) -> dict:
            metadata = row.get("metadata_json")
            if isinstance(metadata, dict):
                return metadata
            if isinstance(metadata, str):
                parsed = _extract_json_block(metadata)
                return parsed if isinstance(parsed, dict) else {}
            return {}

        phase_rows.sort(
            key=lambda row: (
                int(_metadata(row).get("phase_index", row.get("iteration") or 0)),
                int(row.get("iteration") or 0),
            )
        )
        results: list[PhaseResult] = []
        for row in phase_rows:
            metadata = _metadata(row)
            results.append(
                PhaseResult(
                    name=str(metadata.get("phase_name") or row.get("title") or "phase"),
                    summary=str(row.get("content") or ""),
                    review_iterations=int(metadata.get("review_iterations") or 0),
                    review_passed=bool(metadata.get("review_passed", True)),
                    committed=bool(metadata.get("committed", True)),
                )
            )
        return results

    def _parse_phases(self, plan: str) -> list[str]:
        phases: list[str] = []
        for line in plan.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            for prefix in ("phase ", "step "):
                if lower.startswith(prefix):
                    rest = stripped[len(prefix):]
                    for idx, ch in enumerate(rest):
                        if ch in (":", ".", ")") or (ch == " " and idx > 0):
                            candidate = rest[idx + 1:].strip() or rest.strip()
                            phases.append(candidate)
                            break
                    else:
                        phases.append(rest.strip())
                    break
        return phases or ["Implementation"]

    def _run_design(self, requirement: str) -> tuple[str, bool]:
        owner = self._arch or self._dev
        prompt = (
            ARCH_TECH_DESIGN.format(requirement=requirement)
            if owner is self._arch
            else DEV_TECH_DESIGN.format(requirement=requirement)
        )
        stage_run_id = self._queue_stage_task(
            owner=owner,
            stage_name="design",
            prompt=prompt,
            focus=requirement,
            artifact_type="design_doc",
            artifact_title="Technical design",
            summary="Produce technical design",
        )
        completion = self._wait_for_stage_result(stage_run_id=stage_run_id, stage_name="design", owner=owner)
        if completion.status != "completed":
            return "", False

        draft = completion.artifact_content
        reviewers = ([self._dev] if owner is self._arch else []) + self._qas
        reviewed, _iterations, passed = self._run_review_loop(
            owner_stage_name="design",
            review_stage_name="design_review",
            artifact_type="design_doc",
            artifact_title="Technical design",
            initial_artifact=draft,
            initial_artifact_id=completion.artifact_id,
            owner=owner,
            reviewers=reviewers,
            focus=requirement,
            review_prompt_builder=lambda artifact, iteration, reviewer_role: QA_DESIGN_REVIEW.format(
                context=f"Requirement:\n{requirement}",
                design=artifact,
                iteration=iteration,
            ),
            revision_prompt_builder=(
                (lambda artifact, feedback: ARCH_ADDRESS_DESIGN_COMMENTS.format(
                    design=artifact,
                    feedback=feedback,
                ))
                if owner is self._arch
                else (lambda artifact, feedback: DEV_ADDRESS_DESIGN_COMMENTS.format(
                    design=artifact,
                    feedback=feedback,
                ))
            ),
        )
        if not passed:
            return reviewed, False
        reviewed, ok = self._checkpoint(
            reviewed,
            prompt="Design approved by reviewers. Accept? [Y/n/or type feedback] ",
            stage="design",
            owner=owner,
            focus=requirement,
            artifact_type="design_doc",
            artifact_title="Technical design",
            revision_prompt_builder=(
                (lambda artifact, feedback: ARCH_ADDRESS_DESIGN_COMMENTS.format(
                    design=artifact,
                    feedback=feedback,
                ))
                if owner is self._arch
                else (lambda artifact, feedback: DEV_ADDRESS_DESIGN_COMMENTS.format(
                    design=artifact,
                    feedback=feedback,
                ))
            ),
        )
        return reviewed, ok

    def _run_plan(self, requirement: str, design: str) -> tuple[str, bool]:
        owner = self._dev
        stage_run_id = self._queue_stage_task(
            owner=owner,
            stage_name="plan",
            prompt=DEV_IMPLEMENTATION_PLAN.format(requirement=requirement, design=design),
            focus=requirement + "\n\n" + design[:4000],
            artifact_type="implementation_plan",
            artifact_title="Implementation plan",
            summary="Break design into phases",
        )
        completion = self._wait_for_stage_result(stage_run_id=stage_run_id, stage_name="plan", owner=owner)
        if completion.status != "completed":
            return "", False

        reviewed, _iterations, passed = self._run_review_loop(
            owner_stage_name="plan",
            review_stage_name="plan_review",
            artifact_type="implementation_plan",
            artifact_title="Implementation plan",
            initial_artifact=completion.artifact_content,
            initial_artifact_id=completion.artifact_id,
            owner=owner,
            reviewers=self._qas,
            focus=requirement + "\n\n" + design[:4000],
            review_prompt_builder=lambda artifact, iteration, reviewer_role: QA_PLAN_REVIEW.format(
                context=f"Requirement:\n{requirement}\n\nApproved Design:\n{design}",
                plan=artifact,
                iteration=iteration,
            ),
            revision_prompt_builder=lambda artifact, feedback: DEV_ADDRESS_PLAN_COMMENTS.format(
                plan=artifact,
                feedback=feedback,
            ),
        )
        if not passed:
            return reviewed, False
        reviewed, ok = self._checkpoint(
            reviewed,
            prompt="Plan approved by reviewers. Accept? [Y/n/or type feedback] ",
            stage="plan",
            owner=owner,
            focus=requirement + "\n\n" + design[:4000],
            artifact_type="implementation_plan",
            artifact_title="Implementation plan",
            revision_prompt_builder=lambda artifact, feedback: DEV_ADDRESS_PLAN_COMMENTS.format(
                plan=artifact,
                feedback=feedback,
            ),
        )
        return reviewed, ok

    def _run_phase(
        self,
        *,
        requirement: str,
        design: str,
        plan: str,
        phase_name: str,
        phase_index: int,
        total_phases: int,
        completed_summaries: list[str],
    ) -> PhaseResult:
        owner = self._dev
        stage_name = f"phase_{phase_index}"
        completed_phase_context = "\n".join(completed_summaries) or "None"
        prompt = (
            DEV_IMPLEMENT_PHASE.format(
                phase_index=phase_index,
                total_phases=total_phases,
                requirement=requirement,
                design=design,
                plan=plan,
                phase_name=phase_name,
                completed_phases=completed_phase_context,
            )
            + "\n\nAfter completing the work, return a concise implementation summary suitable for QA review."
        )
        stage_run_id = self._queue_stage_task(
            owner=owner,
            stage_name=stage_name,
            prompt=prompt,
            focus=requirement + "\n\n" + phase_name,
            artifact_type="phase_result",
            artifact_title=f"Phase {phase_index}: {phase_name}",
            summary=f"Implement phase {phase_index}: {phase_name}",
            iteration=phase_index,
            metadata_json={"phase_index": phase_index, "phase_name": phase_name},
        )
        completion = self._wait_for_stage_result(stage_run_id=stage_run_id, stage_name=stage_name, owner=owner)
        if completion.status != "completed":
            return PhaseResult(name=phase_name, summary=self._failure_summary or f"Phase {phase_index} failed.")

        reviewed_summary, review_iterations, passed = self._run_review_loop(
            owner_stage_name=stage_name,
            review_stage_name=f"{stage_name}_review",
            artifact_type="phase_result",
            artifact_title=f"Phase {phase_index} summary",
            initial_artifact=completion.artifact_content,
            initial_artifact_id=completion.artifact_id,
            owner=owner,
            reviewers=self._qas,
            focus=requirement + "\n\n" + phase_name,
            review_prompt_builder=lambda artifact, iteration, reviewer_role: QA_CODE_REVIEW.format(
                context=f"Requirement:\n{requirement}\n\nApproved Design:\n{design}\n\nPlan:\n{plan}",
                summary=artifact,
                iteration=iteration,
            ),
            revision_prompt_builder=lambda artifact, feedback: DEV_ADDRESS_CODE_COMMENTS.format(
                summary=artifact,
                feedback=feedback,
            ),
        )
        if not passed:
            return PhaseResult(
                name=phase_name,
                summary=reviewed_summary,
                review_iterations=review_iterations,
                review_passed=False,
                committed=False,
            )

        return PhaseResult(
            name=phase_name,
            summary=reviewed_summary,
            review_iterations=review_iterations,
            review_passed=True,
            committed=True,
        )

    def _run_test(self, requirement: str, design: str, completed_summaries: list[str]) -> GATestResult:
        qa_lead = self._qas[0]
        completed_phase_context = "\n".join(completed_summaries) or "Use the current codebase state."
        stage_run_id = self._queue_stage_task(
            owner=qa_lead,
            stage_name="test_plan",
            prompt=QA_GA_TEST_PLAN.format(
                requirement=requirement,
                design=design,
                dev_summary=completed_phase_context,
            ),
            focus=requirement + "\n\n" + design[:4000],
            artifact_type="test_plan",
            artifact_title="GA test plan",
            summary="Generate GA test plan",
        )
        completion = self._wait_for_stage_result(stage_run_id=stage_run_id, stage_name="test_plan", owner=qa_lead)
        if completion.status != "completed":
            return GATestResult()

        reviewed_plan, _iters, passed = self._run_review_loop(
            owner_stage_name="test_plan",
            review_stage_name="test_plan_review",
            artifact_type="test_plan",
            artifact_title="GA test plan",
            initial_artifact=completion.artifact_content,
            initial_artifact_id=completion.artifact_id,
            owner=qa_lead,
            reviewers=[self._dev],
            focus=requirement + "\n\n" + design[:4000],
            review_prompt_builder=lambda artifact, iteration, reviewer_role: DEV_REVIEW_TEST_PLAN.format(
                context=f"Requirement:\n{requirement}\n\nApproved Design:\n{design}",
                test_plan=artifact,
                iteration=iteration,
            ),
            revision_prompt_builder=lambda artifact, feedback: QA_ADDRESS_TEST_PLAN_COMMENTS.format(
                test_plan=artifact,
                feedback=feedback,
            ),
        )
        if not passed:
            return GATestResult(test_plan=reviewed_plan)
        reviewed_plan, ok = self._checkpoint(
            reviewed_plan,
            prompt="Test plan approved. Start testing? [Y/n/or type feedback] ",
            stage="test_plan",
            owner=qa_lead,
            focus=requirement + "\n\n" + design[:4000],
            artifact_type="test_plan",
            artifact_title="GA test plan",
            revision_prompt_builder=lambda artifact, feedback: QA_ADDRESS_TEST_PLAN_COMMENTS.format(
                test_plan=artifact,
                feedback=feedback,
            ),
        )
        if not ok:
            return GATestResult(test_plan=reviewed_plan)

        execute_stage_id = self._queue_stage_task(
            owner=qa_lead,
            stage_name="ga_test",
            prompt=QA_EXECUTE_GA_TEST.format(test_plan=reviewed_plan),
            focus=reviewed_plan[:4000],
            artifact_type="test_report",
            artifact_title="GA test report",
            summary="Execute GA test plan",
        )
        test_completion = self._wait_for_stage_result(stage_run_id=execute_stage_id, stage_name="ga_test", owner=qa_lead)
        if test_completion.status != "completed":
            return GATestResult(test_plan=reviewed_plan)

        payload = _extract_json_block(test_completion.artifact_content)
        if not isinstance(payload, dict):
            payload = {}
        status = str(payload.get("status") or "fail")
        summary = str(payload.get("summary") or test_completion.summary or test_completion.artifact_content[:4000])
        if status != "pass":
            self._blocked = True
            self._failure_summary = summary
        return GATestResult(
            test_plan=reviewed_plan,
            test_report=test_completion.artifact_content,
            passed=(status == "pass"),
            bugs_found=int(payload.get("tests_failed") or len(payload.get("bugs") or [])),
        )

    def _generate_final_report(self, completed_summaries: list[str]) -> str:
        completed_phase_context = "\n".join(completed_summaries) or "No completed phases."
        stage_run_id = self._queue_stage_task(
            owner=self._dev,
            stage_name="report",
            prompt=DEV_FINAL_REPORT.format(
                completed_phases=completed_phase_context,
            ),
            focus="\n".join(completed_summaries)[:4000],
            artifact_type="final_report",
            artifact_title="Workflow report",
            summary="Generate final report",
        )
        completion = self._wait_for_stage_result(stage_run_id=stage_run_id, stage_name="report", owner=self._dev)
        if completion.status == "completed" and completion.artifact_content.strip():
            return completion.artifact_content
        return "\n".join(completed_summaries) if completed_summaries else "No completed work to report."

    def run(self, requirement: str) -> WorkflowResult:
        result = WorkflowResult(requirement=requirement)
        if self._cancelled():
            result.blocked = True
            result.failure_summary = "Workflow cancelled before start."
            result.final_report = result.failure_summary
            return result

        design = self._load_latest_artifact("design_doc")
        plan = self._load_latest_artifact("implementation_plan")
        completed_phases = self._load_completed_phase_results()
        completed_summaries = [
            f"Phase {idx + 1}: {item.summary[:500]}"
            for idx, item in enumerate(completed_phases)
        ]
        result.phases.extend(completed_phases)

        if self._mode in {WorkMode.full, WorkMode.design} and not design:
            design, ok = self._run_design(requirement)
            result.design = design
            if not ok:
                result.blocked = True
                result.failure_summary = self._failure_summary or "Design stage failed."
                result.final_report = result.failure_summary
                return result
        else:
            result.design = design

        if self._mode in {WorkMode.full, WorkMode.design, WorkMode.develop} and not plan:
            design_ref = result.design or requirement
            plan, ok = self._run_plan(requirement, design_ref)
            result.plan = plan
            if not ok:
                result.blocked = True
                result.failure_summary = self._failure_summary or "Planning stage failed."
                result.final_report = result.failure_summary
                return result
        else:
            result.plan = plan

        if self._mode in {WorkMode.full, WorkMode.develop}:
            phases = self._parse_phases(result.plan or "Phase 1: Implementation")
            start_index = len(completed_phases) + 1
            for phase_index, phase_name in enumerate(phases[start_index - 1:], start_index):
                phase_result = self._run_phase(
                    requirement=requirement,
                    design=result.design or requirement,
                    plan=result.plan or requirement,
                    phase_name=phase_name,
                    phase_index=phase_index,
                    total_phases=len(phases),
                    completed_summaries=completed_summaries,
                )
                result.phases.append(phase_result)
                completed_summaries.append(f"Phase {phase_index}: {phase_result.summary[:500]}")
                if not phase_result.committed:
                    result.blocked = True
                    result.failure_summary = self._failure_summary or f"Phase {phase_index} failed."
                    result.final_report = result.failure_summary
                    return result

        if self._mode in {WorkMode.full, WorkMode.test}:
            ga_test = self._run_test(
                requirement,
                result.design or requirement,
                completed_summaries,
            )
            result.ga_test = ga_test
            if self._blocked:
                result.blocked = True
                result.failure_summary = self._failure_summary or "GA testing failed."
                result.final_report = result.failure_summary
                return result

        if self._mode == WorkMode.design:
            result.final_report = (
                "Design workflow completed.\n\n"
                f"Design:\n{result.design[:4000]}\n\n"
                f"Plan:\n{result.plan[:4000]}"
            )
            result.success = True
            return result

        result.final_report = self._generate_final_report(completed_summaries)
        result.success = not self._blocked
        result.blocked = self._blocked
        result.failure_summary = self._failure_summary
        return result
