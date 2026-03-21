"""Stage-oriented workflow kernel backed by queued MCP assignments."""

from __future__ import annotations

from datetime import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from myswat.large_payloads import resolve_externalized_text, resolve_externalized_value
from myswat.repo_ops import (
    commit_repo_changes,
    list_changed_repo_paths,
    probe_git_repository,
    push_repo_changes,
    write_design_plan_doc,
    write_workflow_report_doc,
)
from myswat.server import (
    ReviewCycleCancellationRequest,
    ReviewRequest,
    ReviewWaitRequest,
    StageRunStart,
    StageRunWaitRequest,
    StatusReport,
)
from myswat.server.workflow_client import WorkflowCoordinator
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
        dev: WorkflowRuntime,
        qas: list[WorkflowRuntime],
        project_id: int,
        work_item_id: int,
        mode: WorkMode = WorkMode.full,
        with_ga_test: bool = False,
        max_review_iterations: int | None = None,
        design_plan_review_limit: int | None = None,
        dev_plan_review_limit: int | None = None,
        dev_code_review_limit: int | None = None,
        ga_plan_review_limit: int | None = None,
        ga_test_review_limit: int | None = None,
        ask_user: Callable[[str], str] | None = None,
        auto_approve: bool = True,
        should_cancel: Callable[[], bool] | None = None,
        arch: WorkflowRuntime | None = None,
        resume_stage: str | None = None,
        on_event: Callable[[WorkflowEvent], None] | None = None,
        assignment_poll_interval_seconds: float = 1.0,
        assignment_timeout_seconds: float | None = None,
        coordinator: WorkflowCoordinator | None = None,
        service: WorkflowCoordinator | None = None,
        repo_path: str | None = None,
    ) -> None:
        self._store = store
        self._coordinator = coordinator or service
        if self._coordinator is None:
            raise ValueError("coordinator is required")
        self._dev = dev
        self._qas = qas
        self._arch = arch
        self._project_id = project_id
        self._work_item_id = work_item_id
        self._mode = WorkMode(mode)
        self._with_ga_test = with_ga_test
        default_review_limits = {
            "design": 10,
            "dev_plan": 10,
            "dev_code": 10,
            "ga_plan": 2,
            "ga_test": 2,
        }
        fallback_review_limit = max_review_iterations
        self._design_plan_review_limit = (
            default_review_limits["design"]
            if design_plan_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if design_plan_review_limit is None else design_plan_review_limit
            )
        )
        self._dev_plan_review_limit = (
            default_review_limits["dev_plan"]
            if dev_plan_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if dev_plan_review_limit is None else dev_plan_review_limit
            )
        )
        self._dev_code_review_limit = (
            default_review_limits["dev_code"]
            if dev_code_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if dev_code_review_limit is None else dev_code_review_limit
            )
        )
        self._ga_plan_review_limit = (
            default_review_limits["ga_plan"]
            if ga_plan_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if ga_plan_review_limit is None else ga_plan_review_limit
            )
        )
        self._ga_test_review_limit = (
            default_review_limits["ga_test"]
            if ga_test_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if ga_test_review_limit is None else ga_test_review_limit
            )
        )
        self._max_review = max(
            self._design_plan_review_limit,
            self._dev_plan_review_limit,
            self._dev_code_review_limit,
            self._ga_plan_review_limit,
            self._ga_test_review_limit,
        )
        self._ask = ask_user or (lambda prompt: input(f"\n{prompt}").strip())
        self._auto_approve = auto_approve
        self._should_cancel = should_cancel
        self._resume_stage = resume_stage
        self._on_event = on_event
        self._blocked = False
        self._failure_summary = ""
        self._last_review_limit_reached = False
        self._last_review_limit_stage = ""
        self._last_review_limit_summary = ""
        self._assignment_poll_interval_seconds = assignment_poll_interval_seconds
        self._assignment_timeout_seconds = assignment_timeout_seconds
        self._repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self._repo_commit_ready = False
        self._repo_commit_checked = False
        self._repo_commit_skip_reason = ""
        self._repo_initial_dirty_paths: set[str] = set()
        self._repo_managed_paths: set[str] = set()
        self._repo_commits_created = False

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

    def _record_repo_event(
        self,
        *,
        event_type: str,
        title: str,
        summary: str,
        stage: str,
        warning: bool = False,
    ) -> None:
        if self._work_item_id:
            self._append_process_event(
                event_type=event_type,
                title=title,
                summary=summary,
                from_role="myswat",
                to_role="user",
            )
        self._emit(
            "warning" if warning else "info",
            summary,
            stage=stage,
        )

    def _ensure_repo_commit_ready(self) -> None:
        if self._repo_commit_checked:
            return
        self._repo_commit_checked = True
        if self._repo_path is None:
            self._repo_commit_skip_reason = "Repository path is not configured; skipping automatic local commits."
            return

        status = probe_git_repository(self._repo_path)
        if not status.available:
            self._repo_commit_skip_reason = status.message or "git CLI is not available; skipping automatic local commits."
        elif not status.is_git_repo:
            self._repo_commit_skip_reason = status.message or "Repository is not under git; skipping automatic local commits."
        else:
            self._repo_commit_ready = True
            if not status.clean:
                dirty = list_changed_repo_paths(self._repo_path)
                if dirty.ok:
                    self._repo_initial_dirty_paths = set(dirty.paths)
                self._record_repo_event(
                    event_type="repo_commit_scope",
                    title="Automatic local commits scoped to workflow paths",
                    summary=(
                        "Repository has pre-existing uncommitted changes; "
                        "MySwat will commit only workflow-owned paths."
                    ),
                    stage="repo",
                    warning=True,
                )
            return

        if self._repo_commit_skip_reason:
            self._record_repo_event(
                event_type="repo_commit_skipped",
                title="Automatic local commits disabled",
                summary=self._repo_commit_skip_reason,
                stage="repo",
                warning=True,
            )

    def _repo_relative_path(self, path: str | Path | None) -> str | None:
        if self._repo_path is None or path is None:
            return None
        candidate = Path(str(path)).expanduser()
        if candidate.is_absolute():
            try:
                return str(candidate.resolve().relative_to(self._repo_path))
            except ValueError:
                return None
        normalized_parts: list[str] = []
        for part in candidate.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                return None
            normalized_parts.append(part)
        if not normalized_parts:
            return None
        return Path(*normalized_parts).as_posix()

    def _register_managed_repo_path(self, path: str | Path | None) -> None:
        relative = self._repo_relative_path(path)
        if relative:
            self._repo_managed_paths.add(relative)

    def _extract_repo_paths_from_text(self, text: str) -> set[str]:
        if self._repo_path is None or not text:
            return set()

        results: set[str] = set()
        for raw in re.findall(r"`([^`\n]+)`", text):
            candidate = raw.strip()
            candidate = re.sub(r":\d+(?::\d+)?$", "", candidate)
            if not candidate:
                continue
            if not Path(candidate).is_absolute() and "/" not in candidate and not (self._repo_path / candidate).exists():
                continue
            relative = self._repo_relative_path(candidate)
            if relative:
                results.add(relative)
        return results

    def _current_workflow_repo_paths(self, *preferred_paths: str | Path) -> list[Path]:
        if self._repo_path is None:
            return []

        changed = list_changed_repo_paths(self._repo_path)
        if not changed.ok:
            return []

        selected = {path for path in changed.paths if path not in self._repo_initial_dirty_paths}
        selected.update(path for path in self._repo_managed_paths if path in changed.paths)
        for preferred in preferred_paths:
            relative = self._repo_relative_path(preferred)
            if relative and relative in changed.paths and relative not in self._repo_initial_dirty_paths:
                selected.add(relative)
        return [self._repo_path / path for path in sorted(selected)]

    def _export_design_plan_to_docs(
        self,
        *,
        requirement: str,
        design: str,
        plan: str,
    ) -> bool:
        if self._repo_path is None or not design.strip() or not plan.strip():
            return True

        try:
            doc_path = write_design_plan_doc(
                self._repo_path,
                requirement=requirement,
                design=design,
                plan=plan,
            )
        except Exception as exc:
            self._blocked = True
            self._failure_summary = f"Failed to export design plan doc: {exc}"
            self._record_repo_event(
                event_type="repo_export_failed",
                title="Design plan export failed",
                summary=self._failure_summary,
                stage="plan",
                warning=True,
            )
            return False

        self._register_managed_repo_path(doc_path)
        summary = f"Exported the approved design plan to {doc_path.relative_to(self._repo_path)}."
        self._record_repo_event(
            event_type="repo_exported",
            title="Design plan exported",
            summary=summary,
            stage="plan",
        )

        if not self._repo_commit_ready:
            return True

        commit_result = commit_repo_changes(
            self._repo_path,
            message="docs: sync myswat design plan",
            paths=[doc_path],
        )
        if not commit_result.ok:
            self._blocked = True
            self._failure_summary = (
                "Failed to commit the exported design plan doc. "
                f"{commit_result.message}".strip()
            )
            self._record_repo_event(
                event_type="repo_commit_failed",
                title="Design plan commit failed",
                summary=self._failure_summary,
                stage="plan",
                warning=True,
            )
            return False

        self._record_repo_event(
            event_type="repo_commit",
            title="Design plan committed" if commit_result.committed else "Design plan commit skipped",
            summary=commit_result.message or "No design-plan doc changes to commit.",
            stage="plan",
        )
        if commit_result.committed:
            self._repo_commits_created = True
        return True

    def _export_final_report_to_docs(self, report: str) -> bool:
        if self._repo_path is None or not report.strip():
            return True

        try:
            doc_path = write_workflow_report_doc(
                self._repo_path,
                report=report,
                work_mode=self._mode.value,
            )
        except Exception as exc:
            self._blocked = True
            self._failure_summary = f"Failed to export workflow report: {exc}"
            self._record_repo_event(
                event_type="repo_export_failed",
                title="Workflow report export failed",
                summary=self._failure_summary,
                stage="report",
                warning=True,
            )
            return False

        self._register_managed_repo_path(doc_path)
        self._record_repo_event(
            event_type="repo_exported",
            title="Workflow report exported",
            summary=f"Exported the workflow report to {doc_path.relative_to(self._repo_path)}.",
            stage="report",
        )

        if not self._repo_commit_ready or self._mode in {WorkMode.full, WorkMode.develop}:
            return True

        commit_result = commit_repo_changes(
            self._repo_path,
            message=f"docs: add myswat {self._mode.value} report",
            paths=[doc_path],
        )
        if not commit_result.ok:
            self._blocked = True
            self._failure_summary = (
                "Failed to commit the exported workflow report. "
                f"{commit_result.message}".strip()
            )
            self._record_repo_event(
                event_type="repo_commit_failed",
                title="Workflow report commit failed",
                summary=self._failure_summary,
                stage="report",
                warning=True,
            )
            return False

        self._record_repo_event(
            event_type="repo_commit",
            title="Workflow report committed" if commit_result.committed else "Workflow report commit skipped",
            summary=commit_result.message or "No workflow-report changes to commit.",
            stage="report",
        )
        if commit_result.committed:
            self._repo_commits_created = True
        return True

    @staticmethod
    def _format_commit_model_label(runtime: WorkflowRuntime | None) -> str:
        if runtime is None:
            return ""

        model_name = runtime.model_name.strip()
        if not model_name:
            return ""

        lowered = model_name.lower()
        if lowered.startswith("gpt-"):
            return "GPT-" + model_name[4:]

        claude_tail = lowered.removeprefix("claude-")
        claude_parts = [part for part in claude_tail.split("-") if part]
        while claude_parts and (claude_parts[-1] == "latest" or (claude_parts[-1].isdigit() and len(claude_parts[-1]) >= 6)):
            claude_parts.pop()

        claude_families = ("opus", "sonnet", "haiku")
        family = next((part for part in claude_parts if part in claude_families), "")
        if family:
            family_index = claude_parts.index(family)
            if family_index == 0:
                version_tokens = [part for part in claude_parts[1:] if part.isdigit()]
            else:
                version_tokens = [part for part in claude_parts[:family_index] if part.isdigit()]

            version = ""
            if len(version_tokens) >= 2:
                version = f"{version_tokens[0]}.{version_tokens[1]}"
            elif version_tokens:
                version = version_tokens[0]

            return f"{family.capitalize()} {version}".strip()

        model_tail = model_name.split("/")[-1]
        cleaned = model_tail.replace("_", " ").replace("-", " ").strip()
        return cleaned.title() if cleaned else model_name

    def _commit_trailers_for(self, runtime: WorkflowRuntime | None) -> list[str]:
        model_label = self._format_commit_model_label(runtime)
        if not model_label:
            return []
        return [f"Co-Authored-By: MySwat Dev ({model_label}) <noreply@myswat.invalid>"]

    def _commit_test_changes(self) -> bool:
        if self._repo_path is None:
            return True
        if not self._repo_commit_ready:
            self._record_repo_event(
                event_type="repo_commit_skipped",
                title="Test workflow local commit skipped",
                summary=self._repo_commit_skip_reason or "Automatic local commits are disabled.",
                stage="ga_test_commit",
                warning=True,
            )
            return True

        # The first QA runtime acts as the GA test lead throughout _run_test,
        # so use the same runtime for commit attribution.
        qa_lead = self._qas[0] if self._qas else None
        commit_paths = self._current_workflow_repo_paths()
        if not commit_paths:
            return True
        commit_result = commit_repo_changes(
            self._repo_path,
            message="test: sync approved test changes",
            paths=commit_paths,
            trailers=self._commit_trailers_for(qa_lead),
        )
        if not commit_result.ok:
            self._blocked = True
            self._failure_summary = (
                "Failed to commit approved test changes. "
                f"{commit_result.message}".strip()
            )
            self._record_repo_event(
                event_type="repo_commit_failed",
                title="Test workflow local commit failed",
                summary=self._failure_summary,
                stage="ga_test_commit",
                warning=True,
            )
            return False

        self._record_repo_event(
            event_type="repo_commit",
            title="Test workflow local changes committed" if commit_result.committed else "Test workflow local commit skipped",
            summary=commit_result.message or "No approved test changes to commit.",
            stage="ga_test_commit",
        )
        if commit_result.committed:
            self._repo_commits_created = True
        return True

    def _commit_phase_changes(self, *, phase_index: int, phase_name: str, summary: str) -> bool:
        if self._repo_path is None:
            return True
        if not self._repo_commit_ready:
            self._record_repo_event(
                event_type="repo_commit_skipped",
                title=f"Phase {phase_index} local commit skipped",
                summary=self._repo_commit_skip_reason or "Automatic local commits are disabled.",
                stage=f"phase_{phase_index}_commit",
                warning=True,
            )
            return True

        preferred_paths = sorted(self._extract_repo_paths_from_text(summary))
        commit_paths = self._current_workflow_repo_paths(*preferred_paths)
        if not commit_paths:
            return True
        commit_result = commit_repo_changes(
            self._repo_path,
            message=f"phase {phase_index}: {phase_name}",
            paths=commit_paths,
            trailers=self._commit_trailers_for(self._dev),
        )
        if not commit_result.ok:
            self._blocked = True
            self._failure_summary = (
                f"Failed to commit phase {phase_index}: {phase_name}. "
                f"{commit_result.message}".strip()
            )
            self._record_repo_event(
                event_type="repo_commit_failed",
                title=f"Phase {phase_index} local commit failed",
                summary=self._failure_summary,
                stage=f"phase_{phase_index}_commit",
                warning=True,
            )
            return False

        self._record_repo_event(
            event_type="repo_commit",
            title=f"Phase {phase_index} local changes committed" if commit_result.committed else f"Phase {phase_index} local commit skipped",
            summary=commit_result.message or "No phase changes to commit.",
            stage=f"phase_{phase_index}_commit",
        )
        if commit_result.committed:
            self._repo_commits_created = True
        return True

    def _finalize_workflow_repo_sync(self) -> bool:
        if self._repo_path is None:
            return True
        if not self._repo_commit_ready:
            self._record_repo_event(
                event_type="repo_push_skipped",
                title="Final workflow push skipped",
                summary=self._repo_commit_skip_reason or "Automatic repo sync is disabled.",
                stage="repo",
                warning=True,
            )
            return True

        commit_paths = self._current_workflow_repo_paths(*sorted(self._repo_managed_paths))
        if commit_paths:
            commit_result = commit_repo_changes(
                self._repo_path,
                message=f"workflow: finalize {self._mode.value}",
                paths=commit_paths,
            )
            if not commit_result.ok:
                self._blocked = True
                self._failure_summary = (
                    f"Failed to commit final {self._mode.value} workflow changes. "
                    f"{commit_result.message}".strip()
                )
                self._record_repo_event(
                    event_type="repo_commit_failed",
                    title="Final workflow commit failed",
                    summary=self._failure_summary,
                    stage="repo",
                    warning=True,
                )
                return False

            self._record_repo_event(
                event_type="repo_commit",
                title="Final workflow changes committed" if commit_result.committed else "Final workflow commit skipped",
                summary=commit_result.message or "No final workflow changes to commit.",
                stage="repo",
            )
            if commit_result.committed:
                self._repo_commits_created = True

        if not self._repo_commits_created:
            self._record_repo_event(
                event_type="repo_push_skipped",
                title="Final workflow push skipped",
                summary="No workflow commits were created during this run; skipping final push.",
                stage="repo",
            )
            return True

        push_result = push_repo_changes(self._repo_path)
        if not push_result.ok:
            self._blocked = True
            self._failure_summary = (
                f"Failed to push final {self._mode.value} workflow changes. "
                f"{push_result.message}".strip()
            )
            self._record_repo_event(
                event_type="repo_push_failed",
                title="Final workflow push failed",
                summary=self._failure_summary,
                stage="repo",
                warning=True,
            )
            return False

        self._record_repo_event(
            event_type="repo_push",
            title="Final workflow changes pushed" if push_result.pushed else "Final workflow push skipped",
            summary=push_result.message or "Pushed final workflow commits.",
            stage="repo",
        )
        return True

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
        self._coordinator.report_status(
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

    def _append_process_event(
        self,
        *,
        event_type: str,
        title: str,
        summary: str,
        from_role: str | None,
        to_role: str | None = None,
        updated_by_agent_id: int | None = None,
    ) -> None:
        self._store.append_work_item_process_event(
            self._work_item_id,
            event_type=event_type,
            title=title,
            summary=summary,
            from_role=from_role,
            to_role=to_role,
            updated_by_agent_id=updated_by_agent_id,
        )

    def _reset_review_loop_state(self) -> None:
        self._last_review_limit_reached = False
        self._last_review_limit_stage = ""
        self._last_review_limit_summary = ""

    def _review_limit_for(self, review_stage_name: str) -> int:
        if review_stage_name == "design_review":
            return self._design_plan_review_limit
        if review_stage_name == "plan_review":
            return self._dev_plan_review_limit
        if review_stage_name == "test_plan_review":
            return self._ga_plan_review_limit
        if review_stage_name.startswith("phase_") and review_stage_name.endswith("_review"):
            return self._dev_code_review_limit
        if review_stage_name.startswith("ga_test"):
            return self._ga_test_review_limit
        return self._max_review

    def _record_review_limit_reached(
        self,
        *,
        artifact_title: str,
        stage: str,
        iteration: int,
        max_iterations: int,
        owner: WorkflowRuntime,
        issues: list[str],
    ) -> None:
        summary = (
            f"Max review iterations reached for {artifact_title} after {iteration} round(s); "
            "continuing with the latest artifact without another review round."
        )
        unresolved = [issue for issue in issues if issue][:8]
        event_summary = summary
        if unresolved:
            event_summary += "\nUnresolved review issues:\n" + "\n".join(f"- {issue}" for issue in unresolved)

        self._last_review_limit_reached = True
        self._last_review_limit_stage = stage
        self._last_review_limit_summary = summary

        self._emit(
            "warning",
            summary,
            stage=stage,
            agent_role=owner.agent_role,
            iteration=iteration,
            max_iterations=max_iterations,
            issue_count=len(issues),
            review_skipped=True,
        )
        self._store.update_work_item_state(
            self._work_item_id,
            current_stage=f"{stage}_skipped",
            latest_summary=summary,
            next_todos=["Continue with the latest artifact despite unresolved review issues"],
            open_issues=issues[:20],
            updated_by_agent_id=owner.agent_id,
        )
        self._append_process_event(
            event_type="review_skipped",
            title=f"{artifact_title} review skipped after max iterations",
            summary=event_summary,
            from_role="myswat",
            to_role=owner.agent_role,
            updated_by_agent_id=owner.agent_id,
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
        stage_run = self._coordinator.start_stage_run(
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
        result = self._coordinator.wait_for_stage_run_completion(
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
        return self._coordinator.wait_for_review_verdicts(
            ReviewWaitRequest(
                cycle_ids=cycle_ids,
                poll_interval_seconds=self._assignment_poll_interval_seconds,
                timeout_seconds=self._assignment_timeout_seconds,
                return_on_failed=True,
            )
        )

    def _cancel_review_cycles(self, cycle_ids: list[int], *, summary: str) -> None:
        if not cycle_ids:
            return
        self._coordinator.cancel_review_cycles(
            ReviewCycleCancellationRequest(
                cycle_ids=cycle_ids,
                summary=summary,
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
        self._reset_review_loop_state()
        artifact = initial_artifact
        artifact_id = initial_artifact_id
        review_limit = self._review_limit_for(review_stage_name)
        approved_reviewers: set[int] = set()
        for iteration in range(1, review_limit + 1):
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

            # Keep approvals sticky within a phase so only unresolved reviewers
            # see later revisions. This intentionally favors shorter review
            # loops over re-reviewing every revision with already-satisfied
            # reviewers in the same phase.
            pending_reviewers = [
                reviewer for reviewer in reviewers if reviewer.agent_id not in approved_reviewers
            ]
            if not pending_reviewers:
                completed_iteration = max(iteration - 1, 0)
                return artifact, completed_iteration, True

            self._emit(
                "review_start",
                f"Queued review iteration {iteration}/{review_limit}",
                stage=review_stage_name,
                agent_role=owner.agent_role,
                iteration=iteration,
                max_iterations=review_limit,
            )
            cycle_ids: list[int] = []
            cycle_to_reviewer: dict[int, WorkflowRuntime] = {}
            for reviewer in pending_reviewers:
                request = self._coordinator.request_review(
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
                cycle_to_reviewer[request.cycle_id] = reviewer

            verdicts = self._wait_for_review_verdicts(cycle_ids)
            failed_verdicts = [verdict for verdict in verdicts if verdict.verdict == "failed"]
            all_lgtm = True
            collected_feedback: list[str] = []
            for verdict in verdicts:
                reviewer = cycle_to_reviewer.get(verdict.cycle_id)
                if reviewer is None:
                    role_matches = [
                        candidate for candidate in pending_reviewers
                        if candidate.agent_role == verdict.reviewer_role
                    ]
                    if len(role_matches) == 1:
                        reviewer = role_matches[0]
                        self._emit(
                            "warning",
                            (
                                f"Recovered reviewer mapping for cycle {verdict.cycle_id} "
                                f"via reviewer_role={verdict.reviewer_role}."
                            ),
                            stage=review_stage_name,
                            agent_role=owner.agent_role,
                            reviewer_role=verdict.reviewer_role,
                            cycle_id=verdict.cycle_id,
                        )
                    elif len(role_matches) > 1:
                        self._emit(
                            "warning",
                            (
                                f"Review verdict cycle {verdict.cycle_id} matched multiple "
                                f"pending reviewers for reviewer_role={verdict.reviewer_role!r}; "
                                "approval tracking skipped and the reviewer will be re-queued."
                            ),
                            stage=review_stage_name,
                            agent_role=owner.agent_role,
                            reviewer_role=verdict.reviewer_role,
                            cycle_id=verdict.cycle_id,
                        )
                    else:
                        self._emit(
                            "warning",
                            (
                                f"Could not map review verdict cycle {verdict.cycle_id} "
                                f"to a pending reviewer; using reviewer_role={verdict.reviewer_role!r}."
                            ),
                            stage=review_stage_name,
                            agent_role=owner.agent_role,
                            reviewer_role=verdict.reviewer_role,
                            cycle_id=verdict.cycle_id,
                        )
                reviewer_role = reviewer.agent_role if reviewer is not None else verdict.reviewer_role
                self._emit(
                    "review_verdict",
                    f"{reviewer_role}: {verdict.verdict.upper()}",
                    stage=review_stage_name,
                    agent_role=reviewer_role,
                    detail=verdict.summary,
                    verdict=verdict.verdict,
                )
                if verdict.verdict == "lgtm":
                    if reviewer is not None:
                        approved_reviewers.add(reviewer.agent_id)
                else:
                    all_lgtm = False
                    if reviewer is not None:
                        approved_reviewers.discard(reviewer.agent_id)
                    issues = verdict.issues or [verdict.summary]
                    collected_feedback.extend(
                        [f"[{reviewer_role}] {issue}" for issue in issues if issue]
                    )

            if failed_verdicts:
                returned_cycle_ids = {verdict.cycle_id for verdict in verdicts}
                remaining_cycle_ids = [cycle_id for cycle_id in cycle_ids if cycle_id not in returned_cycle_ids]
                if remaining_cycle_ids:
                    self._cancel_review_cycles(
                        remaining_cycle_ids,
                        summary=(
                            f"Cancelled remaining {review_stage_name} cycles after a sibling "
                            f"review failed in iteration {iteration}."
                        ),
                    )
                failed = failed_verdicts[0]
                failure_summary = failed.summary or "review failed after retry exhaustion."
                if not failure_summary.lstrip().startswith("["):
                    failure_summary = f"[{failed.reviewer_role}] {failure_summary}"
                self._emit(
                    "error",
                    failure_summary,
                    stage=review_stage_name,
                    agent_role=failed.reviewer_role,
                    detail=failed.summary,
                    verdict=failed.verdict,
                    cancelled_cycle_ids=remaining_cycle_ids,
                )
                self._blocked = True
                self._failure_summary = failure_summary
                return artifact, iteration, False

            if all_lgtm:
                return artifact, iteration, True

            if iteration >= review_limit:
                self._record_review_limit_reached(
                    artifact_title=artifact_title,
                    stage=review_stage_name,
                    iteration=iteration,
                    max_iterations=review_limit,
                    owner=owner,
                    issues=collected_feedback,
                )
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
        return artifact, review_limit, False

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
        review_limit_reached = self._last_review_limit_reached
        if not passed and not review_limit_reached:
            return reviewed, False
        checkpoint_prompt = (
            "Review limit reached for design. Continue with latest design? [Y/n/or type feedback] "
            if review_limit_reached
            else "Design approved by reviewers. Accept? [Y/n/or type feedback] "
        )
        reviewed, ok = self._checkpoint(
            reviewed,
            prompt=checkpoint_prompt,
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
            summary="Create implementation plan",
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
        review_limit_reached = self._last_review_limit_reached
        if not passed and not review_limit_reached:
            return reviewed, False
        checkpoint_prompt = (
            "Review limit reached for plan. Continue with latest plan? [Y/n/or type feedback] "
            if review_limit_reached
            else "Plan approved by reviewers. Accept? [Y/n/or type feedback] "
        )
        reviewed, ok = self._checkpoint(
            reviewed,
            prompt=checkpoint_prompt,
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
        review_limit_reached = self._last_review_limit_reached
        if not passed and not review_limit_reached:
            return PhaseResult(
                name=phase_name,
                summary=reviewed_summary,
                review_iterations=review_iterations,
                review_passed=False,
                committed=False,
            )

        if not self._commit_phase_changes(phase_index=phase_index, phase_name=phase_name, summary=reviewed_summary):
            return PhaseResult(
                name=phase_name,
                summary=reviewed_summary,
                review_iterations=review_iterations,
                review_passed=passed,
                committed=False,
            )

        return PhaseResult(
            name=phase_name,
            summary=reviewed_summary,
            review_iterations=review_iterations,
            review_passed=passed,
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
        review_limit_reached = self._last_review_limit_reached
        if not passed and not review_limit_reached:
            return GATestResult(test_plan=reviewed_plan)
        checkpoint_prompt = (
            "Review limit reached for test plan. Continue with latest test plan? [Y/n/or type feedback] "
            if review_limit_reached
            else "Test plan approved. Start testing? [Y/n/or type feedback] "
        )
        reviewed_plan, ok = self._checkpoint(
            reviewed_plan,
            prompt=checkpoint_prompt,
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
        ga_result = GATestResult(
            test_plan=reviewed_plan,
            test_report=test_completion.artifact_content,
            passed=(status == "pass"),
            bugs_found=int(payload.get("tests_failed") or len(payload.get("bugs") or [])),
        )
        if ga_result.passed:
            self._commit_test_changes()
        return ga_result

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
        self._ensure_repo_commit_ready()

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

        if result.design and result.plan:
            if not self._export_design_plan_to_docs(
                requirement=requirement,
                design=result.design,
                plan=result.plan,
            ):
                result.blocked = True
                result.failure_summary = self._failure_summary or "Failed to export design plan."
                result.final_report = result.failure_summary
                return result

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

        if self._mode == WorkMode.test or (self._mode == WorkMode.full and self._with_ga_test):
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
            if not self._export_final_report_to_docs(result.final_report):
                result.blocked = True
                result.failure_summary = self._failure_summary or "Failed to export workflow report."
                result.final_report = result.failure_summary
                return result
            result.success = True
            return result

        result.final_report = self._generate_final_report(completed_summaries)
        if not self._export_final_report_to_docs(result.final_report):
            result.blocked = True
            result.failure_summary = self._failure_summary or "Failed to export workflow report."
            result.final_report = result.failure_summary
            return result
        if self._mode in {WorkMode.full, WorkMode.develop}:
            if not self._finalize_workflow_repo_sync():
                result.blocked = True
                result.failure_summary = self._failure_summary or "Failed to sync workflow commits."
                result.final_report = result.failure_summary
                return result
        result.success = not self._blocked
        result.blocked = self._blocked
        result.failure_summary = self._failure_summary
        return result
