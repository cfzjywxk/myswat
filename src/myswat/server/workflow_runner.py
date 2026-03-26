"""Daemon-owned workflow execution helpers."""

from __future__ import annotations

import logging
from pathlib import Path
import threading

import typer
from rich.console import Console

from myswat.config.settings import MySwatSettings, get_workflow_review_limit
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.memory.learn_triggers import submit_workflow_summary_learn_request
from myswat.memory.store import MemoryStore
from myswat.server import MySwatToolService
from myswat.server.mcp_stdio import MySwatMCPDispatcher
from myswat.server.workflow_client import LocalMCPToolClient, MCPWorkflowCoordinator
from myswat.workflow.kernel import WorkflowKernel, detect_incomplete_scope_report
from myswat.workflow.modes import WorkMode
from myswat.workflow.prd_support import derive_requirement_title, resolve_prd_requirement
from myswat.workflow.runtime import WorkflowRuntime

console = Console()
LOGGER = logging.getLogger(__name__)


def _normalize_workdir(workdir: str | None) -> str | None:
    if not workdir:
        return None
    return str(Path(workdir).expanduser().resolve())


def load_project_context(
    project_slug: str,
    workdir: str | None,
    *,
    settings: MySwatSettings | None = None,
    store: MemoryStore | None = None,
    project_row: dict | None = None,
) -> tuple[MySwatSettings, MemoryStore, dict, str | None]:
    settings = settings or MySwatSettings()
    if store is None:
        pool = TiDBPool(settings.tidb)
        ensure_schema(pool)
        store = MemoryStore(
            pool,
            tidb_embedding_model=settings.embedding.tidb_model,
            embedding_backend=settings.embedding.backend,
        )

    proj = project_row or store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = _normalize_workdir(workdir or proj.get("repo_path"))
    return settings, store, proj, effective_workdir


def get_workflow_agents(store: MemoryStore, project_id: int) -> tuple[dict, list[dict]]:
    dev_agent = store.get_agent(project_id, "developer")
    if not dev_agent:
        console.print("[red]Developer agent not found.[/red]")
        raise typer.Exit(1)

    qa_agents: list[dict] = []
    for qa_role in ("qa_main", "qa_vice"):
        qa_agent = store.get_agent(project_id, qa_role)
        if qa_agent:
            qa_agents.append(qa_agent)

    if not qa_agents:
        console.print("[red]No QA agents found.[/red]")
        raise typer.Exit(1)

    return dev_agent, qa_agents


def _derive_final_status_and_summary(result, *, cancelled: bool, requested_status: str) -> tuple[str, str]:
    # Pauses/cancellations come only from an explicit external stop request.
    # Everything else that does not end in a complete approved result is
    # blocked, including engine failures and "success=True" runs whose own
    # final report says the approved scope is still incomplete.
    if cancelled:
        final_status = requested_status
        final_summary = "Workflow paused." if requested_status == "paused" else "Workflow cancelled."
        return final_status, final_summary

    if bool(getattr(result, "success", False)):
        final_report = str(getattr(result, "final_report", "") or "")
        incomplete_scope_reasons = detect_incomplete_scope_report(final_report)
        if incomplete_scope_reasons:
            return (
                "blocked",
                "Workflow report says the approved scope is still incomplete: "
                + "; ".join(incomplete_scope_reasons[:3]),
            )
        return "completed", "Workflow completed successfully."

    failure_summary = str(getattr(result, "failure_summary", "") or "").strip()
    return "blocked", (failure_summary or "Workflow finished with unresolved review or test issues.")


def _get_architect_agent(store: MemoryStore, project_id: int) -> dict | None:
    return store.get_agent(project_id, "architect")


def run_workflow(
    project_slug: str,
    requirement: str,
    *,
    workdir: str | None = None,
    work_item_id: int | None = None,
    mode: WorkMode = WorkMode.full,
    with_ga_test: bool = False,
    auto_approve: bool = True,
    external_cancel_event: threading.Event | None = None,
    emit_console_output: bool = True,
    settings: MySwatSettings | None = None,
    store: MemoryStore | None = None,
    project_row: dict | None = None,
    service: MySwatToolService | None = None,
) -> int:
    if with_ga_test and mode != WorkMode.full:
        raise typer.BadParameter("--with-ga-test can only be used with the full workflow.")

    settings, store, proj, effective_workdir = load_project_context(
        project_slug,
        workdir,
        settings=settings,
        store=store,
        project_row=project_row,
    )
    submitted_requirement = requirement
    requirement_for_learn = submitted_requirement
    error_stage = "workflow_requirement_resolution"
    final_status = "blocked"
    final_summary = "Workflow blocked."
    resume_stage: str | None = None

    try:
        requirement_resolution = resolve_prd_requirement(
            store=store,
            project_id=int(proj["id"]),
            requirement=requirement,
        )
        requirement = requirement_resolution.effective_requirement
        requirement_for_learn = requirement
        error_stage = "workflow_setup"

        dev_agent, qa_agents = get_workflow_agents(store, int(proj["id"]))
        arch_agent = (
            _get_architect_agent(store, int(proj["id"]))
            if mode in {WorkMode.full, WorkMode.design}
            else None
        )

        if work_item_id is None:
            item_metadata: dict[str, object] = {
                "work_mode": mode.value,
                "execution_mode": "daemon",
                "submitted_via": "daemon_api",
                "requested_workdir": effective_workdir,
            }
            if requirement_resolution.uses_prd_artifact:
                item_metadata["source_prd_artifact_id"] = requirement_resolution.source_artifact_id
                item_metadata["source_prd_work_item_id"] = requirement_resolution.source_work_item_id
                if requirement_resolution.source_title:
                    item_metadata["source_prd_title"] = requirement_resolution.source_title
            if with_ga_test:
                item_metadata["with_ga_test"] = True
            work_item_id = store.create_work_item(
                project_id=int(proj["id"]),
                title=derive_requirement_title(
                    submitted_requirement=submitted_requirement,
                    resolution=requirement_resolution,
                ),
                description=submitted_requirement,
                item_type="design" if mode == WorkMode.design else "code_change",
                assigned_agent_id=int((arch_agent or dev_agent)["id"]),
                metadata_json=item_metadata,
            )
        else:
            task_state = store.get_work_item_state(work_item_id) or {}
            current_stage = task_state.get("current_stage")
            if isinstance(current_stage, str) and current_stage.strip():
                resume_stage = current_stage.strip()

        store.update_work_item_status(work_item_id, "in_progress")

        arch_runtime = WorkflowRuntime(agent_row=arch_agent) if arch_agent is not None else None
        dev_runtime = WorkflowRuntime(agent_row=dev_agent)
        qa_runtimes = [WorkflowRuntime(agent_row=qa_agent) for qa_agent in qa_agents]

        if emit_console_output:
            console.print(f"[bold]Requirement:[/bold] {submitted_requirement}")
            if requirement_resolution.uses_prd_artifact:
                detail = f"Resolved PRD artifact #{requirement_resolution.source_artifact_id}"
                if requirement_resolution.source_title:
                    detail += f" ({requirement_resolution.source_title})"
                console.print(f"[dim]{detail}[/dim]")
            console.print(f"[dim]Work item: {work_item_id}[/dim]")

        def _should_cancel() -> bool:
            # The daemon cancellation path already cancels open coordination rows,
            # notifies service waiters, and terminates managed worker processes.
            # The server-side orchestrator only needs to observe the request.
            return bool(external_cancel_event is not None and external_cancel_event.is_set())

        service = service or MySwatToolService(store)
        coordinator = MCPWorkflowCoordinator(LocalMCPToolClient(MySwatMCPDispatcher(service)))

        poll_interval_value = getattr(settings.workflow, "assignment_poll_interval_seconds", 1.0)
        try:
            poll_interval_seconds = float(poll_interval_value)
        except (TypeError, ValueError):
            poll_interval_seconds = 1.0

        timeout_value = getattr(settings.workflow, "assignment_timeout_seconds", 0)
        try:
            timeout_seconds_value = float(timeout_value)
        except (TypeError, ValueError):
            timeout_seconds_value = 0.0

        engine = WorkflowKernel(
            store=store,
            coordinator=coordinator,
            dev=dev_runtime,
            qas=qa_runtimes,
            arch=arch_runtime,
            project_id=int(proj["id"]),
            work_item_id=work_item_id,
            mode=mode,
            with_ga_test=with_ga_test,
            design_plan_review_limit=get_workflow_review_limit(
                settings.workflow,
                "design_plan_review_limit",
            ),
            dev_plan_review_limit=get_workflow_review_limit(
                settings.workflow,
                "dev_plan_review_limit",
            ),
            dev_code_review_limit=get_workflow_review_limit(
                settings.workflow,
                "dev_code_review_limit",
            ),
            ga_plan_review_limit=get_workflow_review_limit(
                settings.workflow,
                "ga_plan_review_limit",
            ),
            ga_test_review_limit=get_workflow_review_limit(
                settings.workflow,
                "ga_test_review_limit",
            ),
            auto_approve=auto_approve,
            should_cancel=_should_cancel,
            assignment_poll_interval_seconds=poll_interval_seconds,
            assignment_timeout_seconds=(
                None if timeout_seconds_value <= 0 else timeout_seconds_value
            ),
            repo_path=effective_workdir,
            requirements_skills_root=getattr(settings.workflow, "requirements_skills_root", ""),
            resume_stage=resume_stage,
        )

        error_stage = "workflow_execution"
        result = engine.run(requirement)
        current_item = store.get_work_item(work_item_id) or {}
        requested_status = str(current_item.get("status") or "")
        if requested_status not in {"cancelled", "paused"}:
            requested_status = "cancelled"
        final_status, final_summary = _derive_final_status_and_summary(
            result,
            cancelled=_should_cancel(),
            requested_status=requested_status,
        )

        current_state = store.get_work_item_state(work_item_id) or {}
        if not isinstance(current_state, dict):
            LOGGER.warning(
                "Unexpected work item state type during workflow finalization: work_item_id=%s type=%s",
                work_item_id,
                type(current_state).__name__,
            )
            current_state = {}
        store.update_work_item_status(work_item_id, final_status)
        if final_status == "completed":
            store.update_work_item_state(
                work_item_id,
                current_stage="workflow_completed",
                latest_summary=final_summary,
                next_todos=[],
                open_issues=[],
            )
        else:
            next_todos = current_state.get("next_todos")
            if not isinstance(next_todos, list):
                if next_todos is not None:
                    LOGGER.warning(
                        "Unexpected next_todos type during workflow finalization: work_item_id=%s type=%s",
                        work_item_id,
                        type(next_todos).__name__,
                    )
                next_todos = []
            open_issues = current_state.get("open_issues")
            if not isinstance(open_issues, list):
                if open_issues is not None:
                    LOGGER.warning(
                        "Unexpected open_issues type during workflow finalization: work_item_id=%s type=%s",
                        work_item_id,
                        type(open_issues).__name__,
                    )
                open_issues = []
            store.update_work_item_state(
                work_item_id,
                current_stage=current_state.get("current_stage"),
                latest_summary=final_summary,
                next_todos=next_todos,
                open_issues=open_issues,
            )
    except typer.Exit:
        raise
    except Exception as exc:
        from myswat.workflow.error_handler import WorkflowError, handle_workflow_error

        if work_item_id is None:
            raise

        error_requirement = (
            submitted_requirement if error_stage == "workflow_requirement_resolution" else requirement
        )
        werr = WorkflowError(
            error=exc,
            stage=error_stage,
            context={
                "project": project_slug,
                "requirement": error_requirement[:500],
                "work_item_id": work_item_id,
            },
        )
        handle_workflow_error(werr, store=store, project_id=int(proj["id"]))
        final_status = "blocked"
        final_summary = f"Workflow crashed: {type(exc).__name__}"
        try:
            store.update_work_item_status(work_item_id, final_status)
        except Exception:
            pass
    finally:
        try:
            submit_workflow_summary_learn_request(
                store=store,
                settings=settings,
                project_id=int(proj["id"]),
                source_work_item_id=work_item_id,
                source_session_id=None,
                requirement=requirement_for_learn,
                final_status=final_status,
                final_summary=final_summary,
                mode=mode.value,
                workdir=effective_workdir,
            )
        except Exception:
            pass

    if emit_console_output:
        console.print("\n[dim]Workflow state persisted to TiDB.[/dim]")
    return work_item_id
