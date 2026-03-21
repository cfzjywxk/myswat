"""Daemon-owned workflow execution helpers."""

from __future__ import annotations

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
from myswat.workflow.kernel import WorkflowKernel
from myswat.workflow.modes import WorkMode
from myswat.workflow.runtime import WorkflowRuntime

console = Console()


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

    dev_agent, qa_agents = get_workflow_agents(store, int(proj["id"]))
    arch_agent = _get_architect_agent(store, int(proj["id"])) if mode in {WorkMode.full, WorkMode.design} else None

    if work_item_id is None:
        item_metadata: dict[str, object] = {
            "work_mode": mode.value,
            "execution_mode": "daemon",
            "submitted_via": "daemon_api",
            "requested_workdir": effective_workdir,
        }
        if with_ga_test:
            item_metadata["with_ga_test"] = True
        work_item_id = store.create_work_item(
            project_id=int(proj["id"]),
            title=requirement[:200],
            description=requirement,
            item_type="design" if mode == WorkMode.design else "code_change",
            assigned_agent_id=int((arch_agent or dev_agent)["id"]),
            metadata_json=item_metadata,
        )

    store.update_work_item_status(work_item_id, "in_progress")

    arch_runtime = WorkflowRuntime(agent_row=arch_agent) if arch_agent is not None else None
    dev_runtime = WorkflowRuntime(agent_row=dev_agent)
    qa_runtimes = [WorkflowRuntime(agent_row=qa_agent) for qa_agent in qa_agents]

    if emit_console_output:
        console.print(f"[bold]Requirement:[/bold] {requirement}")
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
    )

    final_status = "blocked"
    final_summary = "Workflow blocked."

    try:
        result = engine.run(requirement)
        if _should_cancel():
            current_item = store.get_work_item(work_item_id) or {}
            requested_status = str(current_item.get("status") or "")
            if requested_status not in {"cancelled", "paused"}:
                requested_status = "cancelled"
            final_status = requested_status
            final_summary = "Workflow paused." if requested_status == "paused" else "Workflow cancelled."
        elif result.success:
            final_status = "completed"
            final_summary = "Workflow completed successfully."
        else:
            final_status = "blocked"
            failure_summary = str(getattr(result, "failure_summary", "") or "").strip()
            final_summary = failure_summary or "Workflow finished with unresolved review or test issues."

        store.update_work_item_status(work_item_id, final_status)
    except Exception as exc:
        from myswat.workflow.error_handler import WorkflowError, handle_workflow_error

        werr = WorkflowError(
            error=exc,
            stage="workflow_execution",
            context={
                "project": project_slug,
                "requirement": requirement[:500],
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
                requirement=requirement,
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
