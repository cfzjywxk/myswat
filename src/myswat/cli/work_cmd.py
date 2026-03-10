"""myswat work — run the full teamwork workflow from CLI."""

from __future__ import annotations

import json
import threading

import typer
from rich.console import Console

from myswat.agents.base import AgentRunner
from myswat.agents.codex_runner import CodexRunner
from myswat.agents.kimi_runner import KimiRunner
from myswat.agents.session_manager import SessionManager
from myswat.cli.progress import _run_with_task_monitor
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import run_migrations
from myswat.memory.compactor import KnowledgeCompactor
from myswat.memory.store import MemoryStore
from myswat.workflow.engine import WorkflowEngine

console = Console()


def _make_runner(agent_row: dict) -> AgentRunner:
    backend = agent_row["cli_backend"]
    cli_path = agent_row["cli_path"]
    model = agent_row["model_name"]
    extra_flags = json.loads(agent_row["cli_extra_args"]) if agent_row.get("cli_extra_args") else []

    if backend == "codex":
        return CodexRunner(cli_path=cli_path, model=model, extra_flags=extra_flags)
    elif backend == "kimi":
        return KimiRunner(cli_path=cli_path, model=model, extra_flags=extra_flags)
    else:
        raise typer.BadParameter(f"Unknown CLI backend: {backend}")


def run_work(
    project_slug: str,
    requirement: str,
    workdir: str | None = None,
) -> None:
    """Run the full teamwork workflow."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    applied = run_migrations(pool)
    if applied:
        console.print(f"[dim]Applied schema migrations: {applied}[/dim]")
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = workdir or proj.get("repo_path")

    # Auto-learn if project hasn't been learned yet
    from myswat.cli.learn_cmd import ensure_learned
    ensure_learned(store, project_slug, proj["id"], effective_workdir)

    # Set up Dev
    dev_agent = store.get_agent(proj["id"], "developer")
    if not dev_agent:
        console.print("[red]Developer agent not found.[/red]")
        raise typer.Exit(1)

    dev_runner = _make_runner(dev_agent)
    dev_runner.workdir = effective_workdir

    # Compactor
    compaction_runner = _make_runner(dev_agent)
    compactor = KnowledgeCompactor(
        store=store,
        runner=compaction_runner,
        threshold_turns=settings.compaction.threshold_turns,
        threshold_tokens=settings.compaction.threshold_tokens,
    )

    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], compactor=compactor,
    )

    # Set up QA(s)
    qa_sms = []
    for qa_role in ("qa_main", "qa_vice"):
        qa_agent = store.get_agent(proj["id"], qa_role)
        if qa_agent:
            qa_runner = _make_runner(qa_agent)
            qa_runner.workdir = effective_workdir
            qa_sm = SessionManager(
                store=store, runner=qa_runner, agent_row=qa_agent,
                project_id=proj["id"], compactor=compactor,
            )
            qa_sms.append(qa_sm)

    if not qa_sms:
        console.print("[red]No QA agents found.[/red]")
        raise typer.Exit(1)

    # Create work item
    work_item_id = store.create_work_item(
        project_id=proj["id"], title=requirement[:200],
        description=requirement, item_type="code_change",
        assigned_agent_id=dev_agent["id"],
    )
    store.update_work_item_status(work_item_id, "in_progress")

    # Create sessions
    dev_sm.create_or_resume(purpose=f"Workflow dev: {requirement[:80]}", work_item_id=work_item_id)
    for qa_sm in qa_sms:
        qa_sm.create_or_resume(purpose=f"Workflow QA: {requirement[:80]}", work_item_id=work_item_id)

    console.print(f"[bold]Requirement:[/bold] {requirement}")
    console.print(
        f"[dim]Dev: {dev_agent['display_name']} ({dev_agent['cli_backend']}/{dev_agent['model_name']})[/dim]"
    )
    for qa_sm in qa_sms:
        qa_row = qa_sm._agent_row
        console.print(
            f"[dim]QA:  {qa_row['display_name']} ({qa_row['cli_backend']}/{qa_row['model_name']})[/dim]"
        )
    console.print(f"[dim]Work item: {work_item_id}[/dim]\n")

    cancel_event = threading.Event()
    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=qa_sms,
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_review_iterations=settings.workflow.max_review_iterations,
        auto_approve=True,
        should_cancel=cancel_event.is_set,
    )

    try:
        work_item_ref: dict[str, int | None] = {"id": work_item_id}
        cancel_targets: list[AgentRunner] = [dev_runner] + [qa_sm._runner for qa_sm in qa_sms]

        def _worker():
            return engine.run(requirement)

        result = _run_with_task_monitor(
            console=console,
            store=store,
            proj=proj,
            label="Running full teamwork workflow",
            worker_fn=_worker,
            work_item_ref=work_item_ref,
            cancel_targets=cancel_targets,
            cancel_event=cancel_event,
        )

        # Update work item
        if cancel_event.is_set():
            store.update_work_item_status(work_item_id, "blocked")
        elif result.success:
            store.update_work_item_status(work_item_id, "completed")
        else:
            store.update_work_item_status(work_item_id, "review")
    except Exception as e:
        from myswat.workflow.error_handler import WorkflowError, handle_workflow_error

        werr = WorkflowError(
            error=e,
            stage="workflow_execution",
            context={
                "project": project_slug,
                "requirement": requirement[:500],
                "work_item_id": work_item_id,
            },
        )
        handle_workflow_error(werr, store=store, project_id=proj["id"])

        try:
            store.update_work_item_status(work_item_id, "blocked")
        except Exception:
            pass
    finally:
        # Always close sessions — prevent orphaned active sessions
        for sm in [dev_sm] + qa_sms:
            try:
                sm.close()
            except Exception:
                pass

    console.print(f"\n[dim]Sessions closed. All turns persisted to TiDB.[/dim]")
