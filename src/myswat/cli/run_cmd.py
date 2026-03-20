"""myswat run — execute agent tasks."""

from __future__ import annotations

import threading

import typer
from rich.console import Console
from rich.panel import Panel

from myswat.agents.base import AgentRunner
from myswat.agents.factory import make_runner_from_row
from myswat.agents.session_manager import SessionManager
from myswat.cli.progress import _fmt_duration, _run_with_task_monitor, _send_with_timer
from myswat.config.settings import MySwatSettings, get_workflow_review_limit
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.memory.learn_triggers import submit_workflow_summary_learn_request
from myswat.memory.store import MemoryStore

console = Console()


def run_single(
    project_slug: str,
    task: str,
    role: str = "developer",
    workdir: str | None = None,
) -> None:
    """Run a single-agent task with session persistence."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    ensure_schema(pool)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    # Resolve project
    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = workdir or proj.get("repo_path")

    # Resolve agent
    agent_row = store.get_agent(proj["id"], role)
    if not agent_row:
        console.print(f"[red]Agent role '{role}' not found in project '{project_slug}'.[/red]")
        raise typer.Exit(1)

    # Create runner with optional workdir override
    runner = make_runner_from_row(agent_row, settings=settings)
    if workdir:
        runner.workdir = workdir
    elif proj.get("repo_path"):
        runner.workdir = proj["repo_path"]

    sm = SessionManager(
        store=store,
        runner=runner,
        agent_row=agent_row,
        project_id=proj["id"],
        settings=settings,
    )

    # Create session and send task
    sm.create_or_resume(purpose=task)
    console.print(
        f"[dim]Session {sm.session.session_uuid} | "
        f"Agent: {agent_row['display_name']} ({agent_row['cli_backend']}/{agent_row['model_name']})[/dim]"
    )
    console.print(f"[bold]Task:[/bold] {task}\n")
    console.print("[bold cyan]Stage 1/1:[/bold cyan] Sending task to agent")

    try:
        response, elapsed = _send_with_timer(
            console,
            sm,
            task,
            task_description=task,
        )
        console.print(
            f"[green]Stage 1/1 complete.[/green] [dim]({_fmt_duration(elapsed)})[/dim]"
        )

        # Display result
        if response.success:
            console.print(Panel(response.content, title="Agent Response", border_style="green"))
        else:
            console.print(Panel(response.content, title="Agent Response (error)", border_style="red"))
            if response.raw_stderr:
                console.print(f"[dim red]stderr: {response.raw_stderr[:500]}[/dim red]")
    except Exception as e:
        from myswat.workflow.error_handler import WorkflowError, handle_workflow_error

        werr = WorkflowError(
            error=e,
            stage="agent_task",
            context={"project": project_slug, "role": role, "task": task[:500]},
        )
        handle_workflow_error(werr, store=store, project_id=proj["id"])
    finally:
        try:
            sm.close()
        except Exception:
            pass

    console.print(f"\n[dim]Session closed. Turns persisted to TiDB.[/dim]")


def run_with_review(
    project_slug: str,
    task: str,
    developer_role: str = "developer",
    reviewer_role: str = "qa_main",
    workdir: str | None = None,
) -> None:
    """Run a task with developer + reviewer feedback loop."""
    from myswat.workflow.review_loop import run_review_loop

    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    ensure_schema(pool)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    # Resolve project
    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = workdir or proj.get("repo_path")

    # Resolve agents
    dev_agent = store.get_agent(proj["id"], developer_role)
    reviewer_agent = store.get_agent(proj["id"], reviewer_role)
    if not dev_agent:
        console.print(f"[red]Developer role '{developer_role}' not found.[/red]")
        raise typer.Exit(1)
    if not reviewer_agent:
        console.print(f"[red]Reviewer role '{reviewer_role}' not found.[/red]")
        raise typer.Exit(1)

    # Create runners
    dev_runner = make_runner_from_row(dev_agent, settings=settings)
    dev_runner.workdir = effective_workdir
    reviewer_runner = make_runner_from_row(reviewer_agent, settings=settings)
    reviewer_runner.workdir = effective_workdir

    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], settings=settings,
    )
    reviewer_sm = SessionManager(
        store=store, runner=reviewer_runner, agent_row=reviewer_agent,
        project_id=proj["id"], settings=settings,
    )

    # Create work item
    work_item_id = store.create_work_item(
        project_id=proj["id"],
        title=task[:200],
        description=task,
        item_type="code_change",
        assigned_agent_id=dev_agent["id"],
    )
    store.update_work_item_status(work_item_id, "in_progress")
    try:
        store.append_work_item_process_event(
            work_item_id,
            event_type="task_request",
            title="Review loop task",
            summary=task,
            from_role="user",
            to_role=dev_agent["role"],
            updated_by_agent_id=dev_agent["id"],
        )
    except Exception:
        pass

    # Create sessions
    dev_sm.create_or_resume(purpose=f"Dev: {task[:100]}", work_item_id=work_item_id)
    reviewer_sm.create_or_resume(purpose=f"Review: {task[:100]}", work_item_id=work_item_id)

    console.print(f"[bold]Task:[/bold] {task}")
    console.print(
        f"[dim]Developer: {dev_agent['display_name']} ({dev_agent['cli_backend']}/{dev_agent['model_name']})[/dim]"
    )
    console.print(
        f"[dim]Reviewer:  {reviewer_agent['display_name']} ({reviewer_agent['cli_backend']}/{reviewer_agent['model_name']})[/dim]"
    )
    console.print(f"[dim]Work item: {work_item_id}[/dim]\n")

    final_status = "blocked"
    final_summary = "Review loop blocked."

    try:
        work_item_ref: dict[str, int | None] = {"id": work_item_id}
        cancel_targets: list[AgentRunner] = [dev_runner, reviewer_runner]
        cancel_event = threading.Event()

        def _worker():
            return run_review_loop(
                store=store,
                dev_sm=dev_sm,
                reviewer_sm=reviewer_sm,
                task=task,
                project_id=proj["id"],
                work_item_id=work_item_id,
                max_iterations=get_workflow_review_limit(
                    settings.workflow,
                    "dev_code_review_limit",
                ),
                should_cancel=cancel_event.is_set,
            )

        # Run the review loop
        verdict = _run_with_task_monitor(
            console=console,
            store=store,
            proj=proj,
            label="Running dev+QA review loop",
            worker_fn=_worker,
            work_item_ref=work_item_ref,
            cancel_targets=cancel_targets,
            cancel_event=cancel_event,
        )

        # Update work item status
        if cancel_event.is_set():
            store.update_work_item_status(work_item_id, "blocked")
            final_status = "blocked"
            final_summary = "Review loop cancelled."
        elif verdict.verdict == "lgtm":
            store.update_work_item_status(work_item_id, "approved")
            final_status = "approved"
            final_summary = verdict.summary or "Review loop approved."
        else:
            store.update_work_item_status(work_item_id, "review")
            final_status = "review"
            final_summary = verdict.summary or verdict.verdict
    except Exception as e:
        from myswat.workflow.error_handler import WorkflowError, handle_workflow_error

        werr = WorkflowError(
            error=e,
            stage="review_loop",
            context={
                "project": project_slug,
                "task": task[:500],
                "work_item_id": work_item_id,
            },
        )
        handle_workflow_error(werr, store=store, project_id=proj["id"])

        try:
            store.update_work_item_status(work_item_id, "blocked")
        except Exception:
            pass
        final_status = "blocked"
        final_summary = f"Review loop crashed: {type(e).__name__}"
    finally:
        for sm in [dev_sm, reviewer_sm]:
            try:
                sm.close()
            except Exception:
                pass
        try:
            submit_workflow_summary_learn_request(
                store=store,
                settings=settings,
                project_id=proj["id"],
                source_work_item_id=work_item_id,
                source_session_id=getattr(getattr(dev_sm, "session", None), "id", None),
                requirement=task,
                final_status=final_status,
                final_summary=final_summary,
                mode="review_loop",
                workdir=effective_workdir,
            )
        except Exception:
            pass

    console.print(f"\n[dim]Sessions closed. All turns persisted to TiDB.[/dim]")
