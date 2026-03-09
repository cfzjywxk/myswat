"""myswat run — execute agent tasks."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel

from myswat.agents.base import AgentRunner
from myswat.agents.codex_runner import CodexRunner
from myswat.agents.kimi_runner import KimiRunner
from myswat.agents.session_manager import SessionManager
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import run_migrations
from myswat.memory.compactor import KnowledgeCompactor
from myswat.memory.store import MemoryStore

console = Console()


def _find_compaction_runner(
    store: MemoryStore, proj: dict, settings: MySwatSettings,
) -> AgentRunner | None:
    """Find a runner suitable for knowledge compaction."""
    agents = store.list_agents(proj["id"])
    for a in agents:
        if a["cli_backend"] == settings.compaction.compaction_backend:
            return _make_runner(a, settings)
    if agents:
        return _make_runner(agents[0], settings)
    return None


def _make_runner(agent_row: dict, settings: MySwatSettings) -> AgentRunner:
    """Create the appropriate AgentRunner from a DB agent row."""
    backend = agent_row["cli_backend"]
    cli_path = agent_row["cli_path"]
    model = agent_row["model_name"]
    extra_flags = json.loads(agent_row["cli_extra_args"]) if agent_row.get("cli_extra_args") else []
    workdir = None

    if backend == "codex":
        return CodexRunner(
            cli_path=cli_path, model=model,
            workdir=workdir, extra_flags=extra_flags,
        )
    elif backend == "kimi":
        return KimiRunner(
            cli_path=cli_path, model=model,
            workdir=workdir, extra_flags=extra_flags,
        )
    else:
        raise typer.BadParameter(f"Unknown CLI backend: {backend}")


def run_single(
    project_slug: str,
    task: str,
    role: str = "developer",
    workdir: str | None = None,
) -> None:
    """Run a single-agent task with session persistence."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    run_migrations(pool)
    store = MemoryStore(pool)

    # Resolve project
    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    # Resolve agent
    agent_row = store.get_agent(proj["id"], role)
    if not agent_row:
        console.print(f"[red]Agent role '{role}' not found in project '{project_slug}'.[/red]")
        raise typer.Exit(1)

    # Create runner with optional workdir override
    runner = _make_runner(agent_row, settings)
    if workdir:
        runner.workdir = workdir
    elif proj.get("repo_path"):
        runner.workdir = proj["repo_path"]

    # Create session manager with compactor
    compaction_runner = _find_compaction_runner(store, proj, settings)
    compactor = KnowledgeCompactor(
        store=store,
        runner=compaction_runner,
        threshold_turns=settings.compaction.threshold_turns,
        threshold_tokens=settings.compaction.threshold_tokens,
    )
    sm = SessionManager(
        store=store,
        runner=runner,
        agent_row=agent_row,
        project_id=proj["id"],
        compactor=compactor,
    )

    # Create session and send task
    sm.create_or_resume(purpose=task)
    console.print(
        f"[dim]Session {sm.session.session_uuid} | "
        f"Agent: {agent_row['display_name']} ({agent_row['cli_backend']}/{agent_row['model_name']})[/dim]"
    )
    console.print(f"[bold]Task:[/bold] {task}\n")

    with console.status("[bold cyan]Agent working...", spinner="dots"):
        response = sm.send(task, task_description=task)

    # Display result
    if response.success:
        console.print(Panel(response.content, title="Agent Response", border_style="green"))
    else:
        console.print(Panel(response.content, title="Agent Response (error)", border_style="red"))
        if response.raw_stderr:
            console.print(f"[dim red]stderr: {response.raw_stderr[:500]}[/dim red]")

    # Close session
    sm.close()

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
    run_migrations(pool)
    store = MemoryStore(pool)

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
    dev_runner = _make_runner(dev_agent, settings)
    dev_runner.workdir = effective_workdir
    reviewer_runner = _make_runner(reviewer_agent, settings)
    reviewer_runner.workdir = effective_workdir

    # Create session managers with compactor
    compaction_runner = _find_compaction_runner(store, proj, settings)
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
    reviewer_sm = SessionManager(
        store=store, runner=reviewer_runner, agent_row=reviewer_agent,
        project_id=proj["id"], compactor=compactor,
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

    # Run the review loop
    verdict = run_review_loop(
        store=store,
        dev_sm=dev_sm,
        reviewer_sm=reviewer_sm,
        task=task,
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_iterations=settings.workflow.max_review_iterations,
    )

    # Update work item status
    if verdict.verdict == "lgtm":
        store.update_work_item_status(work_item_id, "approved")
    else:
        store.update_work_item_status(work_item_id, "review")

    # Close sessions
    dev_sm.close()
    reviewer_sm.close()

    console.print(f"\n[dim]Sessions closed. All turns persisted to TiDB.[/dim]")
