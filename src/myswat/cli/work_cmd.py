"""myswat work — run the full teamwork workflow from CLI."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from myswat.agents.base import AgentRunner
from myswat.agents.codex_runner import CodexRunner
from myswat.agents.kimi_runner import KimiRunner
from myswat.agents.session_manager import SessionManager
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
    store = MemoryStore(pool)

    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = workdir or proj.get("repo_path")

    # Pre-flight: check for project_ops knowledge
    ops = store.list_knowledge(proj["id"], category="project_ops", limit=1)
    if not ops:
        console.print(
            "[yellow]Warning: No project operations knowledge found.\n"
            f"Run 'myswat learn -p {project_slug}' first so agents know how to "
            "build, test, and follow project conventions.[/yellow]\n"
        )

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

    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=qa_sms,
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_review_iterations=settings.workflow.max_review_iterations,
    )

    result = engine.run(requirement)

    # Update work item
    if result.success:
        store.update_work_item_status(work_item_id, "completed")
    else:
        store.update_work_item_status(work_item_id, "review")

    # Close sessions
    dev_sm.close()
    for qa_sm in qa_sms:
        qa_sm.close()

    console.print(f"\n[dim]Sessions closed. All turns persisted to TiDB.[/dim]")
