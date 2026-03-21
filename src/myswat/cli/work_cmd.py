"""Legacy workflow CLI helpers backed entirely by the local daemon."""

from __future__ import annotations

import typer
from rich.console import Console

from myswat.config.settings import MySwatSettings
from myswat.server.control_client import DaemonClient, DaemonClientError
from myswat.workflow.modes import WorkMode

console = Console()


def _print_daemon_error(exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")
    console.print("[dim]Start the daemon first: myswat server[/dim]")


def _validate_daemon_workflow_request(
    *,
    mode: WorkMode,
    with_ga_test: bool,
    auto_approve: bool,
    resume: int | None,
) -> None:
    if resume is not None:
        raise typer.BadParameter("--resume is not supported through the daemon workflow path yet.")
    if not auto_approve:
        raise typer.BadParameter(
            "--interactive-checkpoints is not supported through the daemon workflow path yet."
        )
    if with_ga_test and mode != WorkMode.full:
        raise typer.BadParameter("--with-ga-test can only be used with the full workflow.")


def run_work(
    project_slug: str,
    requirement: str,
    workdir: str | None = None,
    background: bool = False,
    mode: WorkMode = WorkMode.full,
    with_ga_test: bool = False,
    auto_approve: bool = True,
    resume: int | None = None,
    mode_explicit: bool = False,
) -> int:
    """Queue workflow execution through the daemon and return the work item ID."""
    del background, mode_explicit
    _validate_daemon_workflow_request(
        mode=mode,
        with_ga_test=with_ga_test,
        auto_approve=auto_approve,
        resume=resume,
    )

    settings = MySwatSettings()
    client = DaemonClient(settings)
    submit_kwargs = {
        "project": project_slug,
        "requirement": requirement,
        "workdir": workdir,
        "mode": mode.value,
    }
    if with_ga_test:
        submit_kwargs["with_ga_test"] = True

    try:
        result = client.submit_work(**submit_kwargs)
    except DaemonClientError as exc:
        _print_daemon_error(exc)
        raise typer.Exit(1) from exc

    work_item_id = int(result.get("work_item_id") or 0)
    console.print(f"[bold]Queued workflow:[/bold] {requirement}")
    if work_item_id > 0:
        console.print(f"[dim]Work item: {work_item_id}[/dim]")
    workers = result.get("workers") or []
    if isinstance(workers, list) and workers:
        console.print(f"[dim]Workers: {', '.join(str(worker) for worker in workers)}[/dim]")
    console.print(f"[dim]Server: {client.base_url}[/dim]")
    console.print(f"[dim]Track progress: myswat status -p {project_slug} --details[/dim]")
    return work_item_id


def stop_work_item(project_slug: str, work_item_id: int) -> None:
    """Request workflow cancellation through the daemon."""
    settings = MySwatSettings()
    client = DaemonClient(settings)
    try:
        result = client.control_work(project=project_slug, work_item_id=work_item_id, action="cancel")
    except DaemonClientError as exc:
        _print_daemon_error(exc)
        raise typer.Exit(1) from exc

    console.print(
        f"[green]Cancellation requested for work item {result.get('work_item_id') or work_item_id}.[/green]"
    )
