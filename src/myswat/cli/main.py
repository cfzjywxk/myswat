"""MySwat CLI — main entry point."""

from datetime import datetime, timezone
import json
import time

import typer

from myswat.cli.progress import _describe_process_event
from myswat.cli.memory_cmd import memory_app, search as run_search_command
from myswat.workflow.modes import WorkMode, resolve_cli_work_mode

app = typer.Typer(
    name="myswat",
    help="Multi-AI agent co-working system for code development.",
    no_args_is_help=True,
)

_ACTIVE_RUNTIME_WORK_STATUSES = frozenset({"pending", "in_progress", "review"})
_MIN_AWARE_DATETIME = datetime.min.replace(tzinfo=timezone.utc)
_MAX_AWARE_DATETIME = datetime.max.replace(tzinfo=timezone.utc)

app.add_typer(memory_app, name="memory", help="Search and manage project knowledge")


@app.command()
def server():
    """Run the persistent MySwat daemon."""
    from myswat.cli.server_cmd import run_daemon_server

    raise typer.Exit(run_daemon_server())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    source_type: str = typer.Option(None, "--source-type", help="Filter by source type"),
    mode: str = typer.Option("auto", "--mode", help="Search mode"),
    profile: str = typer.Option("standard", "--profile", help="Search profile"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    no_vector: bool = typer.Option(False, "--no-vector", help="Skip vector search (keyword only)"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Search project knowledge with lexical + semantic retrieval."""
    run_search_command(
        query=query,
        project=project,
        category=category,
        source_type=source_type,
        mode=mode,
        profile=profile,
        limit=limit,
        no_vector=no_vector,
        json_output=json_output,
    )


@app.command()
def chat(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    role: str = typer.Option("architect", "--role", help="Initial agent role"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
):
    """Interactive chat session with an agent. Switch roles, trigger reviews, all from the REPL."""
    from myswat.cli.chat_cmd import run_chat
    run_chat(project, role=role, workdir=workdir)


@app.command()
def run(
    task: str = typer.Argument(None, help="Task description (omit for interactive mode)"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    single: bool = typer.Option(False, "--single", help="Single-agent mode (no review loop)"),
    role: str = typer.Option("architect", "--role", help="Agent role to use"),
    reviewer: str = typer.Option("qa_main", "--reviewer", help="Reviewer role (for review loop)"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
):
    """Run a task with AI agents. Omit task to enter interactive mode."""
    if task is None:
        from myswat.cli.chat_cmd import run_chat
        run_chat(project, role=role, workdir=workdir)
    elif single:
        from myswat.cli.run_cmd import run_single
        run_single(project, task, role=role, workdir=workdir)
    else:
        from myswat.cli.run_cmd import run_with_review
        run_with_review(project, task, developer_role=role, reviewer_role=reviewer, workdir=workdir)


def _resolve_work_mode(*, design: bool, develop: bool, test: bool) -> WorkMode:
    try:
        return resolve_cli_work_mode(
            design=design,
            develop=develop,
            test=test,
        )
    except ValueError:
        raise typer.BadParameter(
            "Choose only one of --design/--plan, --develop/--dev, or --test/--ga-test."
        )


def _is_terminal_work_status(status: str) -> bool:
    return status in {"approved", "blocked", "cancelled", "completed", "paused"}


def _follow_work_item_until_terminal(
    *,
    client,
    project: str,
    work_item_id: int,
    poll_interval_seconds: float = 1.0,
    console=None,
) -> dict:
    from rich.console import Console
    from myswat.server.control_client import DaemonClientError

    console = console or Console()
    seen_events = 0
    last_stage = ""
    last_status = ""
    last_poll_error = ""
    while True:
        try:
            payload = client.get_work_item(project=project, work_item_id=work_item_id)
        except DaemonClientError as exc:
            if not getattr(exc, "retryable", False):
                raise
            message = " ".join(str(exc).split())
            if message != last_poll_error:
                console.print(f"[yellow]Waiting for daemon response: {message}[/yellow]")
                last_poll_error = message
            time.sleep(max(0.1, poll_interval_seconds))
            continue
        except TimeoutError as exc:
            message = " ".join(str(exc).split()) or "timed out"
            if message != last_poll_error:
                console.print(f"[yellow]Waiting for daemon response: {message}[/yellow]")
                last_poll_error = message
            time.sleep(max(0.1, poll_interval_seconds))
            continue

        last_poll_error = ""
        item = payload.get("work_item") or {}
        metadata = item.get("metadata_json") if isinstance(item, dict) else {}
        task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
        process_log = task_state.get("process_log") if isinstance(task_state, dict) else []

        if isinstance(process_log, list):
            for event in process_log[seen_events:]:
                if not isinstance(event, dict):
                    continue
                timestamp = event.get("at")
                prefix = f"[{timestamp}] " if timestamp else ""
                console.print(f"{prefix}{_describe_process_event(event, 180)}")
            seen_events = len(process_log)

        stage = str(task_state.get("current_stage") or "")
        status = str(item.get("status") or "")
        if status != last_status or stage != last_stage:
            summary = str(task_state.get("latest_summary") or "")
            status_line = f"status={status or '-'}"
            if stage:
                status_line += f" stage={stage}"
            if summary:
                status_line += f" summary={summary[:180]}"
            console.print(f"[dim]{status_line}[/dim]")
            last_status = status
            last_stage = stage

        if _is_terminal_work_status(status):
            return item
        time.sleep(max(0.1, poll_interval_seconds))


@app.command()
def work(
    requirement: str = typer.Argument(None, help="Requirement description (optional when --resume)"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
    follow: bool = typer.Option(
        False,
        "--follow",
        help="Follow live workflow progress in this client after queueing the work item.",
    ),
    background: bool = typer.Option(
        False,
        "--background",
        hidden=True,
        help="Deprecated alias for detached mode; workflows already detach by default.",
    ),
    design_mode: bool = typer.Option(False, "--design", "--plan", help="Run design + planning only."),
    develop_mode: bool = typer.Option(False, "--develop", "--dev", help="Run development only."),
    test_mode: bool = typer.Option(False, "--test", "--ga-test", help="Run GA testing only."),
    with_ga_test: bool = typer.Option(
        False,
        "--with-ga-test",
        help="Add the GA test stage to the default full workflow.",
    ),
    auto_approve: bool = typer.Option(
        True,
        "--auto-approve/--interactive-checkpoints",
        help="Automatically continue through workflow checkpoints by default. Use --interactive-checkpoints to require manual approval.",
    ),
    resume: int = typer.Option(None, "--resume", help="Resume a blocked/failed work item by ID."),
):
    """Submit a teamwork workflow to the local MySwat daemon."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.server.control_client import DaemonClient, DaemonClientError

    mode = _resolve_work_mode(
        design=design_mode,
        develop=develop_mode,
        test=test_mode,
    )
    mode_explicit = design_mode or develop_mode or test_mode

    if resume is not None:
        if requirement:
            raise typer.BadParameter(
                "Cannot provide a new requirement with --resume. "
                "The original requirement is loaded from the work item. "
                "To start fresh with a different requirement, omit --resume."
            )
        if follow:
            raise typer.BadParameter("--resume cannot be combined with --follow.")
    else:
        if not requirement:
            raise typer.BadParameter("Requirement is required when not using --resume.")

    if background and follow:
        raise typer.BadParameter("--background cannot be combined with --follow.")

    if resume is not None:
        raise typer.BadParameter("--resume is not supported through the daemon workflow path yet.")
    if not auto_approve:
        raise typer.BadParameter("--interactive-checkpoints is not supported through the daemon workflow path yet.")
    if with_ga_test and mode != WorkMode.full:
        raise typer.BadParameter("--with-ga-test can only be used with the default full workflow.")

    console = Console()
    settings = MySwatSettings()
    client = DaemonClient(settings)
    submit_kwargs = {
        "project": project,
        "requirement": requirement or "",
        "workdir": workdir,
        "mode": mode.value,
    }
    if with_ga_test:
        submit_kwargs["with_ga_test"] = True
    try:
        result = client.submit_work(**submit_kwargs)
    except DaemonClientError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"[dim]Start the daemon first: myswat server[/dim]")
        raise typer.Exit(1)

    console.print(f"[bold]Queued workflow:[/bold] {requirement}")
    console.print(f"[dim]Work item: {result.get('work_item_id')}[/dim]")
    workers = result.get("workers") or []
    if isinstance(workers, list) and workers:
        console.print(f"[dim]Workers: {', '.join(str(worker) for worker in workers)}[/dim]")
    console.print(f"[dim]Server: {client.base_url}[/dim]")
    console.print(f"[dim]Track progress: myswat status -p {project} --details[/dim]")
    console.print("[dim]Use --follow to keep this client attached for live progress.[/dim]")

    work_item_id = int(result.get("work_item_id") or 0)
    if background or not follow or work_item_id <= 0:
        return

    console.print("[dim]Following workflow progress. Press Ctrl-C to cancel.[/dim]")
    poll_interval_seconds = float(getattr(settings.workflow, "assignment_poll_interval_seconds", 1.0) or 1.0)
    try:
        final_item = _follow_work_item_until_terminal(
            client=client,
            project=project,
            work_item_id=work_item_id,
            poll_interval_seconds=poll_interval_seconds,
            console=console,
        )
        console.print(f"[bold]Workflow finished with status:[/bold] {final_item.get('status')}")
    except KeyboardInterrupt:
        console.print("[yellow]Cancellation requested. Stopping workflow...[/yellow]")
        try:
            client.control_work(project=project, work_item_id=work_item_id, action="cancel")
        except DaemonClientError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        try:
            _follow_work_item_until_terminal(
                client=client,
                project=project,
                work_item_id=work_item_id,
                poll_interval_seconds=poll_interval_seconds,
                console=console,
            )
        except Exception:
            pass
        raise typer.Exit(130)


@app.command(name="worker", hidden=True)
def worker(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    role: str = typer.Option(..., "--role", help="Agent role handled by this worker"),
    server_url: str = typer.Option(..., "--server-url", help="MySwat daemon base URL"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
):
    """Internal worker entry point supervised by the MySwat daemon."""
    from myswat.cli.worker_cmd import run_worker

    run_worker(
        project_slug=project,
        role=role,
        server_url=server_url,
        workdir=workdir,
    )


@app.command()
def stop(
    work_item_id: int = typer.Argument(..., help="Work item ID"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Request cancellation of a workflow."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.server.control_client import DaemonClient, DaemonClientError

    console = Console()
    settings = MySwatSettings()
    client = DaemonClient(settings)
    try:
        result = client.control_work(project=project, work_item_id=work_item_id, action="cancel")
    except DaemonClientError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"[dim]Start the daemon first: myswat server[/dim]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Cancellation requested for work item {result.get('work_item_id')}.[/green]")


def _format_history_timestamp(value) -> str:
    if hasattr(value, "isoformat"):
        isoformat = value.isoformat
        try:
            return isoformat(sep=" ", timespec="seconds")
        except TypeError:
            return isoformat()
    return str(value)


@app.command()
def gc(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    grace_days: int = typer.Option(7, "--grace-days", help="Grace period before deleting compacted turns."),
    keep_recent: int = typer.Option(50, "--keep-recent", help="Always preserve this many most-recent project turns."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting it."),
):
    """Garbage-collect old raw turns from fully compacted sessions."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.memory.store import MemoryStore

    console = Console()
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    result = store.gc_compacted_turns(
        proj["id"],
        grace_days=grace_days,
        keep_recent=keep_recent,
        dry_run=dry_run,
    )
    verb = "Would delete" if dry_run else "Deleted"
    console.print(
        f"{verb} {result['turns_deleted']} turns from {result['sessions_affected']} sessions"
    )


@app.command()
def history(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    turns: int = typer.Option(50, "--turns", help="Number of recent turns to show."),
    role: str = typer.Option(None, "--role", help="Filter to one agent role."),
):
    """Show recent raw project turns in chronological order."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.memory.store import MemoryStore

    console = Console()
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    rows = store.get_recent_turns_global(proj["id"], limit=turns, role=role)
    if not rows:
        console.print("[dim]No recent turns found.[/dim]")
        return

    for row in rows:
        timestamp = _format_history_timestamp(row.get("created_at"))
        content = str(row.get("content") or "").replace("\n", " ")
        console.print(
            f"[{row.get('agent_role', 'unknown')}] [{timestamp}] {row.get('role', 'unknown')}: {content}",
            markup=False,
        )


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    repo_path: str = typer.Option(None, "--repo", "-r", help="Path to git repo"),
    description: str = typer.Option(None, "--desc", "-d", help="Project description"),
):
    """Initialize a project through the local MySwat daemon."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.server.control_client import DaemonClient, DaemonClientError

    console = Console()
    settings = MySwatSettings()
    client = DaemonClient(settings)
    try:
        result = client.init_project(
            name=name,
            repo_path=repo_path,
            description=description,
        )
    except DaemonClientError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"[dim]Start the daemon first: myswat server[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]Project initialized through daemon.[/green]")
    console.print(f"[dim]Project: {result.get('project') or name}[/dim]")


@app.command()
def cleanup(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete one project and all related TiDB state through the local MySwat daemon."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.server.control_client import DaemonClient, DaemonClientError

    console = Console()
    if not yes:
        console.print(
            f"\n[bold red]WARNING: This will permanently delete project '{project}'.[/bold red]"
        )
        console.print(
            "[red]All project agents, work items, stage runs, review cycles, sessions, "
            "knowledge, runtime registrations, and daemon worker state for this project will be removed.[/red]\n"
        )
        confirm = typer.prompt(f"Type '{project}' to confirm", default="")
        if confirm != project:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    settings = MySwatSettings()
    client = DaemonClient(settings)
    try:
        result = client.cleanup_project(project=project)
    except DaemonClientError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"[dim]Start the daemon first: myswat server[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]Project '{result.get('project') or project}' removed.[/green]")
    deleted = result.get("deleted") or {}
    if isinstance(deleted, dict):
        summary = ", ".join(
            f"{key}={value}"
            for key, value in deleted.items()
            if isinstance(value, int) and value > 0
        )
        if summary:
            console.print(f"[dim]Deleted: {summary}[/dim]")
    removed_runtime_paths = result.get("removed_runtime_paths") or []
    if isinstance(removed_runtime_paths, list) and removed_runtime_paths:
        console.print(f"[dim]Removed runtime state: {', '.join(str(path) for path in removed_runtime_paths)}[/dim]")


@app.command()
def reset(
    project: str = typer.Option(None, "--project", "-p", help="Project slug (re-init after reset)"),
    repo_path: str = typer.Option(None, "--repo", "-r", help="Repo path for re-init"),
    description: str = typer.Option(None, "--desc", "-d", help="Project description for re-init"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Drop the TiDB database and re-create the current schema from scratch.

    WARNING: This destroys ALL data — projects, agents, stage runs,
    coordination events, sessions, knowledge, work items, and review cycles.
    Use only when you want a clean slate.
    """
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.db.schema import ensure_schema

    console = Console()
    settings = MySwatSettings()
    db_name = settings.tidb.database

    if not yes:
        console.print(
            f"\n[bold red]WARNING: This will DROP the entire '{db_name}' database.[/bold red]"
        )
        console.print(
            "[red]All projects, agents, stage runs, coordination events, sessions, "
            "knowledge, work items, and review cycles will be permanently deleted.[/red]\n"
        )
        confirm = typer.prompt(f"Type '{db_name}' to confirm", default="")
        if confirm != db_name:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    pool = TiDBPool(settings.tidb)
    if not pool.health_check():
        console.print("[red]Cannot connect to TiDB. Check your config.[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Dropping database '{db_name}'...[/yellow]")
    pool.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    console.print(f"[green]Database '{db_name}' dropped.[/green]")

    # Re-create with a fresh connection and bootstrap the latest schema
    pool2 = TiDBPool(settings.tidb)
    ensure_schema(pool2)
    console.print(f"[green]Re-created database schema for '{db_name}'.[/green]")

    if project:
        from myswat.cli.init_cmd import run_init
        run_init(project, repo_path, description)
    else:
        console.print(
            "\n[dim]Database schema is ready. Run 'myswat init <name>' to create a project.[/dim]"
        )


def _parse_verdict_payload(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _compact_text(value, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _task_state_dict(item: dict) -> dict:
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    if not isinstance(metadata, dict):
        return {}
    task_state = metadata.get("task_state")
    return task_state if isinstance(task_state, dict) else {}


def _process_log_entries(item: dict) -> list[dict]:
    task_state = _task_state_dict(item)
    process_log = task_state.get("process_log")
    if not isinstance(process_log, list):
        return []
    return [entry for entry in process_log if isinstance(entry, dict)]


def _safe_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _local_timezone():
    return datetime.now().astimezone().tzinfo or timezone.utc


def _normalize_datetime(value: datetime, *, assume_utc_for_naive: bool) -> datetime:
    if value.tzinfo is None:
        if assume_utc_for_naive:
            return value.replace(tzinfo=timezone.utc).astimezone(_local_timezone())
        return value.replace(tzinfo=_local_timezone())
    return value.astimezone(_local_timezone())


def _coerce_datetime(value):
    if isinstance(value, datetime):
        return _normalize_datetime(value, assume_utc_for_naive=True)
    if not value:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _normalize_datetime(
                datetime.fromisoformat(text.replace("Z", "+00:00")),
                assume_utc_for_naive=False,
            )
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return _normalize_datetime(
                    datetime.strptime(text, fmt),
                    assume_utc_for_naive=False,
                )
            except ValueError:
                continue
    return None


def _now_like(value) -> datetime:
    dt = _coerce_datetime(value)
    if dt is not None:
        return datetime.now(dt.tzinfo)
    return datetime.now(_local_timezone())


def _seconds_since(value) -> float | None:
    dt = _coerce_datetime(value)
    if dt is None:
        return None
    return max(0.0, (_now_like(dt) - dt).total_seconds())


def _seconds_until(value) -> float | None:
    dt = _coerce_datetime(value)
    if dt is None:
        return None
    return (dt - _now_like(dt)).total_seconds()


def _format_age_compact(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _format_timestamp_short(value) -> str:
    dt = _coerce_datetime(value)
    if dt is None:
        return "--"
    return dt.strftime("%m-%d %H:%M:%S")


def _format_last_seen(value) -> str:
    age = _seconds_since(value)
    if age is None:
        return "never"
    return f"{_format_age_compact(age)} ago"


def _style_for_item_status(status: str) -> str:
    return {
        "completed": "green",
        "approved": "green",
        "in_progress": "yellow",
        "review": "cyan",
        "pending": "blue",
        "paused": "yellow",
        "cancelled": "red",
        "blocked": "red",
    }.get(status, "white")


def _markup_for_item_status(status: str) -> str:
    style = _style_for_item_status(status)
    return f"[{style}]{status}[/{style}]"


def _role_label(role: str | None) -> str:
    if not role:
        return "System"
    return {
        "user": "User",
        "myswat": "MySwat",
        "architect": "Architect",
        "developer": "Developer",
        "qa_main": "QA",
        "qa_vice": "QA (Vice)",
        "request": "Request",
    }.get(role, str(role).replace("_", " ").title())


def _role_style(role: str | None) -> str:
    return {
        "user": "green",
        "myswat": "magenta",
        "architect": "bright_blue",
        "developer": "bright_cyan",
        "qa_main": "bright_green",
        "qa_vice": "green",
    }.get(role, "white")


def _event_badge(entry: dict) -> tuple[str, str]:
    verdict = str(entry.get("verdict") or "")
    event_type = str(entry.get("type") or "")
    verdict_hint = f"{entry.get('title') or ''} {entry.get('summary') or ''}".lower()
    if not verdict:
        if "changes_requested" in verdict_hint or "request changes" in verdict_hint:
            verdict = "changes_requested"
        elif "lgtm" in verdict_hint:
            verdict = "lgtm"
    if verdict == "lgtm":
        return "LGTM", "black on green"
    if verdict == "changes_requested":
        return "REQUEST CHANGES", "bold white on red"
    if event_type in {"agent_stall", "stage_blocked"}:
        return "CRITICAL", "bold white on red"
    if event_type == "agent_empty_output":
        return "WARNING", "black on yellow"
    if event_type == "review_skipped":
        return "LIMIT", "black on yellow"
    if event_type == "review_requested":
        return "REVIEW", "black on cyan"
    if event_type == "status_report":
        return "STATUS", "black on blue"
    if event_type in {"artifact_submitted", "phase_summary"}:
        return "UPDATE", "black on bright_blue"
    if event_type == "daemon_queued":
        return "QUEUED", "black on white"
    if event_type in {"workflow_completed", "stage_completed"}:
        return "DONE", "black on green"
    return "", ""


def _event_title(entry: dict) -> str:
    title = _compact_text(entry.get("title") or "", 90)
    if title:
        return title
    event_type = str(entry.get("type") or "")
    return {
        "review_requested": "Review requested",
        "review_verdict": "Review response",
        "review_skipped": "Review limit reached",
        "status_report": "Status update",
        "artifact_submitted": "Artifact update",
        "daemon_queued": "Workflow queued",
        "stage_blocked": "Stage blocked",
        "agent_stall": "Agent stalled",
        "agent_empty_output": "Agent returned empty output",
    }.get(event_type, event_type.replace("_", " ").title() or "Update")


def _event_summary(entry: dict) -> str:
    parts: list[str] = []
    summary = str(entry.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    issues = entry.get("issues")
    if isinstance(issues, list) and issues:
        summary_lower = summary.lower()
        unseen_issues = [
            issue
            for issue in issues[:3]
            if isinstance(issue, str) and issue.strip() and issue.strip().lower() not in summary_lower
        ]
        issue_text = "; ".join(_compact_text(issue, 80) for issue in unseen_issues)
        if issue_text:
            parts.append(f"Issues: {issue_text}")
    return _compact_text(" ".join(parts), 320)


def _build_review_cycle_flow_entries(pool, item_id: int) -> list[dict]:
    rows = _safe_list(
        pool.fetch_all(
            "SELECT rc.iteration, rc.stage_name, rc.verdict, rc.verdict_json, "
            "rc.created_at, rc.updated_at, rc.completed_at, "
            "a1.role AS proposer_role, a2.role AS reviewer_role, "
            "art.title AS artifact_title, art.artifact_type "
            "FROM review_cycles rc "
            "JOIN agents a1 ON rc.proposer_agent_id = a1.id "
            "JOIN agents a2 ON rc.reviewer_agent_id = a2.id "
            "LEFT JOIN artifacts art ON rc.artifact_id = art.id "
            "WHERE rc.work_item_id = %s "
            "ORDER BY rc.created_at, rc.id",
            (item_id,),
        )
    )

    events: list[dict] = []
    sequence = 0
    for row in rows:
        artifact_title = str(row.get("artifact_title") or row.get("artifact_type") or "artifact")
        detail_parts = [artifact_title]
        if row.get("stage_name"):
            detail_parts.append(f"stage {row['stage_name']}")
        if row.get("iteration"):
            detail_parts.append(f"iteration {row['iteration']}")
        events.append(
            {
                "at": row.get("created_at"),
                "type": "review_requested",
                "from_role": row.get("proposer_role"),
                "to_role": row.get("reviewer_role"),
                "title": "Review requested",
                "summary": " | ".join(detail_parts),
                "_sequence": sequence,
            }
        )
        sequence += 1

        verdict = str(row.get("verdict") or "")
        if not verdict or verdict == "pending":
            continue
        verdict_payload = _parse_verdict_payload(row.get("verdict_json"))
        issues = verdict_payload.get("issues") if isinstance(verdict_payload.get("issues"), list) else []
        summary = verdict_payload.get("summary") or verdict
        if issues:
            issue_text = "; ".join(str(issue).strip() for issue in issues[:3] if str(issue).strip())
            if issue_text:
                summary = f"{summary} Issues: {issue_text}"
        events.append(
            {
                "at": row.get("completed_at") or row.get("updated_at") or row.get("created_at"),
                "type": "review_verdict",
                "from_role": row.get("reviewer_role"),
                "to_role": row.get("proposer_role"),
                "title": "Review response",
                "summary": summary,
                "issues": issues,
                "verdict": verdict,
                "_sequence": sequence,
            }
        )
        sequence += 1
    return events


def _build_teamwork_flow_entries(pool, item: dict) -> list[dict]:
    review_entries = _build_review_cycle_flow_entries(pool, int(item.get("id") or 0))
    timeline: list[dict] = []
    process_entries = _process_log_entries(item)

    description = str(item.get("description") or "").strip()
    if description and not process_entries:
        timeline.append(
            {
                "at": item.get("created_at"),
                "type": "task_request",
                "from_role": "user",
                "to_role": "myswat",
                "title": "Requirement",
                "summary": description,
                "_sequence": -1,
            }
        )

    for index, event in enumerate(process_entries):
        event_type = str(event.get("type") or "")
        if review_entries and event_type in {"review_requested", "review_verdict"}:
            continue
        timeline.append(
            {
                "at": event.get("at"),
                "type": event_type,
                "from_role": event.get("from_role"),
                "to_role": event.get("to_role"),
                "title": event.get("title"),
                "summary": event.get("summary"),
                "_sequence": index,
            }
        )

    timeline.extend(review_entries)
    return sorted(
        [entry for entry in timeline if isinstance(entry, dict)],
        key=lambda entry: (
            (
                _coerce_datetime(entry.get("at"))
                or (_MIN_AWARE_DATETIME if entry.get("type") == "task_request" else _MAX_AWARE_DATETIME)
            ),
            int(entry.get("_sequence") or 0),
        ),
    )


def _build_timeline_actor(entry: dict):
    from rich.text import Text

    actor = Text()
    from_role = entry.get("from_role")
    to_role = entry.get("to_role")
    actor.append(_role_label(from_role), style=_role_style(from_role))
    if to_role:
        actor.append(" -> ", style="dim")
        actor.append(_role_label(to_role), style=_role_style(to_role))
    return actor


def _build_timeline_message(entry: dict):
    from rich.text import Text

    badge_text, badge_style = _event_badge(entry)
    title = _event_title(entry)
    summary = _event_summary(entry)

    message = Text()
    if badge_text:
        message.append(f" {badge_text} ", style=badge_style)
        message.append(" ")
    message.append(title, style="bold")
    if summary and summary != title:
        message.append("\n")
        message.append(summary, style="dim")
    return message


def _print_message_flow(console, flow_entries: list[dict]) -> None:
    from rich.panel import Panel
    from rich.table import Table

    entries = [entry for entry in flow_entries if isinstance(entry, dict)]
    if not entries:
        console.print(
            Panel(
                "[dim]No recorded message flow yet.[/dim]",
                title="[bold]Message Flow[/bold]",
                border_style="cyan",
            )
        )
        return

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(style="dim", width=14, no_wrap=True)
    grid.add_column(width=22)
    grid.add_column(ratio=1)

    for entry in entries[-30:]:
        grid.add_row(
            _format_timestamp_short(entry.get("at")),
            _build_timeline_actor(entry),
            _build_timeline_message(entry),
        )

    console.print(
        Panel(
            grid,
            title="[bold]Message Flow[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _runtime_health(runtime) -> tuple[str, str]:
    status = str(getattr(runtime, "status", "") or "unknown")
    heartbeat_age = _seconds_since(getattr(runtime, "last_heartbeat_at", None))
    lease_left = _seconds_until(getattr(runtime, "lease_expires_at", None))
    metadata = getattr(runtime, "metadata_json", None)
    metadata = metadata if isinstance(metadata, dict) else {}

    if status != "online":
        stop_reason = str(metadata.get("stop_reason") or "")
        if stop_reason in {"worker_process_exited", "worker_error"}:
            return "quit", "red"
        return status, "dim"
    if (lease_left is not None and lease_left <= 0) or (heartbeat_age is not None and heartbeat_age >= 300):
        return "stalled", "red"
    if (lease_left is not None and lease_left <= 60) or (heartbeat_age is not None and heartbeat_age >= 120):
        return "late", "yellow"
    return "healthy", "green"


def _latest_runtime_rows(store, project_id: int, items: list[dict]) -> list[dict]:
    runtimes = _safe_list(store.list_runtime_registrations(project_id))
    selected: dict[str, object] = {}
    for runtime in runtimes:
        role = str(getattr(runtime, "agent_role", None) or getattr(runtime, "runtime_name", "runtime"))
        existing = selected.get(role)
        runtime_updated = _coerce_datetime(getattr(runtime, "updated_at", None)) or _coerce_datetime(
            getattr(runtime, "last_heartbeat_at", None)
        )
        existing_updated = _coerce_datetime(getattr(existing, "updated_at", None)) or _coerce_datetime(
            getattr(existing, "last_heartbeat_at", None)
        ) if existing is not None else None
        if existing is None or existing_updated is None or (runtime_updated is not None and runtime_updated >= existing_updated):
            selected[role] = runtime

    assignments: dict[int, dict] = {}
    for item in items:
        if str(item.get("status") or "") not in _ACTIVE_RUNTIME_WORK_STATUSES:
            continue
        work_item_id = int(item.get("id") or 0)
        for stage_run in _safe_list(store.list_stage_runs(work_item_id)):
            runtime_id = int(getattr(stage_run, "claimed_by_runtime_id", 0) or 0)
            if runtime_id <= 0 or str(getattr(stage_run, "status", "") or "") != "claimed":
                continue
            assignments[runtime_id] = {
                "work_item_id": work_item_id,
                "activity": f"stage {getattr(stage_run, 'stage_name', '-') or '-'} on #{work_item_id}",
            }
        for cycle in _safe_list(store.get_review_cycles(work_item_id)):
            runtime_id = int(cycle.get("claimed_by_runtime_id") or 0)
            if runtime_id <= 0 or str(cycle.get("status") or "") != "claimed":
                continue
            stage_name = str(cycle.get("stage_name") or "review")
            assignments[runtime_id] = {
                "work_item_id": work_item_id,
                "activity": f"review {stage_name} on #{work_item_id}",
            }

    rows: list[dict] = []
    for role in sorted(selected):
        runtime = selected[role]
        runtime_id = int(getattr(runtime, "id", 0) or 0)
        activity = assignments.get(runtime_id, {}).get("activity", "idle")
        work_item_id = int(assignments.get(runtime_id, {}).get("work_item_id", 0) or 0)
        note = ""
        agent_id = int(getattr(runtime, "agent_id", 0) or 0)
        if agent_id > 0 and work_item_id > 0:
            try:
                session = store.get_active_session(agent_id, work_item_id)
            except Exception:
                session = None
            if session is not None:
                note = _compact_text(getattr(session, "purpose", ""), 90)
        health, style = _runtime_health(runtime)
        metadata = getattr(runtime, "metadata_json", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        stop_reason = str(metadata.get("stop_reason") or "").replace("_", " ")
        exit_code = metadata.get("exit_code")
        if stop_reason and exit_code is not None:
            stop_reason = f"{stop_reason} (exit {exit_code})"
        rows.append(
            {
                "role": role,
                "health": health,
                "health_style": style,
                "last_seen": _format_last_seen(getattr(runtime, "last_heartbeat_at", None)),
                "activity": activity,
                "note": note,
                "runtime_name": str(getattr(runtime, "runtime_name", "") or "-"),
                "stop_reason": stop_reason,
            }
        )
    return rows


def _collect_project_alerts(items: list[dict], runtime_rows: list[dict]) -> list[dict]:
    alerts: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for row in runtime_rows:
        health = str(row.get("health") or "")
        if health == "healthy":
            continue
        severity = "critical" if health in {"stalled", "quit"} else "warning"
        message = f"{_role_label(row.get('role'))} worker is {health}"
        if row.get("activity") and row.get("activity") != "idle":
            message += f" while handling {row['activity']}"
        if row.get("last_seen"):
            message += f" (last heartbeat {row['last_seen']})"
        if row.get("stop_reason"):
            message += f"; {row['stop_reason']}"
        key = (severity, message)
        if key not in seen:
            alerts.append({"severity": severity, "message": message})
            seen.add(key)

    for item in items:
        work_item_id = int(item.get("id") or 0)
        for event in _process_log_entries(item)[-12:]:
            event_type = str(event.get("type") or "")
            if event_type not in {"agent_stall", "agent_empty_output", "stage_blocked"}:
                continue
            severity = "critical" if event_type in {"agent_stall", "stage_blocked"} else "warning"
            message = f"Work item #{work_item_id}: {_compact_text(event.get('summary') or _event_title(event), 180)}"
            key = (severity, message)
            if key not in seen:
                alerts.append({"severity": severity, "message": message})
                seen.add(key)
    return alerts[:8]


def _print_alerts(console, alerts: list[dict]) -> None:
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    border_style = "green"
    if alerts:
        border_style = "red" if any(alert.get("severity") == "critical" for alert in alerts) else "yellow"
        for index, alert in enumerate(alerts):
            severity = str(alert.get("severity") or "warning")
            badge = "CRITICAL" if severity == "critical" else "WARNING"
            badge_style = "bold white on red" if severity == "critical" else "black on yellow"
            text_style = "red" if severity == "critical" else "yellow"
            body.append(f" {badge} ", style=badge_style)
            body.append(" ")
            body.append(str(alert.get("message") or ""), style=text_style)
            if index < len(alerts) - 1:
                body.append("\n")
    else:
        body.append("No critical worker or agent alerts.", style="green")

    console.print(Panel(body, title="[bold]Alerts[/bold]", border_style=border_style, padding=(0, 1)))


def _print_worker_health(console, runtime_rows: list[dict]) -> None:
    from rich.table import Table

    if not runtime_rows:
        console.print("[dim]Worker Health: no active runtime registrations yet.[/dim]")
        return

    table = Table(title="Worker Health")
    table.add_column("Role")
    table.add_column("Health")
    table.add_column("Heartbeat")
    table.add_column("Activity")
    table.add_column("Progress", max_width=70)
    for row in runtime_rows:
        health_style = str(row.get("health_style") or "white")
        health_text = f"[{health_style}]{row.get('health')}[/{health_style}]"
        heartbeat = str(row.get("last_seen") or "never")
        if row.get("stop_reason"):
            heartbeat += f" | {row['stop_reason']}"
        progress = row.get("note") or row.get("runtime_name") or "-"
        table.add_row(
            _role_label(row.get("role")),
            health_text,
            heartbeat,
            str(row.get("activity") or "idle"),
            str(progress),
        )
    console.print(table)


def _print_teamwork_details(pool, item, console, details: bool = False, show_header: bool = True) -> None:
    from rich.table import Table

    item_id = int(item.get("id") or 0)
    status = str(item.get("status") or "unknown")
    if show_header:
        console.print(
            f"\n[bold]Work Item #{item_id}[/bold] — {item.get('title', '')} "
            f"{_markup_for_item_status(status)}"
        )

    flow_entries = _build_teamwork_flow_entries(pool, item)
    _print_message_flow(console, flow_entries if details else flow_entries[-12:])

    artifacts = _safe_list(
        pool.fetch_all(
            "SELECT a.artifact_type, a.title, a.iteration, a.created_at, "
            "ag.role AS agent_role, ag.display_name AS agent_name "
            "FROM artifacts a "
            "JOIN agents ag ON a.agent_id = ag.id "
            "WHERE a.work_item_id = %s "
            "ORDER BY a.created_at",
            (item_id,),
        )
    )
    if artifacts:
        table = Table(title="Artifacts")
        table.add_column("When", style="dim", no_wrap=True)
        table.add_column("Type")
        table.add_column("Title", max_width=48)
        table.add_column("By")
        table.add_column("Version", justify="right")
        for art in artifacts[-8:]:
            table.add_row(
                _format_timestamp_short(art.get("created_at")),
                str(art.get("artifact_type") or "-"),
                str(art.get("title") or "-"),
                str(art.get("agent_name") or art.get("agent_role") or "-"),
                f"v{art.get('iteration') or 1}",
            )
        console.print(table)


def _work_mode_value(item: dict) -> str | None:
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    work_mode = metadata.get("work_mode") if isinstance(metadata, dict) else None
    if isinstance(work_mode, str) and work_mode:
        return work_mode
    return None


def _display_mode(item: dict, fallback: str) -> str:
    work_mode = _work_mode_value(item)
    return work_mode if work_mode else fallback


def _print_task_state(console, item: dict) -> None:
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    background = metadata.get("background") if isinstance(metadata, dict) else {}
    task_state = _task_state_dict(item)
    work_mode = _work_mode_value(item)
    if work_mode:
        console.print(f"[bold]Workflow mode:[/bold] {work_mode}")
    if isinstance(background, dict) and background:
        console.print("[bold]Execution:[/bold]")
        if background.get("mode"):
            console.print(f"  Mode: {background['mode']}")
        if background.get("state"):
            console.print(f"  State: {background['state']}")
        if background.get("pid"):
            console.print(f"  PID: {background['pid']}")
        if background.get("log_path"):
            console.print(f"  Log: {background['log_path']}")
        if background.get("requested_at"):
            console.print(f"  Requested: {background['requested_at']}")
        if background.get("started_at"):
            console.print(f"  Started: {background['started_at']}")
        if background.get("finished_at"):
            console.print(f"  Finished: {background['finished_at']}")

    if not isinstance(task_state, dict) or not task_state:
        if isinstance(background, dict) and background:
            console.print()
        return

    if task_state.get("current_stage"):
        console.print(f"[bold]Stage:[/bold] {task_state['current_stage']}")
    if task_state.get("latest_summary"):
        console.print(f"[bold]Latest summary:[/bold] {_compact_text(task_state['latest_summary'], 600)}")
    if task_state.get("next_todos"):
        console.print("[bold]Next TODOs:[/bold]")
        for todo in task_state["next_todos"][:10]:
            console.print(f"  - {todo}")
    if task_state.get("open_issues"):
        console.print("[bold]Open issues:[/bold]")
        for issue in task_state["open_issues"][:10]:
            console.print(f"  - [yellow]{_compact_text(issue, 220)}[/yellow]")
    console.print()


def _infer_stage_labels(rounds: list[dict]) -> list[str]:
    """Best-effort stage labels from review round patterns.

    The workflow runs review loops in a fixed order:
      1. Design review (dev → QA)
      2. Plan review (dev → QA)
      3. Code review per phase (dev → QA) — can repeat
      4. Test plan review (QA → dev)

    We use the proposer/reviewer direction + sequence to infer labels.
    """
    labels = []
    dev_to_qa_count = 0
    qa_to_dev_count = 0

    for rd in rounds:
        p_role = rd["proposer_role"]
        r_role = rd["reviewer_role"]

        if p_role == "architect" and (r_role == "developer" or r_role.startswith("qa")):
            labels.append("Architect Design Review")
        elif p_role == "developer" and r_role.startswith("qa"):
            dev_to_qa_count += 1
            if dev_to_qa_count == 1:
                labels.append("Design Review")
            elif dev_to_qa_count == 2:
                labels.append("Plan Review")
            else:
                labels.append(f"Code Review (phase {dev_to_qa_count - 2})")
        elif p_role.startswith("qa") and r_role in {"developer", "architect"}:
            qa_to_dev_count += 1
            labels.append("Test Plan Review")
        else:
            labels.append(f"Review ({p_role} → {r_role})")

    return labels


@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    details: bool = typer.Option(False, "--details", help="Show detailed work-item flow and review data"),
):
    """Show project status: active work items, sessions, agents."""
    import socket
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table
    import pymysql.err

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.memory.store import MemoryStore

    console = Console()
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    try:
        proj = store.get_project_by_slug(project)
    except (pymysql.err.OperationalError, socket.gaierror, OSError) as exc:
        console.print("[red]TiDB is unreachable from this environment.[/red]")
        console.print(
            f"[dim]Host: {settings.tidb.host}:{settings.tidb.port} | Error: {type(exc).__name__}: {exc}[/dim]"
        )
        repo_cache = Path.cwd() / "myswat.md"
        if repo_cache.is_file():
            console.print(
                f"[dim]Local project-ops cache is still available at {repo_cache}. "
                "Live status and work-item data require TiDB connectivity.[/dim]"
            )
        raise typer.Exit(1)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Project:[/bold] {proj['name']} ({proj['slug']})")
    if proj.get("repo_path"):
        console.print(f"[bold]Repo:[/bold] {proj['repo_path']}")

    # Agents
    agents = store.list_agents(proj["id"])
    if agents:
        table = Table(title="Agents")
        table.add_column("Role")
        table.add_column("Model")
        table.add_column("Backend")
        for a in agents:
            table.add_row(a["role"], a["model_name"], a["cli_backend"])
        console.print(table)

    # Work items — with work mode (solo vs teamwork)
    items = _safe_list(store.list_work_items(proj["id"]))
    runtime_rows = _latest_runtime_rows(store, int(proj["id"]), items) if details else []
    if details:
        _print_alerts(console, _collect_project_alerts(items, runtime_rows))
        _print_worker_health(console, runtime_rows)

    if items:
        table = Table(title="Work Items")
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Stage")
        table.add_column("Mode")
        table.add_column("Type")
        table.add_column("Agents")
        table.add_column("Title", max_width=50)
        for item in items[:20]:
            # Determine work mode from review_cycles
            cycles = pool.fetch_all(
                "SELECT DISTINCT rc.proposer_agent_id, rc.reviewer_agent_id, "
                "a1.role AS proposer_role, a2.role AS reviewer_role "
                "FROM review_cycles rc "
                "JOIN agents a1 ON rc.proposer_agent_id = a1.id "
                "JOIN agents a2 ON rc.reviewer_agent_id = a2.id "
                "WHERE rc.work_item_id = %s",
                (item["id"],),
            )
            if cycles:
                inferred_mode = "[cyan]team[/cyan]"
                agent_names = set()
                for c in cycles:
                    agent_names.add(c["proposer_role"])
                    agent_names.add(c["reviewer_role"])
                agents_str = ", ".join(sorted(agent_names))
            else:
                # Solo — find the assigned agent or session agent
                assigned = item.get("assigned_agent_id")
                if assigned:
                    agent_row = pool.fetch_one(
                        "SELECT role FROM agents WHERE id = %s", (assigned,),
                    )
                    agents_str = agent_row["role"] if agent_row else "?"
                else:
                    # Check sessions linked to this work item
                    sess_agents = pool.fetch_all(
                        "SELECT DISTINCT a.role FROM sessions s "
                        "JOIN agents a ON s.agent_id = a.id "
                        "WHERE s.work_item_id = %s",
                        (item["id"],),
                    )
                    agents_str = ", ".join(s["role"] for s in sess_agents) if sess_agents else "-"
                inferred_mode = "[dim]solo[/dim]"
            mode = _display_mode(item, inferred_mode)
            metadata = item.get("metadata_json") if isinstance(item, dict) else None
            task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
            stage = task_state.get("current_stage", "-") if isinstance(task_state, dict) else "-"
            table.add_row(
                str(item["id"]), _markup_for_item_status(str(item["status"])), stage, mode,
                item["item_type"], agents_str, item["title"][:50],
            )
        console.print(table)

        if details:
            for item in items[:10]:
                console.print(
                    f"\n[bold]Work Item #{item['id']}[/bold] — {item['title'][:80]} "
                    f"{_markup_for_item_status(str(item.get('status') or 'unknown'))}"
                )
                _print_task_state(console, item)
                _print_teamwork_details(pool, item, console, details=True, show_header=False)
    else:
        console.print("\n[dim]No work items yet.[/dim]")

    # Active sessions
    from myswat.models.session import Session
    active_sessions = pool.fetch_all(
        "SELECT s.*, a.role, a.display_name FROM sessions s "
        "JOIN agents a ON s.agent_id = a.id "
        "WHERE a.project_id = %s AND s.status = 'active' "
        "ORDER BY s.created_at DESC LIMIT 10",
        (proj["id"],),
    )
    if active_sessions:
        table = Table(title="Active Sessions")
        table.add_column("UUID", style="cyan", max_width=12)
        table.add_column("Agent")
        table.add_column("Turns", justify="right")
        table.add_column("Tokens (est)", justify="right")
        table.add_column("Progress", max_width=60)
        for sess in active_sessions:
            turn_count = store.count_session_turns(sess["id"])
            # Check if agent is currently thinking (last turn is user, no agent reply yet)
            last_turn = pool.fetch_one(
                "SELECT role FROM session_turns WHERE session_id = %s "
                "ORDER BY turn_index DESC LIMIT 1",
                (sess["id"],),
            )
            is_thinking = last_turn and last_turn["role"] == "user"
            status_icon = " [bold yellow]⏳ agent thinking[/bold yellow]" if is_thinking else ""
            progress = (sess.get("purpose") or "")[:60]

            table.add_row(
                sess["session_uuid"][:12],
                sess["display_name"],
                str(turn_count),
                str(sess.get("token_count_est", 0)),
                progress,
            )
            if is_thinking:
                # Show what the agent is working on
                pending_turn = pool.fetch_one(
                    "SELECT content FROM session_turns WHERE session_id = %s "
                    "AND role = 'user' ORDER BY turn_index DESC LIMIT 1",
                    (sess["id"],),
                )
                if pending_turn:
                    pending_text = pending_turn["content"][:200].replace("\n", " ")
                    if len(pending_turn["content"]) > 200:
                        pending_text += "..."
        console.print(table)

        # Show recent turns + thinking status from active sessions
        for sess in active_sessions[:3]:
            recent_turns = pool.fetch_all(
                "SELECT role, content, metadata_json, created_at FROM session_turns "
                "WHERE session_id = %s ORDER BY turn_index DESC LIMIT 6",
                (sess["id"],),
            )
            if not recent_turns:
                continue

            last_role = recent_turns[0]["role"]
            thinking = last_role == "user"
            recent_turns.reverse()  # chronological order

            header = (
                f"\n[bold]Recent activity[/bold] — "
                f"{sess['display_name']} ({sess['session_uuid'][:8]})"
            )
            if thinking:
                header += " [bold yellow]⏳ agent thinking...[/bold yellow]"
            console.print(header)

            for t in recent_turns:
                role_label = "[green]user[/green]" if t["role"] == "user" else "[cyan]agent[/cyan]"
                content = t["content"][:150].replace("\n", " ")
                if len(t["content"]) > 150:
                    content += "..."
                # Show elapsed time for agent turns
                time_tag = ""
                if t["role"] == "assistant" and t.get("metadata_json"):
                    try:
                        import json as _json
                        meta = t["metadata_json"]
                        if isinstance(meta, str):
                            meta = _json.loads(meta)
                        elapsed = meta.get("elapsed_seconds")
                        if elapsed is not None:
                            s = int(elapsed)
                            if s < 60:
                                time_tag = f" [dim]({s}s)[/dim]"
                            else:
                                m, s = divmod(s, 60)
                                time_tag = f" [dim]({m}m{s:02d}s)[/dim]"
                    except Exception:
                        pass
                console.print(f"  {role_label}{time_tag}: {content}")

    console.print(
        "\n[dim]Use `myswat task <id> -p "
        f"{proj['slug']}` for one work item, or `myswat status -p {proj['slug']} --details` for full workflow detail.[/dim]"
    )

    # Knowledge stats
    knowledge_row = pool.fetch_one(
        "SELECT COUNT(*) AS cnt FROM knowledge WHERE project_id = %s",
        (proj["id"],),
    )
    knowledge_count = knowledge_row["cnt"] if knowledge_row else 0
    compacted_row = pool.fetch_one(
        "SELECT COUNT(*) AS cnt FROM sessions s "
        "JOIN agents a ON s.agent_id = a.id "
        "WHERE a.project_id = %s AND s.status = 'compacted'",
        (proj["id"],),
    )
    compacted_count = compacted_row["cnt"] if compacted_row else 0
    console.print(
        f"\n[dim]Knowledge: {knowledge_count} entries | "
        f"Compacted sessions: {compacted_count}[/dim]"
    )


@app.command()
def task(
    work_item_id: int = typer.Argument(..., help="Work item ID"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Show detailed status for one work item."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.memory.store import MemoryStore

    console = Console()
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    item = store.get_work_item(work_item_id)
    if not item or item.get("project_id") != proj["id"]:
        console.print(f"[red]Work item {work_item_id} not found in project '{project}'.[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]Work Item #{item['id']}[/bold] — {item['title']} "
        f"{_markup_for_item_status(str(item.get('status') or 'unknown'))}"
    )
    console.print(f"[bold]Type:[/bold] {item['item_type']}")
    if item.get("description"):
        console.print(f"[bold]Description:[/bold] {item['description'][:500]}")

    runtime_rows = _latest_runtime_rows(store, int(proj["id"]), [item])
    _print_alerts(console, _collect_project_alerts([item], runtime_rows))
    _print_worker_health(console, runtime_rows)
    _print_task_state(console, item)
    _print_teamwork_details(pool, item, console, details=True, show_header=False)


if __name__ == "__main__":
    app()
