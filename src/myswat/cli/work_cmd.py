"""myswat work — run the full teamwork workflow from CLI."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

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
from myswat.workflow.engine import WorkMode, WorkflowEngine

console = Console()

_BACKGROUND_SIGNAL_SET = tuple(
    sig
    for name in ("SIGTERM", "SIGHUP", "SIGINT")
    if (sig := getattr(signal, name, None)) is not None
)


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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_workdir(workdir: str | None) -> str | None:
    if not workdir:
        return None
    return str(Path(workdir).expanduser().resolve())


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_project_context(
    project_slug: str,
    workdir: str | None,
) -> tuple[MySwatSettings, MemoryStore, dict, str | None]:
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

    effective_workdir = _normalize_workdir(workdir or proj.get("repo_path"))
    return settings, store, proj, effective_workdir


def _get_workflow_agents(store: MemoryStore, project_id: int) -> tuple[dict, list[dict]]:
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


def _load_item_metadata(store: MemoryStore, item_id: int) -> dict:
    item = store.get_work_item(item_id) or {}
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _update_background_metadata(store: MemoryStore, item_id: int, **fields: object) -> dict:
    metadata = _load_item_metadata(store, item_id)
    background = metadata.get("background")
    if not isinstance(background, dict):
        background = {}

    for key, value in fields.items():
        if value is None:
            background.pop(key, None)
        else:
            background[key] = value

    metadata["background"] = background
    store.update_work_item_metadata(item_id, metadata)
    return background


def _background_runtime_dir(settings: MySwatSettings, project_slug: str) -> Path:
    runtime_dir = Path(settings.config_path).expanduser().parent / "runs" / project_slug
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _background_runtime_paths(
    settings: MySwatSettings,
    project_slug: str,
    work_item_id: int,
) -> tuple[Path, Path]:
    runtime_dir = _background_runtime_dir(settings, project_slug)
    return (
        runtime_dir / f"work-{work_item_id}.log",
        runtime_dir / f"work-{work_item_id}.pid",
    )


def _build_background_env() -> dict[str, str]:
    env = os.environ.copy()
    src_root = _source_root()
    package_dir = src_root / "myswat"
    project_root = src_root.parent
    if package_dir.is_dir() and (project_root / "pyproject.toml").is_file():
        existing = env.get("PYTHONPATH")
        if existing:
            env["PYTHONPATH"] = os.pathsep.join([str(src_root), existing])
        else:
            env["PYTHONPATH"] = str(src_root)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _read_process_argv(pid: int) -> list[str] | None:
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = proc_cmdline.read_bytes()
    except FileNotFoundError:
        raw = b""
    except OSError:
        raw = b""
    if raw:
        return [part.decode(errors="replace") for part in raw.split(b"\0") if part]

    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    cmdline = result.stdout.strip()
    if result.returncode != 0 or not cmdline:
        return None
    try:
        return shlex.split(cmdline)
    except ValueError:
        return cmdline.split()


def _is_background_worker_pid(pid: int, work_item_id: int) -> bool:
    argv = _read_process_argv(pid)
    if not argv:
        return False

    if "work-background-worker" not in argv:
        return False

    expected_id = str(work_item_id)
    for idx, arg in enumerate(argv):
        if arg == "--work-item-id" and idx + 1 < len(argv) and argv[idx + 1] == expected_id:
            return True
        if arg.startswith("--work-item-id=") and arg.split("=", 1)[1] == expected_id:
            return True
    return False


def _cleanup_runtime_file(path_value: object) -> None:
    if not isinstance(path_value, str) or not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except OSError:
        pass


def _print_tracking_commands(project_slug: str, work_item_id: int) -> None:
    console.print("[dim]Query progress:[/dim]")
    console.print(f"[dim]  myswat task {work_item_id} -p {project_slug}[/dim]")
    console.print(f"[dim]  myswat status -p {project_slug}[/dim]\n")


def _install_cancel_signal_handlers(
    cancel_event: threading.Event,
    cancel_targets: list[AgentRunner],
) -> Callable[[], None]:
    previous_handlers = {sig: signal.getsignal(sig) for sig in _BACKGROUND_SIGNAL_SET}

    def _handle_signal(_signum, _frame) -> None:
        cancel_event.set()
        for runner in cancel_targets:
            try:
                runner.cancel()
            except Exception:
                pass

    for sig in _BACKGROUND_SIGNAL_SET:
        signal.signal(sig, _handle_signal)

    def _restore() -> None:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    return _restore


def _finalize_background_run(
    store: MemoryStore,
    work_item_id: int,
    *,
    state: str,
    summary: str,
) -> None:
    try:
        metadata = _load_item_metadata(store, work_item_id)
        background = metadata.get("background") if isinstance(metadata, dict) else None
        pid_path = background.get("pid_path") if isinstance(background, dict) else None
        _cleanup_runtime_file(pid_path)
        _update_background_metadata(
            store,
            work_item_id,
            state=state,
            finished_at=_now_iso(),
        )
        store.append_work_item_process_event(
            work_item_id,
            event_type="background_finished",
            title="Background workflow finished",
            summary=summary,
            from_role="myswat",
            to_role="workflow",
        )
    except Exception:
        pass


def _run_workflow(
    project_slug: str,
    requirement: str,
    *,
    workdir: str | None = None,
    work_item_id: int | None = None,
    show_monitor: bool,
    background_worker: bool,
    mode: WorkMode = WorkMode.full,
) -> int:
    settings, store, proj, effective_workdir = _load_project_context(project_slug, workdir)

    # Auto-learn if project hasn't been learned yet.
    from myswat.cli.learn_cmd import ensure_learned

    ensure_learned(store, project_slug, proj["id"], effective_workdir)

    dev_agent, qa_agents = _get_workflow_agents(store, proj["id"])

    dev_runner = _make_runner(dev_agent)
    dev_runner.workdir = effective_workdir

    compaction_runner = _make_runner(dev_agent)
    compactor = KnowledgeCompactor(
        store=store,
        runner=compaction_runner,
        threshold_turns=settings.compaction.threshold_turns,
        threshold_tokens=settings.compaction.threshold_tokens,
    )

    dev_sm = SessionManager(
        store=store,
        runner=dev_runner,
        agent_row=dev_agent,
        project_id=proj["id"],
        compactor=compactor,
    )

    qa_sms: list[SessionManager] = []
    for qa_agent in qa_agents:
        qa_runner = _make_runner(qa_agent)
        qa_runner.workdir = effective_workdir
        qa_sms.append(
            SessionManager(
                store=store,
                runner=qa_runner,
                agent_row=qa_agent,
                project_id=proj["id"],
                compactor=compactor,
            )
        )

    if work_item_id is None:
        work_item_id = store.create_work_item(
            project_id=proj["id"],
            title=requirement[:200],
            description=requirement,
            item_type="code_change",
            assigned_agent_id=dev_agent["id"],
            metadata_json={"work_mode": mode.value},
        )
    else:
        existing_item = store.get_work_item(work_item_id)
        if not existing_item or existing_item.get("project_id") != proj["id"]:
            console.print(
                f"[red]Work item {work_item_id} not found in project '{project_slug}'.[/red]"
            )
            raise typer.Exit(1)

    store.update_work_item_status(work_item_id, "in_progress")

    if background_worker:
        background = _update_background_metadata(
            store,
            work_item_id,
            state="running",
            pid=os.getpid(),
            started_at=_now_iso(),
            workdir=effective_workdir,
        )
        pid_path = background.get("pid_path")
        if isinstance(pid_path, str) and pid_path:
            Path(pid_path).write_text(f"{os.getpid()}\n", encoding="ascii")
        store.append_work_item_process_event(
            work_item_id,
            event_type="background_started",
            title="Background workflow started",
            summary=f"Detached workflow worker started with PID {os.getpid()}.",
            from_role="myswat",
            to_role="workflow",
        )

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
    _print_tracking_commands(project_slug, work_item_id)

    cancel_event = threading.Event()
    cancel_targets: list[AgentRunner] = [dev_runner] + [qa_sm._runner for qa_sm in qa_sms]
    restore_signal_handlers = (
        _install_cancel_signal_handlers(cancel_event, cancel_targets)
        if background_worker
        else None
    )
    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=qa_sms,
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_review_iterations=settings.workflow.max_review_iterations,
        mode=mode,
        auto_approve=(mode != WorkMode.design),
        should_cancel=cancel_event.is_set,
    )

    final_status = "blocked"
    final_summary = "Workflow blocked."

    try:
        if show_monitor:
            work_item_ref: dict[str, int | None] = {"id": work_item_id}

            def _worker():
                return engine.run(requirement)

            result = _run_with_task_monitor(
                console=console,
                store=store,
                proj=proj,
                label="Running full teamwork workflow" if mode == WorkMode.full else f"Running {mode.value} teamwork workflow",
                worker_fn=_worker,
                work_item_ref=work_item_ref,
                cancel_targets=cancel_targets,
                cancel_event=cancel_event,
            )
        else:
            result = engine.run(requirement)

        if cancel_event.is_set():
            final_status = "blocked"
            final_summary = "Workflow cancelled."
        elif result.success:
            final_status = "completed"
            final_summary = "Workflow completed successfully."
        else:
            final_status = "review"
            final_summary = "Workflow finished with review or unresolved issues."

        store.update_work_item_status(work_item_id, final_status)
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

        final_status = "blocked"
        final_summary = f"Workflow crashed: {type(e).__name__}"
        try:
            store.update_work_item_status(work_item_id, final_status)
        except Exception:
            pass
    finally:
        if restore_signal_handlers is not None:
            restore_signal_handlers()

        for sm in [dev_sm] + qa_sms:
            try:
                sm.close()
            except Exception:
                pass

        if background_worker:
            _finalize_background_run(
                store,
                work_item_id,
                state=final_status,
                summary=final_summary,
            )

    console.print("\n[dim]Sessions closed. All turns persisted to TiDB.[/dim]")
    return work_item_id


def _launch_background_work(
    project_slug: str,
    requirement: str,
    *,
    workdir: str | None = None,
    mode: WorkMode = WorkMode.full,
) -> int:
    if mode == WorkMode.design:
        raise typer.BadParameter("Design mode cannot be combined with --background.")

    settings, store, proj, effective_workdir = _load_project_context(project_slug, workdir)
    dev_agent, _qa_agents = _get_workflow_agents(store, proj["id"])

    work_item_id = store.create_work_item(
        project_id=proj["id"],
        title=requirement[:200],
        description=requirement,
        item_type="code_change",
        assigned_agent_id=dev_agent["id"],
        metadata_json={"work_mode": mode.value},
    )
    log_path, pid_path = _background_runtime_paths(settings, project_slug, work_item_id)

    _update_background_metadata(
        store,
        work_item_id,
        mode="background",
        state="launching",
        requested_at=_now_iso(),
        log_path=str(log_path),
        pid_path=str(pid_path),
        workdir=effective_workdir,
    )
    store.update_work_item_state(
        work_item_id,
        current_stage="background_launch_pending",
        latest_summary=requirement,
        next_todos=["Wait for detached workflow worker to start"],
    )
    store.append_work_item_process_event(
        work_item_id,
        event_type="background_requested",
        title="Background workflow requested",
        summary="Queued a detached workflow worker for this requirement.",
        from_role="user",
        to_role="myswat",
    )

    command = [
        sys.executable,
        "-m",
        "myswat.cli.main",
        "work-background-worker",
        requirement,
        "--project",
        project_slug,
        "--work-item-id",
        str(work_item_id),
        "--mode",
        mode.value,
    ]
    if effective_workdir:
        command.extend(["--workdir", effective_workdir])

    log_file = None
    try:
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            env=_build_background_env(),
        )
    except Exception as exc:
        store.update_work_item_status(work_item_id, "blocked")
        _update_background_metadata(
            store,
            work_item_id,
            state="launch_failed",
            finished_at=_now_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )
        store.append_work_item_process_event(
            work_item_id,
            event_type="background_launch_failed",
            title="Background launch failed",
            summary=f"Detached workflow launch failed: {type(exc).__name__}: {exc}",
            from_role="myswat",
            to_role="workflow",
        )
        raise
    finally:
        if log_file is not None:
            log_file.close()

    pid_path.write_text(f"{proc.pid}\n", encoding="ascii")
    _update_background_metadata(
        store,
        work_item_id,
        state="spawned",
        pid=proc.pid,
        started_at=_now_iso(),
    )
    store.append_work_item_process_event(
        work_item_id,
        event_type="background_spawned",
        title="Background worker spawned",
        summary=f"Detached workflow worker spawned with PID {proc.pid}.",
        from_role="myswat",
        to_role="workflow",
    )

    console.print(f"[bold]Requirement:[/bold] {requirement}")
    console.print(f"[dim]Work item: {work_item_id}[/dim]")
    console.print(f"[dim]PID: {proc.pid}[/dim]")
    console.print(f"[dim]Log: {log_path}[/dim]")
    _print_tracking_commands(project_slug, work_item_id)
    return work_item_id


def run_background_work_item(
    project_slug: str,
    requirement: str,
    *,
    work_item_id: int,
    workdir: str | None = None,
    mode: WorkMode = WorkMode.full,
) -> None:
    """Entry point for the detached workflow worker."""
    if mode == WorkMode.design:
        raise typer.BadParameter("Design mode cannot be combined with --background.")
    _run_workflow(
        project_slug,
        requirement,
        workdir=workdir,
        work_item_id=work_item_id,
        show_monitor=False,
        background_worker=True,
        mode=mode,
    )


def run_work(
    project_slug: str,
    requirement: str,
    workdir: str | None = None,
    background: bool = False,
    mode: WorkMode = WorkMode.full,
) -> None:
    """Run the full teamwork workflow."""
    if background:
        if mode == WorkMode.design:
            raise typer.BadParameter("Design mode cannot be combined with --background.")
        _launch_background_work(project_slug, requirement, workdir=workdir, mode=mode)
        return

    _run_workflow(
        project_slug,
        requirement,
        workdir=workdir,
        show_monitor=True,
        background_worker=False,
        mode=mode,
    )


def stop_work_item(project_slug: str, work_item_id: int) -> None:
    """Request cancellation of a detached work item."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    item = store.get_work_item(work_item_id)
    if not item or item.get("project_id") != proj["id"]:
        console.print(f"[red]Work item {work_item_id} not found in project '{project_slug}'.[/red]")
        raise typer.Exit(1)

    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    background = metadata.get("background") if isinstance(metadata, dict) else None
    if not isinstance(background, dict):
        background = {}

    pid = background.get("pid")
    if not isinstance(pid, int):
        console.print(f"[red]Work item {work_item_id} has no background worker PID.[/red]")
        raise typer.Exit(1)

    if item.get("status") in {"completed", "approved", "blocked"}:
        console.print(
            f"[red]Work item {work_item_id} is already in terminal state '{item['status']}'.[/red]"
        )
        raise typer.Exit(1)

    if not _is_background_worker_pid(pid, work_item_id):
        console.print(
            f"[red]PID {pid} is not the expected myswat background worker for work item {work_item_id}.[/red]"
        )
        raise typer.Exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        console.print(
            f"[red]Background worker PID {pid} is not running. Check `myswat task {work_item_id} -p {project_slug}`.[/red]"
        )
        raise typer.Exit(1)

    store.update_work_item_state(
        work_item_id,
        current_stage="cancellation_requested",
        latest_summary="Cancellation requested from CLI.",
        next_todos=["Wait for the current agent step to stop"],
    )
    store.append_work_item_process_event(
        work_item_id,
        event_type="cancellation_requested",
        title="Stop requested",
        summary=f"Sent SIGTERM to background workflow worker PID {pid}.",
        from_role="user",
        to_role="myswat",
    )
    _update_background_metadata(
        store,
        work_item_id,
        state="stop_requested",
        stop_requested_at=_now_iso(),
    )

    console.print(f"[dim]Sent stop signal to work item {work_item_id} (PID {pid}).[/dim]")
