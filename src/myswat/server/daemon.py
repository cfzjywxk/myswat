"""Persistent local daemon for MySwat workflow submission and worker supervision."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from myswat.cli.init_cmd import _slugify, run_init
from myswat.cli.work_cmd import _run_workflow
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.memory.store import MemoryStore
from myswat.server.mcp_stdio import MySwatMCPDispatcher, dispatch_rpc_request
from myswat.server.service import MySwatToolService
from myswat.workflow.modes import WorkMode

_ACTIVE_WORK_ITEM_STATUSES = frozenset({"pending", "in_progress", "review"})
_TERMINAL_WORK_ITEM_STATUSES = frozenset({"approved", "blocked", "cancelled", "completed", "paused"})
_REQUEST_SLOW_SECONDS = 1.0
_VERBOSE_REQUEST_SECONDS = 10.0
_PROJECT_SNAPSHOT_INTERVAL_SECONDS = 15.0
_NOISY_HTTP_PATHS = frozenset({"/api/work-item"})
_NOISY_MCP_TOOLS = frozenset({
    "heartbeat_runtime",
    "renew_stage_run_lease",
    "renew_review_cycle_lease",
    "claim_next_assignment",
})

LOGGER = logging.getLogger(__name__)
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


def _clip_for_log(value: Any, *, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _format_age(seconds: float | None) -> str:
    total = int(max(0.0, float(seconds or 0.0)))
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _local_timezone():
    return datetime.now().astimezone().tzinfo or timezone.utc


def _normalize_runtime_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).astimezone(_local_timezone())
    return value.astimezone(_local_timezone())


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _worker_env() -> dict[str, str]:
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


@dataclass
class ManagedWorkflow:
    project_slug: str
    cancel_event: threading.Event
    requested_status: str | None = None


@dataclass
class ManagedWorkerProcess:
    process: subprocess.Popen
    workdir: str | None = None


_WORKER_SUPERVISION_INTERVAL_SECONDS = 2.0


class MySwatDaemon:
    def __init__(self, settings: MySwatSettings | None = None) -> None:
        self._settings = settings or MySwatSettings()
        self._pool = TiDBPool(self._settings.tidb)
        ensure_schema(self._pool)
        self._store = MemoryStore(
            self._pool,
            tidb_embedding_model=self._settings.embedding.tidb_model,
            embedding_backend=self._settings.embedding.backend,
        )
        self._service = MySwatToolService(self._store)
        self._dispatcher = MySwatMCPDispatcher(self._service)
        self._lock = threading.RLock()
        self._workers: dict[tuple[str, str], ManagedWorkerProcess] = {}
        self._workflows: dict[int, threading.Thread] = {}
        self._workflow_controls: dict[int, ManagedWorkflow] = {}
        self._supervisor_stop_event = threading.Event()
        self._supervisor_thread: threading.Thread | None = None
        self._last_project_snapshot_at = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self._settings.server.host}:{self._settings.server.port}"

    def _worker_log_path(self, project_slug: str, role: str) -> Path:
        runtime_dir = self._settings.config_path.expanduser().parent / "workers" / project_slug
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir / f"{role}.log"

    def _project_runtime_paths(self, project_slug: str) -> list[Path]:
        runtime_root = self._settings.config_path.expanduser().parent
        return [
            runtime_root / "workers" / project_slug,
            runtime_root / "runs" / project_slug,
        ]

    def _cleanup_project_runtime_files(self, project_slug: str) -> list[str]:
        removed: list[str] = []
        for path in self._project_runtime_paths(project_slug):
            if not path.exists():
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
        return removed

    def _worker_roles_for_mode(self, project_id: int, mode: WorkMode) -> list[str]:
        roles: list[str] = []
        if mode in {WorkMode.full, WorkMode.design} and self._store.get_agent(project_id, "architect"):
            roles.append("architect")
        if self._store.get_agent(project_id, "developer"):
            roles.append("developer")
        for qa_role in ("qa_main", "qa_vice"):
            if self._store.get_agent(project_id, qa_role):
                roles.append(qa_role)
        if mode == WorkMode.test and "developer" not in roles and self._store.get_agent(project_id, "developer"):
            roles.append("developer")
        return roles

    def _start_worker(self, *, project_slug: str, role: str, workdir: str | None) -> None:
        key = (project_slug, role)
        existing = self._workers.get(key)
        if existing is not None and existing.process.poll() is None:
            return

        log_path = self._worker_log_path(project_slug, role)
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            command = [
                sys.executable,
                "-m",
                "myswat.cli.main",
                "worker",
                "--project",
                project_slug,
                "--role",
                role,
                "--server-url",
                self.base_url,
            ]
            if workdir:
                command.extend(["--workdir", workdir])
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
                env=_worker_env(),
            )
        finally:
            log_file.close()
        self._workers[key] = ManagedWorkerProcess(process=proc, workdir=workdir)
        LOGGER.info(
            "Worker started: project=%s role=%s pid=%s workdir=%s log=%s",
            project_slug,
            role,
            proc.pid,
            workdir or "-",
            log_path,
        )

    def _stop_worker_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except OSError:
            pass

    def _stop_project_workers(self, project_slug: str) -> None:
        with self._lock:
            to_stop = [
                (key, worker.process)
                for key, worker in self._workers.items()
                if key[0] == project_slug
            ]
            for key, _proc in to_stop:
                self._workers.pop(key, None)

        if to_stop:
            LOGGER.info(
                "Stopping project workers: project=%s workers=%s",
                project_slug,
                ", ".join(sorted(str(key[1]) for key, _proc in to_stop)),
            )

        for _key, proc in to_stop:
            self._stop_worker_process(proc)

        project = self._store.get_project_by_slug(project_slug)
        if not project:
            return
        for runtime in self._store.list_runtime_registrations(int(project["id"]), status="online"):
            self._store.update_runtime_status(
                int(runtime.id or 0),
                status="offline",
                metadata_json={"stop_reason": "project_worker_recycled"},
            )

    def _mark_dead_worker_runtime_offline(
        self,
        *,
        project_id: int,
        role: str,
        pid: int | None,
        exit_code: int,
    ) -> None:
        for runtime in self._store.list_runtime_registrations(
            project_id,
            agent_role=role,
            status="online",
        ):
            metadata_json = dict(runtime.metadata_json or {})
            runtime_pid = metadata_json.get("pid")
            if pid is not None and runtime_pid is not None and int(runtime_pid) != pid:
                continue
            metadata_json.update(
                {
                    "stop_reason": "worker_process_exited",
                    "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "exit_code": exit_code,
                }
            )
            self._store.update_runtime_status(
                int(runtime.id or 0),
                status="offline",
                metadata_json=metadata_json,
            )

    def _supervise_workers_once(self) -> None:
        with self._lock:
            dead_workers = [
                (key, worker)
                for key, worker in self._workers.items()
                if worker.process.poll() is not None
            ]
            for key, _worker in dead_workers:
                self._workers.pop(key, None)

        for (project_slug, role), worker in dead_workers:
            exit_code = int(worker.process.poll() or 0)
            project = self._store.get_project_by_slug(project_slug)
            if project:
                self._mark_dead_worker_runtime_offline(
                    project_id=int(project["id"]),
                    role=role,
                    pid=worker.process.pid,
                    exit_code=exit_code,
                )
            active_item = self._find_active_work_item(project_slug)
            should_restart = active_item is not None
            LOGGER.warning(
                "Worker exited: project=%s role=%s pid=%s exit_code=%s restart=%s",
                project_slug,
                role,
                worker.process.pid,
                exit_code,
                "yes" if should_restart else "no",
            )
            if should_restart:
                with self._lock:
                    self._start_worker(project_slug=project_slug, role=role, workdir=worker.workdir)

    def _ensure_worker_supervisor_started(self) -> None:
        with self._lock:
            existing = self._supervisor_thread
            if existing is not None and existing.is_alive():
                return
            self._supervisor_stop_event.clear()
            thread = threading.Thread(
                target=self._supervisor_loop,
                name="myswat-worker-supervisor",
                daemon=True,
            )
            self._supervisor_thread = thread
        thread.start()

    def _supervisor_loop(self) -> None:
        while not self._supervisor_stop_event.wait(_WORKER_SUPERVISION_INTERVAL_SECONDS):
            try:
                self._supervise_workers_once()
                self._log_active_project_snapshots()
            except Exception:
                continue

    @staticmethod
    def _task_state_from_item(item: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        metadata = item.get("metadata_json")
        if not isinstance(metadata, dict):
            return {}
        task_state = metadata.get("task_state")
        return task_state if isinstance(task_state, dict) else {}

    @staticmethod
    def _prefer_runtime(candidate, existing) -> bool:
        candidate_online = str(candidate.status or "") == "online"
        existing_online = str(existing.status or "") == "online"
        if candidate_online != existing_online:
            return candidate_online
        candidate_updated = _normalize_runtime_datetime(
            candidate.updated_at or candidate.last_heartbeat_at or candidate.created_at
        )
        existing_updated = _normalize_runtime_datetime(
            existing.updated_at or existing.last_heartbeat_at or existing.created_at
        )
        if candidate_updated is None:
            return False
        if existing_updated is None:
            return True
        return candidate_updated >= existing_updated

    def _latest_runtimes_by_role(self, project_id: int) -> list[Any]:
        runtimes = self._store.list_runtime_registrations(project_id)
        selected: dict[str, Any] = {}
        for runtime in runtimes:
            role = str(runtime.agent_role or runtime.runtime_name or f"runtime-{runtime.id}")
            existing = selected.get(role)
            if existing is None or self._prefer_runtime(runtime, existing):
                selected[role] = runtime
        return [selected[role] for role in sorted(selected)]

    def _heartbeat_label(self, runtime: Any, *, now: datetime) -> str:
        status = str(runtime.status or "unknown")
        last_heartbeat_at = _normalize_runtime_datetime(runtime.last_heartbeat_at)
        lease_expires_at = _normalize_runtime_datetime(runtime.lease_expires_at)

        heartbeat_state = status
        if status == "online":
            heartbeat_state = "healthy"
            if lease_expires_at is not None:
                seconds_until_expiry = (lease_expires_at - now).total_seconds()
                if seconds_until_expiry <= 0:
                    heartbeat_state = "stale"
                elif seconds_until_expiry <= 60:
                    heartbeat_state = "late"
            if heartbeat_state == "healthy" and last_heartbeat_at is not None:
                heartbeat_age = (now - last_heartbeat_at).total_seconds()
                if heartbeat_age >= 300:
                    heartbeat_state = "stale"
                elif heartbeat_age >= 120:
                    heartbeat_state = "late"

        if last_heartbeat_at is None:
            return heartbeat_state
        return f"{heartbeat_state}/{_format_age((now - last_heartbeat_at).total_seconds())}"

    def _build_assignment_index(self, project_id: int) -> dict[int, dict[str, Any]]:
        assignments: dict[int, dict[str, Any]] = {}
        for item in self._store.list_work_items(project_id):
            if str(item.get("status") or "") not in _ACTIVE_WORK_ITEM_STATUSES:
                continue
            work_item_id = int(item.get("id") or 0)
            if work_item_id <= 0:
                continue
            for stage_run in self._store.list_stage_runs(work_item_id):
                runtime_id = int(stage_run.claimed_by_runtime_id or 0)
                if runtime_id <= 0 or str(stage_run.status or "") != "claimed":
                    continue
                assignments[runtime_id] = {
                    "kind": "busy",
                    "work_item_id": work_item_id,
                    "label": f"stage={stage_run.stage_name} work_item=#{work_item_id}",
                }
            for cycle in self._store.get_review_cycles(work_item_id):
                runtime_id = int(cycle.get("claimed_by_runtime_id") or 0)
                if runtime_id <= 0 or str(cycle.get("status") or "") != "claimed":
                    continue
                stage_name = str(cycle.get("stage_name") or "review")
                assignments[runtime_id] = {
                    "kind": "busy",
                    "work_item_id": work_item_id,
                    "label": f"review={stage_name} work_item=#{work_item_id}",
                }
        return assignments

    def _session_note_for_runtime(self, runtime: Any, *, work_item_id: int) -> str:
        agent_id = int(runtime.agent_id or 0)
        if agent_id <= 0 or work_item_id <= 0:
            return ""
        try:
            session = self._store.get_active_session(agent_id, work_item_id)
        except Exception:
            return ""
        if session is None:
            return ""
        return _clip_for_log(getattr(session, "purpose", ""), limit=100)

    def _tracked_project_slugs(self) -> list[str]:
        project_slugs: set[str] = set()
        lock = getattr(self, "_lock", None)
        workers = getattr(self, "_workers", {}) or {}
        controls = getattr(self, "_workflow_controls", {}) or {}
        if lock is not None:
            with lock:
                worker_keys = list(workers.keys())
                control_values = list(controls.values())
        else:
            worker_keys = list(workers.keys())
            control_values = list(controls.values())
        for project_slug, _role in worker_keys:
            project_slugs.add(str(project_slug))
        for control in control_values:
            project_slug = getattr(control, "project_slug", None)
            if project_slug:
                project_slugs.add(str(project_slug))
        return sorted(project_slugs)

    def _project_snapshot_line(self, project_slug: str) -> str | None:
        store = getattr(self, "_store", None)
        if store is None:
            return None

        project = store.get_project_by_slug(project_slug)
        if not project:
            return f"project={project_slug} status=missing"

        project_id = int(project.get("id") or 0)
        active_item = self._find_active_work_item(project_slug)
        task_state = self._task_state_from_item(active_item)
        stage = str(task_state.get("current_stage") or "-")
        summary = _clip_for_log(
            task_state.get("latest_summary") or (active_item or {}).get("title") or "",
            limit=140,
        )

        work_item_part = "workflow=idle"
        if active_item:
            work_item_part = f"work_item=#{active_item['id']} status={active_item['status']} stage={stage}"
            if summary:
                work_item_part += f' summary="{summary}"'

        now = datetime.now(_local_timezone())
        assignment_index = self._build_assignment_index(project_id) if active_item else {}
        worker_parts: list[str] = []
        for runtime in self._latest_runtimes_by_role(project_id):
            role = str(runtime.agent_role or runtime.runtime_name or f"runtime-{runtime.id}")
            heartbeat = self._heartbeat_label(runtime, now=now)
            assignment = assignment_index.get(int(runtime.id or 0))
            if assignment is None:
                worker_parts.append(f"{role}:idle({heartbeat})")
                continue
            note = self._session_note_for_runtime(
                runtime,
                work_item_id=int(assignment.get("work_item_id") or 0),
            )
            worker_line = f"{role}:busy({heartbeat}, {assignment['label']}"
            if note:
                worker_line += f', note="{note}"'
            worker_line += ")"
            worker_parts.append(worker_line)

        if not worker_parts:
            lock = getattr(self, "_lock", None)
            workers = getattr(self, "_workers", {}) or {}
            if lock is not None:
                with lock:
                    worker_keys = list(workers.keys())
            else:
                worker_keys = list(workers.keys())
            tracked_roles = sorted(
                role
                for slug, role in worker_keys
                if slug == project_slug
            )
            worker_parts = [f"{role}:starting" for role in tracked_roles]

        workers_part = "workers=[" + "; ".join(worker_parts) + "]" if worker_parts else "workers=[]"
        return f"project={project_slug} {work_item_part} {workers_part}"

    def _log_active_project_snapshots(self, *, force: bool = False, reason: str = "Project status") -> None:
        if getattr(self, "_store", None) is None:
            return
        now = time.monotonic()
        last_snapshot_at = float(getattr(self, "_last_project_snapshot_at", 0.0) or 0.0)
        if not force and now - last_snapshot_at < _PROJECT_SNAPSHOT_INTERVAL_SECONDS:
            return
        project_slugs = self._tracked_project_slugs()
        if not project_slugs:
            self._last_project_snapshot_at = now
            return
        for project_slug in project_slugs:
            line = self._project_snapshot_line(project_slug)
            if line:
                LOGGER.info("%s: %s", reason, line)
        self._last_project_snapshot_at = now

    @staticmethod
    def _mcp_tool_name(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        if str(payload.get("method") or "") != "tools/call":
            return ""
        params = payload.get("params")
        if not isinstance(params, dict):
            return ""
        return str(params.get("name") or "")

    def _project_ref(self, *, project_slug: str | None = None, project_id: Any = None) -> str:
        if project_slug:
            return f"project={project_slug}"
        try:
            parsed_project_id = int(project_id or 0)
        except (TypeError, ValueError):
            parsed_project_id = 0
        if parsed_project_id <= 0:
            return ""
        store = getattr(self, "_store", None)
        if store is None:
            return f"project_id={parsed_project_id}"
        try:
            project = store.get_project(parsed_project_id)
        except Exception:
            project = None
        if isinstance(project, dict) and project.get("slug"):
            return f"project={project['slug']}"
        return f"project_id={parsed_project_id}"

    def _describe_request(self, path: str, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            payload = {}
        if path == "/api/work":
            parts = [self._project_ref(project_slug=str(payload.get("project") or ""))]
            mode = str(payload.get("mode") or WorkMode.full.value)
            if mode:
                parts.append(f"mode={mode}")
            if payload.get("skip_ga_test"):
                parts.append("skip_ga_test=true")
            requirement = _clip_for_log(payload.get("requirement") or "", limit=100)
            if requirement:
                parts.append(f'requirement="{requirement}"')
            return " ".join(part for part in parts if part)
        if path == "/api/work-item":
            return " ".join(
                part for part in [
                    self._project_ref(project_slug=str(payload.get("project") or "")),
                    f"work_item_id={int(payload.get('work_item_id') or 0)}" if payload.get("work_item_id") else "",
                ] if part
            )
        if path == "/api/work-control":
            return " ".join(
                part for part in [
                    self._project_ref(project_slug=str(payload.get("project") or "")),
                    f"work_item_id={int(payload.get('work_item_id') or 0)}" if payload.get("work_item_id") else "",
                    f"action={payload.get('action')}" if payload.get("action") else "",
                ] if part
            )
        if path == "/api/project-cleanup":
            return self._project_ref(project_slug=str(payload.get("project") or ""))
        if path != "/mcp":
            return ""

        method = str(payload.get("method") or "")
        if method != "tools/call":
            return f"method={method}" if method else ""

        params = payload.get("params")
        if not isinstance(params, dict):
            return "method=tools/call"
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        tool_name = str(params.get("name") or "")
        parts = [f"tool={tool_name}"] if tool_name else ["method=tools/call"]
        project_part = self._project_ref(
            project_slug=str(arguments.get("project") or ""),
            project_id=arguments.get("project_id"),
        )
        if project_part:
            parts.append(project_part)
        for key in ("work_item_id", "stage_run_id", "review_cycle_id", "runtime_registration_id", "agent_role", "stage_name", "status"):
            value = arguments.get(key)
            if value in (None, "", []):
                continue
            parts.append(f"{key}={value}")
        summary = _clip_for_log(arguments.get("summary") or "", limit=80)
        if summary and tool_name in {"report_status", "complete_stage_task", "fail_stage_task", "publish_review_verdict"}:
            parts.append(f'summary="{summary}"')
        return " ".join(parts)

    def _should_log_slow_request(self, *, path: str, payload: dict[str, Any] | None, duration: float) -> bool:
        if duration < _REQUEST_SLOW_SECONDS:
            return False
        if path in _NOISY_HTTP_PATHS and duration < _VERBOSE_REQUEST_SECONDS:
            return False
        if path == "/mcp":
            tool_name = self._mcp_tool_name(payload)
            if tool_name in _NOISY_MCP_TOOLS and duration < _VERBOSE_REQUEST_SECONDS:
                return False
        return True

    def _project_has_live_workflows(
        self,
        project_slug: str,
        *,
        exclude_work_item_id: int | None = None,
    ) -> bool:
        with self._lock:
            handles = list(self._workflow_controls.items())
            threads = dict(self._workflows)
        for work_item_id, handle in handles:
            if exclude_work_item_id is not None and work_item_id == exclude_work_item_id:
                continue
            if handle.project_slug != project_slug:
                continue
            thread = threads.get(work_item_id)
            if thread is not None and thread.is_alive():
                return True
        return False

    def _find_active_work_item(self, project_slug: str) -> dict | None:
        project = self._store.get_project_by_slug(project_slug)
        if not project:
            return None
        for item in self._store.list_work_items(int(project["id"])):
            if str(item.get("status") or "") in _ACTIVE_WORK_ITEM_STATUSES:
                return item
        return None

    def ensure_workers(self, *, project_slug: str, mode: WorkMode, workdir: str | None) -> list[str]:
        project = self._store.get_project_by_slug(project_slug)
        if not project:
            raise ValueError(f"Project not found: {project_slug}")
        roles = self._worker_roles_for_mode(int(project["id"]), mode)
        with self._lock:
            for role in roles:
                self._start_worker(project_slug=project_slug, role=role, workdir=workdir)
        return roles

    def _create_work_item(
        self,
        *,
        project_slug: str,
        requirement: str,
        workdir: str | None,
        mode: WorkMode,
        skip_ga_test: bool = False,
    ) -> int:
        project = self._store.get_project_by_slug(project_slug)
        if not project:
            raise ValueError(f"Project not found: {project_slug}")
        project_id = int(project["id"])
        developer = self._store.get_agent(project_id, "developer")
        if not developer:
            raise ValueError(f"Developer agent not found for project: {project_slug}")
        architect = (
            self._store.get_agent(project_id, "architect")
            if mode in {WorkMode.full, WorkMode.design}
            else None
        )

        item_metadata: dict[str, object] = {
            "work_mode": mode.value,
            "execution_mode": "daemon",
            "submitted_via": "daemon_api",
            "requested_workdir": workdir,
        }
        if skip_ga_test:
            item_metadata["skip_ga_test"] = True

        work_item_id = self._store.create_work_item(
            project_id=project_id,
            title=requirement[:200],
            description=requirement,
            item_type="design" if mode == WorkMode.design else "code_change",
            assigned_agent_id=int((architect or developer)["id"]),
            metadata_json=item_metadata,
        )
        self._store.update_work_item_state(
            work_item_id,
            current_stage="queued",
            latest_summary=requirement,
            next_todos=["Wait for MySwat daemon to dispatch the first stage"],
        )
        self._store.append_work_item_process_event(
            work_item_id,
            event_type="daemon_queued",
            title="Workflow queued",
            summary="MySwat daemon accepted the work item and will orchestrate it asynchronously.",
            from_role="user",
            to_role="myswat",
        )
        return work_item_id

    def _start_workflow_thread(
        self,
        *,
        project_slug: str,
        requirement: str,
        work_item_id: int,
        workdir: str | None,
        mode: WorkMode,
        skip_ga_test: bool = False,
    ) -> None:
        cancel_event = threading.Event()

        def _worker() -> None:
            try:
                _run_workflow(
                    project_slug,
                    requirement,
                    workdir=workdir,
                    work_item_id=work_item_id,
                    show_monitor=False,
                    background_worker=False,
                    mode=mode,
                    skip_ga_test=skip_ga_test,
                    auto_approve=True,
                    external_cancel_event=cancel_event,
                    emit_console_output=False,
                    settings=self._settings,
                    store=self._store,
                    project_row=self._store.get_project_by_slug(project_slug),
                    service=self._service,
                )
            finally:
                with self._lock:
                    self._workflows.pop(work_item_id, None)
                    self._workflow_controls.pop(work_item_id, None)
                if not self._project_has_live_workflows(project_slug):
                    self._stop_project_workers(project_slug)

        thread = threading.Thread(
            target=_worker,
            name=f"myswat-workflow-{work_item_id}",
            daemon=True,
        )
        with self._lock:
            existing = self._workflows.get(work_item_id)
            if existing is not None and existing.is_alive():
                return
            self._workflows[work_item_id] = thread
            self._workflow_controls[work_item_id] = ManagedWorkflow(
                project_slug=project_slug,
                cancel_event=cancel_event,
            )
        thread.start()

    def handle_init(self, *, name: str, repo_path: str | None, description: str | None) -> dict:
        run_init(name, repo_path, description)
        slug = _slugify(name)
        LOGGER.info(
            'Project initialized: project=%s repo=%s description="%s"',
            slug,
            repo_path or "-",
            _clip_for_log(description or "", limit=100),
        )
        return {
            "ok": True,
            "project": slug,
        }

    def handle_work(
        self,
        *,
        project: str,
        requirement: str,
        workdir: str | None,
        mode: str,
        skip_ga_test: bool = False,
    ) -> dict:
        work_mode = WorkMode(mode)
        if skip_ga_test and work_mode != WorkMode.full:
            raise ValueError("--skip-ga-test is only supported for full workflow mode.")
        with self._lock:
            active_item = self._find_active_work_item(project)
            if active_item is not None:
                raise ValueError(
                    f"Project '{project}' already has an active workflow "
                    f"(work item {active_item['id']}, status={active_item['status']})."
                )
            roles = self.ensure_workers(project_slug=project, mode=work_mode, workdir=workdir)
            work_item_id = self._create_work_item(
                project_slug=project,
                requirement=requirement,
                workdir=workdir,
                mode=work_mode,
                skip_ga_test=skip_ga_test,
            )
            self._start_workflow_thread(
                project_slug=project,
                requirement=requirement,
                work_item_id=work_item_id,
                workdir=workdir,
                mode=work_mode,
                skip_ga_test=skip_ga_test,
            )
        LOGGER.info(
            'Workflow queued: project=%s work_item_id=%s mode=%s skip_ga_test=%s workers=%s requirement="%s"',
            project,
            work_item_id,
            work_mode.value,
            str(skip_ga_test).lower(),
            ",".join(roles) if roles else "-",
            _clip_for_log(requirement, limit=120),
        )
        self._log_active_project_snapshots(force=True, reason="Project status")
        return {
            "ok": True,
            "work_item_id": work_item_id,
            "workers": roles,
        }

    def handle_get_work_item(self, *, project: str, work_item_id: int) -> dict:
        project_row = self._store.get_project_by_slug(project)
        if not project_row:
            raise ValueError(f"Project not found: {project}")
        item = self._store.get_work_item(work_item_id)
        if not item or int(item.get("project_id") or 0) != int(project_row["id"]):
            raise ValueError(f"Work item {work_item_id} not found in project '{project}'")
        return {
            "ok": True,
            "work_item": item,
        }

    def handle_control_work(
        self,
        *,
        project: str,
        work_item_id: int,
        action: str,
    ) -> dict:
        if action not in {"cancel", "pause"}:
            raise ValueError(f"Unsupported work control action: {action}")
        project_row = self._store.get_project_by_slug(project)
        if not project_row:
            raise ValueError(f"Project not found: {project}")
        item = self._store.get_work_item(work_item_id)
        if not item or int(item.get("project_id") or 0) != int(project_row["id"]):
            raise ValueError(f"Work item {work_item_id} not found in project '{project}'")

        target_status = "paused" if action == "pause" else "cancelled"
        summary = (
            "Workflow paused by user request."
            if target_status == "paused"
            else "Workflow cancelled by user request."
        )
        current_state = self._store.get_work_item_state(work_item_id) or {}
        self._store.update_work_item_status(work_item_id, target_status)
        self._store.update_work_item_state(
            work_item_id,
            current_stage=current_state.get("current_stage"),
            latest_summary=summary,
            next_todos=[],
            open_issues=[],
        )
        self._store.append_work_item_process_event(
            work_item_id,
            event_type=f"workflow_{target_status}",
            title=f"Workflow {target_status}",
            summary=summary,
            from_role="user",
            to_role="myswat",
        )
        self._store.cancel_open_stage_runs(work_item_id, summary=summary)
        self._store.cancel_open_review_cycles(work_item_id, summary=summary)
        service = getattr(self, "_service", None)
        if service is not None:
            service.notify_work_item_coordination_changed(work_item_id)

        with self._lock:
            control = self._workflow_controls.get(work_item_id)
            if control is not None:
                control.requested_status = target_status
                control.cancel_event.set()

        LOGGER.info(
            "Workflow control: project=%s work_item_id=%s action=%s new_status=%s",
            project,
            work_item_id,
            action,
            target_status,
        )
        self._stop_project_workers(project)
        return {
            "ok": True,
            "work_item_id": work_item_id,
            "status": target_status,
        }

    def handle_cleanup_project(
        self,
        *,
        project: str,
        wait_timeout_seconds: float = 10.0,
    ) -> dict:
        project_row = self._store.get_project_by_slug(project)
        if not project_row:
            raise ValueError(f"Project not found: {project}")

        project_id = int(project_row["id"])
        summary = "Project cleanup requested."
        work_items = self._store.list_work_items(project_id)
        work_item_ids = [int(item.get("id") or 0) for item in work_items if int(item.get("id") or 0) > 0]

        for item in work_items:
            work_item_id = int(item.get("id") or 0)
            if work_item_id <= 0:
                continue
            if str(item.get("status") or "") not in _TERMINAL_WORK_ITEM_STATUSES:
                current_state = self._store.get_work_item_state(work_item_id) or {}
                self._store.update_work_item_status(work_item_id, "cancelled")
                self._store.update_work_item_state(
                    work_item_id,
                    current_stage=current_state.get("current_stage"),
                    latest_summary=summary,
                    next_todos=[],
                    open_issues=[],
                )
                self._store.append_work_item_process_event(
                    work_item_id,
                    event_type="project_cleanup_cancelled",
                    title="Project cleanup cancelled workflow",
                    summary=summary,
                    from_role="myswat",
                    to_role="workflow",
                )
            self._store.cancel_open_stage_runs(work_item_id, summary=summary)
            self._store.cancel_open_review_cycles(work_item_id, summary=summary)
            self._service.notify_work_item_coordination_changed(work_item_id)

        with self._lock:
            tracked_workflows = [
                (work_item_id, self._workflows.get(work_item_id), control)
                for work_item_id, control in self._workflow_controls.items()
                if control.project_slug == project
            ]
            for _work_item_id, _thread, control in tracked_workflows:
                control.requested_status = "cancelled"
                control.cancel_event.set()

        self._stop_project_workers(project)

        deadline = time.monotonic() + max(0.0, wait_timeout_seconds)
        for _work_item_id, thread, _control in tracked_workflows:
            if thread is None:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)

        alive_workflows = [
            work_item_id
            for work_item_id, thread, _control in tracked_workflows
            if thread is not None and thread.is_alive()
        ]
        if alive_workflows:
            raise RuntimeError(
                "Project cleanup blocked waiting for workflows to stop: "
                + ", ".join(str(work_item_id) for work_item_id in alive_workflows)
            )

        deleted = self._store.delete_project(project_id)
        removed_runtime_paths = self._cleanup_project_runtime_files(project)
        LOGGER.info(
            "Project cleanup completed: project=%s work_items=%s deleted=%s removed_paths=%s",
            project,
            len(work_item_ids),
            deleted,
            removed_runtime_paths,
        )
        return {
            "ok": True,
            "project": project,
            "work_item_ids": work_item_ids,
            "deleted": deleted,
            "removed_runtime_paths": removed_runtime_paths,
        }

    @staticmethod
    def _public_error_message(exc: Exception) -> str:
        if isinstance(exc, (ValueError, ValidationError)):
            return str(exc)
        return "internal server error"

    def handle_mcp_request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        request_id = payload.get("id")
        method = str(payload.get("method") or "")
        params = payload.get("params") or {}
        try:
            result = dispatch_rpc_request(
                self._dispatcher,
                method,
                params if isinstance(params, dict) else {},
            )
        except Exception as exc:
            if isinstance(exc, (ValueError, ValidationError)):
                LOGGER.warning("MCP request failed: method=%s error=%s", method, exc)
            else:
                LOGGER.exception("MCP request failed: method=%s", method)
            if request_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": self._public_error_message(exc),
                },
            }
        if request_id is None or result is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def serve_forever(self) -> None:
        self._ensure_worker_supervisor_started()
        daemon = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MySwatDaemon/0.1"

            def log_message(self, format: str, *args) -> None:
                return

            def _request_duration(self, started_at: float) -> float:
                return max(0.0, time.monotonic() - started_at)

            def _log_slow_request(
                self,
                *,
                path: str,
                status: int,
                started_at: float,
                payload: dict[str, Any] | None = None,
            ) -> None:
                duration = self._request_duration(started_at)
                if not daemon._should_log_slow_request(path=path, payload=payload, duration=duration):
                    return
                detail = daemon._describe_request(path, payload)
                LOGGER.warning(
                    "Slow daemon request: %s %s status=%s duration=%.3fs%s",
                    self.command,
                    path,
                    status,
                    duration,
                    f" {detail}" if detail else "",
                )

            def _log_request_failure(
                self,
                *,
                path: str,
                status: int,
                started_at: float,
                exc: Exception,
                payload: dict[str, Any] | None = None,
            ) -> None:
                duration = self._request_duration(started_at)
                detail = daemon._describe_request(path, payload)
                if isinstance(exc, (ValueError, ValidationError)):
                    LOGGER.warning(
                        "Daemon request failed: %s %s status=%s duration=%.3fs%s error=%s",
                        self.command,
                        path,
                        status,
                        duration,
                        f" {detail}" if detail else "",
                        exc,
                    )
                    return
                LOGGER.exception(
                    "Daemon request failed: %s %s status=%s duration=%.3fs%s",
                    self.command,
                    path,
                    status,
                    duration,
                    f" {detail}" if detail else "",
                )

            def _log_client_disconnect(
                self,
                *,
                path: str,
                started_at: float,
                exc: Exception,
                payload: dict[str, Any] | None = None,
            ) -> None:
                detail = daemon._describe_request(path, payload)
                LOGGER.warning(
                    "Daemon client disconnected: %s %s duration=%.3fs%s error=%s",
                    self.command,
                    path,
                    self._request_duration(started_at),
                    f" {detail}" if detail else "",
                    exc,
                )

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length) if length > 0 else b"{}"
                if not body:
                    return {}
                parsed = json.loads(body.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}

            def _write_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                started_at = time.monotonic()
                parsed = urlparse(self.path)
                try:
                    if parsed.path == "/api/health":
                        self._write_json(200, {"ok": True})
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at)
                        return
                    self._write_json(404, {"error": "not found"})
                    self._log_slow_request(path=parsed.path, status=404, started_at=started_at)
                except _CLIENT_DISCONNECT_ERRORS as exc:
                    self._log_client_disconnect(path=parsed.path, started_at=started_at, exc=exc)

            def do_POST(self) -> None:
                started_at = time.monotonic()
                parsed = urlparse(self.path)
                payload: dict[str, Any] = {}
                try:
                    payload = self._read_json()
                    if parsed.path == "/api/init":
                        result = daemon.handle_init(
                            name=str(payload.get("name") or ""),
                            repo_path=payload.get("repo_path"),
                            description=payload.get("description"),
                        )
                        self._write_json(200, result)
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at, payload=payload)
                        return
                    if parsed.path == "/api/work":
                        result = daemon.handle_work(
                            project=str(payload.get("project") or ""),
                            requirement=str(payload.get("requirement") or ""),
                            workdir=payload.get("workdir"),
                            mode=str(payload.get("mode") or WorkMode.full.value),
                            skip_ga_test=bool(payload.get("skip_ga_test")),
                        )
                        self._write_json(200, result)
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at, payload=payload)
                        return
                    if parsed.path == "/api/work-item":
                        result = daemon.handle_get_work_item(
                            project=str(payload.get("project") or ""),
                            work_item_id=int(payload.get("work_item_id") or 0),
                        )
                        self._write_json(200, result)
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at, payload=payload)
                        return
                    if parsed.path == "/api/work-control":
                        result = daemon.handle_control_work(
                            project=str(payload.get("project") or ""),
                            work_item_id=int(payload.get("work_item_id") or 0),
                            action=str(payload.get("action") or "cancel"),
                        )
                        self._write_json(200, result)
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at, payload=payload)
                        return
                    if parsed.path == "/api/project-cleanup":
                        result = daemon.handle_cleanup_project(
                            project=str(payload.get("project") or ""),
                        )
                        self._write_json(200, result)
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at, payload=payload)
                        return
                    if parsed.path == "/mcp":
                        response = daemon.handle_mcp_request(payload)
                        if response is None:
                            self._write_json(202, {})
                            self._log_slow_request(path=parsed.path, status=202, started_at=started_at, payload=payload)
                            return
                        self._write_json(200, response)
                        self._log_slow_request(path=parsed.path, status=200, started_at=started_at, payload=payload)
                        return
                    self._write_json(404, {"error": "not found"})
                    self._log_slow_request(path=parsed.path, status=404, started_at=started_at, payload=payload)
                except (ValueError, ValidationError) as exc:
                    self._log_request_failure(
                        path=parsed.path,
                        status=400,
                        started_at=started_at,
                        exc=exc,
                        payload=payload,
                    )
                    self._write_json(400, {"error": str(exc)})
                except _CLIENT_DISCONNECT_ERRORS as exc:
                    self._log_client_disconnect(path=parsed.path, started_at=started_at, exc=exc, payload=payload)
                except Exception as exc:
                    self._log_request_failure(
                        path=parsed.path,
                        status=500,
                        started_at=started_at,
                        exc=exc,
                        payload=payload,
                    )
                    try:
                        self._write_json(500, {"error": "internal server error"})
                    except _CLIENT_DISCONNECT_ERRORS as disconnect_exc:
                        self._log_client_disconnect(
                            path=parsed.path,
                            started_at=started_at,
                            exc=disconnect_exc,
                            payload=payload,
                        )

        httpd = ThreadingHTTPServer(
            (self._settings.server.host, self._settings.server.port),
            Handler,
        )
        httpd.serve_forever()
