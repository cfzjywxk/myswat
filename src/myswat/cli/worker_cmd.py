"""Internal managed worker for MySwat daemon-supervised agent roles."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable

import typer

from myswat.agents.factory import make_runner_from_row
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.large_payloads import (
    AGENT_FILE_PROMPT,
    maybe_externalize_prompt,
    maybe_externalize_system_context,
    resolve_externalized_text,
    resolve_externalized_value,
)
from myswat.memory.store import MemoryStore
from myswat.server.mcp_http_client import MCPHTTPClient
from myswat.workflow.review_loop import _parse_verdict

_RUNTIME_LEASE_SECONDS = 300
_ASSIGNMENT_LEASE_SECONDS = 300
_KEEPALIVE_INTERVAL_SECONDS = 60.0

LOGGER = logging.getLogger(__name__)


def _build_default_runtime_name(project_slug: str, role: str) -> str:
    return f"{project_slug}-{role}"


def _mark_runtime_offline(
    mcp: MCPHTTPClient,
    *,
    runtime_registration_id: int,
    reason: str,
) -> None:
    mcp.call_tool(
        "update_runtime_status",
        {
            "runtime_registration_id": runtime_registration_id,
            "status": "offline",
            "metadata_json": {
                "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stop_reason": reason,
            },
        },
    )


def _prepare_runner_payloads(
    *,
    assignment: dict[str, Any],
    role: str,
) -> tuple[str, str | None]:
    prompt = str(assignment.get("prompt") or "")
    system_context = str(assignment.get("system_context") or "")
    stage_name = str(assignment.get("stage_name") or assignment.get("assignment_kind") or "task")
    file_aware_system_context = (
        "\n\n---\n\n".join([AGENT_FILE_PROMPT, system_context])
        if system_context
        else AGENT_FILE_PROMPT
    )

    prompt_to_send, _ = maybe_externalize_prompt(
        prompt,
        label=f"{role}-{stage_name}-request",
    )
    system_context_to_send, _ = maybe_externalize_system_context(
        file_aware_system_context,
        label=f"{role}-{stage_name}-context",
    )
    return prompt_to_send, system_context_to_send


def _resolve_runner_response_content(text: str) -> str:
    content = str(text or "")
    stripped = content.strip()
    if not stripped:
        return content

    def _strip_code_fences(value: str) -> str:
        if "```json" in value:
            return value.split("```json", 1)[1].split("```", 1)[0].strip()
        return value

    candidate = _strip_code_fences(stripped)
    try:
        payload = resolve_externalized_value(json.loads(candidate))
    except (json.JSONDecodeError, TypeError):
        return resolve_externalized_text(content)

    rendered = json.dumps(payload, ensure_ascii=False)
    if stripped.startswith("```json"):
        return f"```json\n{rendered}\n```"
    return rendered


def _assignment_keepalive_call(
    *,
    assignment: dict[str, Any],
    mcp: MCPHTTPClient,
    runtime_registration_id: int,
) -> None:
    kind = str(assignment.get("assignment_kind") or "")
    mcp.call_tool(
        "heartbeat_runtime",
        {
            "runtime_registration_id": runtime_registration_id,
            "lease_seconds": _RUNTIME_LEASE_SECONDS,
        },
    )
    if kind == "stage":
        mcp.call_tool(
            "renew_stage_run_lease",
            {
                "stage_run_id": int(assignment["stage_run_id"]),
                "runtime_registration_id": runtime_registration_id,
                "lease_seconds": _ASSIGNMENT_LEASE_SECONDS,
            },
        )
        return
    if kind == "review":
        mcp.call_tool(
            "renew_review_cycle_lease",
            {
                "cycle_id": int(assignment["review_cycle_id"]),
                "runtime_registration_id": runtime_registration_id,
                "lease_seconds": _ASSIGNMENT_LEASE_SECONDS,
            },
        )
        return
    raise RuntimeError(f"Unsupported keepalive assignment kind: {kind}")


class _AssignmentKeepalive:
    def __init__(
        self,
        *,
        assignment: dict[str, Any],
        mcp: MCPHTTPClient,
        runtime_registration_id: int,
        cancel_runner: Callable[[], None] | None = None,
    ) -> None:
        self._assignment = assignment
        self._mcp = mcp
        self._runtime_registration_id = runtime_registration_id
        self._cancel_runner = cancel_runner
        self._stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._error: Exception | None = None
        assignment_kind = str(assignment.get("assignment_kind") or "unknown")
        assignment_id = int(
            assignment.get("stage_run_id")
            or assignment.get("review_cycle_id")
            or 0
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"myswat-worker-keepalive-{assignment_kind}-{assignment_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> Exception | None:
        self._stop_event.set()
        self._thread.join()
        with self._error_lock:
            return self._error

    def _record_error(self, exc: Exception) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _run(self) -> None:
        while not self._stop_event.wait(_KEEPALIVE_INTERVAL_SECONDS):
            try:
                _assignment_keepalive_call(
                    assignment=self._assignment,
                    mcp=self._mcp,
                    runtime_registration_id=self._runtime_registration_id,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Worker assignment keepalive failed: kind=%s stage_run_id=%s review_cycle_id=%s error=%s",
                    self._assignment.get("assignment_kind"),
                    self._assignment.get("stage_run_id"),
                    self._assignment.get("review_cycle_id"),
                    exc,
                )
                self._record_error(exc)
                if self._cancel_runner is not None:
                    try:
                        self._cancel_runner()
                    except Exception:  # pragma: no cover - defensive only
                        LOGGER.exception("Failed to cancel runner after keepalive error")
                return


def _invoke_runner_with_keepalive(
    *,
    assignment: dict[str, Any],
    runner: Any,
    prompt: str,
    system_context: str | None,
    mcp: MCPHTTPClient,
    runtime_registration_id: int,
) -> Any:
    cancel_runner = getattr(runner, "cancel", None)
    keepalive = _AssignmentKeepalive(
        assignment=assignment,
        mcp=mcp,
        runtime_registration_id=runtime_registration_id,
        cancel_runner=cancel_runner if callable(cancel_runner) else None,
    )
    keepalive.start()
    runner_error: Exception | None = None
    response: Any = None
    try:
        response = runner.invoke(prompt, system_context=system_context)
    except Exception as exc:
        runner_error = exc
    keepalive_error = keepalive.stop()
    if runner_error is not None:
        raise runner_error
    if keepalive_error is not None:
        raise keepalive_error
    return response


def _handle_stage_assignment(
    *,
    assignment: dict[str, Any],
    runner: Any,
    mcp: MCPHTTPClient,
    runtime_registration_id: int,
    agent_row: dict[str, Any],
    role: str,
) -> None:
    prompt, system_context = _prepare_runner_payloads(
        assignment=assignment,
        role=role,
    )
    runner.reset_session()
    response = _invoke_runner_with_keepalive(
        assignment=assignment,
        runner=runner,
        prompt=prompt,
        system_context=system_context,
        mcp=mcp,
        runtime_registration_id=runtime_registration_id,
    )
    resolved_content = _resolve_runner_response_content(response.content)

    stage_run_id = int(assignment["stage_run_id"])
    work_item_id = int(assignment["work_item_id"])
    stage_name = str(assignment["stage_name"])
    if response.success:
        mcp.call_tool(
            "complete_stage_task",
            {
                "stage_run_id": stage_run_id,
                "runtime_registration_id": runtime_registration_id,
                "work_item_id": work_item_id,
                "agent_id": int(agent_row["id"]),
                "agent_role": role,
                "iteration": int(assignment.get("iteration") or 1),
                "stage_name": stage_name,
                "artifact_type": str(assignment.get("artifact_type") or "artifact"),
                "title": assignment.get("artifact_title"),
                "content": resolved_content,
                "summary": resolved_content[:4000],
            },
        )
        return

    mcp.call_tool(
        "fail_stage_task",
        {
            "stage_run_id": stage_run_id,
            "runtime_registration_id": runtime_registration_id,
            "work_item_id": work_item_id,
            "agent_id": int(agent_row["id"]),
            "agent_role": role,
            "stage_name": stage_name,
            "summary": resolved_content[:4000] or f"{role} failed during {stage_name}",
        },
    )


def _handle_review_assignment(
    *,
    assignment: dict[str, Any],
    runner: Any,
    mcp: MCPHTTPClient,
    runtime_registration_id: int,
    agent_row: dict[str, Any],
    role: str,
) -> None:
    prompt, system_context = _prepare_runner_payloads(
        assignment=assignment,
        role=role,
    )
    runner.reset_session()
    response = _invoke_runner_with_keepalive(
        assignment=assignment,
        runner=runner,
        prompt=prompt,
        system_context=system_context,
        mcp=mcp,
        runtime_registration_id=runtime_registration_id,
    )
    resolved_content = _resolve_runner_response_content(response.content)

    work_item_id = int(assignment["work_item_id"])
    cycle_id = int(assignment["review_cycle_id"])
    verdict = _parse_verdict(resolved_content if response.success else "")
    mcp.call_tool(
        "publish_review_verdict",
        {
            "cycle_id": cycle_id,
            "work_item_id": work_item_id,
            "reviewer_agent_id": int(agent_row["id"]),
            "reviewer_role": role,
            "verdict": verdict.verdict,
            "issues": verdict.issues,
            "summary": verdict.summary,
            "stage": assignment.get("stage_name") or "",
            "runtime_registration_id": runtime_registration_id,
        },
    )


def run_worker(
    *,
    project_slug: str,
    role: str,
    server_url: str,
    workdir: str | None = None,
    poll_interval_seconds: float = 1.0,
    idle_exit_seconds: float | None = None,
    settings: MySwatSettings | None = None,
    store: MemoryStore | None = None,
    project_row: dict[str, Any] | None = None,
    agent_row: dict[str, Any] | None = None,
    runner: Any | None = None,
    mcp_client: MCPHTTPClient | None = None,
) -> dict[str, int]:
    settings = settings or MySwatSettings()
    if store is None and (project_row is None or agent_row is None):
        pool = TiDBPool(settings.tidb)
        ensure_schema(pool)
        store = MemoryStore(
            pool,
            tidb_embedding_model=settings.embedding.tidb_model,
            embedding_backend=settings.embedding.backend,
        )

    project = project_row or (store.get_project_by_slug(project_slug) if store is not None else None)
    if not project:
        raise typer.BadParameter(f"Project not found: {project_slug}")

    agent_row = agent_row or (store.get_agent(int(project["id"]), role) if store is not None else None)
    if not agent_row:
        raise typer.BadParameter(f"Role '{role}' not found in project '{project_slug}'")

    runner = runner or make_runner_from_row(
        agent_row,
        settings=settings,
        workdir=workdir or project.get("repo_path"),
    )
    mcp = mcp_client or MCPHTTPClient(
        server_url,
        timeout_seconds=settings.server.request_timeout_seconds,
    )
    runtime_registration_id: int | None = None
    runtime = mcp.call_tool(
        "register_runtime",
        {
            "project_id": int(project["id"]),
            "runtime_name": _build_default_runtime_name(project_slug, role),
            "runtime_kind": "managed_worker",
            "agent_role": role,
            "agent_id": int(agent_row["id"]),
            "metadata_json": {
                "pid": os.getpid(),
            },
        },
    )
    runtime_registration_id = int(runtime["runtime_registration_id"])
    idle_started_at = time.monotonic()
    processed_counts = {
        "stage_assignments": 0,
        "review_assignments": 0,
    }

    stop_reason = "idle_exit"
    try:
        while True:
            mcp.call_tool(
                "heartbeat_runtime",
                {
                    "runtime_registration_id": runtime_registration_id,
                    "lease_seconds": _RUNTIME_LEASE_SECONDS,
                },
            )
            assignment = mcp.call_tool(
                "claim_next_assignment",
                {
                    "project_id": int(project["id"]),
                    "agent_role": role,
                    "runtime_registration_id": runtime_registration_id,
                    "lease_seconds": _ASSIGNMENT_LEASE_SECONDS,
                },
            )
            kind = assignment.get("assignment_kind")
            if kind == "none":
                if (
                    idle_exit_seconds is not None
                    and idle_exit_seconds > 0
                    and time.monotonic() - idle_started_at >= idle_exit_seconds
                ):
                    break
                time.sleep(max(0.1, poll_interval_seconds))
                continue

            idle_started_at = time.monotonic()
            if kind == "stage":
                _handle_stage_assignment(
                    assignment=assignment,
                    runner=runner,
                    mcp=mcp,
                    runtime_registration_id=runtime_registration_id,
                    agent_row=agent_row,
                    role=role,
                )
                processed_counts["stage_assignments"] += 1
                continue

            if kind == "review":
                _handle_review_assignment(
                    assignment=assignment,
                    runner=runner,
                    mcp=mcp,
                    runtime_registration_id=runtime_registration_id,
                    agent_row=agent_row,
                    role=role,
                )
                processed_counts["review_assignments"] += 1
                continue

            raise RuntimeError(f"Unsupported assignment kind: {kind}")
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        raise
    except Exception:
        stop_reason = "worker_error"
        raise
    finally:
        try:
            if runtime_registration_id is not None:
                _mark_runtime_offline(
                    mcp,
                    runtime_registration_id=runtime_registration_id,
                    reason=stop_reason,
                )
        except Exception:
            pass

    return processed_counts
