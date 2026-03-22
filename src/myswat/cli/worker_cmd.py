"""Internal managed worker for MySwat daemon-supervised agent roles."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import typer

from myswat.agents.factory import make_runner_from_row
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.large_payloads import (
    AGENT_CONTEXT_USAGE_PROMPT,
    AGENT_FILE_PROMPT,
    maybe_externalize_prompt,
    maybe_externalize_system_context,
    resolve_externalized_text,
    resolve_externalized_value,
)
from myswat.memory.store import MemoryStore
from myswat.models.work_item import ReviewVerdict
from myswat.server.mcp_http_client import MCPHTTPClient
from myswat.workflow.review_parsing import (
    UNSTRUCTURED_REVIEW_ISSUE_LIMIT,
    UNSTRUCTURED_REVIEW_SUMMARY_LIMIT,
    looks_like_structured_review_payload,
    parse_plain_text_lgtm_verdict,
    parse_structured_review_verdict,
    parse_unstructured_changes_requested_verdict,
)

_RUNTIME_LEASE_SECONDS = 300
_ASSIGNMENT_LEASE_SECONDS = 300
_KEEPALIVE_INTERVAL_SECONDS = 60.0
_REVIEW_MAX_ATTEMPTS = 2
_REVIEW_DIAGNOSTIC_TEXT_LIMIT = 2_000
_REVIEW_EXCEPTION_TEXT_LIMIT = 1_000

_REVIEW_OUTPUT_CONTRACT_PROMPT = """## Review Output Contract
- Your final answer MUST be exactly one JSON object matching the requested review schema.
- Never return a top-level markdown review body.
- Never return only a `/tmp/*.md` path or a sentence pointing to a markdown file as the entire answer.
- If review details are long, keep the JSON object and replace only oversized `summary` or `issues` entries with `See /tmp/...md`.
"""

LOGGER = logging.getLogger(__name__)


class ReviewAttemptOutcome(Enum):
    VALID_VERDICT = "valid_verdict"
    RETRYABLE_FAILURE = "retryable_failure"
    NONRETRYABLE_FAILURE = "nonretryable_failure"


@dataclass
class ReviewAttemptResult:
    outcome: ReviewAttemptOutcome
    verdict: ReviewVerdict | None = None
    failure_kind: str | None = None
    summary: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().setLevel(logging.INFO)


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
    system_sections = [AGENT_FILE_PROMPT, AGENT_CONTEXT_USAGE_PROMPT]
    if str(assignment.get("assignment_kind") or "") == "review":
        system_sections.append(_REVIEW_OUTPUT_CONTRACT_PROMPT)
    if system_context:
        system_sections.append(system_context)
    file_aware_system_context = "\n\n---\n\n".join(system_sections)

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


def _clip_diagnostic_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text
    return text[-limit:]


def _looks_nonretryable_review_failure(*parts: str) -> bool:
    text = " ".join(part for part in parts if part).lower()
    return any(
        needle in text
        for needle in (
            "command not found",
            "no such file or directory",
            "permission denied",
            "unknown model",
            "model not found",
            "invalid model",
            "api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "rate limit",
            "quota",
        )
    )


def _classify_review_attempt(
    *,
    response: Any | None = None,
    resolved_content: str = "",
    error: Exception | None = None,
) -> ReviewAttemptResult:
    diagnostics = {
        "runner_exit_code": getattr(response, "exit_code", None),
        "cancelled": bool(getattr(response, "cancelled", False)),
        "stderr_tail": _clip_diagnostic_text(
            getattr(response, "raw_stderr", ""),
            limit=_REVIEW_DIAGNOSTIC_TEXT_LIMIT,
        ),
        "response_excerpt": _clip_diagnostic_text(
            resolved_content or getattr(response, "content", ""),
            limit=_REVIEW_DIAGNOSTIC_TEXT_LIMIT,
        ),
        "exception_type": type(error).__name__ if error is not None else "",
        "exception_message": _clip_diagnostic_text(
            str(error) if error is not None else "",
            limit=_REVIEW_EXCEPTION_TEXT_LIMIT,
        ),
    }
    if error is not None:
        summary = diagnostics["exception_message"] or "Review execution raised an exception."
        failure_kind = "environment_misconfiguration" if _looks_nonretryable_review_failure(summary) else "runner_exception"
        return ReviewAttemptResult(
            outcome=(
                ReviewAttemptOutcome.RETRYABLE_FAILURE
                if failure_kind != "environment_misconfiguration"
                else ReviewAttemptOutcome.NONRETRYABLE_FAILURE
            ),
            failure_kind=failure_kind,
            summary=summary,
            diagnostics=diagnostics,
        )

    if bool(getattr(response, "cancelled", False)):
        return ReviewAttemptResult(
            outcome=ReviewAttemptOutcome.NONRETRYABLE_FAILURE,
            failure_kind="cancelled",
            summary="Review execution was cancelled.",
            diagnostics=diagnostics,
        )

    diagnostics["verdict_source"] = ""
    verdict = parse_structured_review_verdict(resolved_content)
    if verdict is not None:
        diagnostics["verdict_source"] = "structured_json"
    if verdict is None:
        verdict = parse_plain_text_lgtm_verdict(
            resolved_content,
            summary_limit=_REVIEW_DIAGNOSTIC_TEXT_LIMIT,
        )
        if verdict is not None:
            diagnostics["verdict_source"] = "plain_text_lgtm"
    if verdict is None and bool(getattr(response, "success", False)) and str(resolved_content).strip():
        if not looks_like_structured_review_payload(resolved_content):
            verdict = parse_unstructured_changes_requested_verdict(
                resolved_content,
                summary_limit=UNSTRUCTURED_REVIEW_SUMMARY_LIMIT,
                issue_limit=UNSTRUCTURED_REVIEW_ISSUE_LIMIT,
            )
            if verdict is not None:
                diagnostics["verdict_source"] = "unstructured_changes_requested"
    if bool(getattr(response, "success", False)) and verdict is not None:
        return ReviewAttemptResult(
            outcome=ReviewAttemptOutcome.VALID_VERDICT,
            verdict=verdict,
            diagnostics=diagnostics,
        )

    response_excerpt = str(diagnostics["response_excerpt"] or "").strip()
    stderr_tail = str(diagnostics["stderr_tail"] or "").strip()
    failure_text = response_excerpt or stderr_tail
    if bool(getattr(response, "success", False)):
        summary = "Reviewer returned empty output." if not failure_text else "Reviewer returned malformed verdict output."
        return ReviewAttemptResult(
            outcome=ReviewAttemptOutcome.RETRYABLE_FAILURE,
            failure_kind="malformed_output",
            summary=summary,
            diagnostics=diagnostics,
        )

    summary = failure_text or "Review execution failed."
    failure_kind = "environment_misconfiguration" if _looks_nonretryable_review_failure(summary) else "execution_failed"
    return ReviewAttemptResult(
        outcome=(
            ReviewAttemptOutcome.RETRYABLE_FAILURE
            if failure_kind != "environment_misconfiguration"
            else ReviewAttemptOutcome.NONRETRYABLE_FAILURE
        ),
        failure_kind=failure_kind,
        summary=summary,
        diagnostics=diagnostics,
    )


def _build_review_diagnostics(
    *,
    attempt_result: ReviewAttemptResult,
    assignment: dict[str, Any],
    agent_row: dict[str, Any],
    role: str,
    runtime_registration_id: int,
    runner: Any,
    attempt: int,
    attempts: int,
) -> dict[str, Any]:
    diagnostics = dict(attempt_result.diagnostics)
    diagnostics.update(
        {
            "failure_kind": attempt_result.failure_kind,
            "attempt": attempt,
            "attempts": attempts,
            "agent_role": role,
            "agent_id": int(agent_row.get("id") or 0) or None,
            "stage_name": str(assignment.get("stage_name") or ""),
            "runtime_registration_id": runtime_registration_id,
            "worker_pid": os.getpid(),
            "backend": str(agent_row.get("cli_backend") or ""),
            "model": str(agent_row.get("model_name") or ""),
            "workdir": str(getattr(runner, "workdir", "") or ""),
            "diagnosed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    return diagnostics


def _append_review_retry_event(
    *,
    assignment: dict[str, Any],
    mcp: MCPHTTPClient,
    agent_row: dict[str, Any],
    role: str,
    summary: str,
    diagnostics: dict[str, Any],
) -> None:
    mcp.call_tool(
        "append_coordination_event",
        {
            "work_item_id": int(assignment["work_item_id"]),
            "stage_name": assignment.get("stage_name") or "",
            "event_type": "review_retry",
            "summary": summary,
            "from_agent_id": int(agent_row["id"]),
            "from_role": role,
            "payload_json": diagnostics,
        },
    )


def _run_review_with_single_retry(
    *,
    assignment: dict[str, Any],
    runner: Any,
    mcp: MCPHTTPClient,
    runtime_registration_id: int,
    agent_row: dict[str, Any],
    role: str,
    prompt: str,
    system_context: str | None,
) -> None:
    work_item_id = int(assignment["work_item_id"])
    cycle_id = int(assignment["review_cycle_id"])
    stage_name = str(assignment.get("stage_name") or "")

    for attempt in range(1, _REVIEW_MAX_ATTEMPTS + 1):
        response: Any | None = None
        resolved_content = ""
        error: Exception | None = None
        runner.reset_session()
        try:
            response = _invoke_runner_with_keepalive(
                assignment=assignment,
                runner=runner,
                prompt=prompt,
                system_context=system_context,
                mcp=mcp,
                runtime_registration_id=runtime_registration_id,
            )
            resolved_content = _resolve_runner_response_content(str(getattr(response, "content", "") or ""))
        except Exception as exc:
            error = exc

        attempt_result = _classify_review_attempt(
            response=response,
            resolved_content=resolved_content,
            error=error,
        )
        diagnostics = _build_review_diagnostics(
            attempt_result=attempt_result,
            assignment=assignment,
            agent_row=agent_row,
            role=role,
            runtime_registration_id=runtime_registration_id,
            runner=runner,
            attempt=attempt,
            attempts=_REVIEW_MAX_ATTEMPTS,
        )

        if attempt_result.outcome == ReviewAttemptOutcome.VALID_VERDICT and attempt_result.verdict is not None:
            mcp.call_tool(
                "publish_review_verdict",
                {
                    "cycle_id": cycle_id,
                    "work_item_id": work_item_id,
                    "reviewer_agent_id": int(agent_row["id"]),
                    "reviewer_role": role,
                    "verdict": attempt_result.verdict.verdict,
                    "issues": attempt_result.verdict.issues,
                    "summary": attempt_result.verdict.summary,
                    "stage": stage_name,
                    "runtime_registration_id": runtime_registration_id,
                },
            )
            return

        if attempt_result.outcome == ReviewAttemptOutcome.RETRYABLE_FAILURE and attempt < _REVIEW_MAX_ATTEMPTS:
            _append_review_retry_event(
                assignment=assignment,
                mcp=mcp,
                agent_row=agent_row,
                role=role,
                summary=(
                    f"{attempt_result.summary} Retrying review execution "
                    f"({attempt}/{_REVIEW_MAX_ATTEMPTS})."
                ),
                diagnostics=diagnostics,
            )
            continue

        mcp.call_tool(
            "fail_review_cycle",
            {
                "cycle_id": cycle_id,
                "work_item_id": work_item_id,
                "reviewer_agent_id": int(agent_row["id"]),
                "reviewer_role": role,
                "stage": stage_name,
                "runtime_registration_id": runtime_registration_id,
                "summary": attempt_result.summary,
                "failure_kind": attempt_result.failure_kind or "review_failed",
                "attempts": attempt,
                "diagnostics": diagnostics,
            },
        )
        return


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
    _run_review_with_single_retry(
        assignment=assignment,
        runner=runner,
        mcp=mcp,
        runtime_registration_id=runtime_registration_id,
        agent_row=agent_row,
        role=role,
        prompt=prompt,
        system_context=system_context,
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
    _configure_logging()
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
        timeout_seconds=None,
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
