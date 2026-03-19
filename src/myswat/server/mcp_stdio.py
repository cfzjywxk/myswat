"""Minimal stdio MCP server for MySwat workflow tools."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from myswat.server.contracts import (
    ArtifactSubmission,
    ClaimNextAssignmentRequest,
    CompleteStageTaskRequest,
    DecisionPersistenceRequest,
    FailStageTaskRequest,
    KnowledgeSearchRequest,
    ProjectLookupRequest,
    RecentArtifactsRequest,
    ReviewCycleLeaseRenewalRequest,
    ReviewRequest,
    ReviewVerdictSubmission,
    ReviewWaitRequest,
    RuntimeHeartbeatRequest,
    RuntimeRegistrationRequest,
    RuntimeStatusUpdateRequest,
    StageRunStart,
    StageRunLeaseRenewalRequest,
    StageRunWaitRequest,
    StatusReport,
    WorkItemSnapshotRequest,
)
from myswat.server.service import MySwatToolService


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    schema_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]


def _normalize_structured(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, list):
        return [_normalize_structured(item) for item in result]
    if isinstance(result, tuple):
        return [_normalize_structured(item) for item in result]
    if isinstance(result, dict):
        return {key: _normalize_structured(value) for key, value in result.items()}
    return result


def _result_payload(result: Any) -> dict[str, Any]:
    structured = _normalize_structured(result)
    if structured is None:
        structured = {}
    text = json.dumps(structured, ensure_ascii=False, indent=2, default=str)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
    }


class MySwatMCPDispatcher:
    """Dispatch MCP tool calls onto the store-backed MySwat service."""

    def __init__(self, service: MySwatToolService) -> None:
        self._service = service
        self._tools: dict[str, MCPTool] = {
            "resolve_project": MCPTool(
                name="resolve_project",
                description="Resolve a MySwat project slug to project metadata.",
                schema_model=ProjectLookupRequest,
                handler=self._service.resolve_project,
            ),
            "register_runtime": MCPTool(
                name="register_runtime",
                description="Register an external runtime for a workflow role.",
                schema_model=RuntimeRegistrationRequest,
                handler=self._service.register_runtime,
            ),
            "heartbeat_runtime": MCPTool(
                name="heartbeat_runtime",
                description="Refresh a runtime registration heartbeat and lease.",
                schema_model=RuntimeHeartbeatRequest,
                handler=self._service.heartbeat_runtime,
            ),
            "claim_next_assignment": MCPTool(
                name="claim_next_assignment",
                description="Claim the next pending stage or review assignment for a role.",
                schema_model=ClaimNextAssignmentRequest,
                handler=self._service.claim_next_assignment,
            ),
            "update_runtime_status": MCPTool(
                name="update_runtime_status",
                description="Mark a runtime registration online, offline, or errored.",
                schema_model=RuntimeStatusUpdateRequest,
                handler=self._service.update_runtime_status,
            ),
            "get_work_item_snapshot": MCPTool(
                name="get_work_item_snapshot",
                description="Fetch the current work-item context bundle for execution.",
                schema_model=WorkItemSnapshotRequest,
                handler=self._service.get_work_item_snapshot,
            ),
            "search_knowledge": MCPTool(
                name="search_knowledge",
                description="Search project knowledge relevant to the current task.",
                schema_model=KnowledgeSearchRequest,
                handler=self._service.search_knowledge,
            ),
            "get_recent_artifacts": MCPTool(
                name="get_recent_artifacts",
                description="Fetch recent workflow artifacts for a project.",
                schema_model=RecentArtifactsRequest,
                handler=self._service.get_recent_artifacts,
            ),
            "report_status": MCPTool(
                name="report_status",
                description="Report intermediate task status back to MySwat.",
                schema_model=StatusReport,
                handler=self._service.report_status,
            ),
            "start_stage_run": MCPTool(
                name="start_stage_run",
                description="Internal orchestrator tool to queue a stage run.",
                schema_model=StageRunStart,
                handler=self._service.start_stage_run,
            ),
            "wait_for_stage_run_completion": MCPTool(
                name="wait_for_stage_run_completion",
                description="Internal orchestrator tool to wait for a queued stage run.",
                schema_model=StageRunWaitRequest,
                handler=self._service.wait_for_stage_run_completion,
            ),
            "renew_stage_run_lease": MCPTool(
                name="renew_stage_run_lease",
                description="Refresh the lease for a claimed stage run owned by a runtime.",
                schema_model=StageRunLeaseRenewalRequest,
                handler=self._service.renew_stage_run_lease,
            ),
            "submit_artifact": MCPTool(
                name="submit_artifact",
                description="Persist an artifact for a workflow stage.",
                schema_model=ArtifactSubmission,
                handler=self._service.submit_artifact,
            ),
            "complete_stage_task": MCPTool(
                name="complete_stage_task",
                description="Complete a claimed stage task and persist its artifact.",
                schema_model=CompleteStageTaskRequest,
                handler=self._service.complete_stage_task,
            ),
            "fail_stage_task": MCPTool(
                name="fail_stage_task",
                description="Mark a claimed stage task as blocked.",
                schema_model=FailStageTaskRequest,
                handler=self._service.fail_stage_task,
            ),
            "publish_review_verdict": MCPTool(
                name="publish_review_verdict",
                description="Publish a structured review verdict for a claimed review task.",
                schema_model=ReviewVerdictSubmission,
                handler=self._service.publish_review_verdict,
            ),
            "renew_review_cycle_lease": MCPTool(
                name="renew_review_cycle_lease",
                description="Refresh the lease for a claimed review cycle owned by a runtime.",
                schema_model=ReviewCycleLeaseRenewalRequest,
                handler=self._service.renew_review_cycle_lease,
            ),
            "request_review": MCPTool(
                name="request_review",
                description="Internal orchestrator tool to queue a review cycle.",
                schema_model=ReviewRequest,
                handler=self._service.request_review,
            ),
            "wait_for_review_verdicts": MCPTool(
                name="wait_for_review_verdicts",
                description="Internal orchestrator tool to wait for review verdicts.",
                schema_model=ReviewWaitRequest,
                handler=self._service.wait_for_review_verdicts,
            ),
            "persist_decision": MCPTool(
                name="persist_decision",
                description="Store a durable workflow or project decision into knowledge.",
                schema_model=DecisionPersistenceRequest,
                handler=self._service.persist_decision,
            ),
        }

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.schema_model.model_json_schema(),
                }
                for tool in self._tools.values()
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        model = tool.schema_model.model_validate(arguments or {})
        result = tool.handler(model)
        return _result_payload(result)


def _read_message(stdin) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8").strip()
        if not decoded:
            break
        key, _, value = decoded.partition(":")
        headers[key.lower()] = value.strip()
    content_length = int(headers.get("content-length", "0") or 0)
    if content_length <= 0:
        return None
    payload = stdin.read(content_length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _write_message(stdout, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    stdout.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stdout.write(body)
    stdout.flush()


def serve_stdio(service: MySwatToolService) -> int:
    dispatcher = MySwatMCPDispatcher(service)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        message = _read_message(stdin)
        if message is None:
            return 0

        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "myswat", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = dispatcher.list_tools()
            elif method == "tools/call":
                result = dispatcher.call_tool(
                    str(params.get("name") or ""),
                    params.get("arguments") or {},
                )
            elif method == "ping":
                result = {}
            elif method == "initialized":
                continue
            else:
                raise ValueError(f"Unsupported MCP method: {method}")
            if request_id is not None:
                _write_message(stdout, {"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception as exc:
            if request_id is None:
                continue
            _write_message(
                stdout,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": str(exc),
                    },
                },
            )
