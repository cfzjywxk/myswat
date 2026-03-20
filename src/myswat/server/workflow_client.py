"""Local MCP-backed workflow coordination client for the kernel."""

from __future__ import annotations

from typing import Any, Protocol

from myswat.server.contracts import (
    ReviewCycleCancellationRequest,
    ReviewRequest,
    ReviewRequestResult,
    ReviewVerdictEnvelope,
    ReviewWaitRequest,
    StageRunCompletion,
    StageRunResult,
    StageRunStart,
    StageRunWaitRequest,
    StatusReport,
)
from myswat.server.mcp_stdio import MySwatMCPDispatcher


class MCPToolCaller(Protocol):
    """Minimal tool-calling surface shared by local and remote MCP clients."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return its structured content."""


class WorkflowCoordinator(Protocol):
    """Typed coordination surface consumed by the deterministic kernel."""

    def report_status(self, request: StatusReport) -> dict[str, Any]:
        """Persist intermediate work-item state."""

    def start_stage_run(self, request: StageRunStart) -> StageRunResult:
        """Queue a stage run."""

    def wait_for_stage_run_completion(
        self,
        request: StageRunWaitRequest,
    ) -> StageRunCompletion:
        """Wait for a queued stage run to reach a terminal state."""

    def request_review(self, request: ReviewRequest) -> ReviewRequestResult:
        """Queue a review cycle."""

    def wait_for_review_verdicts(
        self,
        request: ReviewWaitRequest,
    ) -> list[ReviewVerdictEnvelope]:
        """Wait for review verdicts to reach terminal states."""

    def cancel_review_cycles(self, request: ReviewCycleCancellationRequest) -> dict[str, Any]:
        """Cancel outstanding review cycles."""


class LocalMCPToolClient:
    """In-process adapter over the dispatcher for kernel-side MCP calls."""

    def __init__(self, dispatcher: MySwatMCPDispatcher) -> None:
        self._dispatcher = dispatcher

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = self._dispatcher.call_tool(name, arguments)
        if not isinstance(result, dict):
            raise RuntimeError(f"Invalid dispatcher result for tool {name}: {result!r}")
        return result.get("structuredContent")


class MCPWorkflowCoordinator:
    """Typed wrapper over MCP tools used by the deterministic workflow kernel."""

    def __init__(self, client: MCPToolCaller) -> None:
        self._client = client

    def report_status(self, request: StatusReport) -> dict[str, Any]:
        result = self._client.call_tool(
            "report_status",
            request.model_dump(exclude_none=True),
        )
        return result if isinstance(result, dict) else {}

    def start_stage_run(self, request: StageRunStart) -> StageRunResult:
        result = self._client.call_tool(
            "start_stage_run",
            request.model_dump(exclude_none=True),
        )
        return StageRunResult.model_validate(result or {})

    def wait_for_stage_run_completion(
        self,
        request: StageRunWaitRequest,
    ) -> StageRunCompletion:
        result = self._client.call_tool(
            "wait_for_stage_run_completion",
            request.model_dump(exclude_none=True),
        )
        return StageRunCompletion.model_validate(result or {})

    def request_review(self, request: ReviewRequest) -> ReviewRequestResult:
        result = self._client.call_tool(
            "request_review",
            request.model_dump(exclude_none=True),
        )
        return ReviewRequestResult.model_validate(result or {})

    def wait_for_review_verdicts(
        self,
        request: ReviewWaitRequest,
    ) -> list[ReviewVerdictEnvelope]:
        result = self._client.call_tool(
            "wait_for_review_verdicts",
            request.model_dump(exclude_none=True),
        )
        if result is None:
            return []
        if not isinstance(result, list):
            raise RuntimeError(f"Invalid review verdict payload: {result!r}")
        return [ReviewVerdictEnvelope.model_validate(item) for item in result]

    def cancel_review_cycles(self, request: ReviewCycleCancellationRequest) -> dict[str, Any]:
        result = self._client.call_tool(
            "cancel_review_cycles",
            request.model_dump(exclude_none=True),
        )
        return result if isinstance(result, dict) else {}
