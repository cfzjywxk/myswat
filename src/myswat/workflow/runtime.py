"""Workflow participants for the server-first orchestration loop."""

from __future__ import annotations

from typing import Any


class WorkflowRuntime:
    """Lightweight role/profile wrapper used by the workflow kernel.

    In the MCP-oriented architecture the kernel no longer invokes the runtime
    directly. It queues work for an external runtime registered against this
    role, and waits for the result through the server/service layer.
    """

    def __init__(
        self,
        *,
        agent_row: dict[str, Any],
    ) -> None:
        self._agent_row = agent_row

    @property
    def agent_id(self) -> int:
        return int(self._agent_row["id"])

    @property
    def agent_role(self) -> str:
        return str(self._agent_row["role"])

    @property
    def display_name(self) -> str:
        return str(self._agent_row.get("display_name") or self.agent_role)

    @property
    def cli_backend(self) -> str:
        return str(self._agent_row.get("cli_backend") or "")

    @property
    def model_name(self) -> str:
        return str(self._agent_row.get("model_name") or "")

    @property
    def agent_row(self) -> dict[str, Any]:
        return self._agent_row
