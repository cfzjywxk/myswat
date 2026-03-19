"""Opt-in live Codex worker smoke tests for the MCP worker loop."""

from __future__ import annotations

import json
import os
import shutil
from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from myswat.agents.codex_runner import CodexRunner
from myswat.cli.worker_cmd import run_worker
from myswat.server.contracts import AssignmentEnvelope, StageRunCompletion
from myswat.server.mcp_stdio import MySwatMCPDispatcher


def _require_live_codex() -> tuple[str, str]:
    if os.environ.get("MYSWAT_RUN_LIVE_CODEX_TESTS") != "1":
        pytest.skip("Set MYSWAT_RUN_LIVE_CODEX_TESTS=1 to run live Codex smoke tests.")
    codex_path = shutil.which("codex")
    if not codex_path:
        pytest.skip("codex CLI is not available on PATH.")
    model = os.environ.get("MYSWAT_LIVE_CODEX_MODEL", "gpt-5.4")
    return codex_path, model


class _DirectMCPClient:
    def __init__(self, service: Mock) -> None:
        self._dispatcher = MySwatMCPDispatcher(service)

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._dispatcher.call_tool(name, arguments)
        wire_body = json.dumps({"result": result}, ensure_ascii=False, default=str)
        parsed = json.loads(wire_body)
        structured = parsed["result"].get("structuredContent")
        return {} if structured is None else structured


def _agent_row(role: str) -> dict:
    return {
        "id": 3,
        "role": role,
        "display_name": role,
        "cli_backend": "codex",
        "model_name": "gpt-5.4",
        "cli_path": "codex",
    }


def _assignment_stream(*head: AssignmentEnvelope):
    idle = AssignmentEnvelope(
        assignment_kind="none",
        runtime_registration_id=int(head[-1].runtime_registration_id if head else 1),
        project_id=int(head[-1].project_id if head else 1),
    )
    return chain(head, repeat(idle))


def test_live_codex_worker_completes_stage_roundtrip(tmp_path):
    codex_path, model = _require_live_codex()
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 71}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=71,
            project_id=1,
            work_item_id=11,
            stage_run_id=22,
            stage_name="design",
            agent_id=3,
            agent_role="developer",
            iteration=1,
            prompt=(
                "Reply with a short design note that includes the exact token "
                "`FIB_LIVE_STAGE_OK` and mentions Rust."
            ),
            system_context="kind=stage;stage=design",
            artifact_type="design_doc",
            artifact_title="Technical design",
        )
    )
    service.complete_stage_task.return_value = StageRunCompletion(
        stage_run_id=22,
        work_item_id=11,
        stage_name="design",
        status="completed",
        summary="done",
        artifact_id=99,
        artifact_content="",
    )
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 71, "status": "offline"}

    runner = CodexRunner(
        cli_path=codex_path,
        model=model,
        workdir=str(tmp_path),
        extra_flags=["--effort", "low"],
        timeout=180,
    )
    result = run_worker(
        project_slug="fib-demo",
        role="developer",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": str(tmp_path)},
        agent_row=_agent_row("developer") | {"cli_path": codex_path, "model_name": model},
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 1, "review_assignments": 0}
    request = service.complete_stage_task.call_args.args[0]
    assert "FIB_LIVE_STAGE_OK" in request.content
    assert "rust" in request.content.lower()
