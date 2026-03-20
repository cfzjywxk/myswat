"""Tests for the managed worker MCP loop."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from myswat.cli.worker_cmd import run_worker
from myswat.large_payloads import extract_markdown_path, read_markdown_file
from myswat.server.contracts import AssignmentEnvelope, StageRunCompletion
from myswat.server.mcp_stdio import MySwatMCPDispatcher


class _FakeRunner:
    def __init__(
        self,
        responses: list[tuple[bool, str] | tuple[bool, str, dict] | Exception | SimpleNamespace],
    ) -> None:
        self._responses = list(responses)
        self.reset_calls = 0
        self.invocations: list[tuple[str, str | None]] = []

    def reset_session(self) -> None:
        self.reset_calls += 1

    def invoke(self, prompt: str, system_context: str | None = None):
        self.invocations.append((prompt, system_context))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            if len(response) == 2:
                success, content = response
                extras = {}
            else:
                success, content, extras = response
            return SimpleNamespace(success=success, content=content, **dict(extras))
        return response


class _BlockingRunner:
    def __init__(
        self,
        response: tuple[bool, str],
        *,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        self._response = response
        self._started = started
        self._release = release
        self.reset_calls = 0
        self.invocations: list[tuple[str, str | None]] = []

    def reset_session(self) -> None:
        self.reset_calls += 1

    def invoke(self, prompt: str, system_context: str | None = None):
        self.invocations.append((prompt, system_context))
        self._started.set()
        if not self._release.wait(timeout=2):
            raise RuntimeError("blocking runner was not released")
        success, content = self._response
        return SimpleNamespace(success=success, content=content)


class _CancelableBlockingRunner(_BlockingRunner):
    def __init__(
        self,
        response: tuple[bool, str],
        *,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(response, started=started, release=release)
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._release.set()


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
        "cli_backend": "fake",
        "model_name": "fake-model",
        "cli_path": "fake",
    }


def _assignment_stream(*head: AssignmentEnvelope):
    idle = AssignmentEnvelope(
        assignment_kind="none",
        runtime_registration_id=int(head[-1].runtime_registration_id if head else 1),
        project_id=int(head[-1].project_id if head else 1),
    )
    return chain(head, repeat(idle))


def test_run_worker_completes_stage_assignment_roundtrip():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 7}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=7,
            project_id=1,
            work_item_id=11,
            stage_run_id=22,
            stage_name="design",
            agent_id=3,
            agent_role="architect",
            iteration=1,
            prompt="Draft the design",
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
        artifact_content="# Design\n",
    )
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 7, "status": "offline"}

    runner = _FakeRunner([(True, "# Design\nUse an iterative Rust implementation.\n")])
    result = run_worker(
        project_slug="fib-demo",
        role="architect",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("architect"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 1, "review_assignments": 0}
    request = service.complete_stage_task.call_args.args[0]
    assert request.stage_name == "design"
    assert request.artifact_type == "design_doc"
    assert "iterative Rust implementation" in request.content
    status_request = service.update_runtime_status.call_args.args[0]
    assert status_request.status == "offline"


def test_run_worker_reports_failed_stage_roundtrip():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 8}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=8,
            project_id=1,
            work_item_id=12,
            stage_run_id=24,
            stage_name="phase_1",
            agent_id=3,
            agent_role="developer",
            iteration=2,
            prompt="Implement the generator",
            system_context="kind=stage;stage=phase_1",
            artifact_type="phase_result",
            artifact_title="Implementation result",
        )
    )
    service.fail_stage_task.return_value = StageRunCompletion(
        stage_run_id=24,
        work_item_id=12,
        stage_name="phase_1",
        status="blocked",
        summary="cargo test failed",
    )
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 8, "status": "offline"}

    runner = _FakeRunner([(False, "cargo test failed due to borrow checker errors")])
    result = run_worker(
        project_slug="fib-demo",
        role="developer",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("developer"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 1, "review_assignments": 0}
    request = service.fail_stage_task.call_args.args[0]
    assert request.stage_run_id == 24
    assert request.stage_name == "phase_1"
    assert "borrow checker errors" in request.summary


def test_run_worker_renews_stage_lease_while_assignment_is_running():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 801}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=801,
            project_id=1,
            work_item_id=44,
            stage_run_id=54,
            stage_name="phase_1",
            agent_id=3,
            agent_role="developer",
            iteration=1,
            prompt="Implement phase 1",
            system_context="kind=stage;stage=phase_1",
            artifact_type="phase_result",
            artifact_title="Phase 1 result",
        )
    )
    service.complete_stage_task.return_value = StageRunCompletion(
        stage_run_id=54,
        work_item_id=44,
        stage_name="phase_1",
        status="completed",
        summary="done",
        artifact_id=144,
        artifact_content="done",
    )
    service.heartbeat_runtime.return_value = None
    service.renew_stage_run_lease.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 801, "status": "offline"}

    started = threading.Event()
    release = threading.Event()
    runner = _BlockingRunner((True, "done"), started=started, release=release)
    result_holder: dict[str, object] = {}

    def _run() -> None:
        try:
            result_holder["result"] = run_worker(
                project_slug="fib-demo",
                role="developer",
                server_url="http://unused",
                project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
                agent_row=_agent_row("developer"),
                runner=runner,
                mcp_client=_DirectMCPClient(service),
                idle_exit_seconds=0.05,
            )
        except Exception as exc:  # pragma: no cover - failure path surfaced below
            result_holder["error"] = exc

    with patch("myswat.cli.worker_cmd._KEEPALIVE_INTERVAL_SECONDS", 0.01):
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        assert started.wait(timeout=1) is True
        time.sleep(0.05)
        release.set()
        thread.join(timeout=2)

    assert thread.is_alive() is False
    assert "error" not in result_holder
    assert result_holder["result"] == {"stage_assignments": 1, "review_assignments": 0}
    renew_request = service.renew_stage_run_lease.call_args.args[0]
    assert renew_request.stage_run_id == 54
    assert renew_request.runtime_registration_id == 801


def test_run_worker_cancels_runner_when_stage_keepalive_fails():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 811}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=811,
            project_id=1,
            work_item_id=46,
            stage_run_id=66,
            stage_name="phase_1",
            agent_id=3,
            agent_role="developer",
            iteration=1,
            prompt="Implement phase 1",
            system_context="kind=stage;stage=phase_1",
            artifact_type="phase_result",
            artifact_title="Phase 1 result",
        )
    )
    service.heartbeat_runtime.return_value = None
    service.renew_stage_run_lease.side_effect = RuntimeError("lease lost")
    service.update_runtime_status.return_value = {"runtime_registration_id": 811, "status": "offline"}

    started = threading.Event()
    release = threading.Event()
    runner = _CancelableBlockingRunner((True, "done"), started=started, release=release)
    result_holder: dict[str, object] = {}

    def _run() -> None:
        try:
            result_holder["result"] = run_worker(
                project_slug="fib-demo",
                role="developer",
                server_url="http://unused",
                project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
                agent_row=_agent_row("developer"),
                runner=runner,
                mcp_client=_DirectMCPClient(service),
                idle_exit_seconds=0.05,
            )
        except Exception as exc:
            result_holder["error"] = exc

    with patch("myswat.cli.worker_cmd._KEEPALIVE_INTERVAL_SECONDS", 0.01):
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        assert started.wait(timeout=1) is True
        thread.join(timeout=2)

    assert thread.is_alive() is False
    assert "result" not in result_holder
    assert isinstance(result_holder.get("error"), RuntimeError)
    assert "lease lost" in str(result_holder["error"])
    assert runner.cancel_calls == 1


def test_run_worker_externalizes_large_stage_prompt_and_context_to_markdown_files():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 81}
    )
    large_prompt = "Implement the design.\n" + ("prompt-body\n" * 140) + "PROMPT-TAIL-MARKER"
    large_context = "System context.\n" + ("context-body\n" * 140) + "CONTEXT-TAIL-MARKER"
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=81,
            project_id=1,
            work_item_id=31,
            stage_run_id=41,
            stage_name="plan",
            agent_id=3,
            agent_role="developer",
            iteration=1,
            prompt=large_prompt,
            system_context=large_context,
            artifact_type="implementation_plan",
            artifact_title="Implementation plan",
        )
    )
    service.complete_stage_task.return_value = StageRunCompletion(
        stage_run_id=41,
        work_item_id=31,
        stage_name="plan",
        status="completed",
        summary="done",
        artifact_id=88,
        artifact_content="Phase 1",
    )
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 81, "status": "offline"}

    runner = _FakeRunner([(True, "Phase 1")])
    result = run_worker(
        project_slug="fib-demo",
        role="developer",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("developer"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 1, "review_assignments": 0}
    sent_prompt, sent_context = runner.invocations[0]
    prompt_path = extract_markdown_path(sent_prompt)
    context_path = extract_markdown_path(sent_context or "")
    assert prompt_path is not None
    assert context_path is not None
    assert "Read that markdown file completely" in sent_prompt
    assert "Read it before responding" in (sent_context or "")
    assert "PROMPT-TAIL-MARKER" in read_markdown_file(prompt_path)
    assert "CONTEXT-TAIL-MARKER" in read_markdown_file(context_path)
    Path(prompt_path).unlink(missing_ok=True)
    Path(context_path).unlink(missing_ok=True)


def test_run_worker_resolves_externalized_stage_response_before_persisting_artifact():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 82}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="stage",
            runtime_registration_id=82,
            project_id=1,
            work_item_id=32,
            stage_run_id=42,
            stage_name="design",
            agent_id=3,
            agent_role="architect",
            iteration=1,
            prompt="Draft the design",
            system_context="kind=stage;stage=design",
            artifact_type="design_doc",
            artifact_title="Technical design",
        )
    )
    service.complete_stage_task.return_value = StageRunCompletion(
        stage_run_id=42,
        work_item_id=32,
        stage_name="design",
        status="completed",
        summary="done",
        artifact_id=89,
        artifact_content="# Design\n",
    )
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 82, "status": "offline"}

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write("# Design\nUse a Rust iterative generator.\n")
        response_path = handle.name

    runner = _FakeRunner([(True, f"The detailed response is in `{response_path}`.")])
    result = run_worker(
        project_slug="fib-demo",
        role="architect",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("architect"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 1, "review_assignments": 0}
    request = service.complete_stage_task.call_args.args[0]
    assert request.content == "# Design\nUse a Rust iterative generator.\n"
    assert request.summary == "# Design\nUse a Rust iterative generator.\n"
    Path(response_path).unlink(missing_ok=True)


def test_run_worker_publishes_review_verdict_roundtrip():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 9}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="review",
            runtime_registration_id=9,
            project_id=1,
            work_item_id=12,
            review_cycle_id=33,
            stage_name="design_review",
            agent_id=4,
            agent_role="qa_main",
            iteration=1,
            prompt="Review the design",
            system_context="kind=review;stage=design_review",
        )
    )
    service.publish_review_verdict.return_value = {"ok": True}
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 9, "status": "offline"}

    runner = _FakeRunner([(True, '{"verdict":"lgtm","issues":[],"summary":"Looks good."}')])
    result = run_worker(
        project_slug="fib-demo",
        role="qa_main",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("qa_main"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 1}
    request = service.publish_review_verdict.call_args.args[0]
    assert request.cycle_id == 33
    assert request.verdict == "lgtm"
    assert request.summary == "Looks good."


def test_run_worker_renews_review_lease_while_assignment_is_running():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 901}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="review",
            runtime_registration_id=901,
            project_id=1,
            work_item_id=45,
            review_cycle_id=65,
            stage_name="phase_1_review",
            agent_id=4,
            agent_role="qa_main",
            iteration=1,
            prompt="Review phase 1",
            system_context="kind=review;stage=phase_1_review",
        )
    )
    service.publish_review_verdict.return_value = {"ok": True}
    service.heartbeat_runtime.return_value = None
    service.renew_review_cycle_lease.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 901, "status": "offline"}

    started = threading.Event()
    release = threading.Event()
    runner = _BlockingRunner(
        (True, '{"verdict":"lgtm","issues":[],"summary":"Looks good."}'),
        started=started,
        release=release,
    )
    result_holder: dict[str, object] = {}

    def _run() -> None:
        try:
            result_holder["result"] = run_worker(
                project_slug="fib-demo",
                role="qa_main",
                server_url="http://unused",
                project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
                agent_row=_agent_row("qa_main"),
                runner=runner,
                mcp_client=_DirectMCPClient(service),
                idle_exit_seconds=0.05,
            )
        except Exception as exc:  # pragma: no cover - failure path surfaced below
            result_holder["error"] = exc

    with patch("myswat.cli.worker_cmd._KEEPALIVE_INTERVAL_SECONDS", 0.01):
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        assert started.wait(timeout=1) is True
        time.sleep(0.05)
        release.set()
        thread.join(timeout=2)

    assert thread.is_alive() is False
    assert "error" not in result_holder
    assert result_holder["result"] == {"stage_assignments": 0, "review_assignments": 1}
    renew_request = service.renew_review_cycle_lease.call_args.args[0]
    assert renew_request.cycle_id == 65
    assert renew_request.runtime_registration_id == 901


def test_run_worker_resolves_externalized_review_verdict_fields():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 83}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="review",
            runtime_registration_id=83,
            project_id=1,
            work_item_id=33,
            review_cycle_id=43,
            stage_name="code_review",
            agent_id=4,
            agent_role="qa_main",
            iteration=1,
            prompt="Review the implementation",
            system_context="kind=review;stage=code_review",
        )
    )
    service.publish_review_verdict.return_value = {"ok": True}
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 83, "status": "offline"}

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as summary_handle:
        summary_handle.write("Need better overflow handling.\n")
        summary_path = summary_handle.name
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as issue_handle:
        issue_handle.write("Handle n > 92 without wrapping.\n")
        issue_path = issue_handle.name

    runner = _FakeRunner(
        [(
            True,
            json.dumps(
                {
                    "verdict": "changes_requested",
                    "issues": [f"See `{issue_path}`"],
                    "summary": f"See `{summary_path}`",
                }
            ),
        )]
    )
    result = run_worker(
        project_slug="fib-demo",
        role="qa_main",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("qa_main"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 1}
    request = service.publish_review_verdict.call_args.args[0]
    assert request.verdict == "changes_requested"
    assert request.summary == "Need better overflow handling.\n"
    assert request.issues == ["Handle n > 92 without wrapping.\n"]
    Path(summary_path).unlink(missing_ok=True)
    Path(issue_path).unlink(missing_ok=True)


def test_run_worker_retries_malformed_review_output_once_then_publishes_verdict():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 10}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="review",
            runtime_registration_id=10,
            project_id=1,
            work_item_id=13,
            review_cycle_id=34,
            stage_name="code_review",
            agent_id=4,
            agent_role="qa_main",
            iteration=2,
            prompt="Review the implementation",
            system_context="kind=review;stage=code_review",
        )
    )
    service.publish_review_verdict.return_value = {"ok": True}
    service.append_coordination_event.return_value = {"ok": True}
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 10, "status": "offline"}

    runner = _FakeRunner(
        [
            (True, "Needs better error handling around overflow."),
            (True, '{"verdict":"changes_requested","issues":["Handle overflow."],"summary":"Need overflow guard."}'),
        ]
    )
    result = run_worker(
        project_slug="fib-demo",
        role="qa_main",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("qa_main"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 1}
    request = service.publish_review_verdict.call_args.args[0]
    assert request.cycle_id == 34
    assert request.verdict == "changes_requested"
    assert request.issues == ["Handle overflow."]
    assert runner.reset_calls == 2
    service.append_coordination_event.assert_called_once()
    service.fail_review_cycle.assert_not_called()


def test_run_worker_marks_review_failed_after_repeated_malformed_output():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 10}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="review",
            runtime_registration_id=10,
            project_id=1,
            work_item_id=13,
            review_cycle_id=34,
            stage_name="code_review",
            agent_id=4,
            agent_role="qa_main",
            iteration=2,
            prompt="Review the implementation",
            system_context="kind=review;stage=code_review",
        )
    )
    service.append_coordination_event.return_value = {"ok": True}
    service.fail_review_cycle.return_value = {"ok": True}
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 10, "status": "offline"}

    runner = _FakeRunner(
        [
            (True, "Needs better error handling around overflow."),
            (True, "Still not returning JSON."),
        ]
    )
    result = run_worker(
        project_slug="fib-demo",
        role="qa_main",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("qa_main"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 1}
    request = service.fail_review_cycle.call_args.args[0]
    assert request.cycle_id == 34
    assert request.failure_kind == "malformed_output"
    assert request.attempts == 2
    assert request.diagnostics["attempt"] == 2
    assert runner.reset_calls == 2
    service.publish_review_verdict.assert_not_called()
    service.append_coordination_event.assert_called_once()


def test_run_worker_marks_review_failed_without_retry_for_nonretryable_error():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 10}
    )
    service.claim_next_assignment.side_effect = _assignment_stream(
        AssignmentEnvelope(
            assignment_kind="review",
            runtime_registration_id=10,
            project_id=1,
            work_item_id=13,
            review_cycle_id=34,
            stage_name="code_review",
            agent_id=4,
            agent_role="qa_main",
            iteration=2,
            prompt="Review the implementation",
            system_context="kind=review;stage=code_review",
        )
    )
    service.fail_review_cycle.return_value = {"ok": True}
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 10, "status": "offline"}

    runner = _FakeRunner([RuntimeError("command not found: codex")])
    result = run_worker(
        project_slug="fib-demo",
        role="qa_main",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("qa_main"),
        runner=runner,
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 1}
    request = service.fail_review_cycle.call_args.args[0]
    assert request.failure_kind == "environment_misconfiguration"
    assert request.attempts == 1
    assert runner.reset_calls == 1
    service.append_coordination_event.assert_not_called()
    service.publish_review_verdict.assert_not_called()


def test_run_worker_marks_runtime_offline_after_idle_exit():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 15}
    )
    service.claim_next_assignment.return_value = AssignmentEnvelope(
        assignment_kind="none",
        runtime_registration_id=15,
        project_id=1,
    )
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 15, "status": "offline"}

    result = run_worker(
        project_slug="fib-demo",
        role="developer",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("developer"),
        runner=_FakeRunner([]),
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 0}
    status_request = service.update_runtime_status.call_args.args[0]
    assert status_request.status == "offline"


def test_run_worker_bootstraps_store_when_project_context_not_injected():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 18}
    )
    service.claim_next_assignment.side_effect = _assignment_stream()
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 18, "status": "offline"}

    store = Mock()
    store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"}
    store.get_agent.return_value = _agent_row("developer")
    settings = SimpleNamespace(
        tidb=object(),
        embedding=SimpleNamespace(tidb_model="tidb", backend="none"),
        server=SimpleNamespace(request_timeout_seconds=5),
    )

    with (
        patch("myswat.cli.worker_cmd.TiDBPool", return_value=Mock()) as mock_pool,
        patch("myswat.cli.worker_cmd.ensure_schema") as mock_ensure_schema,
        patch("myswat.cli.worker_cmd.MemoryStore", return_value=store) as mock_store_cls,
    ):
        result = run_worker(
            project_slug="fib-demo",
            role="developer",
            server_url="http://unused",
            settings=settings,
            runner=_FakeRunner([]),
            mcp_client=_DirectMCPClient(service),
            idle_exit_seconds=0.05,
        )

    assert result == {"stage_assignments": 0, "review_assignments": 0}
    mock_pool.assert_called_once_with(settings.tidb)
    mock_ensure_schema.assert_called_once()
    mock_store_cls.assert_called_once()


def test_run_worker_raises_when_project_is_missing():
    service = Mock()
    settings = SimpleNamespace(
        server=SimpleNamespace(request_timeout_seconds=5),
    )
    store = Mock()
    store.get_project_by_slug.return_value = None

    with pytest.raises(Exception, match="Project not found: fib-demo"):
        run_worker(
            project_slug="fib-demo",
            role="developer",
            server_url="http://unused",
            settings=settings,
            store=store,
            runner=_FakeRunner([]),
            mcp_client=_DirectMCPClient(service),
            idle_exit_seconds=0.05,
        )


def test_run_worker_raises_when_role_is_missing():
    service = Mock()
    settings = SimpleNamespace(
        server=SimpleNamespace(request_timeout_seconds=5),
    )
    store = Mock()
    store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"}
    store.get_agent.return_value = None

    with pytest.raises(Exception, match="Role 'developer' not found in project 'fib-demo'"):
        run_worker(
            project_slug="fib-demo",
            role="developer",
            server_url="http://unused",
            settings=settings,
            store=store,
            runner=_FakeRunner([]),
            mcp_client=_DirectMCPClient(service),
            idle_exit_seconds=0.05,
        )


def test_run_worker_marks_worker_error_for_unknown_assignment_kind():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 19}
    )
    service.claim_next_assignment.return_value = AssignmentEnvelope(
        assignment_kind="none",
        runtime_registration_id=19,
        project_id=1,
    ).model_copy(update={"assignment_kind": "mystery"})
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 19, "status": "offline"}

    with pytest.raises(RuntimeError, match="Unsupported assignment kind: mystery"):
        run_worker(
            project_slug="fib-demo",
            role="developer",
            server_url="http://unused",
            project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
            agent_row=_agent_row("developer"),
            runner=_FakeRunner([]),
            mcp_client=_DirectMCPClient(service),
            idle_exit_seconds=0.05,
        )

    status_request = service.update_runtime_status.call_args.args[0]
    assert status_request.metadata_json["stop_reason"] == "worker_error"


def test_run_worker_marks_keyboard_interrupt_before_reraising():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 20}
    )
    service.claim_next_assignment.side_effect = KeyboardInterrupt()
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.return_value = {"runtime_registration_id": 20, "status": "offline"}

    with pytest.raises(KeyboardInterrupt):
        run_worker(
            project_slug="fib-demo",
            role="developer",
            server_url="http://unused",
            project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
            agent_row=_agent_row("developer"),
            runner=_FakeRunner([]),
            mcp_client=_DirectMCPClient(service),
            idle_exit_seconds=0.05,
        )

    status_request = service.update_runtime_status.call_args.args[0]
    assert status_request.metadata_json["stop_reason"] == "keyboard_interrupt"


def test_run_worker_swallows_offline_update_failures():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 21}
    )
    service.claim_next_assignment.side_effect = _assignment_stream()
    service.heartbeat_runtime.return_value = None
    service.update_runtime_status.side_effect = RuntimeError("network down")

    result = run_worker(
        project_slug="fib-demo",
        role="developer",
        server_url="http://unused",
        project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
        agent_row=_agent_row("developer"),
        runner=_FakeRunner([]),
        mcp_client=_DirectMCPClient(service),
        idle_exit_seconds=0.05,
    )

    assert result == {"stage_assignments": 0, "review_assignments": 0}


def test_run_worker_propagates_registration_failure_without_offline_update():
    service = Mock()
    service.register_runtime.side_effect = RuntimeError("registration failed")

    with pytest.raises(RuntimeError, match="registration failed"):
        run_worker(
            project_slug="fib-demo",
            role="developer",
            server_url="http://unused",
            project_row={"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"},
            agent_row=_agent_row("developer"),
            runner=_FakeRunner([]),
            mcp_client=_DirectMCPClient(service),
            idle_exit_seconds=0.05,
        )

    service.update_runtime_status.assert_not_called()
