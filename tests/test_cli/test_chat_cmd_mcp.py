"""Focused tests for MCP-backed chat session handling."""

from __future__ import annotations

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from myswat.cli.chat_cmd import (
    SessionManager,
    _RemoteRunnerStub,
    _print_daemon_error,
    _public_chat_work_mode,
    _run_remote_workflow,
    _run_workflow,
    _send_with_timer,
)
from myswat.workflow.modes import WorkMode


def _agent_row(role: str = "architect") -> dict:
    return {
        "id": 3,
        "role": role,
        "display_name": "Architect",
        "cli_backend": "codex",
        "model_name": "gpt-5",
    }


def _project() -> dict:
    return {"id": 9, "slug": "proj", "name": "Proj", "repo_path": "/tmp/repo"}


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.workflow.assignment_poll_interval_seconds = 0.2
    settings.server.host = "127.0.0.1"
    settings.server.port = 8080
    settings.server.request_timeout_seconds = 10
    return settings


def test_remote_runner_stub_methods_are_noops():
    runner = _RemoteRunnerStub()
    runner.live_output.extend(["a", "b"])

    runner.clear_live_output()
    runner.cancel()

    assert runner.live_output == []


def test_remote_session_manager_round_trips_mcp_calls():
    mcp = MagicMock()
    mcp.call_tool.side_effect = [
        {
            "session_id": 41,
            "session_uuid": "uuid-41",
            "agent_id": 3,
            "agent_role": "architect",
        },
        {
            "session_id": 41,
            "session_uuid": "uuid-41",
            "agent_id": 3,
            "agent_role": "architect",
            "content": "hello back",
            "exit_code": 0,
            "raw_stdout": "stdout",
            "raw_stderr": "",
            "token_usage": {"prompt": 12},
            "cancelled": False,
        },
        {"session_id": 41, "session_uuid": "uuid-41", "ok": True},
        {"session_id": 41, "session_uuid": "uuid-41", "ok": True},
    ]
    manager = SessionManager(
        store=MagicMock(),
        mcp=mcp,
        project_row=_project(),
        agent_row=_agent_row(),
        settings=MagicMock(),
        workdir="/tmp/repo",
    )

    session = manager.create_or_resume(purpose="Interactive chat (architect)")
    response = manager.send("hello", task_description="chat turn")
    manager.reset_ai_session()
    manager.close()

    assert session.id == 41
    assert response.content == "hello back"
    assert mcp.call_tool.call_args_list[0].args[0] == "open_chat_session"
    assert mcp.call_tool.call_args_list[1].args[0] == "send_chat_message"
    assert mcp.call_tool.call_args_list[2].args[0] == "reset_chat_session"
    assert mcp.call_tool.call_args_list[3].args[0] == "close_chat_session"


def test_remote_session_manager_reuses_open_session():
    mcp = MagicMock()
    mcp.call_tool.return_value = {
        "session_id": 41,
        "session_uuid": "uuid-41",
        "agent_id": 3,
        "agent_role": "architect",
    }
    manager = SessionManager(
        store=MagicMock(),
        mcp=mcp,
        project_row=_project(),
        agent_row=_agent_row(),
        settings=MagicMock(),
        workdir="/tmp/repo",
    )

    session_a = manager.create_or_resume(purpose="Interactive chat (architect)")
    session_b = manager.create_or_resume(purpose="ignored")

    assert session_a is session_b
    mcp.call_tool.assert_called_once()


def test_session_manager_builds_timeoutless_mcp_client():
    settings = _settings()
    with patch("myswat.cli.chat_cmd.MCPHTTPClient") as mock_client_cls:
        manager = SessionManager(
            store=MagicMock(),
            project_row=_project(),
            agent_row=_agent_row(),
            settings=settings,
            workdir="/tmp/repo",
        )

    mock_client_cls.assert_called_once_with(
        "http://127.0.0.1:8080",
        timeout_seconds=None,
    )
    assert manager._mcp is mock_client_cls.return_value


def test_send_with_timer_uses_status_spinner_for_remote_sessions():
    mcp = MagicMock()
    mcp.call_tool.side_effect = [
        {
            "session_id": 41,
            "session_uuid": "uuid-41",
            "agent_id": 3,
            "agent_role": "architect",
        },
        {
            "session_id": 41,
            "session_uuid": "uuid-41",
            "agent_id": 3,
            "agent_role": "architect",
            "content": "done",
            "exit_code": 0,
            "raw_stdout": "",
            "raw_stderr": "",
            "token_usage": {},
            "cancelled": False,
        },
    ]
    manager = SessionManager(
        store=MagicMock(),
        mcp=mcp,
        project_row=_project(),
        agent_row=_agent_row(),
        settings=MagicMock(),
        workdir="/tmp/repo",
    )
    console = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.return_value = None
    status_context.__exit__.return_value = False
    console.status.return_value = status_context

    manager.create_or_resume(purpose="Interactive chat (architect)")
    response, elapsed = _send_with_timer(console, manager, "hello")

    assert response.content == "done"
    assert elapsed >= 0
    console.status.assert_called_once()


def test_session_manager_properties_and_guard_paths():
    mcp = MagicMock()
    manager = SessionManager(
        store=MagicMock(),
        mcp=mcp,
        project_row=_project(),
        agent_row=_agent_row("qa_main"),
        settings=_settings(),
        workdir="/tmp/repo",
    )

    assert manager.session is None
    assert manager.agent_role == "qa_main"
    assert manager.agent_id == 3

    manager.reset_ai_session()
    manager.close()

    mcp.call_tool.assert_not_called()

    with pytest.raises(ValueError, match="work-item-scoped"):
        manager.create_or_resume(work_item_id=7)


def test_session_manager_send_auto_opens_session_when_missing():
    mcp = MagicMock()
    mcp.call_tool.side_effect = [
        {
            "session_id": 51,
            "session_uuid": "uuid-51",
            "agent_id": 3,
            "agent_role": "architect",
        },
        {
            "session_id": 51,
            "session_uuid": "uuid-51",
            "agent_id": 3,
            "agent_role": "architect",
            "content": "ok",
            "exit_code": 0,
            "raw_stdout": "",
            "raw_stderr": "",
            "token_usage": {},
            "cancelled": False,
        },
    ]
    manager = SessionManager(
        store=MagicMock(),
        mcp=mcp,
        project_row=_project(),
        agent_row=_agent_row(),
        settings=_settings(),
        workdir="/tmp/repo",
    )

    response = manager.send("hello", task_description="first turn")

    assert response.content == "ok"
    assert mcp.call_tool.call_args_list[0].args[0] == "open_chat_session"
    assert mcp.call_tool.call_args_list[1].args[0] == "send_chat_message"


def test_send_with_timer_waits_for_worker_loop():
    started = threading.Event()
    release = threading.Event()
    ticks = iter([10.0, 10.1, 10.25])

    class _SlowManager:
        def send(self, prompt: str, task_description: str | None = None):
            started.set()
            release.wait(timeout=1)
            return MagicMock(content="done", exit_code=0, success=True, cancelled=False)

    console = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.side_effect = lambda: release.set() if started.wait(timeout=1) else None
    status_context.__exit__.return_value = False
    console.status.return_value = status_context

    with patch("myswat.cli.chat_cmd.time.monotonic", side_effect=lambda: next(ticks)):
        response, elapsed = _send_with_timer(console, _SlowManager(), "hello")

    assert response.content == "done"
    assert elapsed == 0.25


def test_send_with_timer_reraises_worker_errors():
    class _BrokenManager:
        def send(self, prompt: str, task_description: str | None = None):
            raise RuntimeError("boom")

    console = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.return_value = None
    status_context.__exit__.return_value = False
    console.status.return_value = status_context

    with pytest.raises(RuntimeError, match="boom"):
        _send_with_timer(console, _BrokenManager(), "hello")


def test_send_with_timer_keyboard_interrupt_returns_cancelled_response():
    class _SlowManager:
        def __init__(self):
            self._runner = MagicMock()

        def send(self, prompt: str, task_description: str | None = None):
            time.sleep(2)
            return MagicMock(content="done", exit_code=0, success=True, cancelled=False)

    console = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.side_effect = KeyboardInterrupt()
    status_context.__exit__.return_value = False
    console.status.return_value = status_context

    response, elapsed = _send_with_timer(console, _SlowManager(), "hello")

    assert response.cancelled is True
    assert elapsed >= 0


def test_send_with_timer_keyboard_interrupt_returns_finished_result():
    class _FastManager:
        def __init__(self):
            self._runner = MagicMock()

        def send(self, prompt: str, task_description: str | None = None):
            return MagicMock(content="done", exit_code=0, success=True, cancelled=False)

    console = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.side_effect = KeyboardInterrupt()
    status_context.__exit__.return_value = False
    console.status.return_value = status_context

    response, _elapsed = _send_with_timer(console, _FastManager(), "hello")

    assert response.content == "done"


def test_send_with_timer_keyboard_interrupt_reraises_worker_error():
    class _BrokenManager:
        def __init__(self):
            self._runner = MagicMock()

        def send(self, prompt: str, task_description: str | None = None):
            raise RuntimeError("boom")

    console = MagicMock()
    status_context = MagicMock()
    status_context.__enter__.side_effect = KeyboardInterrupt()
    status_context.__exit__.return_value = False
    console.status.return_value = status_context

    with pytest.raises(RuntimeError, match="boom"):
        _send_with_timer(console, _BrokenManager(), "hello")


def test_print_daemon_error_prints_help():
    printed: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("myswat.cli.chat_cmd.console.print", lambda message: printed.append(str(message)))
        _print_daemon_error(RuntimeError("MCP endpoint is unavailable at http://127.0.0.1:8765/mcp"))
    assert "unavailable" in printed[0]
    assert "myswat server" in printed[1]


def test_print_daemon_error_clarifies_timeout_without_start_hint():
    printed: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("myswat.cli.chat_cmd.console.print", lambda message: printed.append(str(message)))
        _print_daemon_error(RuntimeError("MCP request timed out after 30s: send_chat_message"))
    assert "timed out" in printed[0]
    assert "still in progress or blocked" in printed[1]
    assert all("myswat server" not in line for line in printed[1:])


def test_public_chat_work_mode_maps_internal_modes():
    assert _public_chat_work_mode(WorkMode.architect_design) == WorkMode.design
    assert _public_chat_work_mode(WorkMode.testplan_design) == WorkMode.test
    assert _public_chat_work_mode(WorkMode.full) == WorkMode.full


def test_run_remote_workflow_returns_terminal_work_item(monkeypatch):
    client = MagicMock()
    client.submit_work.return_value = {"work_item_id": 41}
    client.get_work_item.return_value = {"work_item": {"status": "completed"}}
    monkeypatch.setattr("myswat.cli.chat_cmd.DaemonClient", lambda settings: client)
    created: list[int] = []

    work_item_id = _run_remote_workflow(
        store=MagicMock(),
        proj=_project(),
        workdir="/tmp/repo",
        settings=_settings(),
        requirement="design auth",
        mode=WorkMode.architect_design,
        on_work_item_created=created.append,
    )

    assert work_item_id == 41
    assert created == [41]
    client.submit_work.assert_called_once_with(
        project="proj",
        requirement="design auth",
        workdir="/tmp/repo",
        mode="design",
    )


def test_run_remote_workflow_cancels_and_uses_default_poll_interval(monkeypatch):
    client = MagicMock()
    client.submit_work.return_value = {"work_item_id": 52}
    client.get_work_item.side_effect = [
        {"work_item": {"status": "in_progress"}},
        {"work_item": {"status": "cancelled"}},
    ]
    monkeypatch.setattr("myswat.cli.chat_cmd.DaemonClient", lambda settings: client)
    sleep_calls: list[float] = []
    monkeypatch.setattr("myswat.cli.chat_cmd.time.sleep", lambda seconds: sleep_calls.append(seconds))
    settings = _settings()
    settings.workflow.assignment_poll_interval_seconds = "bad"

    work_item_id = _run_remote_workflow(
        store=MagicMock(),
        proj=_project(),
        workdir="/tmp/repo",
        settings=settings,
        requirement="test auth",
        mode=WorkMode.testplan_design,
        should_cancel=lambda: True,
    )

    assert work_item_id == 52
    assert sleep_calls == [1.0]
    client.control_work.assert_called_once_with(project="proj", work_item_id=52, action="cancel")


def test_run_remote_workflow_rejects_missing_work_item_id(monkeypatch):
    client = MagicMock()
    client.submit_work.return_value = {"work_item_id": 0}
    monkeypatch.setattr("myswat.cli.chat_cmd.DaemonClient", lambda settings: client)

    with pytest.raises(RuntimeError, match="work item ID"):
        _run_remote_workflow(
            store=MagicMock(),
            proj=_project(),
            workdir="/tmp/repo",
            settings=_settings(),
            requirement="ship auth",
            mode=WorkMode.full,
        )


def test_run_remote_workflow_waits_for_terminal_status_without_stale_timeout(monkeypatch):
    client = MagicMock()
    client.submit_work.return_value = {"work_item_id": 77}
    client.get_work_item.side_effect = [
        {"work_item": {"status": "in_progress"}},
        {"work_item": {"status": "in_progress"}},
        {"work_item": {"status": "completed"}},
    ]
    monkeypatch.setattr("myswat.cli.chat_cmd.DaemonClient", lambda settings: client)
    settings = _settings()

    work_item_id = _run_remote_workflow(
        store=MagicMock(),
        proj=_project(),
        workdir="/tmp/repo",
        settings=settings,
        requirement="ship auth",
        mode=WorkMode.full,
    )

    assert work_item_id == 77
    assert client.get_work_item.call_count == 3
