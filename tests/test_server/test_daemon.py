"""Focused tests for daemon-side worker supervision helpers."""

from __future__ import annotations

import threading
from unittest.mock import Mock

from myswat.server.daemon import ManagedWorkerProcess, MySwatDaemon
from myswat.workflow.modes import WorkMode


def test_worker_roles_for_full_mode_include_arch_dev_and_available_qas():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._store = Mock()
    daemon._store.get_agent.side_effect = lambda project_id, role: {"id": 1, "role": role} if role in {
        "architect",
        "developer",
        "qa_main",
    } else None

    roles = daemon._worker_roles_for_mode(1, WorkMode.full)

    assert roles == ["architect", "developer", "qa_main"]


def test_worker_roles_for_develop_mode_skip_architect():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._store = Mock()
    daemon._store.get_agent.side_effect = lambda project_id, role: {"id": 1, "role": role} if role in {
        "architect",
        "developer",
        "qa_main",
        "qa_vice",
    } else None

    roles = daemon._worker_roles_for_mode(1, WorkMode.develop)

    assert roles == ["developer", "qa_main", "qa_vice"]


def test_handle_work_queues_workers_and_starts_in_process_workflow():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._lock = threading.RLock()
    daemon._find_active_work_item = Mock(return_value=None)
    daemon.ensure_workers = Mock(return_value=["architect", "developer", "qa_main"])
    daemon._create_work_item = Mock(return_value=88)
    daemon._start_workflow_thread = Mock()

    result = daemon.handle_work(
        project="fib-demo",
        requirement="implement fibonacci",
        workdir="/tmp/fib-demo",
        mode=WorkMode.full.value,
    )

    assert result == {
        "ok": True,
        "work_item_id": 88,
        "workers": ["architect", "developer", "qa_main"],
    }
    daemon.ensure_workers.assert_called_once_with(
        project_slug="fib-demo",
        mode=WorkMode.full,
        workdir="/tmp/fib-demo",
    )
    daemon._create_work_item.assert_called_once_with(
        project_slug="fib-demo",
        requirement="implement fibonacci",
        workdir="/tmp/fib-demo",
        mode=WorkMode.full,
    )
    daemon._start_workflow_thread.assert_called_once_with(
        project_slug="fib-demo",
        requirement="implement fibonacci",
        work_item_id=88,
        workdir="/tmp/fib-demo",
        mode=WorkMode.full,
    )


def test_handle_work_rejects_when_project_already_has_active_workflow():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._lock = threading.RLock()
    daemon._find_active_work_item = Mock(return_value={"id": 12, "status": "in_progress"})

    try:
        daemon.handle_work(
            project="fib-demo",
            requirement="implement fibonacci",
            workdir="/tmp/fib-demo",
            mode=WorkMode.full.value,
        )
    except ValueError as exc:
        assert "already has an active workflow" in str(exc)
    else:
        raise AssertionError("Expected active workflow rejection")


def test_handle_work_serializes_concurrent_submissions():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._lock = threading.RLock()
    daemon.ensure_workers = Mock(return_value=["developer"])
    daemon._start_workflow_thread = Mock()
    created_work_items: list[int] = []
    create_gate = threading.Event()

    def _find_active_work_item(_project: str):
        if not created_work_items:
            return None
        return {"id": created_work_items[0], "status": "in_progress"}

    def _create_work_item(**_kwargs):
        create_gate.wait(timeout=1)
        work_item_id = 88 + len(created_work_items)
        created_work_items.append(work_item_id)
        return work_item_id

    daemon._find_active_work_item = Mock(side_effect=_find_active_work_item)
    daemon._create_work_item = Mock(side_effect=_create_work_item)

    results: list[dict] = []
    errors: list[Exception] = []

    def _submit() -> None:
        try:
            results.append(
                daemon.handle_work(
                    project="fib-demo",
                    requirement="implement fibonacci",
                    workdir="/tmp/fib-demo",
                    mode=WorkMode.full.value,
                )
            )
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=_submit, daemon=True)
    second = threading.Thread(target=_submit, daemon=True)
    first.start()
    second.start()
    create_gate.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert len(results) == 1
    assert len(errors) == 1
    assert "already has an active workflow" in str(errors[0])


def test_handle_control_work_marks_cancelled_and_recycles_workers():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._store = Mock()
    daemon._lock = threading.Lock()
    daemon._workflow_controls = {
        7: type(
            "Handle",
            (),
            {
                "project_slug": "fib-demo",
                "cancel_event": threading.Event(),
                "requested_status": None,
            },
        )()
    }
    daemon._store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo"}
    daemon._store.get_work_item.return_value = {"id": 7, "project_id": 1, "status": "in_progress"}
    daemon._store.get_work_item_state.return_value = {"current_stage": "design"}
    daemon._stop_project_workers = Mock()

    result = daemon.handle_control_work(
        project="fib-demo",
        work_item_id=7,
        action="cancel",
    )

    assert result == {"ok": True, "work_item_id": 7, "status": "cancelled"}
    daemon._store.update_work_item_status.assert_called_once_with(7, "cancelled")
    daemon._store.cancel_open_stage_runs.assert_called_once()
    daemon._store.cancel_open_review_cycles.assert_called_once()
    assert daemon._workflow_controls[7].cancel_event.is_set() is True
    daemon._stop_project_workers.assert_called_once_with("fib-demo")


def test_handle_control_work_marks_paused_and_recycles_workers():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._store = Mock()
    daemon._lock = threading.Lock()
    daemon._workflow_controls = {
        8: type(
            "Handle",
            (),
            {
                "project_slug": "fib-demo",
                "cancel_event": threading.Event(),
                "requested_status": None,
            },
        )()
    }
    daemon._store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo"}
    daemon._store.get_work_item.return_value = {"id": 8, "project_id": 1, "status": "review"}
    daemon._store.get_work_item_state.return_value = {}
    daemon._stop_project_workers = Mock()

    result = daemon.handle_control_work(
        project="fib-demo",
        work_item_id=8,
        action="pause",
    )

    assert result == {"ok": True, "work_item_id": 8, "status": "paused"}
    daemon._store.update_work_item_status.assert_called_once_with(8, "paused")
    daemon._store.cancel_open_stage_runs.assert_called_once()
    daemon._store.cancel_open_review_cycles.assert_called_once()
    assert daemon._workflow_controls[8].cancel_event.is_set() is True
    assert daemon._workflow_controls[8].requested_status == "paused"
    daemon._stop_project_workers.assert_called_once_with("fib-demo")


def test_handle_cleanup_project_cancels_active_workflows_stops_workers_and_deletes_data():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._store = Mock()
    daemon._service = Mock()
    daemon._lock = threading.Lock()
    thread = Mock()
    thread.is_alive.return_value = False
    daemon._workflows = {7: thread}
    daemon._workflow_controls = {
        7: type(
            "Handle",
            (),
            {
                "project_slug": "fib-demo",
                "cancel_event": threading.Event(),
                "requested_status": None,
            },
        )()
    }
    daemon._store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo"}
    daemon._store.list_work_items.return_value = [
        {"id": 7, "project_id": 1, "status": "in_progress"},
        {"id": 8, "project_id": 1, "status": "completed"},
    ]
    daemon._store.get_work_item_state.return_value = {"current_stage": "design"}
    daemon._store.delete_project.return_value = {"projects": 1, "agents": 3}
    daemon._stop_project_workers = Mock()
    daemon._cleanup_project_runtime_files = Mock(return_value=["/tmp/workers/fib-demo"])

    result = daemon.handle_cleanup_project(project="fib-demo")

    assert result == {
        "ok": True,
        "project": "fib-demo",
        "work_item_ids": [7, 8],
        "deleted": {"projects": 1, "agents": 3},
        "removed_runtime_paths": ["/tmp/workers/fib-demo"],
    }
    daemon._store.update_work_item_status.assert_called_once_with(7, "cancelled")
    assert daemon._workflow_controls[7].cancel_event.is_set() is True
    assert daemon._workflow_controls[7].requested_status == "cancelled"
    daemon._stop_project_workers.assert_called_once_with("fib-demo")
    daemon._store.delete_project.assert_called_once_with(1)
    assert daemon._store.cancel_open_stage_runs.call_count == 2
    assert daemon._store.cancel_open_review_cycles.call_count == 2
    assert daemon._service.notify_work_item_coordination_changed.call_count == 2
    thread.join.assert_called_once()


def test_handle_cleanup_project_rejects_when_workflow_thread_does_not_stop():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._store = Mock()
    daemon._service = Mock()
    daemon._lock = threading.Lock()
    thread = Mock()
    thread.is_alive.return_value = True
    daemon._workflows = {7: thread}
    daemon._workflow_controls = {
        7: type(
            "Handle",
            (),
            {
                "project_slug": "fib-demo",
                "cancel_event": threading.Event(),
                "requested_status": None,
            },
        )()
    }
    daemon._store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo"}
    daemon._store.list_work_items.return_value = [
        {"id": 7, "project_id": 1, "status": "in_progress"},
    ]
    daemon._store.get_work_item_state.return_value = {"current_stage": "design"}
    daemon._stop_project_workers = Mock()

    try:
        daemon.handle_cleanup_project(project="fib-demo", wait_timeout_seconds=0.01)
    except RuntimeError as exc:
        assert "waiting for workflows to stop" in str(exc)
    else:
        raise AssertionError("Expected cleanup to fail when workflow thread stays alive")

    daemon._store.delete_project.assert_not_called()


def test_handle_mcp_request_returns_success_payload():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._dispatcher = Mock()
    daemon._dispatcher.list_tools.return_value = {"tools": [{"name": "ping"}]}

    result = daemon.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
            "params": {},
        }
    )

    assert result == {
        "jsonrpc": "2.0",
        "id": 4,
        "result": {"tools": [{"name": "ping"}]},
    }


def test_handle_mcp_request_sanitizes_internal_errors():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    dispatcher = Mock()
    dispatcher.call_tool.side_effect = RuntimeError("db password leaked")
    daemon._dispatcher = dispatcher

    result = daemon.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "complete_stage_task", "arguments": {}},
        }
    )

    assert result == {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {
            "code": -32000,
            "message": "internal server error",
        },
    }


def test_handle_mcp_request_returns_none_for_initialized_notification():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._dispatcher = Mock()

    result = daemon.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "initialized",
            "params": {},
        }
    )

    assert result is None


def test_supervise_workers_once_marks_dead_worker_offline_and_restarts_active_project_worker():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._lock = threading.Lock()
    proc = Mock()
    proc.poll.return_value = 17
    proc.pid = 4321
    daemon._workers = {
        ("fib-demo", "developer"): ManagedWorkerProcess(
            process=proc,
            workdir="/tmp/fib-demo",
        )
    }
    daemon._store = Mock()
    daemon._store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo"}
    daemon._store.list_runtime_registrations.return_value = [
        type("Runtime", (), {"id": 12, "metadata_json": {"pid": 4321}})()
    ]
    daemon._find_active_work_item = Mock(return_value={"id": 7, "status": "in_progress"})
    daemon._start_worker = Mock()

    daemon._supervise_workers_once()

    assert daemon._workers == {}
    daemon._start_worker.assert_called_once_with(
        project_slug="fib-demo",
        role="developer",
        workdir="/tmp/fib-demo",
    )
    runtime_update = daemon._store.update_runtime_status.call_args
    assert runtime_update.args[0] == 12
    assert runtime_update.kwargs["status"] == "offline"
    assert runtime_update.kwargs["metadata_json"]["stop_reason"] == "worker_process_exited"
    assert runtime_update.kwargs["metadata_json"]["exit_code"] == 17


def test_supervise_workers_once_does_not_restart_when_project_has_no_active_work():
    daemon = MySwatDaemon.__new__(MySwatDaemon)
    daemon._lock = threading.Lock()
    proc = Mock()
    proc.poll.return_value = 9
    proc.pid = 7654
    daemon._workers = {
        ("fib-demo", "qa_main"): ManagedWorkerProcess(
            process=proc,
            workdir="/tmp/fib-demo",
        )
    }
    daemon._store = Mock()
    daemon._store.get_project_by_slug.return_value = {"id": 1, "slug": "fib-demo"}
    daemon._store.list_runtime_registrations.return_value = []
    daemon._find_active_work_item = Mock(return_value=None)
    daemon._start_worker = Mock()

    daemon._supervise_workers_once()

    assert daemon._workers == {}
    daemon._start_worker.assert_not_called()
