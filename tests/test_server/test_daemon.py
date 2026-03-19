"""Focused tests for daemon-side worker supervision helpers."""

from __future__ import annotations

import threading
from unittest.mock import Mock

from myswat.server.daemon import MySwatDaemon
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
