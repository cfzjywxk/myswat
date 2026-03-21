"""Tests for daemon-owned workflow execution helpers."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.server.workflow_runner import (
    get_workflow_agents,
    load_project_context,
    run_workflow,
)
from myswat.workflow.modes import WorkMode


def _project(repo_path: str) -> dict:
    return {
        "id": 1,
        "slug": "proj",
        "name": "Proj",
        "repo_path": repo_path,
    }


def _agent_row(role: str, *, agent_id: int) -> dict:
    return {
        "id": agent_id,
        "role": role,
        "display_name": role,
        "cli_backend": "codex",
        "model_name": "gpt-5.4",
    }


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        workflow=SimpleNamespace(
            design_plan_review_limit=9,
            dev_plan_review_limit=8,
            dev_code_review_limit=7,
            ga_plan_review_limit=6,
            ga_test_review_limit=5,
            assignment_poll_interval_seconds=0.25,
            assignment_timeout_seconds=30,
        )
    )


def test_load_project_context_requires_known_project():
    store = Mock()
    store.get_project_by_slug.return_value = None

    with pytest.raises(ClickExit):
        load_project_context("missing", None, settings=_settings(), store=store)


def test_load_project_context_normalizes_workdir(tmp_path):
    store = Mock()
    project = _project(str(tmp_path / "repo"))
    workdir = str(tmp_path / "nested" / ".." / "repo")

    settings, resolved_store, resolved_project, resolved_workdir = load_project_context(
        "proj",
        workdir,
        settings=_settings(),
        store=store,
        project_row=project,
    )

    assert settings is not None
    assert resolved_store is store
    assert resolved_project is project
    assert resolved_workdir == str(Path(workdir).resolve())


def test_get_workflow_agents_requires_qa():
    store = Mock()
    store.get_agent.side_effect = lambda _project_id, role: {
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": None,
        "qa_vice": None,
    }.get(role)

    with pytest.raises(ClickExit):
        get_workflow_agents(store, 1)


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_creates_item_and_wires_kernel(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.create_work_item.return_value = 42
    store.get_agent.side_effect = lambda _project_id, role: {
        "architect": _agent_row("architect", agent_id=30),
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)
    kernel = Mock()
    kernel.run.return_value = SimpleNamespace(success=True)
    mock_kernel_cls.return_value = kernel
    service = MagicMock()

    work_item_id = run_workflow(
        "proj",
        "ship it",
        workdir=None,
        work_item_id=None,
        mode=WorkMode.full,
        with_ga_test=True,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=service,
    )

    assert work_item_id == 42
    assert store.create_work_item.call_args.kwargs["metadata_json"] == {
        "work_mode": "full",
        "execution_mode": "daemon",
        "submitted_via": "daemon_api",
        "requested_workdir": str(tmp_path.resolve()),
        "with_ga_test": True,
    }
    assert store.create_work_item.call_args.kwargs["assigned_agent_id"] == 30
    store.update_work_item_status.assert_any_call(42, "in_progress")
    store.update_work_item_status.assert_any_call(42, "completed")
    kernel_kwargs = mock_kernel_cls.call_args.kwargs
    assert kernel_kwargs["mode"] == WorkMode.full
    assert kernel_kwargs["with_ga_test"] is True
    assert kernel_kwargs["repo_path"] == str(tmp_path.resolve())
    assert kernel_kwargs["assignment_poll_interval_seconds"] == 0.25
    assert kernel_kwargs["assignment_timeout_seconds"] == 30.0
    mock_submit_learn.assert_called_once_with(
        store=store,
        settings=ANY,
        project_id=1,
        source_work_item_id=42,
        source_session_id=None,
        requirement="ship it",
        final_status="completed",
        final_summary="Workflow completed successfully.",
        mode="full",
        workdir=str(tmp_path.resolve()),
    )


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_uses_requested_pause_status_when_cancelled(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.get_work_item.return_value = {"status": "paused"}
    store.get_agent.side_effect = lambda _project_id, role: {
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)
    mock_kernel_cls.return_value = Mock(run=Mock(return_value=SimpleNamespace(success=True)))
    cancel_event = threading.Event()
    cancel_event.set()

    work_item_id = run_workflow(
        "proj",
        "pause this",
        work_item_id=77,
        mode=WorkMode.develop,
        auto_approve=True,
        external_cancel_event=cancel_event,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 77
    store.create_work_item.assert_not_called()
    store.update_work_item_status.assert_any_call(77, "in_progress")
    store.update_work_item_status.assert_any_call(77, "paused")
    mock_submit_learn.assert_called_once_with(
        store=store,
        settings=ANY,
        project_id=1,
        source_work_item_id=77,
        source_session_id=None,
        requirement="pause this",
        final_status="paused",
        final_summary="Workflow paused.",
        mode="develop",
        workdir=str(tmp_path.resolve()),
    )


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.workflow.error_handler.handle_workflow_error")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_handles_engine_failure_and_status_update_error(
    mock_kernel_cls,
    mock_handle_error,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.get_agent.side_effect = lambda _project_id, role: {
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)
    store.update_work_item_status.side_effect = [None, RuntimeError("db down")]
    mock_kernel_cls.return_value = Mock(run=Mock(side_effect=RuntimeError("boom")))

    work_item_id = run_workflow(
        "proj",
        "broken",
        work_item_id=91,
        mode=WorkMode.develop,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 91
    mock_handle_error.assert_called_once()
    mock_submit_learn.assert_called_once_with(
        store=store,
        settings=ANY,
        project_id=1,
        source_work_item_id=91,
        source_session_id=None,
        requirement="broken",
        final_status="blocked",
        final_summary="Workflow crashed: RuntimeError",
        mode="develop",
        workdir=str(tmp_path.resolve()),
    )


def test_run_workflow_rejects_with_ga_test_for_non_full_mode(tmp_path):
    store = Mock()
    project = _project(str(tmp_path))

    with pytest.raises(typer.BadParameter, match="--with-ga-test"):
        run_workflow(
            "proj",
            "invalid",
            mode=WorkMode.test,
            with_ga_test=True,
            emit_console_output=False,
            settings=_settings(),
            store=store,
            project_row=project,
        )


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_marks_blocked_when_engine_returns_failure_summary(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.get_agent.side_effect = lambda _project_id, role: {
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)
    mock_kernel_cls.return_value = Mock(
        run=Mock(return_value=SimpleNamespace(success=False, failure_summary="review failed")),
    )

    work_item_id = run_workflow(
        "proj",
        "needs review",
        work_item_id=52,
        mode=WorkMode.develop,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 52
    store.update_work_item_status.assert_any_call(52, "in_progress")
    store.update_work_item_status.assert_any_call(52, "blocked")
    mock_submit_learn.assert_called_once_with(
        store=store,
        settings=ANY,
        project_id=1,
        source_work_item_id=52,
        source_session_id=None,
        requirement="needs review",
        final_status="blocked",
        final_summary="review failed",
        mode="develop",
        workdir=str(tmp_path.resolve()),
    )
