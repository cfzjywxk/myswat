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
    _derive_final_status_and_summary,
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


@patch("myswat.server.workflow_runner.MemoryStore")
@patch("myswat.server.workflow_runner.ensure_schema")
@patch("myswat.server.workflow_runner.TiDBPool")
def test_load_project_context_builds_store_when_missing(
    mock_pool_cls,
    mock_ensure_schema,
    mock_store_cls,
):
    settings = SimpleNamespace(
        tidb=object(),
        embedding=SimpleNamespace(tidb_model="tidb-model", backend="local"),
    )
    store = Mock()
    store.get_project_by_slug.return_value = {
        "id": 1,
        "slug": "proj",
        "name": "Proj",
        "repo_path": None,
    }
    mock_store_cls.return_value = store

    resolved_settings, resolved_store, resolved_project, resolved_workdir = load_project_context(
        "proj",
        None,
        settings=settings,
    )

    assert resolved_settings is settings
    assert resolved_store is store
    assert resolved_project == store.get_project_by_slug.return_value
    assert resolved_workdir is None
    mock_pool_cls.assert_called_once_with(settings.tidb)
    mock_ensure_schema.assert_called_once_with(mock_pool_cls.return_value)
    mock_store_cls.assert_called_once_with(
        mock_pool_cls.return_value,
        tidb_embedding_model="tidb-model",
        embedding_backend="local",
    )


def test_get_workflow_agents_requires_developer():
    store = Mock()
    store.get_agent.side_effect = lambda _project_id, role: {
        "developer": None,
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)

    with pytest.raises(ClickExit):
        get_workflow_agents(store, 1)


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
def test_run_workflow_existing_item_passes_resume_stage_to_kernel(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.get_work_item_state.return_value = {"current_stage": "design_review"}
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
        work_item_id=42,
        mode=WorkMode.full,
        with_ga_test=False,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=service,
    )

    assert work_item_id == 42
    store.create_work_item.assert_not_called()
    kernel_kwargs = mock_kernel_cls.call_args.kwargs
    assert kernel_kwargs["resume_stage"] == "design_review"
    store.update_work_item_state.assert_called_once_with(
        42,
        current_stage="workflow_completed",
        latest_summary="Workflow completed successfully.",
        next_todos=[],
        open_issues=[],
    )


def test_derive_final_status_blocks_when_successful_result_reports_incomplete_scope():
    result = SimpleNamespace(
        success=True,
        final_report=(
            "## Scope completeness\n"
            "Status: INCOMPLETE\n"
            "The checked repository currently implements a narrower subset.\n"
        ),
    )

    final_status, final_summary = _derive_final_status_and_summary(
        result,
        cancelled=False,
        requested_status="cancelled",
    )

    assert final_status == "blocked"
    assert "scope is still incomplete" in final_summary


def test_derive_final_status_pauses_only_for_explicit_pause_request():
    result = SimpleNamespace(success=True, final_report="# done")

    final_status, final_summary = _derive_final_status_and_summary(
        result,
        cancelled=True,
        requested_status="paused",
    )

    assert final_status == "paused"
    assert final_summary == "Workflow paused."


def test_derive_final_status_blocks_unsuccessful_results():
    result = SimpleNamespace(success=False, failure_summary="review failed")

    final_status, final_summary = _derive_final_status_and_summary(
        result,
        cancelled=False,
        requested_status="cancelled",
    )

    assert final_status == "blocked"
    assert final_summary == "review failed"


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_resolves_prd_artifact_reference_before_kernel_run(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.create_work_item.return_value = 43
    store.get_artifact.return_value = {
        "id": 12,
        "work_item_id": 7,
        "artifact_type": "prd_doc",
        "title": "PRD: Billing Revamp",
        "content": "# PRD: Billing Revamp\n\n## Problem Statement\n\nLegacy billing is brittle.",
    }
    store.get_work_item.side_effect = lambda item_id: (
        {"id": 7, "project_id": 1} if item_id == 7 else None
    )
    store.get_agent.side_effect = lambda _project_id, role: {
        "architect": _agent_row("architect", agent_id=30),
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)
    kernel = Mock()
    kernel.run.return_value = SimpleNamespace(success=True)
    mock_kernel_cls.return_value = kernel

    work_item_id = run_workflow(
        "proj",
        "PRD_ARTIFACT: 12\nImplement the first billing slice.",
        workdir=None,
        work_item_id=None,
        mode=WorkMode.full,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 43
    assert store.create_work_item.call_args.kwargs["title"] == "Implement the first billing slice."
    assert store.create_work_item.call_args.kwargs["metadata_json"]["source_prd_artifact_id"] == 12
    kernel.run.assert_called_once()
    effective_requirement = kernel.run.call_args.args[0]
    assert "# PRD: Billing Revamp" in effective_requirement
    assert "## Additional Run Instructions" in effective_requirement
    assert "Implement the first billing slice." in effective_requirement
    mock_submit_learn.assert_called_once()


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


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.workflow.error_handler.handle_workflow_error")
def test_run_workflow_blocks_existing_item_when_prd_resolution_fails(
    mock_handle_error,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.get_artifact.return_value = None

    work_item_id = run_workflow(
        "proj",
        "PRD_ARTIFACT: 999",
        work_item_id=67,
        mode=WorkMode.develop,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 67
    mock_handle_error.assert_called_once()
    store.update_work_item_status.assert_called_once_with(67, "blocked")
    mock_submit_learn.assert_called_once_with(
        store=store,
        settings=ANY,
        project_id=1,
        source_work_item_id=67,
        source_session_id=None,
        requirement="PRD_ARTIFACT: 999",
        final_status="blocked",
        final_summary="Workflow crashed: ValueError",
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
    store.get_work_item_state.return_value = {
        "current_stage": "plan",
        "next_todos": ["resume execution"],
        "open_issues": ["known issue"],
    }
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
    store.update_work_item_state.assert_called_with(
        52,
        current_stage="plan",
        latest_summary="review failed",
        next_todos=["resume execution"],
        open_issues=["known issue"],
    )
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


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_reads_state_before_setting_blocked_status(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    current_state = {
        "current_stage": "plan",
        "next_todos": ["resume execution"],
        "open_issues": ["known issue"],
    }
    store.get_work_item_state.return_value = current_state
    def _clobber_status(_item_id, status):
        if status == "blocked":
            store.get_work_item_state.return_value = {
                "current_stage": "workflow_finished_with_issues",
                "next_todos": [],
                "open_issues": [],
            }
    store.update_work_item_status.side_effect = _clobber_status
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
        work_item_id=54,
        mode=WorkMode.develop,
        auto_approve=True,
        emit_console_output=False,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 54
    store.update_work_item_state.assert_called_with(
        54,
        current_stage="plan",
        latest_summary="review failed",
        next_todos=["resume execution"],
        open_issues=["known issue"],
    )
    mock_submit_learn.assert_called_once()


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_defaults_cancelled_status_and_assignment_fallbacks(
    mock_kernel_cls,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    settings = _settings()
    settings.workflow.assignment_poll_interval_seconds = object()
    settings.workflow.assignment_timeout_seconds = object()
    store.get_work_item.return_value = {"status": "pending_cancel"}
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
        "cancel this",
        work_item_id=78,
        mode=WorkMode.develop,
        auto_approve=True,
        external_cancel_event=cancel_event,
        emit_console_output=False,
        settings=settings,
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 78
    kernel_kwargs = mock_kernel_cls.call_args.kwargs
    assert kernel_kwargs["assignment_poll_interval_seconds"] == 1.0
    assert kernel_kwargs["assignment_timeout_seconds"] is None
    store.update_work_item_status.assert_any_call(78, "cancelled")
    mock_submit_learn.assert_called_once_with(
        store=store,
        settings=ANY,
        project_id=1,
        source_work_item_id=78,
        source_session_id=None,
        requirement="cancel this",
        final_status="cancelled",
        final_summary="Workflow cancelled.",
        mode="develop",
        workdir=str(tmp_path.resolve()),
    )


@patch("myswat.server.workflow_runner.submit_workflow_summary_learn_request")
@patch("myswat.server.workflow_runner.console.print")
@patch("myswat.server.workflow_runner.WorkflowKernel")
def test_run_workflow_prints_progress_and_ignores_summary_learn_failures(
    mock_kernel_cls,
    mock_console_print,
    mock_submit_learn,
    tmp_path,
):
    store = Mock()
    project = _project(str(tmp_path))
    store.get_work_item_state.return_value = {
        "current_stage": "phase_2",
        "next_todos": ["keep going"],
        "open_issues": [],
    }
    store.get_agent.side_effect = lambda _project_id, role: {
        "developer": _agent_row("developer", agent_id=10),
        "qa_main": _agent_row("qa_main", agent_id=20),
        "qa_vice": None,
    }.get(role)
    mock_kernel_cls.return_value = Mock(
        run=Mock(return_value=SimpleNamespace(success=False, failure_summary="   ")),
    )
    mock_submit_learn.side_effect = RuntimeError("learn down")

    work_item_id = run_workflow(
        "proj",
        "needs help",
        work_item_id=53,
        mode=WorkMode.develop,
        auto_approve=True,
        emit_console_output=True,
        settings=_settings(),
        store=store,
        project_row=project,
        service=MagicMock(),
    )

    assert work_item_id == 53
    store.update_work_item_status.assert_any_call(53, "in_progress")
    store.update_work_item_status.assert_any_call(53, "blocked")
    store.update_work_item_state.assert_called_with(
        53,
        current_stage="phase_2",
        latest_summary="Workflow finished with unresolved review or test issues.",
        next_todos=["keep going"],
        open_issues=[],
    )
    assert mock_submit_learn.call_args.kwargs["final_summary"] == (
        "Workflow finished with unresolved review or test issues."
    )
    mock_console_print.assert_any_call("[bold]Requirement:[/bold] needs help")
    mock_console_print.assert_any_call("[dim]Work item: 53[/dim]")
    mock_console_print.assert_any_call("\n[dim]Workflow state persisted to TiDB.[/dim]")
