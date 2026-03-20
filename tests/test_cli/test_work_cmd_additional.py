"""Additional coverage-focused tests for myswat.cli.work_cmd."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.workflow.modes import WorkMode
from myswat.cli.work_cmd import (
    _launch_background_work,
    _cleanup_runtime_file,
    _finalize_background_run,
    _is_background_worker_pid,
    _read_process_argv,
    _run_workflow,
    run_background_work_item,
    stop_work_item,
)


def _agent_row(role: str = "developer", backend: str = "codex") -> dict:
    return {
        "id": 1,
        "role": role,
        "display_name": f"Agent-{role}",
        "cli_backend": backend,
        "model_name": "gpt-5",
        "cli_path": backend,
        "cli_extra_args": None,
    }


def _proj() -> dict:
    return {"id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp"}


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.workflow.max_review_iterations = 3
    settings.embedding.tidb_model = "built-in"
    settings.compaction.threshold_turns = 200
    return settings


def test_read_process_argv_handles_proc_oserror_and_ps_failure():
    with patch("pathlib.Path.read_bytes", side_effect=OSError):
        with patch("myswat.cli.work_cmd.subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
            assert _read_process_argv(123) is None


def test_read_process_argv_returns_none_when_ps_raises():
    with patch("pathlib.Path.read_bytes", side_effect=FileNotFoundError):
        with patch("myswat.cli.work_cmd.subprocess.run", side_effect=RuntimeError("ps failed")):
            assert _read_process_argv(123) is None


def test_is_background_worker_pid_returns_false_without_argv():
    with patch("myswat.cli.work_cmd._read_process_argv", return_value=None):
        assert _is_background_worker_pid(123, 42) is False


def test_cleanup_runtime_file_ignores_unlink_oserror():
    with patch("pathlib.Path.unlink", side_effect=OSError("busy")):
        _cleanup_runtime_file("/tmp/work.pid")


def test_finalize_background_run_swallows_internal_errors():
    with patch("myswat.cli.work_cmd._load_item_metadata", side_effect=RuntimeError("boom")):
        _finalize_background_run(MagicMock(), 7, state="completed", summary="done")


@patch("myswat.cli.work_cmd.submit_workflow_summary_learn_request")
@patch("myswat.cli.work_cmd.WorkflowEngine")
def test_run_workflow_resume_parses_invalid_metadata_and_reuses_work_item(
    mock_engine_cls,
    mock_submit,
):
    settings = _settings()
    store = MagicMock()
    proj = _proj()
    existing_item = {
        "id": 42,
        "project_id": 1,
        "metadata_json": "{",
        "description": "existing requirement",
    }
    store.get_work_item.side_effect = [existing_item, existing_item]
    store.get_work_item_state.return_value = {"current_stage": "proposal_review"}
    store.get_agent.side_effect = lambda _pid, role: {
        "developer": _agent_row("developer"),
        "qa_main": _agent_row("qa_main", "kimi"),
        "architect": _agent_row("architect"),
        "qa_vice": None,
    }.get(role)
    mock_engine_cls.return_value = MagicMock(run=MagicMock(return_value=SimpleNamespace(success=True)))

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(settings, store, proj, "/tmp")):
        work_item_id = _run_workflow(
            "proj",
            "",
            show_monitor=False,
            background_worker=False,
            resume=42,
            mode=WorkMode.full,
        )

    assert work_item_id == 42
    store.create_work_item.assert_not_called()
    store.update_work_item_status.assert_any_call(42, "completed")


@patch("myswat.cli.work_cmd.WorkflowEngine")
def test_run_workflow_existing_work_item_must_exist(
    mock_engine_cls,
):
    settings = _settings()
    store = MagicMock()
    proj = _proj()
    store.get_agent.side_effect = lambda _pid, role: {
        "developer": _agent_row("developer"),
        "qa_main": _agent_row("qa_main", "kimi"),
        "qa_vice": None,
    }.get(role)

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(settings, store, proj, "/tmp")):
        with pytest.raises(ClickExit):
            _run_workflow(
                "proj",
                "do work",
                work_item_id=42,
                show_monitor=False,
                background_worker=False,
                mode=WorkMode.develop,
            )

    mock_engine_cls.assert_not_called()


@patch("myswat.cli.work_cmd.submit_workflow_summary_learn_request")
@patch("myswat.cli.work_cmd._run_with_task_monitor")
@patch("myswat.cli.work_cmd.WorkflowEngine")
def test_run_workflow_marks_cancelled_and_ignores_close_failures(
    mock_engine_cls,
    mock_task_monitor,
    mock_submit,
):
    settings = _settings()
    store = MagicMock()
    proj = _proj()
    store.get_agent.side_effect = lambda _pid, role: {
        "developer": _agent_row("developer"),
        "qa_main": _agent_row("qa_main", "kimi"),
        "qa_vice": None,
        "architect": None,
    }.get(role)
    store.create_work_item.return_value = 77
    mock_engine_cls.return_value = MagicMock()

    def _monitor(**kwargs):
        kwargs["cancel_event"].set()
        return SimpleNamespace(success=True)

    mock_task_monitor.side_effect = _monitor

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(settings, store, proj, "/tmp")):
        work_item_id = _run_workflow(
            "proj",
            "do work",
            show_monitor=True,
            background_worker=False,
            mode=WorkMode.develop,
        )

    assert work_item_id == 77
    store.update_work_item_status.assert_any_call(77, "cancelled")


@patch("myswat.cli.work_cmd.submit_workflow_summary_learn_request")
@patch("myswat.workflow.error_handler.handle_workflow_error")
@patch("myswat.cli.work_cmd.WorkflowEngine")
def test_run_workflow_exception_path_ignores_status_update_failure(
    mock_engine_cls,
    mock_handle_error,
    mock_submit,
):
    settings = _settings()
    store = MagicMock()
    proj = _proj()
    store.get_agent.side_effect = lambda _pid, role: {
        "developer": _agent_row("developer"),
        "qa_main": _agent_row("qa_main", "kimi"),
        "qa_vice": None,
        "architect": None,
    }.get(role)
    store.create_work_item.return_value = 12
    store.update_work_item_status.side_effect = [None, RuntimeError("db down")]
    mock_engine_cls.return_value = MagicMock(run=MagicMock(side_effect=RuntimeError("boom")))

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(settings, store, proj, "/tmp")):
        work_item_id = _run_workflow(
            "proj",
            "do work",
            show_monitor=False,
            background_worker=False,
            mode=WorkMode.develop,
        )

    assert work_item_id == 12
    assert mock_submit.called


@patch("myswat.cli.work_cmd._run_workflow")
def test_run_background_work_item_allows_design_mode(mock_run_workflow):
    run_background_work_item("proj", "req", work_item_id=1, mode=WorkMode.design)
    mock_run_workflow.assert_called_once_with(
        "proj",
        "req",
        workdir=None,
        work_item_id=1,
        show_monitor=False,
        background_worker=True,
        mode=WorkMode.design,
        skip_ga_test=False,
        auto_approve=True,
    )


@patch("myswat.cli.work_cmd.subprocess.Popen")
def test_launch_background_work_allows_design_mode_and_assigns_architect(mock_popen, tmp_path):
    settings = _settings()
    settings.config_path = tmp_path / "config.toml"
    store = MagicMock()
    proj = _proj()
    store.get_agent.side_effect = lambda _pid, role: {
        "architect": {**_agent_row("architect"), "id": 7},
        "developer": {**_agent_row("developer"), "id": 9},
        "qa_main": _agent_row("qa_main", "kimi"),
        "qa_vice": None,
    }.get(role)
    store.create_work_item.return_value = 21

    proc = MagicMock()
    proc.pid = 555
    mock_popen.return_value = proc

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(settings, store, proj, "/tmp")):
        work_item_id = _launch_background_work("proj", "req", mode=WorkMode.design)

    assert work_item_id == 21
    assert store.create_work_item.call_args.kwargs["item_type"] == "design"
    assert store.create_work_item.call_args.kwargs["assigned_agent_id"] == 7
    command = mock_popen.call_args.args[0]
    mode_index = command.index("--mode")
    assert command[mode_index + 1] == "design"


@patch("myswat.cli.work_cmd.MySwatSettings")
@patch("myswat.cli.work_cmd.TiDBPool")
@patch("myswat.cli.work_cmd.MemoryStore")
def test_stop_work_item_rejects_missing_project_item_and_terminal_state(
    mock_store_cls,
    mock_pool_cls,
    mock_settings_cls,
):
    settings = _settings()
    mock_settings_cls.return_value = settings
    store = MagicMock()
    mock_store_cls.return_value = store

    store.get_project_by_slug.return_value = None
    with pytest.raises(ClickExit):
        stop_work_item("proj", 42)

    store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
    store.get_work_item.return_value = None
    with pytest.raises(ClickExit):
        stop_work_item("proj", 42)

    store.get_work_item.return_value = {
        "id": 42,
        "project_id": 1,
        "status": "completed",
        "metadata_json": {"background": {"pid": 123}},
    }
    with pytest.raises(ClickExit):
        stop_work_item("proj", 42)
