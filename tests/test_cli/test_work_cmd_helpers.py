"""Extra helper and resume-path tests for myswat.cli.work_cmd."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit

from myswat.workflow.modes import WorkMode
from myswat.cli.work_cmd import (
    _cleanup_runtime_file,
    _finalize_background_run,
    _install_cancel_signal_handlers,
    _is_background_worker_pid,
    _normalize_workdir,
    _read_process_argv,
    _run_workflow,
    _update_background_metadata,
)


def test_normalize_workdir_handles_none_and_expands_home(tmp_path):
    assert _normalize_workdir(None) is None
    with patch("myswat.cli.work_cmd.Path.home", return_value=tmp_path):
        normalized = _normalize_workdir(str(tmp_path / ".." / tmp_path.name))
    assert normalized == str(tmp_path.resolve())


def test_update_background_metadata_adds_and_removes_fields():
    store = MagicMock()
    store.get_work_item.return_value = {
        "metadata_json": {"background": {"pid": 123, "state": "running"}}
    }

    background = _update_background_metadata(store, 7, state="done", pid=None, finished_at="now")

    assert background == {"state": "done", "finished_at": "now"}
    store.update_work_item_metadata.assert_called_once_with(
        7,
        {"background": {"state": "done", "finished_at": "now"}},
    )


def test_read_process_argv_prefers_proc_cmdline():
    with patch(
        "myswat.cli.work_cmd.Path.read_bytes",
        return_value=b"python\0-m\0myswat.cli.main\0",
    ):
        argv = _read_process_argv(123)

    assert argv == ["python", "-m", "myswat.cli.main"]


def test_read_process_argv_falls_back_to_ps_and_splits_plainly_on_value_error():
    with patch("pathlib.Path.read_bytes", side_effect=FileNotFoundError):
        with patch(
            "myswat.cli.work_cmd.subprocess.run",
            return_value=MagicMock(returncode=0, stdout='python -m "broken'),
        ):
            argv = _read_process_argv(123)

    assert argv == ["python", "-m", '"broken']


def test_is_background_worker_pid_matches_both_work_item_id_forms():
    with patch("myswat.cli.work_cmd._read_process_argv", return_value=[
        "python", "-m", "myswat.cli.main", "work-background-worker", "--work-item-id", "42",
    ]):
        assert _is_background_worker_pid(123, 42) is True

    with patch("myswat.cli.work_cmd._read_process_argv", return_value=[
        "python", "-m", "myswat.cli.main", "work-background-worker", "--work-item-id=42",
    ]):
        assert _is_background_worker_pid(123, 42) is True

    with patch("myswat.cli.work_cmd._read_process_argv", return_value=[
        "python", "-m", "myswat.cli.main", "work-background-worker", "--work-item-id", "77",
    ]):
        assert _is_background_worker_pid(123, 42) is False


def test_cleanup_runtime_file_ignores_missing_and_non_string(tmp_path):
    missing = tmp_path / "missing.pid"
    _cleanup_runtime_file(None)
    _cleanup_runtime_file("")
    _cleanup_runtime_file(str(missing))
    assert not missing.exists()


def test_install_cancel_signal_handlers_sets_event_cancels_and_restores():
    cancel_event = MagicMock()
    runner_ok = MagicMock()
    runner_bad = MagicMock()
    runner_bad.cancel.side_effect = RuntimeError("boom")
    previous = MagicMock()
    installed = {}

    def fake_signal(sig, handler):
        installed[sig] = handler

    with patch("myswat.cli.work_cmd.signal.getsignal", return_value=previous):
        with patch("myswat.cli.work_cmd.signal.signal", side_effect=fake_signal):
            restore = _install_cancel_signal_handlers(cancel_event, [runner_ok, runner_bad])

    first_sig = next(iter(installed))
    installed[first_sig](None, None)
    cancel_event.set.assert_called_once()
    runner_ok.cancel.assert_called_once()
    runner_bad.cancel.assert_called_once()

    installed.clear()
    with patch("myswat.cli.work_cmd.signal.signal", side_effect=fake_signal):
        restore()
    assert installed


def test_finalize_background_run_is_best_effort(tmp_path):
    pid_path = tmp_path / "worker.pid"
    pid_path.write_text("123\n", encoding="ascii")
    store = MagicMock()
    store.get_work_item.return_value = {
        "metadata_json": {"background": {"pid_path": str(pid_path)}}
    }

    _finalize_background_run(store, 7, state="completed", summary="done")

    assert not pid_path.exists()
    store.update_work_item_metadata.assert_called_once()
    store.append_work_item_process_event.assert_called_once()


def test_run_workflow_resume_rejects_internal_mode():
    store = MagicMock()
    proj = {"id": 1, "repo_path": "/tmp"}
    store.get_work_item.return_value = {
        "id": 42,
        "project_id": 1,
        "metadata_json": {"work_mode": WorkMode.architect_design.value},
        "description": "existing requirement",
    }

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(MagicMock(), store, proj, "/tmp")):
        with pytest.raises(ClickExit):
            _run_workflow(
                "proj",
                "",
                show_monitor=True,
                background_worker=False,
                resume=42,
            )


def test_run_workflow_resume_rejects_explicit_mode_mismatch():
    store = MagicMock()
    proj = {"id": 1, "repo_path": "/tmp"}
    store.get_work_item.return_value = {
        "id": 42,
        "project_id": 1,
        "metadata_json": {"work_mode": WorkMode.design.value},
        "description": "existing requirement",
    }

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(MagicMock(), store, proj, "/tmp")):
        with pytest.raises(ClickExit):
            _run_workflow(
                "proj",
                "",
                show_monitor=True,
                background_worker=False,
                resume=42,
                mode=WorkMode.full,
                mode_explicit=True,
            )


def test_run_workflow_resume_rejects_missing_requirement():
    store = MagicMock()
    proj = {"id": 1, "repo_path": "/tmp"}
    store.get_work_item.return_value = {
        "id": 42,
        "project_id": 1,
        "metadata_json": {"work_mode": WorkMode.design.value},
        "description": "",
        "title": "",
    }

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(MagicMock(), store, proj, "/tmp")):
        with pytest.raises(ClickExit):
            _run_workflow(
                "proj",
                "",
                show_monitor=True,
                background_worker=False,
                resume=42,
            )


def test_run_workflow_resume_rejects_missing_existing_item():
    store = MagicMock()
    proj = {"id": 1, "repo_path": "/tmp"}
    store.get_work_item.return_value = None

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(MagicMock(), store, proj, "/tmp")):
        with pytest.raises(ClickExit):
            _run_workflow(
                "proj",
                "",
                show_monitor=True,
                background_worker=False,
                resume=42,
            )


def test_launch_background_work_propagates_spawn_failure(tmp_path):
    from myswat.cli.work_cmd import _launch_background_work

    settings = MagicMock()
    settings.config_path = tmp_path / "config.toml"
    settings.embedding.tidb_model = "built-in"
    store = MagicMock()
    proj = {"id": 1, "repo_path": "/tmp"}
    dev_agent = {"id": 11}

    with patch("myswat.cli.work_cmd._load_project_context", return_value=(settings, store, proj, "/tmp")):
        with patch("myswat.cli.work_cmd._get_workflow_agents", return_value=(dev_agent, [])):
            with patch("myswat.cli.work_cmd._get_architect_agent", return_value=None):
                with patch("myswat.cli.work_cmd.subprocess.Popen", side_effect=OSError("spawn failed")):
                    with pytest.raises(OSError, match="spawn failed"):
                        _launch_background_work("proj", "do stuff", mode=WorkMode.develop)

    store.update_work_item_status.assert_called()
    store.append_work_item_process_event.assert_called()


def test_stop_work_item_reports_missing_process_after_validation():
    from myswat.cli.work_cmd import stop_work_item

    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    store = MagicMock()
    store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
    store.get_work_item.return_value = {
        "id": 42,
        "project_id": 1,
        "status": "in_progress",
        "metadata_json": {"background": {"pid": 123}},
    }

    with patch("myswat.cli.work_cmd.MySwatSettings", return_value=settings):
        with patch("myswat.cli.work_cmd.TiDBPool"):
            with patch("myswat.cli.work_cmd.MemoryStore", return_value=store):
                with patch("myswat.cli.work_cmd._is_background_worker_pid", return_value=True):
                    with patch("myswat.cli.work_cmd.os.kill", side_effect=ProcessLookupError):
                        with pytest.raises(ClickExit):
                            stop_work_item("proj", 42)
