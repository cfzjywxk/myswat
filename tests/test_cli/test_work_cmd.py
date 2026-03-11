"""Tests for myswat.cli.work_cmd."""

from __future__ import annotations

import json
import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.workflow.engine import WorkMode

from myswat.cli.work_cmd import (
    _build_background_env,
    _make_runner,
    run_background_work_item,
    run_work,
    stop_work_item,
)


# ---------------------------------------------------------------------------
# _make_runner
# ---------------------------------------------------------------------------
class TestMakeRunner:
    def test_codex(self):
        row = {
            "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5", "cli_extra_args": None,
        }
        assert _make_runner(row) is not None

    def test_kimi(self):
        row = {
            "cli_backend": "kimi", "cli_path": "kimi",
            "model_name": "k2", "cli_extra_args": None,
        }
        assert _make_runner(row) is not None

    def test_unknown_raises(self):
        row = {
            "cli_backend": "unknown", "cli_path": "x",
            "model_name": "m", "cli_extra_args": None,
        }
        with pytest.raises(typer.BadParameter):
            _make_runner(row)

    def test_extra_args(self):
        row = {
            "cli_backend": "codex", "cli_path": "codex",
            "model_name": "gpt-5",
            "cli_extra_args": json.dumps(["--flag"]),
        }
        assert _make_runner(row) is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _agent_row(role="developer", backend="codex"):
    return {
        "id": 1, "role": role, "display_name": f"Agent-{role}",
        "cli_backend": backend, "model_name": "gpt-5",
        "cli_path": backend, "cli_extra_args": None,
    }


# ---------------------------------------------------------------------------
# run_work
# ---------------------------------------------------------------------------
class TestRunWork:
    def test_build_background_env_for_source_checkout(self, tmp_path):
        source_root = tmp_path / "src"
        package_dir = source_root / "myswat"
        package_dir.mkdir(parents=True)
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myswat'\n", encoding="utf-8")

        with patch.dict("os.environ", {}, clear=True):
            with patch("myswat.cli.work_cmd._source_root", return_value=source_root):
                env = _build_background_env()

        assert env["PYTHONPATH"] == str(source_root)
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_build_background_env_without_source_checkout(self, tmp_path):
        site_root = tmp_path / "site-packages"
        (site_root / "myswat").mkdir(parents=True)

        with patch.dict("os.environ", {}, clear=True):
            with patch("myswat.cli.work_cmd._source_root", return_value=site_root):
                env = _build_background_env()

        assert "PYTHONPATH" not in env
        assert env["PYTHONUNBUFFERED"] == "1"

    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                                mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_work("missing", "do stuff")

    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_dev_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                            mock_settings_cls, mock_learn):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        mock_store.get_agent.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_work("proj", "do stuff")

    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_no_qa_agents(self, mock_store_cls, mock_mig, mock_pool_cls,
                           mock_settings_cls, mock_learn):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_work("proj", "do stuff")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_success(self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
                      mock_pool_cls, mock_settings_cls, mock_learn,
                      mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "completed")
        assert mock_store.create_work_item.call_args.kwargs["metadata_json"] == {"work_mode": "full"}
        assert mock_engine_cls.call_args.kwargs["mode"] == WorkMode.full
        assert mock_engine_cls.call_args.kwargs["auto_approve"] is True

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_foreground_run_prints_tracking_commands(
        self,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_learn,
        mock_engine_cls,
    ):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None

        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        with patch("myswat.cli.work_cmd.console.print") as mock_console_print:
            run_work("proj", "do stuff")

        rendered = [str(call.args[0]) for call in mock_console_print.call_args_list if call.args]
        assert any("myswat task 42 -p proj" in line for line in rendered)
        assert any("myswat status -p proj" in line for line in rendered)

    @patch("myswat.cli.work_cmd._run_with_task_monitor")
    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_foreground_run_uses_task_monitor(
        self,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_learn,
        mock_engine_cls,
        mock_task_monitor,
    ):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None

        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        mock_task_monitor.return_value = SimpleNamespace(success=True)

        run_work("proj", "do stuff")

        mock_task_monitor.assert_called_once()
        kwargs = mock_task_monitor.call_args.kwargs
        assert kwargs["label"] == "Running full teamwork workflow"
        assert kwargs["proj"]["id"] == 1

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_failure(self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
                      mock_pool_cls, mock_settings_cls, mock_learn,
                      mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=False)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "review")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_exception(self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
                        mock_pool_cls, mock_settings_cls, mock_learn,
                        mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.side_effect = RuntimeError("engine crash")
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")
        mock_store.update_work_item_status.assert_any_call(42, "blocked")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_migrations_applied_printed(self, mock_comp, mock_sm_cls,
                                         mock_store_cls, mock_mig,
                                         mock_pool_cls, mock_settings_cls,
                                         mock_learn, mock_engine_cls):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings
        mock_mig.return_value = ["v001"]

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }
        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None
        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff")

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_design_mode_threads_to_engine_and_persists_metadata(
        self, mock_comp, mock_sm_cls, mock_store_cls, mock_mig,
        mock_pool_cls, mock_settings_cls, mock_learn, mock_engine_cls,
    ):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/tmp",
        }

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None

        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        run_work("proj", "do stuff", mode=WorkMode.design)

        assert mock_store.create_work_item.call_args.kwargs["metadata_json"] == {"work_mode": "design"}
        assert mock_engine_cls.call_args.kwargs["mode"] == WorkMode.design
        assert mock_engine_cls.call_args.kwargs["auto_approve"] is False

    @patch("myswat.cli.work_cmd.subprocess.Popen")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_background_launch(
        self,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_popen,
        tmp_path,
    ):
        settings = MagicMock()
        settings.config_path = tmp_path / "config.toml"
        settings.embedding.tidb_model = "built-in"
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1,
            "repo_path": "/tmp",
        }

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None

        mock_store.get_agent.side_effect = get_agent_side
        mock_store.create_work_item.return_value = 42
        mock_store_cls.return_value = mock_store

        proc = MagicMock()
        proc.pid = 12345
        mock_popen.return_value = proc

        with patch("myswat.cli.work_cmd.console.print") as mock_console_print:
            run_work("proj", "do stuff", background=True)

        assert mock_popen.called
        args, kwargs = mock_popen.call_args
        assert args[0][0]
        assert args[0][1:4] == ["-m", "myswat.cli.main", "work-background-worker"]
        assert "--work-item-id" in args[0]
        assert "42" in args[0]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is not None
        assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        mock_store.update_work_item_state.assert_any_call(
            42,
            current_stage="background_launch_pending",
            latest_summary="do stuff",
            next_todos=["Wait for detached workflow worker to start"],
        )
        assert (tmp_path / "runs" / "proj" / "work-42.pid").read_text(encoding="ascii").strip() == "12345"
        rendered = [str(call.args[0]) for call in mock_console_print.call_args_list if call.args]
        assert any("myswat task 42 -p proj" in line for line in rendered)
        assert any("myswat status -p proj" in line for line in rendered)

    @patch("myswat.cli.work_cmd.WorkflowEngine")
    @patch("myswat.cli.learn_cmd.ensure_learned")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.run_migrations")
    @patch("myswat.cli.work_cmd.MemoryStore")
    @patch("myswat.cli.work_cmd.SessionManager")
    @patch("myswat.cli.work_cmd.KnowledgeCompactor")
    def test_background_worker_reuses_existing_work_item(
        self,
        mock_comp,
        mock_sm_cls,
        mock_store_cls,
        mock_mig,
        mock_pool_cls,
        mock_settings_cls,
        mock_learn,
        mock_engine_cls,
        tmp_path,
    ):
        settings = MagicMock()
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000
        settings.workflow.max_review_iterations = 5
        settings.embedding.tidb_model = "built-in"
        mock_settings_cls.return_value = settings

        dev_row = _agent_row("developer")
        qa_row = _agent_row("qa_main", "kimi")
        pid_path = tmp_path / "worker.pid"

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1,
            "repo_path": "/tmp",
        }

        def get_agent_side(pid, role):
            if role == "developer":
                return dev_row
            if role == "qa_main":
                return qa_row
            return None

        mock_store.get_agent.side_effect = get_agent_side
        mock_store.get_work_item.return_value = {
            "id": 42,
            "project_id": 1,
            "metadata_json": {
                "background": {
                    "pid_path": str(pid_path),
                }
            },
        }
        mock_store_cls.return_value = mock_store

        sm = MagicMock()
        sm.session = SimpleNamespace(session_uuid="uuid")
        sm._agent_row = qa_row
        mock_sm_cls.return_value = sm

        engine = MagicMock()
        engine.run.return_value = SimpleNamespace(success=True)
        mock_engine_cls.return_value = engine

        run_background_work_item("proj", "do stuff", work_item_id=42)

        mock_store.update_work_item_status.assert_any_call(42, "completed")
        assert not pid_path.exists()

    @patch("myswat.cli.work_cmd._read_process_argv", return_value=[
        "python3",
        "-m",
        "myswat.cli.main",
        "work-background-worker",
        "do stuff",
        "--project",
        "proj",
        "--work-item-id",
        "42",
    ])
    @patch("myswat.cli.work_cmd.os.kill")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_stop_background_work_item(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
        mock_kill,
        mock_read_process_argv,
    ):
        settings = MagicMock()
        settings.embedding.tidb_model = "built-in"
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.get_work_item.return_value = {
            "id": 42,
            "project_id": 1,
            "status": "in_progress",
            "metadata_json": {
                "background": {
                    "pid": 12345,
                }
            },
        }
        mock_store_cls.return_value = mock_store

        stop_work_item("proj", 42)

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        mock_store.update_work_item_state.assert_called_once()

    @patch("myswat.cli.work_cmd._read_process_argv", return_value=[
        "python3",
        "-m",
        "python.http.server",
    ])
    @patch("myswat.cli.work_cmd.os.kill")
    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_stop_background_work_item_rejects_stale_pid(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
        mock_kill,
        mock_read_process_argv,
    ):
        settings = MagicMock()
        settings.embedding.tidb_model = "built-in"
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.get_work_item.return_value = {
            "id": 42,
            "project_id": 1,
            "status": "in_progress",
            "metadata_json": {
                "background": {
                    "pid": 12345,
                }
            },
        }
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            stop_work_item("proj", 42)

        mock_kill.assert_not_called()
        mock_store.update_work_item_state.assert_not_called()

    @patch("myswat.cli.work_cmd.MySwatSettings")
    @patch("myswat.cli.work_cmd.TiDBPool")
    @patch("myswat.cli.work_cmd.MemoryStore")
    def test_stop_background_work_item_without_pid(
        self,
        mock_store_cls,
        mock_pool_cls,
        mock_settings_cls,
    ):
        settings = MagicMock()
        settings.embedding.tidb_model = "built-in"
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 1, "slug": "proj"}
        mock_store.get_work_item.return_value = {
            "id": 42,
            "project_id": 1,
            "status": "in_progress",
            "metadata_json": {},
        }
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            stop_work_item("proj", 42)
