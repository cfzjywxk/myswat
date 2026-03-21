"""Additional coverage-focused tests for myswat.cli.main."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from unittest.mock import MagicMock, patch

import pytest
import pymysql.err
import typer
from click.exceptions import Exit as ClickExit
from rich.console import Console
from typer.testing import CliRunner

from myswat.cli.main import (
    _build_teamwork_flow_entries,
    _format_timestamp_short,
    _format_history_timestamp,
    _infer_stage_labels,
    _print_message_flow,
    _print_task_state,
    _runtime_health,
    _select_status_flow_item,
    app,
)


def test_work_command_requires_requirement_without_resume():
    from myswat.cli.main import work

    with pytest.raises(typer.BadParameter, match="Requirement is required"):
        work(
            project="proj",
            requirement=None,
            follow=False,
            background=False,
            resume=None,
            design_mode=False,
            develop_mode=False,
            test_mode=False,
            auto_approve=False,
            workdir=None,
        )


def test_work_command_rejects_requirement_with_resume():
    from myswat.cli.main import work

    with pytest.raises(typer.BadParameter, match="Cannot provide a new requirement with --resume"):
        work(
            project="proj",
            requirement="new req",
            follow=False,
            background=False,
            resume=7,
            design_mode=False,
            develop_mode=False,
            test_mode=False,
            auto_approve=False,
            workdir=None,
        )


def test_work_command_rejects_resume_with_follow():
    from myswat.cli.main import work

    with pytest.raises(typer.BadParameter, match="--resume cannot be combined with --follow"):
        work(
            project="proj",
            requirement=None,
            follow=True,
            background=False,
            resume=7,
            design_mode=False,
            develop_mode=False,
            test_mode=False,
            auto_approve=False,
            workdir=None,
        )


def test_format_history_timestamp_falls_back_to_plain_isoformat():
    assert _format_history_timestamp(date(2026, 3, 18)) == "2026-03-18"


@patch("myswat.memory.store.MemoryStore")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_gc_command_rejects_unknown_project(mock_settings_cls, mock_pool_cls, mock_store_cls):
    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = None
    mock_store_cls.return_value = store

    result = CliRunner().invoke(app, ["gc", "--project", "missing"])

    assert result.exit_code == 1
    assert "Project 'missing' not found." in result.stdout


@patch("myswat.memory.store.MemoryStore")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_history_command_handles_missing_project_and_empty_rows(
    mock_settings_cls,
    mock_pool_cls,
    mock_store_cls,
):
    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.side_effect = [None, {"id": 1}]
    store.get_recent_turns_global.return_value = []
    mock_store_cls.return_value = store
    runner = CliRunner()

    missing = runner.invoke(app, ["history", "--project", "missing"])
    empty = runner.invoke(app, ["history", "--project", "proj"])

    assert missing.exit_code == 1
    assert "Project 'missing' not found." in missing.stdout
    assert empty.exit_code == 0
    assert "No recent turns found." in empty.stdout


@patch("myswat.db.schema.ensure_schema", return_value=["v001"])
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_reset_without_project_prints_reinit_hint(mock_settings_cls, mock_pool_cls, mock_migrations):
    settings = MagicMock()
    settings.tidb.database = "myswat"
    mock_settings_cls.return_value = settings
    first_pool = MagicMock()
    first_pool.health_check.return_value = True
    second_pool = MagicMock()
    mock_pool_cls.side_effect = [first_pool, second_pool]

    result = CliRunner().invoke(app, ["reset", "--yes"])

    assert result.exit_code == 0
    assert "Database schema is ready. Run 'myswat init <name>' to create a project." in result.stdout


def test_build_teamwork_flow_entries_returns_empty_without_rows():
    pool = MagicMock()
    pool.fetch_all.return_value = []

    assert _build_teamwork_flow_entries(pool, {"id": 7, "description": ""}) == []


def test_build_teamwork_flow_entries_uses_description_as_requirement_when_no_process_log():
    pool = MagicMock()
    pool.fetch_all.return_value = []

    entries = _build_teamwork_flow_entries(
        pool,
        {
            "id": 7,
            "created_at": "2026-03-21T22:51:31",
            "description": "Implement the feature",
            "metadata_json": {"task_state": {"process_log": []}},
        },
    )

    assert entries == [
        {
            "at": "2026-03-21T22:51:31",
            "type": "task_request",
            "from_role": "user",
            "to_role": "myswat",
            "title": "Requirement",
            "summary": "Implement the feature",
            "_sequence": -1,
        },
    ]


def test_select_status_flow_item_falls_back_to_description_before_last_item():
    items = [
        {"id": 1, "description": "", "metadata_json": {"task_state": {"process_log": []}}},
        {"id": 2, "description": "Implement the feature", "metadata_json": {"task_state": {"process_log": []}}},
        {"id": 3, "description": "", "metadata_json": {"task_state": {"process_log": []}}},
    ]

    assert _select_status_flow_item(items)["id"] == 2


def test_print_message_flow_renders_timeline_panel():
    output = Console(record=True, force_terminal=False, width=120)

    _print_message_flow(
        output,
        [
            "skip-me",
            {
                "type": "review_verdict",
                "verdict": "changes_requested",
                "from_role": "developer",
                "to_role": "architect",
                "summary": "Need more detail",
                "at": "2026-03-18T10:00:00",
            },
        ],
    )

    rendered = output.export_text()
    assert "Message Flow" in rendered
    assert "Developer -> Architect" in rendered
    assert "REQUEST CHANGES" in rendered


def test_format_timestamp_short_normalizes_db_utc_datetimes_but_keeps_local_strings():
    with patch("myswat.cli.main._local_timezone", return_value=timezone(timedelta(hours=8))):
        assert _format_timestamp_short(datetime(2026, 3, 20, 0, 31, 50)) == "03-20 08:31:50"
        assert _format_timestamp_short("2026-03-20T08:25:59") == "03-20 08:25:59"


def test_runtime_health_does_not_mark_recent_utc_heartbeat_as_stalled():
    now_utc = datetime.now(timezone.utc)
    runtime = type(
        "Runtime",
        (),
        {
            "status": "online",
            "last_heartbeat_at": (now_utc - timedelta(seconds=30)).replace(tzinfo=None),
            "lease_expires_at": (now_utc + timedelta(seconds=240)).replace(tzinfo=None),
            "metadata_json": {},
        },
    )()

    health, style = _runtime_health(runtime)

    assert (health, style) == ("healthy", "green")


def test_print_task_state_renders_background_and_open_issues():
    output = []

    class _Console:
        def print(self, message=""):
            output.append(str(message))

    _print_task_state(
        _Console(),
        {
            "metadata_json": {
                "work_mode": "develop",
                "background": {
                    "mode": "test",
                    "state": "running",
                    "pid": 1234,
                    "log_path": "/tmp/work.log",
                    "requested_at": "2026-03-18T10:00:00",
                    "started_at": "2026-03-18T10:01:00",
                    "finished_at": "2026-03-18T10:02:00",
                },
                "task_state": {
                    "current_stage": "phase_2",
                    "latest_summary": "working",
                    "next_todos": ["ship"],
                    "open_issues": ["retry"],
                },
            }
        },
    )

    rendered = "\n".join(output)
    assert "Execution:" in rendered
    assert "Finished:" in rendered
    assert "Open issues:" in rendered


def test_print_task_state_prints_blank_line_for_background_only():
    output = []

    class _Console:
        def print(self, message=""):
            output.append(str(message))

    _print_task_state(_Console(), {"metadata_json": {"background": {"state": "running"}}})

    assert output[-1] == ""


def test_infer_stage_labels_uses_generic_fallback():
    labels = _infer_stage_labels([{"proposer_role": "analyst", "reviewer_role": "operator"}])

    assert labels == ["Review (analyst → operator)"]


@patch("myswat.config.settings.MySwatSettings")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.memory.store.MemoryStore")
def test_status_handles_tidb_unreachable_and_local_cache(
    mock_store_cls,
    mock_pool_cls,
    mock_settings_cls,
    tmp_path,
):
    settings = MagicMock()
    settings.tidb.host = "tidb.example"
    settings.tidb.port = 4000
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.side_effect = pymysql.err.OperationalError(2003, "boom")
    mock_store_cls.return_value = store
    runner = CliRunner()
    repo_cache = tmp_path / "myswat.md"
    repo_cache.write_text("cached", encoding="ascii")

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["status", "--project", "proj"])

    assert result.exit_code == 1
    assert "TiDB is unreachable from this environment." in result.stdout
    assert str(repo_cache) in result.stdout


@patch("myswat.config.settings.MySwatSettings")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.memory.store.MemoryStore")
def test_status_renders_session_truncation_short_elapsed_and_invalid_metadata(
    mock_store_cls,
    mock_pool_cls,
    mock_settings_cls,
):
    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = {"id": 1, "slug": "proj", "name": "Proj"}
    store.list_agents.return_value = []
    store.list_work_items.return_value = []
    store.count_session_turns.return_value = 2
    mock_store_cls.return_value = store
    long_pending = "x" * 210
    long_content = "y" * 160
    pool = MagicMock()
    pool.fetch_all.side_effect = [
        [
            {"session_uuid": "uuid-12345678", "display_name": "Dev", "id": 1, "purpose": "doing things", "token_count_est": 99},
            {"session_uuid": "uuid-87654321", "display_name": "QA", "id": 2, "purpose": "waiting", "token_count_est": 12},
        ],
        [],
        [
            {"role": "assistant", "content": long_content, "metadata_json": json.dumps({"elapsed_seconds": 59}), "created_at": "2026-03-18"},
            {"role": "user", "content": "hi", "metadata_json": None, "created_at": "2026-03-18"},
            {"role": "assistant", "content": "broken", "metadata_json": "{", "created_at": "2026-03-18"},
        ],
    ]
    pool.fetch_one.side_effect = [
        {"role": "user"},
        {"content": long_pending},
        {"role": "assistant"},
        {"cnt": 0},
        {"cnt": 0},
    ]
    mock_pool_cls.return_value = pool

    result = CliRunner().invoke(app, ["status", "--project", "proj", "--details"])

    assert result.exit_code == 0
    assert "(59s)" in result.stdout
    assert "..." in result.stdout


@patch("myswat.config.settings.MySwatSettings")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.memory.store.MemoryStore")
def test_status_overview_handles_terminal_item_without_flow_data(
    mock_store_cls,
    mock_pool_cls,
    mock_settings_cls,
):
    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = {"id": 1, "slug": "proj", "name": "Proj"}
    store.list_work_items.return_value = [
        {
            "id": 1,
            "status": "completed",
            "item_type": "code_change",
            "title": "Completed task",
            "description": "",
            "metadata_json": {"task_state": {"current_stage": "report", "process_log": []}},
        },
    ]
    store.list_runtime_registrations.return_value = []
    mock_store_cls.return_value = store
    pool = MagicMock()
    pool.fetch_all.return_value = []
    mock_pool_cls.return_value = pool

    result = CliRunner().invoke(app, ["status", "--project", "proj"])

    assert result.exit_code == 0
    assert "No active work item." in result.stdout
    assert "Recent Messages" in result.stdout
    assert "No recorded message flow yet." in result.stdout


@patch("myswat.cli.main._print_teamwork_details")
@patch("myswat.config.settings.MySwatSettings")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.memory.store.MemoryStore")
def test_task_command_without_description_still_prints_state(
    mock_store_cls,
    mock_pool_cls,
    mock_settings_cls,
    mock_teamwork,
):
    store = MagicMock()
    store.get_project_by_slug.return_value = {"id": 1, "slug": "proj", "name": "Proj"}
    store.get_work_item.return_value = {
        "id": 7,
        "project_id": 1,
        "status": "in_progress",
        "item_type": "code_change",
        "title": "Implement feature",
        "description": "",
        "metadata_json": {},
    }
    mock_store_cls.return_value = store

    result = CliRunner().invoke(app, ["task", "7", "--project", "proj"])

    assert result.exit_code == 0
    mock_teamwork.assert_called_once()
