"""Extra CLI.main tests for hidden commands and helper branches."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit
from rich.console import Console
from rich.tree import Tree
from typer.testing import CliRunner

from myswat.cli.main import (
    _build_teamwork_flow_entries,
    _format_history_timestamp,
    _parse_verdict_payload,
    _print_message_flow,
    app,
)


def test_search_command_routes_to_hidden_search_handler():
    with patch("myswat.cli.main.run_search_command") as mock_search:
        result = CliRunner().invoke(app, ["search", "auth", "--project", "proj"])

    assert result.exit_code == 0
    mock_search.assert_called_once_with(
        query="auth",
        project="proj",
        category=None,
        source_type=None,
        mode="auto",
        profile="standard",
        limit=10,
        no_vector=False,
        json_output=False,
    )


@patch("myswat.memory.store.MemoryStore")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_gc_command_supports_dry_run(mock_settings_cls, mock_pool_cls, mock_store_cls):
    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = {"id": 1}
    store.gc_compacted_turns.return_value = {"turns_deleted": 5, "sessions_affected": 2}
    mock_store_cls.return_value = store

    result = CliRunner().invoke(app, ["gc", "--project", "proj", "--dry-run"])

    assert result.exit_code == 0
    assert "Would delete 5 turns from 2 sessions" in result.output


@patch("myswat.memory.store.MemoryStore")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_history_command_renders_rows(mock_settings_cls, mock_pool_cls, mock_store_cls):
    settings = MagicMock()
    settings.embedding.tidb_model = "built-in"
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = {"id": 1}
    store.get_recent_turns_global.return_value = [
        {
            "created_at": "2026-03-18 12:00:00",
            "agent_role": "developer",
            "role": "assistant",
            "content": "hello\nworld",
        }
    ]
    mock_store_cls.return_value = store

    result = CliRunner().invoke(app, ["history", "--project", "proj"])

    assert result.exit_code == 0
    assert "[developer] [2026-03-18 12:00:00] assistant: hello world" in result.output


@patch("myswat.db.schema.ensure_schema")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_reset_aborts_when_confirmation_does_not_match(mock_settings_cls, mock_pool_cls, mock_migrations):
    settings = MagicMock()
    settings.tidb.database = "myswat"
    mock_settings_cls.return_value = settings

    with patch("myswat.cli.main.typer.prompt", return_value="nope"):
        with pytest.raises(ClickExit) as exc_info:
            from myswat.cli.main import reset

            reset(project=None, repo_path=None, description=None, yes=False)

    assert exc_info.value.exit_code == 0
    mock_pool_cls.assert_not_called()


@patch("myswat.db.schema.ensure_schema")
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_reset_rejects_unhealthy_connection(mock_settings_cls, mock_pool_cls, mock_migrations):
    settings = MagicMock()
    settings.tidb.database = "myswat"
    mock_settings_cls.return_value = settings
    pool = MagicMock()
    pool.health_check.return_value = False
    mock_pool_cls.return_value = pool

    with pytest.raises(ClickExit):
        from myswat.cli.main import reset

        reset(project=None, repo_path=None, description=None, yes=True)


@patch("myswat.cli.init_cmd.run_init")
@patch("myswat.db.schema.ensure_schema", return_value=["v001"])
@patch("myswat.db.connection.TiDBPool")
@patch("myswat.config.settings.MySwatSettings")
def test_reset_recreates_database_and_optionally_reinits_project(
    mock_settings_cls,
    mock_pool_cls,
    mock_migrations,
    mock_run_init,
):
    settings = MagicMock()
    settings.tidb.database = "myswat"
    mock_settings_cls.return_value = settings
    first_pool = MagicMock()
    first_pool.health_check.return_value = True
    second_pool = MagicMock()
    mock_pool_cls.side_effect = [first_pool, second_pool]

    from myswat.cli.main import reset

    reset(project="proj", repo_path="/tmp/repo", description="desc", yes=True)

    first_pool.execute.assert_called_once()
    mock_run_init.assert_called_once_with("proj", "/tmp/repo", "desc")


def test_format_history_timestamp_formats_datetime_values():
    value = datetime(2026, 3, 18, 12, 34, 56, 789000)

    assert _format_history_timestamp(value) == "2026-03-18 12:34:56"


def test_parse_verdict_payload_handles_dict_string_and_invalid_input():
    assert _parse_verdict_payload({"verdict": "lgtm"}) == {"verdict": "lgtm"}
    assert _parse_verdict_payload('{"verdict":"lgtm"}') == {"verdict": "lgtm"}
    assert _parse_verdict_payload("not json") == {}
    assert _parse_verdict_payload(3) == {}


def test_build_teamwork_flow_entries_prefers_process_log():
    pool = MagicMock()
    item = {
        "id": 7,
        "metadata_json": {
            "task_state": {
                "process_log": [{"type": "task_request"}, "skip-me"],
            }
        },
    }

    assert _build_teamwork_flow_entries(pool, item) == [{"type": "task_request"}]
    pool.fetch_all.assert_not_called()


def test_build_teamwork_flow_entries_reconstructs_from_review_cycles():
    pool = MagicMock()
    pool.fetch_all.return_value = [
        {
            "iteration": 1,
            "verdict": "changes_requested",
            "verdict_json": '{"summary":"Needs fixes","issues":["missing tests"]}',
            "proposer_role": "developer",
            "reviewer_role": "qa_main",
            "artifact_title": "Iteration 1",
            "artifact_type": "proposal",
            "artifact_content": "draft design",
        }
    ]
    item = {"id": 7, "description": "Build auth"}

    entries = _build_teamwork_flow_entries(pool, item)

    assert entries[0]["type"] == "task_request"
    assert entries[1]["type"] == "review_request"
    assert entries[2]["type"] == "review_response"
    assert "missing tests" in entries[2]["summary"]


def test_print_message_flow_handles_reaction_and_summary():
    tree = Tree("root")
    _print_message_flow(
        tree,
        [
            {"type": "review_request", "from_role": "developer", "to_role": "qa_main", "title": "Draft", "summary": "Ready"},
            {"type": "reaction", "from_role": "myswat", "summary": "Ask for revision"},
        ],
    )

    console = Console(record=True, force_terminal=False)
    console.print(tree)
    rendered = console.export_text()
    assert "Draft" in rendered
    assert "Ask for revision" in rendered
