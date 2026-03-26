"""Tests for daemon-backed workflow CLI helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.cli.work_cmd import run_work, stop_work_item
from myswat.server.control_client import DaemonClientError
from myswat.workflow.modes import WorkMode


@patch("myswat.cli.work_cmd.DaemonClient")
@patch("myswat.cli.work_cmd.MySwatSettings")
def test_run_work_submits_through_daemon(mock_settings_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client.base_url = "http://127.0.0.1:8765"
    mock_client.submit_work.return_value = {
        "work_item_id": 41,
        "workers": ["architect", "developer", "qa_main"],
    }
    mock_client_cls.return_value = mock_client

    work_item_id = run_work("proj", "ship feature", mode=WorkMode.full, with_ga_test=True)

    assert work_item_id == 41
    mock_client.submit_work.assert_called_once_with(
        project="proj",
        requirement="ship feature",
        workdir=None,
        mode="full",
        with_ga_test=True,
    )


def test_run_work_rejects_with_ga_test_for_non_full_mode():
    with pytest.raises(typer.BadParameter, match="--with-ga-test"):
        run_work("proj", "ship feature", mode=WorkMode.test, with_ga_test=True)


def test_run_work_rejects_interactive_checkpoints():
    with pytest.raises(typer.BadParameter, match="interactive-checkpoints"):
        run_work("proj", "ship feature", auto_approve=False)


@patch("myswat.cli.work_cmd.DaemonClient")
@patch("myswat.cli.work_cmd.MySwatSettings")
def test_run_work_resume_submits_existing_work_item(mock_settings_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client.base_url = "http://127.0.0.1:8765"
    mock_client.submit_work.return_value = {
        "work_item_id": 7,
        "workers": ["architect", "developer", "qa_main"],
    }
    mock_client_cls.return_value = mock_client

    work_item_id = run_work("proj", "", resume=7)

    assert work_item_id == 7
    mock_client.submit_work.assert_called_once_with(
        project="proj",
        requirement="",
        workdir=None,
        mode="full",
        resume_work_item_id=7,
    )


@patch("myswat.cli.work_cmd.console.print")
@patch("myswat.cli.work_cmd.DaemonClient")
@patch("myswat.cli.work_cmd.MySwatSettings")
def test_run_work_surfaces_daemon_error(mock_settings_cls, mock_client_cls, mock_print):
    mock_client = MagicMock()
    mock_client.submit_work.side_effect = DaemonClientError(
        "MySwat daemon is unavailable at http://127.0.0.1:8765: <urlopen error [Errno 111] connection refused>",
        retryable=True,
    )
    mock_client_cls.return_value = mock_client

    with pytest.raises(ClickExit):
        run_work("proj", "ship feature")

    rendered = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "unavailable" in rendered
    assert "myswat server" in rendered


@patch("myswat.cli.work_cmd.console.print")
@patch("myswat.cli.work_cmd.DaemonClient")
@patch("myswat.cli.work_cmd.MySwatSettings")
def test_run_work_timeout_does_not_print_start_hint(mock_settings_cls, mock_client_cls, mock_print):
    mock_client = MagicMock()
    mock_client.submit_work.side_effect = DaemonClientError(
        "MySwat daemon request timed out: POST http://127.0.0.1:8765/api/work",
        retryable=True,
    )
    mock_client_cls.return_value = mock_client

    with pytest.raises(ClickExit):
        run_work("proj", "ship feature")

    rendered = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "timed out" in rendered
    assert "still in progress or blocked" in rendered
    assert "myswat server" not in rendered


@patch("myswat.cli.work_cmd.console.print")
@patch("myswat.cli.work_cmd.DaemonClient")
@patch("myswat.cli.work_cmd.MySwatSettings")
def test_stop_work_item_cancels_through_daemon(mock_settings_cls, mock_client_cls, mock_print):
    mock_client = MagicMock()
    mock_client.control_work.return_value = {"work_item_id": 99}
    mock_client_cls.return_value = mock_client

    stop_work_item("proj", 99)

    mock_client.control_work.assert_called_once_with(
        project="proj",
        work_item_id=99,
        action="cancel",
    )
    assert any("Cancellation requested" in str(call.args[0]) for call in mock_print.call_args_list if call.args)


@patch("myswat.cli.work_cmd.console.print")
@patch("myswat.cli.work_cmd.DaemonClient")
@patch("myswat.cli.work_cmd.MySwatSettings")
def test_stop_work_item_requires_daemon(mock_settings_cls, mock_client_cls, mock_print):
    mock_client = MagicMock()
    mock_client.control_work.side_effect = DaemonClientError(
        "MySwat daemon is unavailable at http://127.0.0.1:8765: <urlopen error [Errno 111] connection refused>",
        retryable=True,
    )
    mock_client_cls.return_value = mock_client

    with pytest.raises(ClickExit):
        stop_work_item("proj", 99)

    rendered = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "unavailable" in rendered
    assert "myswat server" in rendered
