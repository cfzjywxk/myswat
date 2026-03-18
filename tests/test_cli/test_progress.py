"""Tests for myswat.cli.progress."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from rich.text import Text

from myswat.agents.base import AgentResponse
from myswat.cli.progress import _build_task_snapshot_display, _run_with_task_monitor, _send_with_timer


class _FakeLive:
    def __init__(self, *args, **kwargs):
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, renderable):
        self.updates.append(renderable)


def test_build_task_snapshot_display_includes_summary_and_todos():
    renderable = _build_task_snapshot_display(
        proj={"slug": "myswat"},
        work_item_id=42,
        item={"status": "approved"},
        state={
            "current_stage": "review_loop_approved",
            "latest_summary": "Finalized WORK_MODE_DESIGN.md as the approved implementation plan.",
            "next_todos": ["Reopen the architect session", "Hand off the final doc"],
            "open_issues": ["Confirm rollout notes wording"],
        },
    )

    assert renderable.plain
    assert "Current task" in renderable.plain
    assert "Work item: #42" in renderable.plain
    assert "Stage: review_loop_approved" in renderable.plain
    assert "Summary: Finalized WORK_MODE_DESIGN.md" in renderable.plain
    assert "Reopen the architect session" in renderable.plain
    assert "Confirm rollout notes wording" in renderable.plain


def test_run_with_task_monitor_prints_final_snapshot_after_worker_finishes():
    console = MagicMock()
    store = MagicMock()
    store.get_work_item.return_value = {"id": 42, "status": "approved"}
    store.get_work_item_state.return_value = {
        "current_stage": "review_loop_approved",
        "latest_summary": "Approved the finalized work-mode design.",
        "next_todos": ["Resume architect chat"],
    }

    with patch("myswat.cli.progress.Live", _FakeLive):
        result = _run_with_task_monitor(
            console=console,
            store=store,
            proj={"slug": "myswat"},
            label="Running dev+QA review loop",
            worker_fn=lambda: "done",
            work_item_ref={"id": 42},
            cancel_targets=[],
            cancel_event=threading.Event(),
        )

    assert result == "done"

    printed = [call.args[0] for call in console.print.call_args_list if call.args]
    assert any(isinstance(arg, Text) and "Current task" in arg.plain for arg in printed)
    assert any(isinstance(arg, Text) and "Approved the finalized work-mode design." in arg.plain for arg in printed)


class _SyncThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._alive = False

    def start(self):
        self._alive = True
        try:
            if self._target is not None:
                self._target()
        finally:
            self._alive = False

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self._alive


def test_send_with_timer_clears_live_output_before_worker_runs():
    runner = MagicMock()
    runner._cleared = False

    def clear_live_output():
        runner._cleared = True

    runner.clear_live_output.side_effect = clear_live_output
    runner.live_output = ["stale line"]
    runner.cancel.return_value = None

    sm = MagicMock()
    sm._runner = runner

    def send(prompt, task_description=None):
        assert runner._cleared is True
        return AgentResponse(content="ok")

    sm.send.side_effect = send

    console = MagicMock()
    with patch("myswat.cli.progress.threading.Thread", _SyncThread):
        with patch("myswat.cli.progress.Live", _FakeLive):
            with patch("myswat.cli.progress.termios.tcgetattr", side_effect=Exception("no tty")):
                response, elapsed = _send_with_timer(console, sm, "hello")

    assert response.content == "ok"
    assert elapsed >= 0
    runner.clear_live_output.assert_called_once()
    sm.send.assert_called_once_with("hello", task_description=None)
