"""Additional tests for myswat.cli.progress helper branches."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from rich.text import Text

from myswat.agents.base import AgentResponse
from myswat.cli.progress import (
    TaskMonitorPromptBridge,
    _build_live_display,
    _build_task_monitor_display,
    _check_esc,
    _coerce_live_lines,
    _collapse_text,
    _describe_process_event,
    _fmt_duration,
    _load_task_monitor_snapshot,
    _preview_text,
    _print_task_monitor_snapshot,
    _run_with_task_monitor,
    _send_with_timer,
)


class _FakeLive:
    def __init__(self, *args, **kwargs):
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, renderable):
        self.updates.append(renderable)


class _ControlledLive(_FakeLive):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_calls = 0
        self.start_calls = []
        self.__class__.instances.append(self)

    def stop(self):
        self.stop_calls += 1

    def start(self, refresh=False):
        self.start_calls.append(refresh)


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


class _IdleThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._alive = False

    def start(self):
        self._alive = True

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self._alive


class _JoinStopsThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._alive = False

    def start(self):
        self._alive = True

    def join(self, timeout=None):
        self._alive = False

    def is_alive(self):
        return self._alive


class _CancelThenFinishThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._alive = False

    def start(self):
        self._alive = True

    def join(self, timeout=None):
        if timeout and timeout >= 1 and self._alive:
            try:
                if self._target is not None:
                    self._target()
            finally:
                self._alive = False

    def is_alive(self):
        return self._alive


def test_check_esc_reads_escape_character():
    with patch("myswat.cli.progress.select.select", return_value=([True], [], [])):
        with patch("myswat.cli.progress.sys.stdin.read", return_value="\x1b"):
            assert _check_esc() is True


def test_fmt_duration_formats_hours():
    assert _fmt_duration(3665) == "1h01m05s"


def test_coerce_live_lines_accepts_list_and_tuple_and_filters_blank():
    assert _coerce_live_lines(["one", "", " two "]) == ["one", " two "]
    assert _coerce_live_lines(("a", " ", 3)) == ["a", "3"]
    assert _coerce_live_lines("not-a-sequence") == []


def test_collapse_and_preview_text_normalize_and_truncate():
    assert _collapse_text("hello \n  world") == "hello world"
    assert _preview_text("alpha beta gamma", limit=8) == "alpha be..."


def test_build_live_display_shows_overflow_prefix():
    live_lines = [f"line {i}" for i in range(12)]
    renderable = _build_live_display(0, 5, live_lines)

    assert isinstance(renderable, Text)
    assert "Waiting for AI agent... (5s)" in renderable.plain
    assert "... (4 earlier lines)" in renderable.plain
    assert "line 11" in renderable.plain


def test_build_live_display_shows_waiting_message_when_empty():
    renderable = _build_live_display(1, 0, [])
    assert "Waiting for the current agent step to finish" in renderable.plain


def test_describe_process_event_variants():
    both = _describe_process_event(
        {"from_role": "dev", "to_role": "qa", "title": "Draft", "summary": "Looks ready"}
    )
    only_to = _describe_process_event({"to_role": "qa", "summary": "Queued"})
    fallback = _describe_process_event({"type": "review_request"})

    assert both == "dev -> qa: Draft: Looks ready"
    assert only_to == "qa: Queued"
    assert fallback == "review_request"


def test_load_task_monitor_snapshot_handles_none_and_non_dict_rows():
    store = MagicMock()
    assert _load_task_monitor_snapshot(store, None) == (None, {}, {})

    store.get_work_item.return_value = "bad"
    store.get_work_item_state.return_value = "bad"
    assert _load_task_monitor_snapshot(store, 7) == (7, {}, {})


def test_build_task_monitor_display_shows_preparing_and_cancellation():
    renderable = _build_task_monitor_display(
        proj={"slug": "proj"},
        work_item_id=None,
        item={},
        state={},
        label="Running workflow",
        frame_idx=2,
        elapsed=3,
        cancel_requested=True,
    )

    assert "Running workflow (3s)" in renderable.plain
    assert "Preparing work item and sessions" in renderable.plain
    assert "Cancellation requested" in renderable.plain
    assert "myswat status -p proj --details" in renderable.plain


def test_build_task_monitor_display_shows_summary_todos_issues_and_flow():
    renderable = _build_task_monitor_display(
        proj={"slug": "proj"},
        work_item_id=42,
        item={"status": "in_progress", "title": "Implement auth"},
        state={
            "current_stage": "reviewing",
            "latest_summary": "Need to validate auth tokens and clean error paths.",
            "next_todos": ["Ask QA to review"],
            "open_issues": ["Missing token expiry case"],
            "process_log": [
                {"from_role": "developer", "to_role": "qa_main", "summary": "Submitted draft"}
            ],
        },
        label="Running workflow",
        frame_idx=0,
        elapsed=2,
    )

    assert "Project: proj  Work item: #42 [in_progress]" in renderable.plain
    assert "Summary: Need to validate auth tokens" in renderable.plain
    assert "Ask QA to review" in renderable.plain
    assert "Missing token expiry case" in renderable.plain
    assert "developer -> qa_main: Submitted draft" in renderable.plain


def test_print_task_monitor_snapshot_skips_when_no_work_item():
    console = MagicMock()
    store = MagicMock()

    _print_task_monitor_snapshot(console, store, {"slug": "proj"}, None)

    console.print.assert_not_called()


def test_print_task_monitor_snapshot_renders_snapshot():
    console = MagicMock()
    store = MagicMock()
    store.get_work_item.return_value = {"status": "completed"}
    store.get_work_item_state.return_value = {"current_stage": "done"}

    _print_task_monitor_snapshot(console, store, {"slug": "proj"}, 42, heading="Snapshot")

    printed = [call.args[0] for call in console.print.call_args_list if call.args]
    assert any(isinstance(arg, Text) and "Snapshot" in arg.plain for arg in printed)


def test_task_monitor_prompt_bridge_falls_back_to_direct_prompt_when_inactive():
    prompt_cb = MagicMock(return_value="y")
    bridge = TaskMonitorPromptBridge(prompt_cb)

    assert bridge.ask("Approve?") == "y"
    prompt_cb.assert_called_once_with("Approve?")


def test_task_monitor_prompt_bridge_service_round_trip():
    prompt_cb = MagicMock(return_value="approved")
    bridge = TaskMonitorPromptBridge(prompt_cb)
    bridge.activate()

    result_holder = {}

    def _worker():
        result_holder["value"] = bridge.ask("Proceed?")

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    for _ in range(20):
        if bridge.has_pending_request():
            break
        threading.Event().wait(0.01)

    assert bridge.service_pending_request() is True
    worker.join(timeout=1)

    assert result_holder["value"] == "approved"
    prompt_cb.assert_called_once_with("Proceed?")
    bridge.deactivate()


def test_run_with_task_monitor_propagates_worker_error():
    console = MagicMock()
    store = MagicMock()

    with patch("myswat.cli.progress.Live", _FakeLive):
        with patch("myswat.cli.progress.termios.tcgetattr", side_effect=Exception("no tty")):
            with pytest.raises(RuntimeError, match="boom"):
                _run_with_task_monitor(
                    console=console,
                    store=store,
                    proj={"slug": "proj"},
                    label="Running workflow",
                    worker_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                    work_item_ref={"id": None},
                    cancel_targets=[],
                    cancel_event=threading.Event(),
                )


def test_run_with_task_monitor_cancels_targets_on_escape():
    console = MagicMock()
    store = MagicMock()
    store.get_work_item.return_value = {"id": 42, "status": "in_progress"}
    store.get_work_item_state.return_value = {"current_stage": "working"}

    runner = MagicMock()

    with patch("myswat.cli.progress.threading.Thread", _JoinStopsThread):
        with patch("myswat.cli.progress.Live", _FakeLive):
            with patch("myswat.cli.progress.sys.stdin.fileno", return_value=0):
                with patch("myswat.cli.progress.termios.tcgetattr", return_value=object()):
                    with patch("myswat.cli.progress.termios.tcsetattr"):
                        with patch("myswat.cli.progress.tty.setcbreak"):
                            with patch("myswat.cli.progress._check_esc", return_value=True):
                                result = _run_with_task_monitor(
                                    console=console,
                                    store=store,
                                    proj={"slug": "proj"},
                                    label="Running workflow",
                                    worker_fn=lambda: "done",
                                    work_item_ref={"id": 42},
                                    cancel_targets=[runner],
                                    cancel_event=threading.Event(),
                                )

    assert result is None
    runner.cancel.assert_called_once()


def test_run_with_task_monitor_services_prompt_bridge_requests():
    console = MagicMock()
    store = MagicMock()
    prompt_cb = MagicMock(return_value="y")
    bridge = TaskMonitorPromptBridge(prompt_cb)
    _ControlledLive.instances.clear()

    with patch("myswat.cli.progress.Live", _ControlledLive):
        with patch("myswat.cli.progress.sys.stdin.fileno", return_value=0):
            with patch("myswat.cli.progress.termios.tcgetattr", return_value=object()):
                with patch("myswat.cli.progress.termios.tcsetattr") as mock_tcsetattr:
                    with patch("myswat.cli.progress.tty.setcbreak"):
                        result = _run_with_task_monitor(
                            console=console,
                            store=store,
                            proj={"slug": "proj"},
                            label="Running workflow",
                            worker_fn=lambda: bridge.ask("Approve?"),
                            work_item_ref={"id": None},
                            cancel_targets=[],
                            cancel_event=threading.Event(),
                            prompt_bridge=bridge,
                        )

    assert result == "y"
    prompt_cb.assert_called_once_with("Approve?")
    assert _ControlledLive.instances
    assert _ControlledLive.instances[0].stop_calls >= 1
    assert mock_tcsetattr.call_count >= 1


def test_send_with_timer_propagates_send_error():
    sm = MagicMock()
    sm._runner = MagicMock()
    sm._runner.clear_live_output.return_value = None
    sm.send.side_effect = RuntimeError("send boom")

    with patch("myswat.cli.progress.threading.Thread", _SyncThread):
        with patch("myswat.cli.progress.Live", _FakeLive):
            with patch("myswat.cli.progress.termios.tcgetattr", side_effect=Exception("no tty")):
                with pytest.raises(RuntimeError, match="send boom"):
                    _send_with_timer(MagicMock(), sm, "hello")


def test_send_with_timer_returns_cancelled_response_when_worker_keeps_running():
    runner = MagicMock()
    runner.live_output = []
    runner.clear_live_output.return_value = None
    runner.cancel.return_value = None

    sm = MagicMock()
    sm._runner = runner

    with patch("myswat.cli.progress.threading.Thread", _IdleThread):
        with patch("myswat.cli.progress.Live", _FakeLive):
            with patch("myswat.cli.progress.sys.stdin.fileno", return_value=0):
                with patch("myswat.cli.progress.termios.tcgetattr", return_value=object()):
                    with patch("myswat.cli.progress.termios.tcsetattr"):
                        with patch("myswat.cli.progress.tty.setcbreak"):
                            with patch("myswat.cli.progress._check_esc", return_value=True):
                                response, _elapsed = _send_with_timer(MagicMock(), sm, "hello")

    assert response.cancelled is True
    assert response.exit_code == -1
    runner.cancel.assert_called_once()


def test_send_with_timer_waits_for_cancelled_worker_result_before_fabricating_response():
    runner = MagicMock()
    runner.live_output = []
    runner.clear_live_output.return_value = None
    runner.cancel.return_value = None

    sm = MagicMock()
    sm._runner = runner
    sm.send.return_value = AgentResponse(
        content="Request cancelled.",
        exit_code=-15,
        cancelled=True,
    )

    with patch("myswat.cli.progress.threading.Thread", _CancelThenFinishThread):
        with patch("myswat.cli.progress.Live", _FakeLive):
            with patch("myswat.cli.progress.sys.stdin.fileno", return_value=0):
                with patch("myswat.cli.progress.termios.tcgetattr", return_value=object()):
                    with patch("myswat.cli.progress.termios.tcsetattr"):
                        with patch("myswat.cli.progress.tty.setcbreak"):
                            with patch("myswat.cli.progress._check_esc", return_value=True):
                                response, _elapsed = _send_with_timer(MagicMock(), sm, "hello")

    assert response.cancelled is True
    assert response.exit_code == -15
    sm.send.assert_called_once_with("hello", task_description=None)
    runner.cancel.assert_called_once()
