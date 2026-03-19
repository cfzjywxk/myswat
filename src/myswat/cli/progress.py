"""Shared terminal progress helpers for CLI commands."""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.live import Live
from rich.text import Text

from myswat.agents.base import AgentResponse, AgentRunner

if TYPE_CHECKING:
    from myswat.agents.session_manager import SessionManager
    from myswat.cli.workflow_display import WorkflowDisplay
    from myswat.memory.store import MemoryStore


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MAX_LIVE_LINES = 8
_TASK_MONITOR_SUMMARY_CHARS = 220
_TASK_MONITOR_PROCESS_LINES = 4
_TASK_MONITOR_REFRESH_PER_SECOND = 8
_TASK_MONITOR_LOOP_INTERVAL = 0.1
_TASK_MONITOR_STORE_POLL_SECONDS = 1.0
_SEND_CANCEL_GRACE_SECONDS = 5.0


class TaskMonitorPromptBridge:
    """Route workflow checkpoint prompts from a worker thread to the main thread."""

    def __init__(self, prompt_callback: Callable[[str], str]) -> None:
        self._prompt_callback = prompt_callback
        self._lock = threading.Lock()
        self._request_ready = threading.Event()
        self._response_ready = threading.Event()
        self._active = False
        self._prompt_text = ""
        self._response_text = ""

    def activate(self) -> None:
        with self._lock:
            self._active = True

    def deactivate(self, default_response: str = "n") -> None:
        with self._lock:
            self._active = False
            if self._request_ready.is_set():
                self._response_text = default_response
                self._request_ready.clear()
                self._response_ready.set()

    def ask(self, prompt_text: str) -> str:
        with self._lock:
            active = self._active
        if not active:
            return self._prompt_callback(prompt_text)

        with self._lock:
            self._prompt_text = prompt_text
            self._response_text = ""
            self._response_ready.clear()
            self._request_ready.set()

        self._response_ready.wait()
        with self._lock:
            response = self._response_text
            self._prompt_text = ""
        return response

    def has_pending_request(self) -> bool:
        return self._request_ready.is_set()

    def service_pending_request(self) -> bool:
        with self._lock:
            if not self._request_ready.is_set():
                return False
            prompt_text = self._prompt_text

        try:
            response = self._prompt_callback(prompt_text)
        except Exception:
            response = "n"

        with self._lock:
            self._response_text = response
            self._request_ready.clear()
            self._response_ready.set()
        return True


def _check_esc() -> bool:
    """Non-blocking check if ESC key was pressed."""
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                return True
    except Exception:
        pass
    return False


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _coerce_live_lines(value: object) -> list[str]:
    """Normalize runner live output into a list of non-empty strings."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    else:
        return []
    return [str(line) for line in items if str(line).strip()]


def _collapse_text(text: str | None) -> str:
    """Collapse arbitrary text into a single whitespace-normalized line."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _preview_text(text: str | None, limit: int = _TASK_MONITOR_SUMMARY_CHARS) -> str:
    """Collapse whitespace and trim text to a single preview line."""
    collapsed = _collapse_text(text)
    if len(collapsed) > limit:
        return collapsed[:limit] + "..."
    return collapsed


def _build_live_display(
    frame_idx: int,
    elapsed: float,
    live_lines: list[str],
) -> Text:
    """Build the Rich Text renderable for simple live agent output."""
    frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
    text = Text(
        f" {frame}  Waiting for AI agent... ({_fmt_duration(elapsed)})  ESC to cancel\n",
        style="bold cyan",
    )

    tail = live_lines[-_MAX_LIVE_LINES:] if live_lines else []
    if len(live_lines) > _MAX_LIVE_LINES:
        tail = [f"  ... ({len(live_lines) - _MAX_LIVE_LINES} earlier lines)"] + tail

    if tail:
        body = "\n".join(f"  {line}" for line in tail)
        text.append(body + "\n", style="dim")
    else:
        text.append("  Waiting for the current agent step to finish...\n", style="dim")
    return text


def _single_line_preview(text: str | None, limit: int = _TASK_MONITOR_SUMMARY_CHARS) -> str:
    return _preview_text(text, limit)


def _describe_process_event(event: dict, summary_limit: int = 120) -> str:
    from_role = _single_line_preview(event.get("from_role"), 30)
    to_role = _single_line_preview(event.get("to_role"), 30)
    title = _single_line_preview(event.get("title"), 80)
    summary = _single_line_preview(event.get("summary"), summary_limit)

    if from_role and to_role:
        prefix = f"{from_role} -> {to_role}"
    elif from_role:
        prefix = from_role
    elif to_role:
        prefix = to_role
    else:
        prefix = _single_line_preview(event.get("type"), 30) or "event"

    details = title or summary
    if title and summary and summary != title:
        details = f"{title}: {summary}"
    return f"{prefix}: {details}" if details else prefix


def _load_task_monitor_snapshot(
    store: "MemoryStore",
    work_item_id: int | None,
) -> tuple[int | None, dict, dict]:
    """Fetch the latest work-item snapshot for the task monitor."""
    if work_item_id is None:
        return None, {}, {}

    item = store.get_work_item(work_item_id) or {}
    if not isinstance(item, dict):
        item = {}
    state = store.get_work_item_state(work_item_id) or {}
    if not isinstance(state, dict):
        state = {}
    return work_item_id, item, state


def _build_task_monitor_display(
    proj: dict,
    work_item_id: int | None,
    item: dict,
    state: dict,
    label: str,
    frame_idx: int,
    elapsed: float,
    cancel_requested: bool = False,
) -> Text:
    """Build the Rich Text renderable for workflow/review monitoring."""
    frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
    text = Text(
        f" {frame}  {label} ({_fmt_duration(elapsed)})  ESC to cancel current step\n",
        style="bold cyan",
    )

    if work_item_id is None:
        text.append("  Preparing work item and sessions...\n", style="dim")
    else:
        status = item.get("status", "unknown")
        title = _single_line_preview(item.get("title") or f"Work item {work_item_id}", 80)
        stage = _single_line_preview(state.get("current_stage"), 80) or "starting"

        text.append(f"  Project: {proj['slug']}  Work item: #{work_item_id} [{status}]\n", style="bold")
        text.append(f"  Title: {title}\n", style="bold")
        text.append(f"  Stage: {stage}\n")

        summary = _single_line_preview(state.get("latest_summary"))
        if summary:
            text.append(f"  Summary: {summary}\n", style="dim")

        next_todos = state.get("next_todos") or []
        if next_todos:
            text.append("  Next:\n", style="green")
            for todo in next_todos[:3]:
                text.append(f"    - {_single_line_preview(todo, 100)}\n", style="green")

        open_issues = state.get("open_issues") or []
        if open_issues:
            text.append("  Open issues:\n", style="yellow")
            for issue in open_issues[:3]:
                text.append(f"    - {_single_line_preview(issue, 100)}\n", style="yellow")

        process_log = state.get("process_log") or []
        if process_log:
            text.append("  Flow:\n", style="cyan")
            for event in process_log[-_TASK_MONITOR_PROCESS_LINES:]:
                if isinstance(event, dict):
                    text.append(f"    - {_describe_process_event(event, 90)}\n", style="cyan")

    if cancel_requested:
        text.append(
            "  Cancellation requested. Waiting for the current agent step to stop...\n",
            style="bold yellow",
        )

    text.append("\n  Query from another terminal:\n", style="bold")
    text.append(f"    myswat status -p {proj['slug']}\n", style="dim")
    text.append(f"    myswat status -p {proj['slug']} --details\n", style="dim")
    return text


def _build_task_snapshot_display(
    proj: dict,
    work_item_id: int | None,
    item: dict,
    state: dict,
    heading: str = "Current task",
) -> Text:
    """Build a stable post-run snapshot of the current work item."""
    text = Text(f" {heading}\n", style="bold")

    if work_item_id is None:
        text.append("  Work item not created yet.\n", style="dim")
        return text

    text.append(f"  Project: {proj['slug']}  Work item: #{work_item_id}\n", style="bold")

    status = _single_line_preview(item.get("status"), 40)
    if status and status != "in_progress":
        text.append(f"  Status: {status}\n")

    stage = _single_line_preview(state.get("current_stage"), 80)
    if stage:
        text.append(f"  Stage: {stage}\n")

    summary = _single_line_preview(state.get("latest_summary"))
    if summary:
        text.append(f"  Summary: {summary}\n", style="dim")

    next_todos = state.get("next_todos") or []
    if next_todos:
        text.append("  Next:\n", style="green")
        for todo in next_todos[:3]:
            text.append(f"    - {_single_line_preview(todo, 100)}\n", style="green")

    open_issues = state.get("open_issues") or []
    if open_issues:
        text.append("  Open issues:\n", style="yellow")
        for issue in open_issues[:3]:
            text.append(f"    - {_single_line_preview(issue, 100)}\n", style="yellow")

    text.append("\n  Query from another terminal:\n", style="bold")
    text.append(f"    myswat status -p {proj['slug']}\n", style="dim")
    text.append(f"    myswat status -p {proj['slug']} --details\n", style="dim")
    return text


def _print_task_monitor_snapshot(
    console: Console,
    store: "MemoryStore",
    proj: dict,
    work_item_id: int | None,
    heading: str = "Current task",
) -> None:
    """Print a non-transient snapshot after task monitor exit."""
    current_id, item, state = _load_task_monitor_snapshot(store, work_item_id)
    if current_id is None:
        return

    console.print()
    console.print(
        _build_task_snapshot_display(
            proj=proj,
            work_item_id=current_id,
            item=item,
            state=state,
            heading=heading,
        )
    )


def _run_with_task_monitor(
    console: Console,
    store: "MemoryStore",
    proj: dict,
    label: str,
    worker_fn: Callable[[], object | None],
    work_item_ref: dict[str, int | None],
    cancel_targets: list[AgentRunner],
    cancel_event: threading.Event,
    workflow_display: "WorkflowDisplay | None" = None,
    prompt_bridge: TaskMonitorPromptBridge | None = None,
) -> object | None:
    """Run a long task while showing work-item progress.

    When *workflow_display* is provided the live display is driven by
    structured events from the engine (no DB polling needed).  Otherwise
    the legacy DB-polling display is used.
    """
    result = [None]
    error = [None]
    start = time.monotonic()
    use_event_display = workflow_display is not None

    # -- DB-polling state (only used when workflow_display is None) --
    snapshot_lock = threading.Lock()
    snapshot: dict = {
        "work_item_id": None,
        "item": {},
        "state": {},
    }
    stop_polling = threading.Event()

    def _run():
        try:
            result[0] = worker_fn()
        except Exception as exc:
            error[0] = exc

    def _poll_task_state():
        last_refresh_at = 0.0
        last_work_item_id = object()

        while not stop_polling.is_set():
            try:
                work_item_id = work_item_ref.get("id")
                now = time.monotonic()
                should_refresh = (
                    work_item_id != last_work_item_id
                    or now - last_refresh_at >= _TASK_MONITOR_STORE_POLL_SECONDS
                )

                if should_refresh:
                    current_id, item, state = _load_task_monitor_snapshot(store, work_item_id)
                    with snapshot_lock:
                        snapshot["work_item_id"] = current_id
                        snapshot["item"] = item
                        snapshot["state"] = state
                    last_work_item_id = work_item_id
                    last_refresh_at = now
            except Exception:
                # Keep the last good snapshot and retry on the next poll cycle.
                pass

            stop_polling.wait(_TASK_MONITOR_LOOP_INTERVAL)

    if prompt_bridge is not None:
        prompt_bridge.activate()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    poller: threading.Thread | None = None
    if not use_event_display:
        poller = threading.Thread(target=_poll_task_state, daemon=True)
        poller.start()

    fd = None
    old_settings = None
    use_cbreak = False
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        use_cbreak = True
    except Exception:
        use_cbreak = False

    frame_idx = 0
    try:
        if use_cbreak and fd is not None:
            tty.setcbreak(fd)
        with Live(
            console=console,
            refresh_per_second=_TASK_MONITOR_REFRESH_PER_SECOND,
            transient=True,
        ) as live:
            while worker.is_alive():
                if prompt_bridge is not None and prompt_bridge.has_pending_request():
                    if use_cbreak and fd is not None and old_settings is not None:
                        termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
                    if hasattr(live, "stop"):
                        live.stop()
                    prompt_bridge.service_pending_request()
                    if worker.is_alive() and hasattr(live, "start"):
                        live.start(refresh=True)
                    if use_cbreak and fd is not None:
                        tty.setcbreak(fd)
                    continue

                elapsed = time.monotonic() - start

                if use_event_display:
                    live.update(
                        workflow_display.build_live_renderable(
                            proj_slug=proj["slug"],
                            work_item_id=work_item_ref.get("id"),
                            frame_idx=frame_idx,
                            elapsed=elapsed,
                            cancel_requested=cancel_event.is_set(),
                        )
                    )
                else:
                    with snapshot_lock:
                        current_id = snapshot["work_item_id"]
                        item = dict(snapshot["item"])
                        state = dict(snapshot["state"])
                    live.update(
                        _build_task_monitor_display(
                            proj=proj,
                            work_item_id=current_id,
                            item=item,
                            state=state,
                            label=label,
                            frame_idx=frame_idx,
                            elapsed=elapsed,
                            cancel_requested=cancel_event.is_set(),
                        )
                    )
                frame_idx += 1

                if use_cbreak and _check_esc() and not cancel_event.is_set():
                    cancel_event.set()
                    for runner in cancel_targets:
                        try:
                            runner.cancel()
                        except Exception:
                            pass
                    break

                worker.join(timeout=_TASK_MONITOR_LOOP_INTERVAL)

            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
    except KeyboardInterrupt:
        if not cancel_event.is_set():
            cancel_event.set()
            for runner in cancel_targets:
                try:
                    runner.cancel()
                except Exception:
                    pass
    finally:
        stop_polling.set()
        if poller is not None:
            poller.join(timeout=1)
        if prompt_bridge is not None:
            prompt_bridge.deactivate()
        try:
            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
        except Exception:
            pass

    # Post-run snapshot
    try:
        if use_event_display:
            console.print()
            console.print(
                workflow_display.build_final_snapshot(
                    proj_slug=proj["slug"],
                    work_item_id=work_item_ref.get("id"),
                )
            )
        else:
            _print_task_monitor_snapshot(
                console=console,
                store=store,
                proj=proj,
                work_item_id=work_item_ref.get("id"),
            )
    except Exception:
        # Best-effort snapshot only; do not mask the task outcome.
        pass

    if error[0] is not None:
        raise error[0]
    return result[0]


def _send_with_timer(
    console: Console,
    sm: "SessionManager",
    prompt: str,
    task_description: str | None = None,
) -> tuple[AgentResponse, float]:
    """Send a message to the agent while showing live output and elapsed time."""
    result = [None]
    error = [None]
    start = time.monotonic()

    def _run():
        try:
            result[0] = sm.send(prompt, task_description=task_description)
        except Exception as exc:
            error[0] = exc

    # Clear stale live output from the previous invocation before the worker
    # can start appending fresh lines. Clearing after worker.start() would fix
    # the stale-render race but can also erase the new invocation's first lines.
    sm._runner.clear_live_output()
    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    fd = None
    old_settings = None
    use_cbreak = False
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        use_cbreak = True
    except Exception:
        use_cbreak = False

    cancelled = False
    frame_idx = 0
    try:
        if use_cbreak and fd is not None:
            tty.setcbreak(fd)
        with Live(
            console=console,
            refresh_per_second=_TASK_MONITOR_REFRESH_PER_SECOND,
            transient=True,
        ) as live:
            while worker.is_alive():
                elapsed = time.monotonic() - start
                live_lines = _coerce_live_lines(sm._runner.live_output)
                live.update(_build_live_display(frame_idx, elapsed, live_lines))
                frame_idx += 1

                if use_cbreak and _check_esc():
                    cancelled = True
                    sm._runner.cancel()
                    break

                worker.join(timeout=_TASK_MONITOR_LOOP_INTERVAL)

            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
    except KeyboardInterrupt:
        cancelled = True
        sm._runner.cancel()
    finally:
        try:
            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
        except Exception:
            pass

    elapsed = time.monotonic() - start

    if error[0] is not None:
        raise error[0]

    if cancelled and result[0] is None:
        worker.join(timeout=_SEND_CANCEL_GRACE_SECONDS)
        elapsed = time.monotonic() - start
        if error[0] is not None:
            raise error[0]
        if result[0] is not None:
            return result[0], elapsed
        return AgentResponse(content="Request cancelled.", exit_code=-1, cancelled=True), elapsed

    return result[0], elapsed
