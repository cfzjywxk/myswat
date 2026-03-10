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
    from myswat.memory.store import MemoryStore


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MAX_LIVE_LINES = 8
_TASK_MONITOR_SUMMARY_CHARS = 220


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
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[:limit] + "..."
    return collapsed


def _build_task_monitor_display(
    store: "MemoryStore",
    proj: dict,
    work_item_id: int | None,
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
        item = store.get_work_item(work_item_id) or {}
        if not isinstance(item, dict):
            item = {}
        state = store.get_work_item_state(work_item_id) or {}
        if not isinstance(state, dict):
            state = {}

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

    if cancel_requested:
        text.append(
            "  Cancellation requested. Waiting for the current agent step to stop...\n",
            style="bold yellow",
        )

    text.append("\n  Query from another terminal:\n", style="bold")
    text.append(f"    myswat task {work_item_id or '<id>'} -p {proj['slug']}\n", style="dim")
    text.append(f"    myswat status -p {proj['slug']}\n", style="dim")
    return text


def _run_with_task_monitor(
    console: Console,
    store: "MemoryStore",
    proj: dict,
    label: str,
    worker_fn: Callable[[], object | None],
    work_item_ref: dict[str, int | None],
    cancel_targets: list[AgentRunner],
    cancel_event: threading.Event,
) -> object | None:
    """Run a long task while showing work-item progress."""
    result = [None]
    error = [None]
    start = time.monotonic()

    def _run():
        try:
            result[0] = worker_fn()
        except Exception as exc:
            error[0] = exc

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

    frame_idx = 0
    try:
        if use_cbreak and fd is not None:
            tty.setcbreak(fd)
        with Live(console=console, refresh_per_second=4, transient=True) as live:
            while worker.is_alive():
                live.update(
                    _build_task_monitor_display(
                        store=store,
                        proj=proj,
                        work_item_id=work_item_ref.get("id"),
                        label=label,
                        frame_idx=frame_idx,
                        elapsed=time.monotonic() - start,
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

                worker.join(timeout=0.25)

            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
    finally:
        try:
            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
        except Exception:
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
        with Live(console=console, refresh_per_second=4, transient=True) as live:
            while worker.is_alive():
                elapsed = time.monotonic() - start
                live_lines = _coerce_live_lines(sm._runner.live_output)
                live.update(_build_live_display(frame_idx, elapsed, live_lines))
                frame_idx += 1

                if use_cbreak and _check_esc():
                    cancelled = True
                    live.update(Text(" Cancelling...", style="bold yellow"))
                    sm._runner.cancel()
                    worker.join(timeout=5)
                    break

                worker.join(timeout=0.25)

            if use_cbreak and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
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
        return AgentResponse(content="Request cancelled.", exit_code=-1, cancelled=True), elapsed

    return result[0], elapsed
