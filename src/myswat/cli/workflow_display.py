"""Unified workflow display renderer.

Consumes WorkflowEvents from the engine and builds Rich renderables
for the Live terminal display.  Thread-safe: the engine calls
``handle_event()`` from its worker thread while the main thread calls
``build_live_renderable()`` at 8 fps.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.text import Text

from myswat.cli.progress import _fmt_duration, _SPINNER_FRAMES

if TYPE_CHECKING:
    from myswat.workflow.events import WorkflowEvent


# ── Display state helpers ────────────────────────────────────────────────

@dataclass
class _ReviewerVerdict:
    role: str
    verdict: str  # "lgtm" | "changes_requested" | "failed" | "skipped"
    issues: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class _ReviewState:
    iteration: int = 0
    max_iterations: int = 10
    verdicts: list[_ReviewerVerdict] = field(default_factory=list)
    all_approved: bool = False


@dataclass
class _PhaseState:
    index: int = 0
    total: int = 0
    name: str = ""
    status: str = "working"  # working | reviewing | committed | failed | cancelled
    review_iterations: int = 0
    review_passed: bool = False
    elapsed: float = 0.0
    start_time: float = 0.0


@dataclass
class _StageState:
    name: str = ""
    label: str = ""
    agent_role: str | None = None
    agent_action: str = ""
    agent_start: float = 0.0
    finished: bool = False
    failed: bool = False
    result_summary: str = ""
    review: _ReviewState | None = None
    phases: list[_PhaseState] = field(default_factory=list)
    current_phase_idx: int = -1


# ── Main renderer ────────────────────────────────────────────────────────

class WorkflowDisplay:
    """Thread-safe workflow display renderer."""

    def __init__(self, workflow_start: float | None = None) -> None:
        self._lock = threading.Lock()
        self._workflow_start = workflow_start or time.monotonic()
        self._stages: list[_StageState] = []
        self._current_stage: _StageState | None = None
        self._last_info: str = ""
        self._last_warning: str = ""

    # ── Event handler (called from worker thread) ────────────────────

    def handle_event(self, event: WorkflowEvent) -> None:
        with self._lock:
            self._process_event(event)

    def _process_event(self, event: WorkflowEvent) -> None:
        t = event.event_type

        if t == "stage_start":
            self._start_stage(event)
        elif t == "agent_working":
            self._set_agent_working(event)
        elif t == "agent_done":
            self._set_agent_done(event)
        elif t == "agent_error":
            self._set_agent_done(event, error=True)
        elif t == "review_start":
            self._start_review(event)
        elif t == "review_verdict":
            self._add_verdict(event)
        elif t == "review_complete":
            self._complete_review(event)
        elif t == "revision_start":
            self._set_agent_working(event)
        elif t == "phase_start":
            self._start_phase(event)
        elif t == "phase_done":
            self._complete_phase(event)
        elif t == "stage_complete":
            self._complete_stage(event)
        elif t == "info":
            self._last_info = event.message
        elif t == "warning":
            self._last_warning = event.message

    def _start_stage(self, event: WorkflowEvent) -> None:
        if self._current_stage and not self._current_stage.finished:
            self._current_stage.finished = True
        stage = _StageState(
            name=event.stage,
            label=event.message,
        )
        self._stages.append(stage)
        self._current_stage = stage

    def _set_agent_working(self, event: WorkflowEvent) -> None:
        if self._current_stage:
            self._current_stage.agent_role = event.agent_role
            self._current_stage.agent_action = event.message
            self._current_stage.agent_start = event.timestamp
            self._current_stage.finished = False

    def _set_agent_done(self, event: WorkflowEvent, error: bool = False) -> None:
        if self._current_stage:
            self._current_stage.result_summary = event.message
            if error:
                self._current_stage.failed = True
            self._current_stage.agent_action = ""
            self._current_stage.agent_role = None

    def _start_review(self, event: WorkflowEvent) -> None:
        if not self._current_stage:
            return
        # Clear agent state — review is a distinct activity.
        self._current_stage.agent_action = ""
        self._current_stage.agent_role = None
        iteration = event.metadata.get("iteration", 1)
        max_iter = event.metadata.get("max_iterations", 10)
        if self._current_stage.review is None:
            self._current_stage.review = _ReviewState(
                iteration=iteration, max_iterations=max_iter,
            )
        else:
            self._current_stage.review.iteration = iteration
            self._current_stage.review.verdicts.clear()
            self._current_stage.review.all_approved = False

    def _add_verdict(self, event: WorkflowEvent) -> None:
        if not self._current_stage or not self._current_stage.review:
            return
        self._current_stage.review.verdicts.append(_ReviewerVerdict(
            role=event.agent_role or "reviewer",
            verdict=event.metadata.get("verdict", "unknown"),
            issues=event.metadata.get("issues", []),
            summary=event.metadata.get("summary", ""),
        ))

    def _complete_review(self, event: WorkflowEvent) -> None:
        if self._current_stage and self._current_stage.review:
            self._current_stage.review.all_approved = event.metadata.get("approved", False)

    def _start_phase(self, event: WorkflowEvent) -> None:
        if not self._current_stage:
            return
        # Clear stale review state from the previous phase.
        self._current_stage.review = None
        self._current_stage.agent_action = ""
        self._current_stage.agent_role = None
        if not self._current_stage.failed:
            self._current_stage.result_summary = ""
        phase = _PhaseState(
            index=event.metadata.get("phase_index", 0),
            total=event.metadata.get("total_phases", 0),
            name=event.metadata.get("phase_name", ""),
            start_time=event.timestamp,
        )
        self._current_stage.phases.append(phase)
        self._current_stage.current_phase_idx = len(self._current_stage.phases) - 1

    def _complete_phase(self, event: WorkflowEvent) -> None:
        if not self._current_stage or self._current_stage.current_phase_idx < 0:
            return
        phase = self._current_stage.phases[self._current_stage.current_phase_idx]
        phase.status = event.metadata.get("status", "done")
        phase.review_iterations = event.metadata.get("review_iterations", 0)
        phase.review_passed = event.metadata.get("review_passed", False)
        phase.elapsed = event.timestamp - phase.start_time
        # Propagate phase failure to the enclosing stage.
        if phase.status == "failed":
            self._current_stage.failed = True
            self._current_stage.result_summary = event.message
        elif phase.status in ("committed", "done") and not self._current_stage.failed:
            self._current_stage.result_summary = event.message

    def _complete_stage(self, event: WorkflowEvent) -> None:
        if not self._current_stage:
            return
        event_failed = event.metadata.get("failed", False)
        stage_was_failed = self._current_stage.failed
        if event_failed:
            self._current_stage.failed = True
        if event.message and (event_failed or not stage_was_failed or not self._current_stage.result_summary):
            self._current_stage.result_summary = event.message
        self._current_stage.agent_action = ""
        self._current_stage.agent_role = None
        self._current_stage.finished = True

    # ── Renderable builder (called from main thread) ─────────────────

    def build_live_renderable(
        self,
        proj_slug: str,
        work_item_id: int | None,
        frame_idx: int,
        elapsed: float,
        cancel_requested: bool = False,
    ) -> Text:
        """Build the Rich renderable for the Live display."""
        with self._lock:
            return self._render(
                proj_slug, work_item_id, frame_idx, elapsed, cancel_requested,
            )

    def _render(
        self,
        proj_slug: str,
        work_item_id: int | None,
        frame_idx: int,
        elapsed: float,
        cancel_requested: bool,
    ) -> Text:
        spinner = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
        text = Text()

        # ── Header ──
        text.append(
            f" {spinner}  Workflow running ({_fmt_duration(elapsed)})  ESC to cancel\n",
            style="bold cyan",
        )
        if work_item_id is not None:
            text.append(f"    Project: {proj_slug}  Work item: #{work_item_id}\n", style="dim")
        text.append("\n")

        # ── Completed stages (compact) ──
        for stage in self._stages:
            if stage is self._current_stage:
                break
            if stage.finished:
                self._render_completed_stage(text, stage)

        # ── Current stage (detailed) ──
        if self._current_stage:
            self._render_current_stage(text, self._current_stage)

        # ── Footer ──
        if cancel_requested:
            text.append(
                "\n    Cancellation requested, waiting for agent to stop...\n",
                style="bold yellow",
            )

        text.append(f"\n    Query: myswat status -p {proj_slug} --details\n", style="dim")
        return text

    def _render_completed_stage(self, text: Text, stage: _StageState) -> None:
        """Render a completed stage as a single compact line."""
        if stage.failed:
            icon = "\u2717"  # ✗
            icon_style = "red"
        else:
            icon = "\u2713"  # ✓
            icon_style = "green"
        summary = stage.result_summary or ("failed" if stage.failed else "done")
        if len(summary) > 80:
            summary = summary[:77] + "..."
        text.append(f"    {icon} {stage.label}", style=icon_style)
        if summary and summary not in ("done", "failed"):
            text.append(f"  {summary}", style="dim red" if stage.failed else "dim")
        text.append("\n")

    def _render_current_stage(self, text: Text, stage: _StageState) -> None:
        """Render the currently active stage with full detail."""
        text.append(f"    {stage.label}\n", style="bold")

        # Show phases if present
        if stage.phases:
            self._render_phases(text, stage)
        # Show review state
        elif stage.review:
            self._render_review(text, stage.review, indent=8)

        # Show active agent
        if stage.agent_action and stage.agent_role:
            agent_elapsed = _fmt_duration(time.monotonic() - stage.agent_start)
            text.append(f"      {stage.agent_role}", style="cyan")
            text.append(f"  {stage.agent_action}")
            text.append(f"  ({agent_elapsed})\n", style="dim")

        # Show last result
        if stage.result_summary:
            summary = stage.result_summary
            if len(summary) > 120:
                summary = summary[:117] + "..."
            text.append(f"      {summary}\n", style="dim red" if stage.failed else "dim")

    def _render_phases(self, text: Text, stage: _StageState) -> None:
        """Render development phases within a stage."""
        for i, phase in enumerate(stage.phases):
            is_current = (i == stage.current_phase_idx and phase.status == "working")
            prefix = "\u25b6" if is_current else ("\u2713" if phase.status in ("committed", "done") else "\u2717")  # ▶ or ✓ or ✗
            style = "bold" if is_current else ("green" if phase.status in ("committed", "done") else "red" if phase.status == "failed" else "dim")

            text.append(f"      {prefix} Phase {phase.index}/{phase.total}: {phase.name}", style=style)

            if phase.status in ("committed", "done") and phase.elapsed > 0:
                text.append(f"  ({_fmt_duration(phase.elapsed)})", style="dim")
                if phase.review_iterations > 0:
                    text.append(f"  {phase.review_iterations} review iter", style="dim")
            text.append("\n")

        # Show review state for current phase
        if stage.review and stage.current_phase_idx >= 0:
            current_phase = stage.phases[stage.current_phase_idx]
            if current_phase.status == "working":
                self._render_review(text, stage.review, indent=10)

    def _render_review(self, text: Text, review: _ReviewState, indent: int = 8) -> None:
        """Render review iteration state."""
        pad = " " * indent
        text.append(
            f"{pad}Review iteration {review.iteration}/{review.max_iterations}\n",
            style="dim",
        )
        for v in review.verdicts:
            if v.verdict == "lgtm":
                text.append(f"{pad}  {v.role}: ", style="dim")
                text.append("LGTM\n", style="bold green")
            elif v.verdict == "changes_requested":
                text.append(f"{pad}  {v.role}: ", style="dim")
                text.append("CHANGES REQUESTED\n", style="bold yellow")
                for issue in v.issues[:3]:
                    issue_text = issue if len(issue) <= 80 else issue[:77] + "..."
                    text.append(f"{pad}    - {issue_text}\n", style="yellow")
            elif v.verdict == "failed":
                text.append(f"{pad}  {v.role}: ", style="dim")
                text.append("FAILED\n", style="bold red")
                detail = v.summary or (v.issues[0] if v.issues else "")
                if detail:
                    if len(detail) > 80:
                        detail = detail[:77] + "..."
                    text.append(f"{pad}    {detail}\n", style="red")
            elif v.verdict == "skipped":
                text.append(f"{pad}  {v.role}: ", style="dim")
                text.append("skipped\n", style="dim")
                if v.summary:
                    detail = v.summary if len(v.summary) <= 80 else v.summary[:77] + "..."
                    text.append(f"{pad}    {detail}\n", style="dim")

        if review.all_approved:
            text.append(f"{pad}  All reviewers approved\n", style="bold green")

    # ── Snapshot for post-run display ────────────────────────────────

    def _render_incomplete_stage(self, text: Text, stage: _StageState) -> None:
        """Render a stage that never finished (cancelled/aborted/stopped)."""
        icon = "\u2012"  # ‒ (figure dash)
        summary = stage.result_summary or "incomplete"
        if len(summary) > 80:
            summary = summary[:77] + "..."
        text.append(f"    {icon} {stage.label}", style="yellow")
        if summary and summary != "incomplete":
            text.append(f"  {summary}", style="dim yellow")
        text.append("\n")

    def build_final_snapshot(self, proj_slug: str, work_item_id: int | None) -> Text:
        """Build a non-transient summary after workflow completes."""
        with self._lock:
            text = Text()
            text.append(" Workflow summary\n\n", style="bold")
            if work_item_id is not None:
                text.append(f"    Project: {proj_slug}  Work item: #{work_item_id}\n\n", style="dim")

            for stage in self._stages:
                if stage.finished or stage.failed:
                    self._render_completed_stage(text, stage)
                else:
                    self._render_incomplete_stage(text, stage)

            text.append(f"\n    Details: myswat status -p {proj_slug} --details\n", style="dim")
            return text
