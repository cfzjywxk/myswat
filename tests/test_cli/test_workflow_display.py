"""Tests for myswat.cli.workflow_display."""

from __future__ import annotations

import time

from myswat.cli.workflow_display import WorkflowDisplay, _StageState
from myswat.workflow.events import WorkflowEvent


def _make_event(event_type: str, message: str, **kwargs) -> WorkflowEvent:
    return WorkflowEvent(event_type=event_type, message=message, **kwargs)


def test_stage_start_and_completed_rendering():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Technical Design", stage="design"))
    display.handle_event(_make_event("agent_working", "Producing design...", stage="design", agent_role="developer"))
    display.handle_event(_make_event("agent_done", "Design submitted", stage="design", agent_role="developer"))
    display.handle_event(_make_event("stage_start", "Design Review", stage="design_review"))

    text = display.build_live_renderable("proj", 7, 0, 42.0)
    rendered = text.plain

    # Completed stage shows checkmark
    assert "\u2713" in rendered
    assert "Technical Design" in rendered
    # Current stage shows its label
    assert "Design Review" in rendered
    # Header with elapsed
    assert "42s" in rendered
    assert "proj" in rendered
    assert "#7" in rendered


def test_review_verdict_rendering():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design Review", stage="design_review"))
    display.handle_event(_make_event(
        "review_start", "Starting review",
        stage="design_review",
        metadata={"iteration": 1, "max_iterations": 3},
    ))
    display.handle_event(_make_event(
        "review_verdict", "QA reviewed",
        stage="design_review",
        agent_role="qa_main",
        metadata={"verdict": "lgtm"},
    ))
    display.handle_event(_make_event(
        "review_verdict", "Dev reviewed",
        stage="design_review",
        agent_role="developer",
        metadata={"verdict": "changes_requested", "issues": ["missing tests"]},
    ))

    text = display.build_live_renderable("proj", None, 1, 10.0)
    rendered = text.plain

    assert "1/3" in rendered
    assert "LGTM" in rendered
    assert "CHANGES REQUESTED" in rendered
    assert "missing tests" in rendered


def test_phase_rendering():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Starting phase",
        stage="development",
        metadata={"phase_index": 1, "total_phases": 3, "phase_name": "Auth module"},
    ))
    display.handle_event(_make_event(
        "agent_working", "Implementing...",
        stage="development",
        agent_role="developer",
    ))

    text = display.build_live_renderable("proj", 1, 2, 120.0)
    rendered = text.plain

    assert "Phase 1/3" in rendered
    assert "Auth module" in rendered
    assert "developer" in rendered
    assert "2m00s" in rendered


def test_cancel_requested_shows_message():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design", stage="design"))

    text = display.build_live_renderable("proj", 1, 0, 5.0, cancel_requested=True)
    rendered = text.plain

    assert "cancel" in rendered.lower() or "Cancel" in rendered


def test_final_snapshot_shows_all_stages():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design", stage="design"))
    display.handle_event(_make_event("agent_done", "Done", stage="design", agent_role="dev"))
    display.handle_event(_make_event("stage_start", "Review", stage="review"))
    display.handle_event(_make_event("agent_done", "Approved", stage="review", agent_role="qa"))

    text = display.build_final_snapshot("proj", 7)
    rendered = text.plain

    assert "Design" in rendered
    assert "#7" in rendered


def test_no_work_item_id_skips_project_line():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design", stage="design"))

    text = display.build_live_renderable("proj", None, 0, 1.0)
    rendered = text.plain

    assert "#" not in rendered


def test_phase_done_records_elapsed_and_review_info():
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Dev", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1",
        stage="development",
        metadata={"phase_index": 1, "total_phases": 2, "phase_name": "Setup"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase completed",
        stage="development",
        metadata={"status": "committed", "review_iterations": 2, "review_passed": True},
    ))
    display.handle_event(_make_event(
        "phase_start", "Phase 2",
        stage="development",
        metadata={"phase_index": 2, "total_phases": 2, "phase_name": "Core"},
    ))

    text = display.build_live_renderable("proj", 1, 0, 60.0)
    rendered = text.plain

    assert "Setup" in rendered
    assert "Core" in rendered
    assert "2 review iter" in rendered


# ── Issue 1: failed stages must render as failures, not success ──────


def test_agent_error_marks_stage_failed():
    """agent_error sets the failed flag on the stage."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Technical Design", stage="design"))
    display.handle_event(_make_event("agent_error", "Agent failed (exit=1)", stage="design", agent_role="developer"))

    # Stage state should be marked failed
    with display._lock:
        assert display._current_stage is not None
        assert display._current_stage.failed is True


def test_failed_completed_stage_renders_cross_not_check():
    """A failed stage shows ✗ in red, not ✓ in green."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Technical Design", stage="design"))
    display.handle_event(_make_event("agent_error", "Agent failed (exit=1)", stage="design", agent_role="developer"))
    display.handle_event(_make_event("stage_start", "Design Review", stage="design_review"))

    text = display.build_live_renderable("proj", 1, 0, 10.0)
    rendered = text.plain

    assert "\u2717" in rendered  # ✗
    assert "\u2713" not in rendered  # no ✓
    assert "Agent failed" in rendered


def test_failed_stage_in_final_snapshot():
    """build_final_snapshot uses ✗ for failed stages."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design", stage="design"))
    display.handle_event(_make_event("agent_done", "Done", stage="design", agent_role="dev"))
    display.handle_event(_make_event("stage_start", "Review", stage="review"))
    display.handle_event(_make_event("agent_error", "Reviewer crashed", stage="review", agent_role="qa"))

    text = display.build_final_snapshot("proj", 7)
    rendered = text.plain

    # First stage passed, second failed
    assert "\u2713" in rendered
    assert "\u2717" in rendered
    assert "Reviewer crashed" in rendered


# ── Issue 2: quiet console + real_console for user-facing content ────


def test_engine_stores_real_console():
    """When on_event is set, engine keeps _real_console pointing to the original."""
    import io
    from unittest.mock import MagicMock

    from myswat.workflow.engine import WorkflowEngine

    store = MagicMock()
    dev_sm = MagicMock()
    dev_sm.agent_role = "developer"
    dev_sm.agent_id = 1
    qa_sm = MagicMock()
    qa_sm.agent_role = "qa_main"
    qa_sm.agent_id = 2

    events = []
    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=[qa_sm],
        project_id=1,
        on_event=events.append,
    )

    # The quiet console writes to a StringIO, not the terminal.
    assert isinstance(engine._console.file, io.StringIO)
    # The real console is the module-level console.
    from myswat.workflow.engine import console as module_console
    assert engine._real_console is module_console


def test_engine_without_on_event_shares_consoles():
    """When on_event is None, _console and _real_console are the same."""
    from unittest.mock import MagicMock

    from myswat.workflow.engine import WorkflowEngine

    store = MagicMock()
    dev_sm = MagicMock()
    dev_sm.agent_role = "developer"
    dev_sm.agent_id = 1
    qa_sm = MagicMock()
    qa_sm.agent_role = "qa_main"
    qa_sm.agent_id = 2

    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=[qa_sm],
        project_id=1,
    )

    assert engine._console is engine._real_console


# ── Issue 3: phase display must not leak stale state ─────────────────


def test_phase_start_clears_stale_review_state():
    """Starting a new phase clears the review state from the previous phase."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="dev"))

    # Phase 1 with review
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="dev",
        metadata={"phase_index": 1, "total_phases": 2, "phase_name": "Setup"},
    ))
    display.handle_event(_make_event(
        "review_start", "Review", stage="dev",
        metadata={"iteration": 1, "max_iterations": 3},
    ))
    display.handle_event(_make_event(
        "review_verdict", "QA", stage="dev",
        agent_role="qa_main",
        metadata={"verdict": "changes_requested", "issues": ["bad code"]},
    ))

    # Phase 2 starts — should not carry phase 1's review
    display.handle_event(_make_event(
        "phase_start", "Phase 2", stage="dev",
        metadata={"phase_index": 2, "total_phases": 2, "phase_name": "Core"},
    ))

    text = display.build_live_renderable("proj", 1, 0, 30.0)
    rendered = text.plain

    # Phase 2 is current and should NOT show phase 1's review verdict
    assert "bad code" not in rendered
    assert "CHANGES REQUESTED" not in rendered
    assert "Core" in rendered


def test_review_start_clears_agent_action():
    """Starting review clears the previous agent action (e.g. 'Summarizing...')."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Review Stage", stage="review"))
    display.handle_event(_make_event(
        "agent_working", "Summarizing changes...", stage="review", agent_role="developer",
    ))

    # Review starts — agent state should be cleared
    display.handle_event(_make_event(
        "review_start", "Review", stage="review",
        metadata={"iteration": 1, "max_iterations": 3},
    ))

    text = display.build_live_renderable("proj", 1, 0, 10.0)
    rendered = text.plain

    assert "Summarizing changes" not in rendered
    assert "developer" not in rendered
    # Review iteration should show
    assert "1/3" in rendered


def test_phase_start_clears_agent_state():
    """Starting a new phase clears the previous agent action."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Dev", stage="dev"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="dev",
        metadata={"phase_index": 1, "total_phases": 2, "phase_name": "Setup"},
    ))
    display.handle_event(_make_event(
        "agent_working", "Committing phase 1...", stage="dev", agent_role="developer",
    ))

    # Phase 2 starts — should clear agent state from phase 1
    display.handle_event(_make_event(
        "phase_start", "Phase 2", stage="dev",
        metadata={"phase_index": 2, "total_phases": 2, "phase_name": "Core"},
    ))

    text = display.build_live_renderable("proj", 1, 0, 20.0)
    rendered = text.plain

    assert "Committing phase 1" not in rendered
    assert "Core" in rendered


# ── Phase failure propagates to enclosing stage ──────────────────────


def test_phase_failure_marks_enclosing_stage_failed():
    """A failed phase_done propagates failure to the Development stage."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="development",
        metadata={"phase_index": 1, "total_phases": 2, "phase_name": "Auth"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 1 commit failed", stage="development",
        metadata={"status": "failed", "review_iterations": 1, "review_passed": True},
    ))

    with display._lock:
        assert display._current_stage is not None
        assert display._current_stage.failed is True
        assert "commit failed" in display._current_stage.result_summary


def test_phase_failure_renders_cross_in_completed_stage():
    """After a phase failure, the Development stage shows ✗ when completed."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="development",
        metadata={"phase_index": 1, "total_phases": 1, "phase_name": "Auth"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 1 commit failed", stage="development",
        metadata={"status": "failed", "review_iterations": 0, "review_passed": False},
    ))
    # Next stage starts, finishing the Development stage
    display.handle_event(_make_event("stage_start", "Final Report", stage="report"))

    text = display.build_live_renderable("proj", 1, 0, 60.0)
    rendered = text.plain

    assert "\u2717" in rendered  # ✗ for the failed Development stage
    assert "commit failed" in rendered


def test_phase_success_updates_stage_summary():
    """A successful phase_done updates the stage result_summary."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="development",
        metadata={"phase_index": 1, "total_phases": 1, "phase_name": "Core"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 1 committed", stage="development",
        metadata={"status": "committed", "review_iterations": 2, "review_passed": True},
    ))

    with display._lock:
        assert display._current_stage.result_summary == "Phase 1 committed"
        assert display._current_stage.failed is False


def test_failed_development_stage_in_final_snapshot():
    """Final snapshot shows ✗ for the Development stage when a phase failed."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design", stage="design"))
    display.handle_event(_make_event("agent_done", "Done", stage="design", agent_role="dev"))
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="development",
        metadata={"phase_index": 1, "total_phases": 1, "phase_name": "Auth"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 1 commit failed", stage="development",
        metadata={"status": "failed"},
    ))
    display.handle_event(_make_event("stage_start", "Final Report", stage="report"))
    display.handle_event(_make_event("agent_done", "Report generated", stage="report", agent_role="dev"))

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    assert "\u2713" in rendered  # Design passed
    assert "\u2717" in rendered  # Development failed
    assert "commit failed" in rendered


# ── Unfinished stages must not render as completed ───────────────────


def test_unfinished_stage_renders_incomplete_in_snapshot():
    """A stage that never finished shows ‒ (incomplete), not ✓ (done)."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Design", stage="design"))
    display.handle_event(_make_event("agent_done", "Done", stage="design", agent_role="dev"))
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    # Workflow is cancelled/aborted — no more events, Development never finishes.

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    # Design completed normally
    assert "\u2713" in rendered
    # Development is incomplete — must NOT show ✓
    assert "Development" in rendered
    # The incomplete icon ‒
    assert "\u2012" in rendered


def test_cancelled_stage_shows_incomplete_not_done():
    """Cancelled current stage should show 'incomplete' not 'done'."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "GA Test", stage="ga_test"))
    display.handle_event(_make_event("agent_working", "Executing tests...", stage="ga_test", agent_role="qa_main"))
    # Cancelled — no agent_done, no new stage_start

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    # Must NOT show ✓ or "done"
    assert "\u2713" not in rendered
    assert "\u2012" in rendered  # incomplete icon
    assert "GA Test" in rendered


def test_aborted_stage_with_summary_shows_summary():
    """An incomplete stage with a result_summary should display it."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "GA Test", stage="ga_test"))
    display.handle_event(_make_event("agent_done", "3 of 5 tests passed", stage="ga_test", agent_role="qa"))
    # No new stage_start — stage is incomplete

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    assert "\u2012" in rendered
    assert "3 of 5 tests passed" in rendered


def test_stage_complete_marks_final_report_finished():
    """A terminal stage_complete event renders the final report as finished."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Final Report", stage="report"))
    display.handle_event(_make_event("stage_complete", "Report generated", stage="report"))

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    assert "\u2713 Final Report" in rendered
    assert "Report generated" in rendered
    assert "\u2012 Final Report" not in rendered


def test_failed_phase_summary_survives_later_successful_phases():
    """A later successful phase must not overwrite an earlier failure summary."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="development",
        metadata={"phase_index": 1, "total_phases": 2, "phase_name": "Setup"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 1 commit failed", stage="development",
        metadata={"status": "failed"},
    ))
    display.handle_event(_make_event(
        "phase_start", "Phase 2", stage="development",
        metadata={"phase_index": 2, "total_phases": 2, "phase_name": "Core"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 2 committed", stage="development",
        metadata={"status": "committed"},
    ))
    display.handle_event(_make_event("stage_start", "Final Report", stage="report"))

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    assert "\u2717 Development" in rendered
    assert "Phase 1 commit failed" in rendered
    assert "Phase 2 committed" not in rendered


def test_stage_complete_preserves_existing_failure_summary():
    """A generic stage_complete must not overwrite a prior failure summary."""
    display = WorkflowDisplay()
    display.handle_event(_make_event("stage_start", "Development", stage="development"))
    display.handle_event(_make_event(
        "phase_start", "Phase 1", stage="development",
        metadata={"phase_index": 1, "total_phases": 1, "phase_name": "Setup"},
    ))
    display.handle_event(_make_event(
        "phase_done", "Phase 1 commit failed", stage="development",
        metadata={"status": "failed"},
    ))
    display.handle_event(_make_event("stage_complete", "Development done", stage="development"))

    text = display.build_final_snapshot("proj", 1)
    rendered = text.plain

    assert "\u2717 Development" in rendered
    assert "Phase 1 commit failed" in rendered
    assert "Development done" not in rendered
