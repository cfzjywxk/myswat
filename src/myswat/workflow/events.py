"""Structured workflow events emitted by WorkflowEngine.

The engine emits events instead of printing directly to console.
A display renderer consumes these events to build a unified terminal display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowEvent:
    """A single structured event from the workflow engine."""

    event_type: str
    """One of: stage_start, stage_complete, agent_working, agent_done, agent_error,
    review_start, review_verdict, review_complete, revision_start,
    phase_start, phase_done, user_checkpoint, info, warning, error."""

    message: str
    """Human-readable summary of the event."""

    stage: str = ""
    """Current workflow stage (e.g. 'design', 'design_review', 'phase_2')."""

    agent_role: str | None = None
    """Which agent is involved (e.g. 'developer', 'architect', 'qa_main')."""

    detail: str | None = None
    """Extended content (e.g. approved design text, error traceback)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary structured data (iteration, verdict, phase_index, etc.)."""

    timestamp: float = field(default_factory=time.monotonic)
    """Monotonic timestamp for elapsed-time calculations."""
