"""WorkflowEngine — drives the teamwork workflow regulated by myswat.

myswat acts as the Architect/regulator. The user provides requirements
and approves key decisions. Dev and QA are AI agents driven by myswat.

Workflow:
  1. Dev produces tech design
  2. QA reviews design (loop until all LGTM)
  3. User sees approved design, can intervene
  4. Dev breaks into implementation phases
  5. User + QA review the plan (loop until all LGTM)
  6. For each phase:
     a. Dev implements
     b. Dev summarizes
     c. QA reviews (loop until LGTM)
     d. Dev commits locally
  7. GA Test:
     a. QA generates test plan
     b. Dev + User review test plan (loop until LGTM)
     c. QA executes tests
     d. Bug fix loop (dev fixes, QA re-tests; >5 bugs = abort)
  8. Final E2E report (dev + test results combined)
"""

from __future__ import annotations

import io
import inspect
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from myswat.cli.progress import _collapse_text
from myswat.large_payloads import (
    maybe_externalize_list,
    maybe_externalize_response,
    maybe_externalize_summary,
)
from myswat.models.work_item import ReviewVerdict
from myswat.workflow.events import WorkflowEvent
from myswat.workflow.modes import WorkMode
from myswat.workflow.prompts import (
    ARCH_ADDRESS_DESIGN_COMMENTS,
    ARCH_TECH_DESIGN,
    DESIGN_REVIEW,
    DEV_ADDRESS_CODE_COMMENTS,
    DEV_ADDRESS_DESIGN_COMMENTS,
    DEV_ADDRESS_PLAN_COMMENTS,
    DEV_COMMIT_PHASE,
    DEV_ESTIMATE_BUG,
    DEV_FINAL_REPORT,
    DEV_FIX_BUG_SIMPLE,
    DEV_IMPLEMENT_PHASE,
    DEV_IMPLEMENTATION_PLAN,
    DEV_REVIEW_TEST_PLAN,
    DEV_SUMMARIZE_BUG_FIX,
    TEST_PLAN_REVIEW,
    DEV_SUMMARIZE_PHASE,
    DEV_TECH_DESIGN,
    QA_ADDRESS_TEST_PLAN_COMMENTS,
    QA_CODE_REVIEW,
    QA_CONTINUE_GA_TEST,
    QA_DESIGN_REVIEW,
    QA_DESIGN_TEST_PLAN,
    QA_EXECUTE_GA_TEST,
    QA_GA_TEST_PLAN,
    QA_GA_TEST_REPORT,
    QA_PLAN_REVIEW,
)

if TYPE_CHECKING:
    from myswat.agents.base import AgentResponse
    from myswat.agents.session_manager import SessionManager
    from myswat.memory.store import MemoryStore

console = Console()

# Maximum bugs before aborting GA test phase
MAX_GA_BUGS = 5
_DELEGATE_BLOCK_RE = re.compile(r"```delegate\b", re.IGNORECASE)


@dataclass
class PhaseResult:
    name: str
    summary: str
    review_iterations: int
    review_passed: bool = False
    committed: bool = False


@dataclass
class BugFixResult:
    title: str
    arch_change: bool = False
    fixed: bool = False
    summary: str = ""


@dataclass
class GATestResult:
    test_plan: str = ""
    test_plan_review_iterations: int = 0
    test_plan_review_passed: bool = False
    test_report: str = ""
    bugs_found: int = 0
    bugs_fixed: int = 0
    bug_fixes: list[BugFixResult] = field(default_factory=list)
    passed: bool = False
    aborted: bool = False


@dataclass
class WorkflowResult:
    requirement: str
    design: str = ""
    design_review_iterations: int = 0
    design_review_passed: bool = False
    plan: str = ""
    plan_review_iterations: int = 0
    plan_review_passed: bool = False
    phases: list[PhaseResult] = field(default_factory=list)
    ga_test: GATestResult | None = None
    final_report: str = ""
    success: bool = False
    blocked: bool = False
    failure_summary: str = ""


# ── Helpers ──

def _default_ask(prompt: str) -> str:
    try:
        return input(f"\n{prompt}").strip()
    except (EOFError, KeyboardInterrupt):
        return "n"


def _parse_verdict(raw: str) -> ReviewVerdict:
    from myswat.workflow.review_loop import _parse_verdict as _pv
    return _pv(raw)


def _extract_json_block(text: str) -> dict | list | None:
    """Extract a JSON object or array from text that may contain markdown."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    # Try to find JSON
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        start = text.find(start_ch)
        end = text.rfind(end_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class WorkflowEngine:
    """Drives the full teamwork workflow regulated by myswat code."""

    def __init__(
        self,
        store: MemoryStore,
        dev_sm: SessionManager,
        qa_sms: list[SessionManager],
        project_id: int,
        work_item_id: int | None = None,
        max_review_iterations: int | None = None,
        design_plan_review_limit: int | None = None,
        dev_plan_review_limit: int | None = None,
        dev_code_review_limit: int | None = None,
        ga_plan_review_limit: int | None = None,
        ga_test_review_limit: int | None = None,
        mode: WorkMode = WorkMode.full,
        ask_user: Callable[[str], str] | None = None,
        auto_approve: bool = False,
        should_cancel: Callable[[], bool] | None = None,
        arch_sm: SessionManager | None = None,
        resume_stage: str | None = None,
        on_event: Callable[[WorkflowEvent], None] | None = None,
    ) -> None:
        self._store = store
        self._dev = dev_sm
        self._qas = qa_sms
        self._project_id = project_id
        self._work_item_id = work_item_id
        default_review_limits = {
            "design": 10,
            "dev_plan": 10,
            "dev_code": 10,
            "ga_plan": 2,
            "ga_test": 2,
        }
        fallback_review_limit = max_review_iterations
        self._design_plan_review_limit = (
            default_review_limits["design"]
            if design_plan_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if design_plan_review_limit is None else design_plan_review_limit
            )
        )
        self._dev_plan_review_limit = (
            default_review_limits["dev_plan"]
            if dev_plan_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if dev_plan_review_limit is None else dev_plan_review_limit
            )
        )
        self._dev_code_review_limit = (
            default_review_limits["dev_code"]
            if dev_code_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if dev_code_review_limit is None else dev_code_review_limit
            )
        )
        self._ga_plan_review_limit = (
            default_review_limits["ga_plan"]
            if ga_plan_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if ga_plan_review_limit is None else ga_plan_review_limit
            )
        )
        self._ga_test_review_limit = (
            default_review_limits["ga_test"]
            if ga_test_review_limit is None and fallback_review_limit is None
            else (
                fallback_review_limit if ga_test_review_limit is None else ga_test_review_limit
            )
        )
        self._max_review = max(
            self._design_plan_review_limit,
            self._dev_plan_review_limit,
            self._dev_code_review_limit,
            self._ga_plan_review_limit,
            self._ga_test_review_limit,
        )
        self._mode = WorkMode(mode)
        self._arch = arch_sm
        self._resume_stage = resume_stage
        if self._mode in {WorkMode.architect_design, WorkMode.testplan_design} and self._arch is None:
            raise ValueError("arch_sm is required for architect_design and testplan_design modes")
        if self._mode == WorkMode.testplan_design and not qa_sms:
            raise ValueError("testplan_design mode requires at least one qa session")
        self._ask = ask_user or _default_ask
        self._auto_approve = auto_approve
        self._should_cancel = should_cancel
        self._blocked = False
        self._failure_summary = ""
        self._blocked_stage = ""
        self._last_review_limit_reached = False
        self._last_review_limit_stage = ""
        self._last_review_limit_summary = ""
        self._on_event = on_event
        # When an event handler is attached, use a quiet console to suppress
        # status prints (the display renderer shows structured output instead).
        # The real console is kept for user-facing content (artifact panels
        # before checkpoints, final reports) which Rich Live renders above
        # the transient display.
        self._real_console = console
        if on_event is not None:
            self._console = Console(file=io.StringIO(), quiet=True)
        else:
            self._console = console

    def _emit(
        self,
        event_type: str,
        message: str,
        *,
        stage: str = "",
        agent_role: str | None = None,
        detail: str | None = None,
        **metadata: object,
    ) -> None:
        if self._on_event is not None:
            self._on_event(WorkflowEvent(
                event_type=event_type,
                message=message,
                stage=stage,
                agent_role=agent_role,
                detail=detail,
                metadata=metadata,
            ))

    def _persist_task_state(
        self,
        *,
        current_stage: str,
        latest_summary: str | None = None,
        next_todos: list[str] | None = None,
        open_issues: list[str] | None = None,
        last_artifact_id: int | None = None,
        updated_by_agent_id: int | None = None,
    ) -> None:
        if not self._work_item_id:
            return
        latest_summary = maybe_externalize_summary(
            latest_summary,
            label=f"{current_stage}-summary",
        ) if latest_summary is not None else None
        next_todos = maybe_externalize_list(next_todos, label=f"{current_stage}-todo")
        open_issues = maybe_externalize_list(open_issues, label=f"{current_stage}-issue")
        try:
            self._store.update_work_item_state(
                self._work_item_id,
                current_stage=current_stage,
                latest_summary=latest_summary,
                next_todos=next_todos,
                open_issues=open_issues,
                last_artifact_id=last_artifact_id,
                updated_by_agent_id=updated_by_agent_id,
            )
        except Exception as e:
            self._console.print(f"[dim red]Warning: Failed to persist task state: {e}[/dim red]")

    @staticmethod
    def _first_lines(text: str, limit: int = 4) -> list[str]:
        lines = [line.strip("-* ") for line in text.splitlines() if line.strip()]
        return lines[:limit]

    def _cancelled(self) -> bool:
        return bool(self._should_cancel and self._should_cancel())

    @staticmethod
    def _checkpoint_stage_name(artifact_type: str) -> str:
        mapping = {
            "arch_design": "arch_design_user_checkpoint",
            "design": "design_user_checkpoint",
            "plan": "plan_user_checkpoint",
            "test_plan": "test_plan_user_checkpoint",
        }
        return mapping.get(artifact_type, f"{artifact_type}_user_checkpoint")

    def _persist_user_checkpoint_state(
        self,
        *,
        artifact: str,
        artifact_type: str,
        next_todo: str,
        updated_by_agent_id: int | None,
    ) -> None:
        self._persist_task_state(
            current_stage=self._checkpoint_stage_name(artifact_type),
            latest_summary=artifact[:4000],
            next_todos=[next_todo],
            open_issues=[],
            updated_by_agent_id=updated_by_agent_id,
        )

    @staticmethod
    def _validate_reviewable_design(artifact_type: str, content: str) -> str | None:
        if artifact_type not in {"design", "arch_design"}:
            return None

        text = (content or "").strip()
        if not text:
            return "The design draft is empty."

        if _DELEGATE_BLOCK_RE.search(text):
            return "The response is a `delegate` block, not a concrete technical design proposal."

        if artifact_type != "arch_design":
            return None

        normalized = text.lower()
        required_topics = {
            "problem/goals": ("problem statement", "goal", "goals", "objective", "scope"),
            "architecture/approach": ("architecture", "architectural", "overview", "approach"),
            "decisions/trade-offs": ("trade-off", "tradeoff", "decision", "decisions"),
            "interfaces/data flow": ("interface", "interfaces", "api", "component", "components", "data flow"),
            "dependencies/risks": ("dependency", "dependencies", "risk", "risks", "constraint", "constraints"),
        }
        matched_topics = [
            label
            for label, terms in required_topics.items()
            if any(term in normalized for term in terms)
        ]
        if len(text) < 400 or len(matched_topics) < 4:
            missing_topics = [label for label in required_topics if label not in matched_topics]
            missing_text = ", ".join(missing_topics[:3]) or "core design sections"
            return (
                "The architect response is not reviewable as a technical design yet; "
                f"it is missing core design content such as {missing_text}."
            )

        return None

    def _record_invalid_design_failure(
        self,
        *,
        artifact_type: str,
        content: str,
        owner: "SessionManager",
        stage: str,
        title: str,
    ) -> None:
        issue = self._validate_reviewable_design(artifact_type, content)
        if issue is None:
            return
        summary = f"[{owner.agent_role}] produced a non-reviewable {artifact_type}: {issue}"
        self._emit("agent_error", issue, stage=stage, agent_role=owner.agent_role)
        self._console.print(f"[red]{issue}[/red]")
        self._record_blocked_failure(
            stage=stage,
            summary=summary,
            updated_by_agent_id=owner.agent_id,
            from_role=owner.agent_role,
            to_role="myswat",
            event_type="proposal_failure",
            title=title,
        )

    def _append_process_event(
        self,
        *,
        event_type: str,
        summary: str,
        from_role: str | None = None,
        to_role: str | None = None,
        title: str | None = None,
        updated_by_agent_id: int | None = None,
    ) -> None:
        if not self._work_item_id:
            return
        summary = maybe_externalize_summary(summary, label=f"{event_type}-summary")
        try:
            self._store.append_work_item_process_event(
                self._work_item_id,
                event_type=event_type,
                title=title,
                summary=_collapse_text(summary),
                from_role=from_role,
                to_role=to_role,
                updated_by_agent_id=updated_by_agent_id,
            )
        except Exception:
            pass

    def _make_status_callback(
        self,
        stage: str,
        sm: "SessionManager",
    ) -> Callable[[str, dict[str, object]], None]:
        def _callback(event_type: str, payload: dict[str, object]) -> None:
            attempt = int(payload.get("attempt") or 0)
            max_attempts = int(payload.get("max_attempts") or 0)

            if event_type == "agent_stalled":
                timeout = payload.get("timeout")
                message = f"{sm.agent_role} stalled"
                if attempt and max_attempts:
                    message += f" ({attempt}/{max_attempts})"
                if timeout:
                    message += f" after {timeout}s without output"
                self._emit(
                    "warning",
                    message,
                    stage=stage,
                    agent_role=sm.agent_role,
                    **payload,
                )
                self._emit(
                    "agent_working",
                    message,
                    stage=stage,
                    agent_role=sm.agent_role,
                    **payload,
                )
                self._append_process_event(
                    event_type="agent_stall",
                    title="Agent stalled",
                    summary=message,
                    from_role=sm.agent_role,
                    to_role="myswat",
                    updated_by_agent_id=sm.agent_id,
                )
                return

            if event_type == "agent_empty_output":
                message = f"{sm.agent_role} returned empty output"
                if attempt and max_attempts:
                    message += f" ({attempt}/{max_attempts})"
                self._emit(
                    "warning",
                    message,
                    stage=stage,
                    agent_role=sm.agent_role,
                    **payload,
                )
                self._emit(
                    "agent_working",
                    message,
                    stage=stage,
                    agent_role=sm.agent_role,
                    **payload,
                )
                self._append_process_event(
                    event_type="agent_empty_output",
                    title="Agent returned empty output",
                    summary=message,
                    from_role=sm.agent_role,
                    to_role="myswat",
                    updated_by_agent_id=sm.agent_id,
                )
                return

            if event_type == "agent_retry":
                next_attempt = int(payload.get("next_attempt") or 0)
                next_timeout = payload.get("next_timeout")
                message = f"{sm.agent_role} retrying"
                if next_attempt and max_attempts:
                    message += f" ({next_attempt}/{max_attempts})"
                if next_timeout:
                    message += f" with {next_timeout}s timeout"
                self._emit(
                    "info",
                    message,
                    stage=stage,
                    agent_role=sm.agent_role,
                    **payload,
                )
                self._emit(
                    "agent_working",
                    message,
                    stage=stage,
                    agent_role=sm.agent_role,
                    **payload,
                )
                self._append_process_event(
                    event_type="agent_retry",
                    title="Agent retry",
                    summary=message,
                    from_role="myswat",
                    to_role=sm.agent_role,
                    updated_by_agent_id=sm.agent_id,
                )

        return _callback

    def _send_agent(
        self,
        sm: "SessionManager",
        prompt: str,
        *,
        task_description: str,
        stage: str,
    ):
        send_kwargs: dict[str, object] = {
            "task_description": task_description,
        }
        try:
            signature = inspect.signature(sm.send)
        except (TypeError, ValueError):
            signature = None
        if signature is None or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ) or "status_callback" in signature.parameters:
            send_kwargs["status_callback"] = self._make_status_callback(stage, sm)
        return sm.send(prompt, **send_kwargs)

    def _print_markdown_panel(
        self,
        content: str,
        *,
        title: str,
        border_style: str,
        label: str,
    ) -> None:
        rendered, _ = maybe_externalize_response(content, label=label)
        self._real_console.print(
            Panel(Markdown(rendered), title=title, border_style=border_style),
        )

    def _record_blocked_failure(
        self,
        *,
        stage: str,
        summary: str,
        updated_by_agent_id: int | None = None,
        from_role: str | None = None,
        to_role: str | None = None,
        event_type: str = "agent_failure",
        title: str | None = None,
    ) -> None:
        self._blocked = True
        self._failure_summary = summary[:500]
        self._blocked_stage = stage[:128]
        self._persist_task_state(
            current_stage=stage,
            latest_summary=summary,
            next_todos=["Resolve workflow failure and retry"],
            open_issues=[summary],
            updated_by_agent_id=updated_by_agent_id,
        )
        self._append_process_event(
            event_type=event_type,
            title=title or stage.replace("_", " "),
            summary=summary,
            from_role=from_role,
            to_role=to_role,
            updated_by_agent_id=updated_by_agent_id,
        )

    def _reset_review_loop_state(self) -> None:
        self._last_review_limit_reached = False
        self._last_review_limit_stage = ""
        self._last_review_limit_summary = ""

    def _record_review_limit_reached(
        self,
        *,
        artifact_type: str,
        stage: str,
        iteration: int,
        max_iterations: int,
        proposer: "SessionManager",
        issues: list[str],
    ) -> None:
        summary = (
            f"Max review iterations reached for {artifact_type} after {iteration} round(s); "
            "continuing with the latest artifact without another review round."
        )
        unresolved = issues[:8]
        event_summary = summary
        if unresolved:
            event_summary += "\nUnresolved review issues:\n" + "\n".join(f"- {issue}" for issue in unresolved)

        self._last_review_limit_reached = True
        self._last_review_limit_stage = stage
        self._last_review_limit_summary = summary

        self._emit(
            "warning",
            summary,
            stage=stage,
            agent_role=proposer.agent_role,
            iteration=iteration,
            max_iterations=max_iterations,
            issue_count=len(issues),
            review_skipped=True,
        )
        self._append_process_event(
            event_type="review_skipped",
            title=f"{artifact_type} review skipped after max iterations",
            summary=event_summary,
            from_role="myswat",
            to_role=proposer.agent_role,
            updated_by_agent_id=proposer.agent_id,
        )
        self._persist_task_state(
            current_stage=f"{stage}_review_skipped",
            latest_summary=summary,
            next_todos=["Continue the workflow with the latest artifact despite unresolved review issues"],
            open_issues=issues,
            updated_by_agent_id=proposer.agent_id,
        )

    def _sync_result_failure_state(self, result: WorkflowResult) -> WorkflowResult:
        result.blocked = self._blocked
        result.failure_summary = self._failure_summary
        return result

    # ════════════════════════════════════════════════════════════════
    # Resume helpers
    # ════════════════════════════════════════════════════════════════

    def _resume_entry_point(self) -> str:
        """Map the persisted current_stage to a high-level resume point.

        Returns one of: "start", "design", "design_review", "design_checkpoint",
        "plan", "plan_review", "plan_checkpoint", "phases", "ga_test", "done".
        """
        if not self._resume_stage:
            return "start"

        s = self._resume_stage

        # Terminal state — truly completed
        if s == "workflow_completed":
            return "done"

        # Finished with issues — re-run from the beginning (not "done")
        if s == "workflow_finished_with_issues":
            return "start"

        # GA test stages
        if s.startswith("ga_test") or s.startswith("test_plan"):
            return "ga_test"

        # Phase stages (phase_1_implementing, phase_2_committed, etc.)
        if s.startswith("phase_"):
            return "phases"

        # Reviewer-approved plan still needs the user checkpoint on resume.
        if s in ("plan_approved", "proposal_approved", "plan_user_checkpoint"):
            return "plan_checkpoint"

        # Plan review-loop states (proposal_review, proposal_revision_ready, etc.)
        if s.startswith("proposal_") or (
            s.startswith("plan_") and s not in ("plan_draft", "plan_draft_blocked")
        ):
            return "plan_review"

        # plan_draft or plan_draft_blocked → re-run planning
        if s in ("plan_draft", "plan_draft_blocked"):
            return "plan"

        # Reviewer-approved design still needs the user checkpoint on resume.
        if s in (
            "design_approved",
            "arch_design_approved",
            "design_user_checkpoint",
            "arch_design_user_checkpoint",
        ):
            return "design_checkpoint"

        # Design review-loop states (not design_draft/design_draft_blocked)
        if (
            s.startswith("design_") and s not in ("design_draft", "design_draft_blocked")
        ) or s.startswith("arch_design_"):
            return "design_review"

        # design_draft or design_draft_blocked → re-run design
        if s in ("design_draft", "design_draft_blocked"):
            return "design"

        # Code review-loop states
        if s.startswith("code_"):
            return "phases"

        return "start"

    def _resume_phase_index(self) -> int:
        """Parse the 1-based starting phase index from a phase_N_* stage name.

        For committed phases (phase_N_committed), returns N+1 because N is done.
        For in-progress phases, returns N to re-run that phase.
        """
        if not self._resume_stage or not self._resume_stage.startswith("phase_"):
            return 1
        parts = self._resume_stage.split("_")
        try:
            n = int(parts[1])
        except (IndexError, ValueError):
            return 1
        suffix = "_".join(parts[2:])
        if suffix == "committed":
            return n + 1
        return n

    @staticmethod
    def _parse_artifact_meta(art: dict) -> dict:
        """Parse metadata_json from a raw artifact row.

        list_artifacts() returns raw rows where metadata_json is a JSON string.
        """
        raw = art.get("metadata_json")
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    def _load_latest_artifact(self, artifact_type: str) -> dict | None:
        """Load the latest artifact of a given type from the work item."""
        if not self._work_item_id:
            return None
        return self._store.get_latest_artifact_by_type(self._work_item_id, artifact_type)

    def _load_completed_phase_records(
        self,
        before_phase: int | None = None,
    ) -> list[tuple[int, PhaseResult]]:
        """Load completed phase results with their original phase indexes."""
        if not self._work_item_id:
            return []
        artifacts = self._store.list_artifacts(self._work_item_id)
        phase_map: dict[int, PhaseResult] = {}
        for art in artifacts:
            if art.get("artifact_type") != "phase_result":
                continue
            phase_idx = art.get("iteration", 0)
            if before_phase is not None and phase_idx >= before_phase:
                continue
            meta = self._parse_artifact_meta(art)
            phase_map[phase_idx] = PhaseResult(
                name=meta.get("name", f"Phase {phase_idx}"),
                summary=(art.get("content") or "")[:2000],
                review_iterations=meta.get("review_iterations", 0),
                review_passed=meta.get("review_passed", False),
                committed=meta.get("committed", False),
            )
        return [(idx, phase_map[idx]) for idx in sorted(phase_map)]

    def _load_completed_phases(
        self,
        before_phase: int | None = None,
    ) -> list[PhaseResult]:
        """Load completed PhaseResult objects from artifacts table."""
        return [phase for _, phase in self._load_completed_phase_records(before_phase)]

    def _load_completed_phase_summaries(
        self,
        before_phase: int | None = None,
    ) -> list[str]:
        """Reconstruct completed_summaries from phase_result artifacts."""
        return [
            f"Phase {idx} ({phase.name}): {phase.summary[:500]}"
            for idx, phase in self._load_completed_phase_records(before_phase)
        ]

    # ════════════════════════════════════════════════════════════════
    # Main workflow
    # ════════════════════════════════════════════════════════════════

    def run(self, requirement: str) -> WorkflowResult:
        self._blocked = False
        self._failure_summary = ""
        self._blocked_stage = ""
        self._reset_review_loop_state()
        result = WorkflowResult(requirement=requirement)
        if self._cancelled():
            result.final_report = "Workflow cancelled before start."
            return result
        startup_owner = self._dev
        startup_todos = ["Produce technical design"]
        if self._mode == WorkMode.full and self._arch is not None:
            startup_owner = self._arch
            startup_todos = ["Architect produce design, then full workflow"]
        elif self._mode == WorkMode.architect_design and self._arch is not None:
            startup_owner = self._arch
            startup_todos = ["Architect produce design"]
        elif self._mode == WorkMode.testplan_design and self._qas:
            startup_owner = self._qas[0]
            startup_todos = ["QA produce test plan"]

        if not self._resume_stage:
            self._persist_task_state(
                current_stage="workflow_started",
                latest_summary=requirement,
                next_todos=startup_todos,
                updated_by_agent_id=startup_owner.agent_id,
            )
            self._append_process_event(
                event_type="task_request",
                title="Workflow requirement",
                summary=requirement,
                from_role="user",
                to_role=startup_owner.agent_role,
                updated_by_agent_id=startup_owner.agent_id,
            )
        result = self._dispatch_mode(requirement, result)
        return self._sync_result_failure_state(result)

    def _dispatch_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        if self._mode == WorkMode.full:
            return self._run_full(requirement, result)
        if self._mode == WorkMode.design:
            return self._run_design_mode(requirement, result)
        if self._mode == WorkMode.develop:
            return self._run_develop_mode(requirement, result)
        if self._mode == WorkMode.test:
            return self._run_test_mode(requirement, result)
        if self._mode == WorkMode.architect_design:
            return self._run_architect_design_mode(requirement, result)
        if self._mode == WorkMode.testplan_design:
            return self._run_testplan_design_mode(requirement, result)
        raise NotImplementedError(f"Workflow mode '{self._mode.value}' is not implemented yet.")

    def _run_architect_design_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        assert self._arch is not None

        self._emit("stage_start", "Stage 1: Technical Design", stage="design")
        self._console.print(Panel("[bold]Stage 1: Technical Design[/bold]", border_style="blue"))
        self._emit("agent_working", "Producing technical design...", stage="design", agent_role=self._arch.agent_role)
        self._console.print("[yellow]Architect producing technical design...[/yellow]")
        prompt = ARCH_TECH_DESIGN.format(requirement=requirement)
        response = self._send_agent(
            self._arch,
            prompt,
            task_description=f"Architect design: {requirement[:100]}",
            stage="design",
        )
        if not response.success:
            self._emit("agent_error", f"Agent failed (exit={response.exit_code})", stage="design", agent_role=self._arch.agent_role)
            self._console.print(f"[red]Architect agent failed (exit={response.exit_code})[/red]")
            self._record_blocked_failure(
                stage="design_draft_blocked",
                summary=f"[{self._arch.agent_role}] design draft failed (exit={response.exit_code})",
                updated_by_agent_id=self._arch.agent_id,
                from_role=self._arch.agent_role,
                to_role="myswat",
                event_type="proposal_failure",
                title="Technical design draft failed",
            )
            result.final_report = self._generate_report(result, [])
            return self._sync_result_failure_state(result)

        self._emit("agent_done", "Design draft submitted", stage="design", agent_role=self._arch.agent_role)
        design = response.content
        issue = self._validate_reviewable_design("arch_design", design)
        if issue is not None:
            result.design = design
            self._record_invalid_design_failure(
                artifact_type="arch_design",
                content=design,
                owner=self._arch,
                stage="design_draft_blocked",
                title="Technical design draft invalid",
            )
            result.final_report = self._generate_report(result, [])
            return self._sync_result_failure_state(result)
        result.design = design
        self._persist_task_state(
            current_stage="design_draft",
            latest_summary=design[:4000],
            next_todos=["Run team design review"],
            updated_by_agent_id=self._arch.agent_id,
        )
        reviewers = [self._dev, *self._qas]
        first_reviewer_role = reviewers[0].agent_role if reviewers else "reviewer"
        self._append_process_event(
            event_type="design_draft",
            title="Technical design draft",
            summary=design,
            from_role=self._arch.agent_role,
            to_role=first_reviewer_role,
            updated_by_agent_id=self._arch.agent_id,
        )
        if self._cancelled():
            result.final_report = "Architect-design workflow cancelled during technical design."
            return self._sync_result_failure_state(result)

        self._emit("stage_start", "Stage 2: Design Review", stage="design_review")
        self._console.print(Panel("[bold]Stage 2: Design Review[/bold]", border_style="blue"))
        design, iters, design_review_passed = self._run_review_loop(
            artifact=design,
            artifact_type="arch_design",
            context=f"Requirement:\n{requirement}",
            proposer=self._arch,
            reviewers=reviewers,
            abort_on_agent_failure=True,
        )
        result.design = design
        result.design_review_iterations = iters
        result.design_review_passed = design_review_passed
        review_limit_reached = self._last_review_limit_reached
        if self._cancelled():
            result.final_report = "Architect-design workflow cancelled during design review."
            return self._sync_result_failure_state(result)

        if not design_review_passed and not review_limit_reached:
            if not self._blocked:
                self._persist_task_state(
                    current_stage="design_review_failed",
                    latest_summary=design[:4000],
                    next_todos=["Review unresolved design issues"],
                    open_issues=self._first_lines(design, limit=8),
                    updated_by_agent_id=self._arch.agent_id,
                )
            result.final_report = self._generate_report(result, [])
            return self._sync_result_failure_state(result)

        self._print_markdown_panel(
            design,
            title="Reviewed Design",
            border_style="green",
            label="reviewed-design",
        )
        self._persist_user_checkpoint_state(
            artifact=design,
            artifact_type="arch_design",
            next_todo="User approve reviewed design to finish the workflow",
            updated_by_agent_id=self._arch.agent_id,
        )
        design = self._user_checkpoint(
            design,
            "arch_design",
            "Design reviewed by the team. Accept? [Y/n/or type feedback] ",
            proposer=self._arch,
        )
        if design is None:
            self._console.print("[yellow]Workflow stopped by user.[/yellow]")
            self._persist_task_state(
                current_stage="design_rejected_by_user",
                latest_summary=result.design[:4000],
                next_todos=["Review rejected design and user feedback"],
                updated_by_agent_id=self._arch.agent_id,
            )
            result.final_report = "Architect-design workflow stopped by user after design review."
            return self._sync_result_failure_state(result)

        result.design = design
        result.success = True
        self._emit("stage_start", "Final Report", stage="report")
        self._console.print(Panel("[bold]Stage 3: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, [])
        result.final_report = report
        self._print_markdown_panel(
            report,
            title="Architect Design Workflow Report",
            border_style="green",
            label="architect-design-report",
        )
        self._emit("stage_complete", "Report generated", stage="report")
        self._persist_task_state(
            current_stage="workflow_completed",
            latest_summary=report[:4000],
            next_todos=[],
            open_issues=[],
            updated_by_agent_id=self._arch.agent_id,
        )
        return self._sync_result_failure_state(result)

    def _run_testplan_design_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        assert self._arch is not None
        qa_lead = self._qas[0]

        self._emit("stage_start", "Test Plan Design", stage="testplan_design")
        self._console.print(Panel("[bold]Stage 1: Test Plan Design[/bold]", border_style="blue"))
        self._emit("agent_working", "Producing formal test plan...", stage="testplan_design", agent_role=qa_lead.agent_role)
        self._console.print("[yellow]QA producing formal test plan...[/yellow]")
        prompt = QA_DESIGN_TEST_PLAN.format(requirement=requirement)
        response = self._send_agent(
            qa_lead,
            prompt,
            task_description=f"QA test plan: {requirement[:100]}",
            stage="testplan_design",
        )
        if not response.success:
            self._emit("agent_error", f"Agent failed (exit={response.exit_code})", stage="testplan_design", agent_role=qa_lead.agent_role)
            self._console.print(f"[red]QA agent failed (exit={response.exit_code})[/red]")
            self._record_blocked_failure(
                stage="testplan_draft_blocked",
                summary=f"[{qa_lead.agent_role}] test plan draft failed (exit={response.exit_code})",
                updated_by_agent_id=qa_lead.agent_id,
                from_role=qa_lead.agent_role,
                to_role="myswat",
                event_type="proposal_failure",
                title="Test plan draft failed",
            )
            result.final_report = self._generate_report(result, [])
            return self._sync_result_failure_state(result)

        self._emit("agent_done", "Test plan draft submitted", stage="testplan_design", agent_role=qa_lead.agent_role)
        test_plan = response.content
        self._persist_task_state(
            current_stage="testplan_draft",
            latest_summary=test_plan[:4000],
            next_todos=["Run team test plan review"],
            updated_by_agent_id=qa_lead.agent_id,
        )
        reviewers = [self._arch, self._dev]
        first_reviewer_role = reviewers[0].agent_role if reviewers else "reviewer"
        self._append_process_event(
            event_type="testplan_draft",
            title="Test plan draft",
            summary=test_plan,
            from_role=qa_lead.agent_role,
            to_role=first_reviewer_role,
            updated_by_agent_id=qa_lead.agent_id,
        )
        if self._cancelled():
            result.final_report = "Testplan-design workflow cancelled during test plan drafting."
            result.ga_test = GATestResult(test_plan=test_plan)
            return self._sync_result_failure_state(result)

        self._emit("stage_start", "Test Plan Review", stage="testplan_review")
        self._console.print(Panel("[bold]Stage 2: Test Plan Review[/bold]", border_style="blue"))
        test_plan, iters, test_plan_review_passed = self._run_review_loop(
            artifact=test_plan,
            artifact_type="test_plan",
            context=f"Requirement:\n{requirement}",
            proposer=qa_lead,
            reviewers=reviewers,
            abort_on_agent_failure=True,
        )
        result.ga_test = GATestResult(
            test_plan=test_plan,
            test_plan_review_iterations=iters,
            test_plan_review_passed=test_plan_review_passed,
        )
        review_limit_reached = self._last_review_limit_reached
        if self._cancelled():
            result.final_report = "Testplan-design workflow cancelled during test plan review."
            return self._sync_result_failure_state(result)

        if not test_plan_review_passed and not review_limit_reached:
            if not self._blocked:
                self._persist_task_state(
                    current_stage="testplan_review_failed",
                    latest_summary=test_plan[:4000],
                    next_todos=["Review unresolved test plan issues"],
                    open_issues=self._first_lines(test_plan, limit=8),
                    updated_by_agent_id=qa_lead.agent_id,
                )
            result.final_report = self._generate_report(result, [])
            return self._sync_result_failure_state(result)

        self._print_markdown_panel(
            test_plan,
            title="Reviewed Test Plan",
            border_style="green",
            label="reviewed-test-plan",
        )
        test_plan = self._user_checkpoint(
            test_plan,
            "test_plan",
            "Test plan reviewed by the team. Accept? [Y/n/or type feedback] ",
            proposer=qa_lead,
        )
        if test_plan is None:
            self._console.print("[yellow]Workflow stopped by user.[/yellow]")
            self._persist_task_state(
                current_stage="testplan_rejected_by_user",
                latest_summary=result.ga_test.test_plan[:4000] if result.ga_test else "",
                next_todos=["Review rejected test plan and user feedback"],
                updated_by_agent_id=qa_lead.agent_id,
            )
            result.final_report = "Testplan-design workflow stopped by user after test plan review."
            return self._sync_result_failure_state(result)

        if result.ga_test is None:
            result.ga_test = GATestResult()
        result.ga_test.test_plan = test_plan
        result.ga_test.test_plan_review_iterations = iters
        result.ga_test.test_plan_review_passed = True
        result.success = True
        self._emit("stage_start", "Final Report", stage="report")
        self._console.print(Panel("[bold]Stage 3: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, [])
        result.final_report = report
        self._print_markdown_panel(
            report,
            title="Test Plan Workflow Report",
            border_style="green",
            label="test-plan-report",
        )
        self._emit("stage_complete", "Report generated", stage="report")
        self._persist_task_state(
            current_stage="workflow_completed",
            latest_summary=report[:4000],
            next_todos=[],
            open_issues=[],
            updated_by_agent_id=qa_lead.agent_id,
        )
        return self._sync_result_failure_state(result)

    def _run_full(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        entry = self._resume_entry_point()

        if entry == "done":
            self._console.print("[dim]Workflow already completed — nothing to resume.[/dim]")
            result.success = True
            result.final_report = "Workflow was already completed."
            return result

        if self._resume_stage:
            self._console.print(f"[dim]Resuming from stage: {self._resume_stage} (entry={entry})[/dim]")

        # ── Stage 1: Tech design ──
        if entry in ("start", "design"):
            self._emit("stage_start", "Technical Design", stage="design")
            self._console.print(Panel("[bold]Stage 1: Technical Design[/bold]", border_style="blue"))
            if self._arch is not None:
                self._emit("agent_working", "Producing technical design...", stage="design", agent_role=self._arch.agent_role)
                self._console.print("[yellow]Architect producing technical design...[/yellow]")
                prompt = ARCH_TECH_DESIGN.format(requirement=requirement)
                response = self._send_agent(
                    self._arch,
                    prompt,
                    task_description=f"Architect design: {requirement[:100]}",
                    stage="design",
                )
                if not response.success:
                    self._emit("agent_error", f"Agent failed (exit={response.exit_code})", stage="design", agent_role=self._arch.agent_role)
                    self._console.print(f"[red]Architect agent failed (exit={response.exit_code})[/red]")
                    self._record_blocked_failure(
                        stage="design_draft_blocked",
                        summary=f"[{self._arch.agent_role}] design draft failed (exit={response.exit_code})",
                        updated_by_agent_id=self._arch.agent_id,
                        from_role=self._arch.agent_role,
                        to_role="myswat",
                        event_type="proposal_failure",
                        title="Technical design draft failed",
                    )
                    result.final_report = self._generate_report(result, [])
                    return self._sync_result_failure_state(result)
                self._emit("agent_done", "Design draft submitted", stage="design", agent_role=self._arch.agent_role)
                design = response.content
                issue = self._validate_reviewable_design("arch_design", design)
                if issue is not None:
                    result.design = design
                    self._record_invalid_design_failure(
                        artifact_type="arch_design",
                        content=design,
                        owner=self._arch,
                        stage="design_draft_blocked",
                        title="Technical design draft invalid",
                    )
                    result.final_report = self._generate_report(result, [])
                    return self._sync_result_failure_state(result)
                result.design = design
                self._persist_task_state(
                    current_stage="design_draft",
                    latest_summary=design[:4000],
                    next_todos=["Run team design review"],
                    updated_by_agent_id=self._arch.agent_id,
                )
                self._append_process_event(
                    event_type="design_draft",
                    title="Technical design draft",
                    summary=design,
                    from_role=self._arch.agent_role,
                    to_role="developer",
                    updated_by_agent_id=self._arch.agent_id,
                )
            else:
                design = self._run_design(requirement)
                if not design:
                    self._console.print("[red]Dev failed to produce design. Aborting.[/red]")
                    return result
                result.design = design
            if self._cancelled():
                result.final_report = "Workflow cancelled during technical design."
                return result
        else:
            # Load design from artifacts (design_review or later)
            art = self._load_latest_artifact("design_doc")
            design = art["content"] if art else ""
            result.design = design

        # ── Stage 2: Design review ──
        if entry in ("start", "design", "design_review"):
            self._emit("stage_start", "Design Review", stage="design_review")
            self._console.print(Panel("[bold]Stage 2: Design Review[/bold]", border_style="blue"))
            if self._arch is not None:
                reviewers = [self._dev, *self._qas]
                design, iters, design_review_passed = self._run_review_loop(
                    artifact=design,
                    artifact_type="arch_design",
                    context=f"Requirement:\n{requirement}",
                    proposer=self._arch,
                    reviewers=reviewers,
                    abort_on_agent_failure=True,
                )
            else:
                design, iters, design_review_passed = self._run_review_loop(
                    artifact=design,
                    artifact_type="design",
                    context=f"Requirement:\n{requirement}",
                )
            result.design = design
            result.design_review_iterations = iters
            result.design_review_passed = design_review_passed
            review_limit_reached = self._last_review_limit_reached
            if self._cancelled():
                result.final_report = "Workflow cancelled during design review."
                return result

            if not design_review_passed and not review_limit_reached:
                if not self._blocked:
                    self._persist_task_state(
                        current_stage="design_review_failed",
                        latest_summary=design[:4000],
                        next_todos=["Review unresolved design issues"],
                        open_issues=self._first_lines(design, limit=8),
                        updated_by_agent_id=(self._arch or self._dev).agent_id,
                    )
                result.final_report = self._generate_report(result, [])
                return self._sync_result_failure_state(result)
        elif entry == "design_checkpoint":
            result.design_review_passed = True
        else:
            result.design_review_passed = True

        if entry in ("start", "design", "design_review", "design_checkpoint"):
            checkpoint_title = "Reviewed Design" if self._arch else "QA-Approved Design"
            checkpoint_artifact_type = "arch_design" if self._arch else "design"
            checkpoint_owner = (self._arch or self._dev)
            self._print_markdown_panel(
                design,
                title=checkpoint_title,
                border_style="green",
                label="approved-design",
            )
            self._persist_user_checkpoint_state(
                artifact=design,
                artifact_type=checkpoint_artifact_type,
                next_todo="User approve reviewed design to proceed to planning",
                updated_by_agent_id=checkpoint_owner.agent_id,
            )
            design = self._user_checkpoint(
                design,
                checkpoint_artifact_type,
                "Design reviewed by the team. Proceed to planning? [Y/n/or type feedback] ",
                proposer=self._arch,
            )
            if design is None:
                self._console.print("[yellow]Workflow stopped by user.[/yellow]")
                self._persist_task_state(
                    current_stage="design_rejected_by_user",
                    latest_summary=result.design[:4000],
                    next_todos=["Review rejected design and user feedback"],
                    updated_by_agent_id=checkpoint_owner.agent_id,
                )
                result.final_report = "Workflow stopped by user after design review."
                return result
            result.design = design

        # ── Stage 3: Implementation planning ──
        if entry in ("start", "design", "design_review", "design_checkpoint", "plan"):
            self._emit("stage_start", "Implementation Planning", stage="planning")
            self._console.print(Panel("[bold]Stage 3: Implementation Planning[/bold]", border_style="blue"))
            plan = self._run_planning(design, requirement)
            if not plan:
                self._console.print("[red]Dev failed to produce plan. Aborting.[/red]")
                return result
            result.plan = plan
            if self._cancelled():
                result.final_report = "Workflow cancelled during planning."
                return result
        else:
            # Load plan from artifacts (plan_review or later)
            art = self._load_latest_artifact("proposal")
            plan = art["content"] if art else ""
            result.plan = plan

        # ── Stage 4: Plan review ──
        if entry in ("start", "design", "design_review", "design_checkpoint", "plan", "plan_review"):
            self._emit("stage_start", "Plan Review", stage="plan_review")
            self._console.print(Panel("[bold]Stage 4: Plan Review[/bold]", border_style="blue"))
            plan, iters, plan_review_passed = self._run_review_loop(
                artifact=plan,
                artifact_type="plan",
                context=f"Requirement:\n{requirement[:2000]}\n\nApproved Design:\n{design[:4000]}",
            )
            result.plan = plan
            result.plan_review_iterations = iters
            result.plan_review_passed = plan_review_passed
            if self._cancelled():
                result.final_report = "Workflow cancelled during plan review."
                return result

        elif entry == "plan_checkpoint":
            result.plan_review_passed = True
        else:
            result.plan_review_passed = True

        if entry in (
            "start",
            "design",
            "design_review",
            "design_checkpoint",
            "plan",
            "plan_review",
            "plan_checkpoint",
        ):
            self._print_markdown_panel(
                plan,
                title="QA-Approved Plan",
                border_style="green",
                label="approved-plan",
            )
            self._persist_user_checkpoint_state(
                artifact=plan,
                artifact_type="plan",
                next_todo="User approve reviewed plan to start development",
                updated_by_agent_id=self._dev.agent_id,
            )
            plan = self._user_checkpoint(
                plan, "plan", "Plan approved by QA. Start development? [Y/n/or type feedback] "
            )
            if plan is None:
                self._console.print("[yellow]Workflow stopped by user.[/yellow]")
                return result
            result.plan = plan

        # ── Stage 5: Phased development ──
        if entry in (
            "start",
            "design",
            "design_review",
            "design_checkpoint",
            "plan",
            "plan_review",
            "plan_checkpoint",
            "phases",
        ):
            self._emit("stage_start", "Development", stage="development")
            self._console.print(Panel("[bold]Stage 5: Development[/bold]", border_style="blue"))
            phases = self._parse_phases(plan)
            self._console.print(f"[dim]Parsed {len(phases)} phase(s) from plan.[/dim]")

            start_phase = self._resume_phase_index() if entry == "phases" else 1
            # Reconstruct state for already-completed phases
            completed_summaries = self._load_completed_phase_summaries(start_phase) if start_phase > 1 else []
            result.phases = self._load_completed_phases(start_phase) if start_phase > 1 else []

            for i, phase in enumerate(phases, 1):
                if i < start_phase:
                    self._console.print(f"[dim]Phase {i}/{len(phases)}: already completed, skipping[/dim]")
                    continue
                if self._cancelled():
                    result.final_report = f"Workflow cancelled before phase {i}."
                    return result
                self._console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
                self._console.print(f"[bold cyan]Phase {i}/{len(phases)}: {phase}[/bold cyan]")
                self._console.print(f"[bold cyan]{'='*60}[/bold cyan]")

                phase_result = self._run_phase(
                    phase_name=phase,
                    phase_index=i,
                    total_phases=len(phases),
                    requirement=requirement,
                    design=design,
                    plan=plan,
                    completed_summaries=completed_summaries,
                )
                result.phases.append(phase_result)
                completed_summaries.append(f"Phase {i} ({phase}): {phase_result.summary[:500]}")
                if self._cancelled():
                    result.final_report = f"Workflow cancelled during phase {i}."
                    return result
        else:
            # All phases done — reconstruct from artifacts
            completed_summaries = self._load_completed_phase_summaries()
            result.phases = self._load_completed_phases()

        # ── Stage 6: GA Test ──
        self._emit("stage_start", "GA Test", stage="ga_test")
        self._console.print(Panel("[bold]Stage 6: GA Test[/bold]", border_style="blue"))
        dev_summary = "\n\n".join(completed_summaries)
        ga_result = self._run_ga_test_phase(requirement, design, plan, dev_summary)
        result.ga_test = ga_result
        if self._cancelled():
            result.final_report = "Workflow cancelled during GA testing."
            return result

        if ga_result.aborted:
            self._real_console.print(Panel(
                f"[bold red]GA Test aborted.[/bold red]\n"
                f"Bugs found: {ga_result.bugs_found}, Fixed: {ga_result.bugs_fixed}\n"
                f"User intervention required.",
                title="GA Test Aborted",
                border_style="red",
            ))
        elif ga_result.passed:
            self._console.print("[bold green]GA Test passed![/bold green]")

        # ── Stage 7: Final report ──
        self._emit("stage_start", "Final Report", stage="report")
        self._console.print(Panel("[bold]Stage 7: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, completed_summaries)
        result.final_report = report
        result.success = (
            all(p.committed for p in result.phases)
            and ga_result.passed
        ) if result.phases else False

        self._print_markdown_panel(
            report,
            title="E2E Workflow Report",
            border_style="green",
            label="e2e-workflow-report",
        )
        self._emit("stage_complete", "Report generated", stage="report")
        self._persist_task_state(
            current_stage="workflow_completed" if result.success else "workflow_finished_with_issues",
            latest_summary=report[:4000],
            next_todos=[] if result.success else ["Review final report and unresolved issues"],
            open_issues=[] if result.success else self._first_lines(report, limit=8),
            updated_by_agent_id=self._dev.agent_id,
        )
        return result

    def _run_design_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        entry = self._resume_entry_point()

        if entry == "done":
            self._console.print("[dim]Workflow already completed — nothing to resume.[/dim]")
            result.success = True
            result.final_report = "Workflow was already completed."
            return result

        if self._resume_stage:
            self._console.print(f"[dim]Resuming from stage: {self._resume_stage} (entry={entry})[/dim]")

        # Stage 1: Tech design
        if entry in ("start", "design"):
            self._emit("stage_start", "Technical Design", stage="design")
            self._console.print(Panel("[bold]Stage 1: Technical Design[/bold]", border_style="blue"))
            design = self._run_design(requirement)
            if not design:
                self._console.print("[red]Dev failed to produce design. Aborting.[/red]")
                result.final_report = "Design workflow failed: developer did not produce a technical design."
                return result
            result.design = design
            if self._cancelled():
                result.final_report = "Design workflow cancelled during technical design."
                return result
        else:
            art = self._load_latest_artifact("design_doc")
            design = art["content"] if art else ""
            result.design = design

        # Stage 2: Design review
        if entry in ("start", "design", "design_review"):
            self._emit("stage_start", "Design Review", stage="design_review")
            self._console.print(Panel("[bold]Stage 2: Design Review[/bold]", border_style="blue"))
            design, iters, design_review_passed = self._run_review_loop(
                artifact=design,
                artifact_type="design",
                context=f"Requirement:\n{requirement}",
            )
            result.design = design
            result.design_review_iterations = iters
            result.design_review_passed = design_review_passed
            if self._cancelled():
                result.final_report = "Design workflow cancelled during design review."
                return result
        elif entry == "design_checkpoint":
            result.design_review_passed = True
        else:
            result.design_review_passed = True

        if entry in ("start", "design", "design_review", "design_checkpoint"):
            self._print_markdown_panel(
                design,
                title="Reviewed Design",
                border_style="green",
                label="reviewed-design",
            )
            self._persist_user_checkpoint_state(
                artifact=design,
                artifact_type="design",
                next_todo="User approve reviewed design to proceed to planning",
                updated_by_agent_id=self._dev.agent_id,
            )
            design = self._user_checkpoint(
                design,
                "design",
                "Design reviewed by QA. Proceed to planning? [Y/n/or type feedback] ",
            )
            if design is None:
                self._console.print("[yellow]Workflow stopped by user.[/yellow]")
                result.final_report = "Design workflow stopped by user after design review."
                return result
            result.design = design

        # Stage 3: Planning
        if entry in ("start", "design", "design_review", "design_checkpoint", "plan"):
            self._emit("stage_start", "Implementation Planning", stage="planning")
            self._console.print(Panel("[bold]Stage 3: Implementation Planning[/bold]", border_style="blue"))
            plan = self._run_planning(design, requirement)
            if not plan:
                self._console.print("[red]Dev failed to produce plan. Aborting.[/red]")
                result.final_report = "Design workflow failed: developer did not produce an implementation plan."
                return result
            result.plan = plan
            if self._cancelled():
                result.final_report = "Design workflow cancelled during planning."
                return result
        else:
            art = self._load_latest_artifact("proposal")
            plan = art["content"] if art else ""
            result.plan = plan

        # Stage 4: Plan review
        if entry in ("start", "design", "design_review", "design_checkpoint", "plan", "plan_review"):
            self._emit("stage_start", "Plan Review", stage="plan_review")
            self._console.print(Panel("[bold]Stage 4: Plan Review[/bold]", border_style="blue"))
            plan, iters, plan_review_passed = self._run_review_loop(
                artifact=plan,
                artifact_type="plan",
                context=f"Requirement:\n{requirement[:2000]}\n\nApproved Design:\n{design[:4000]}",
            )
            result.plan = plan
            result.plan_review_iterations = iters
            result.plan_review_passed = plan_review_passed
            if self._cancelled():
                result.final_report = "Design workflow cancelled during plan review."
                return result
        elif entry == "plan_checkpoint":
            result.plan_review_passed = True
        else:
            result.plan_review_passed = True

        if entry in (
            "start",
            "design",
            "design_review",
            "design_checkpoint",
            "plan",
            "plan_review",
            "plan_checkpoint",
        ):
            self._print_markdown_panel(
                plan,
                title="Reviewed Plan",
                border_style="green",
                label="reviewed-plan",
            )
            self._persist_user_checkpoint_state(
                artifact=plan,
                artifact_type="plan",
                next_todo="User approve reviewed plan to finish the design workflow",
                updated_by_agent_id=self._dev.agent_id,
            )
            plan = self._user_checkpoint(
                plan,
                "plan",
                "Plan reviewed by QA. Finish design workflow? [Y/n/or type feedback] ",
            )
            if plan is None:
                self._console.print("[yellow]Workflow stopped by user.[/yellow]")
                result.final_report = "Design workflow stopped by user after plan review."
                return result
            result.plan = plan

        self._emit("stage_start", "Final Report", stage="report")
        self._console.print(Panel("[bold]Stage 5: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, [])
        result.final_report = report
        result.success = result.design_review_passed and result.plan_review_passed

        self._print_markdown_panel(
            report,
            title="Design Workflow Report",
            border_style="green",
            label="design-workflow-report",
        )
        self._emit("stage_complete", "Report generated", stage="report")
        self._persist_task_state(
            current_stage="workflow_completed" if result.success else "workflow_finished_with_issues",
            latest_summary=report[:4000],
            next_todos=[] if result.success else ["Review design report and unresolved issues"],
            open_issues=[] if result.success else self._first_lines(report, limit=8),
            updated_by_agent_id=self._dev.agent_id,
        )
        return result

    def _run_develop_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        entry = self._resume_entry_point()

        if entry == "done":
            self._console.print("[dim]Workflow already completed — nothing to resume.[/dim]")
            result.success = True
            result.final_report = "Workflow was already completed."
            return result

        if self._resume_stage:
            self._console.print(f"[dim]Resuming from stage: {self._resume_stage} (entry={entry})[/dim]")

        self._emit("stage_start", "Development", stage="development")
        self._console.print(Panel("[bold]Stage 1: Development[/bold]", border_style="blue"))
        phases = self._parse_phases(requirement)
        self._console.print(f"[dim]Parsed {len(phases)} phase(s) from requirement.[/dim]")

        start_phase = self._resume_phase_index() if entry == "phases" else 1
        completed_summaries = self._load_completed_phase_summaries(start_phase) if start_phase > 1 else []
        result.phases = self._load_completed_phases(start_phase) if start_phase > 1 else []

        for i, phase in enumerate(phases, 1):
            if i < start_phase:
                self._console.print(f"[dim]Phase {i}/{len(phases)}: already completed, skipping[/dim]")
                continue
            if self._cancelled():
                result.final_report = f"Development workflow cancelled before phase {i}."
                return result
            self._console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            self._console.print(f"[bold cyan]Phase {i}/{len(phases)}: {phase}[/bold cyan]")
            self._console.print(f"[bold cyan]{'='*60}[/bold cyan]")

            phase_result = self._run_phase(
                phase_name=phase,
                phase_index=i,
                total_phases=len(phases),
                requirement=requirement,
                design=requirement,
                plan=requirement,
                completed_summaries=completed_summaries,
            )
            result.phases.append(phase_result)
            completed_summaries.append(f"Phase {i} ({phase}): {phase_result.summary[:500]}")
            if self._cancelled():
                result.final_report = f"Development workflow cancelled during phase {i}."
                return result

        self._emit("stage_start", "Final Report", stage="report")
        self._console.print(Panel("[bold]Stage 2: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, completed_summaries)
        result.final_report = report
        result.success = all(p.committed for p in result.phases) if result.phases else False

        self._print_markdown_panel(
            report,
            title="Development Workflow Report",
            border_style="green",
            label="development-workflow-report",
        )
        self._emit("stage_complete", "Report generated", stage="report")
        self._persist_task_state(
            current_stage="workflow_completed" if result.success else "workflow_finished_with_issues",
            latest_summary=report[:4000],
            next_todos=[] if result.success else ["Review development report and unresolved issues"],
            open_issues=[] if result.success else self._first_lines(report, limit=8),
            updated_by_agent_id=self._dev.agent_id,
        )
        return result

    def _run_test_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult:
        entry = self._resume_entry_point()

        if entry == "done":
            self._console.print("[dim]Workflow already completed — nothing to resume.[/dim]")
            result.success = True
            result.final_report = "Workflow was already completed."
            return result

        if self._resume_stage:
            self._console.print(f"[dim]Resuming from stage: {self._resume_stage} (entry={entry})[/dim]")

        # GA test always re-runs from start (sub-stage resume not yet supported)
        self._emit("stage_start", "GA Test", stage="ga_test")
        self._console.print(Panel("[bold]Stage 1: GA Test[/bold]", border_style="blue"))
        ga_result = self._run_ga_test_phase(
            requirement,
            requirement,
            "",
            requirement,
            allow_arch_fix=False,
        )
        result.ga_test = ga_result
        if self._cancelled():
            result.final_report = "Test workflow cancelled during GA testing."
            return result

        self._emit("stage_start", "Final Report", stage="report")
        self._console.print(Panel("[bold]Stage 2: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, [])
        result.final_report = report
        result.success = ga_result.passed

        self._print_markdown_panel(
            report,
            title="Test Workflow Report",
            border_style="green",
            label="test-workflow-report",
        )
        self._emit("stage_complete", "Report generated", stage="report")
        self._persist_task_state(
            current_stage="workflow_completed" if result.success else "workflow_finished_with_issues",
            latest_summary=report[:4000],
            next_todos=[] if result.success else ["Review test report and unresolved issues"],
            open_issues=[] if result.success else self._first_lines(report, limit=8),
            updated_by_agent_id=self._dev.agent_id,
        )
        return result

    # ════════════════════════════════════════════════════════════════
    # Stage implementations
    # ════════════════════════════════════════════════════════════════

    def _run_design(self, requirement: str, abort_on_failure: bool = False) -> str:
        if self._cancelled():
            return ""
        self._emit("agent_working", "Producing technical design...", stage="design", agent_role=self._dev.agent_role)
        self._console.print("[yellow]Dev producing technical design...[/yellow]")
        prompt = DEV_TECH_DESIGN.format(requirement=requirement)
        response = self._send_agent(
            self._dev,
            prompt,
            task_description=f"Tech design: {requirement[:100]}",
            stage="design",
        )
        if not response.success:
            self._emit("agent_error", f"Agent failed (exit={response.exit_code})", stage="design", agent_role=self._dev.agent_role)
            self._console.print(f"[red]Dev agent failed (exit={response.exit_code})[/red]")
            if abort_on_failure:
                self._record_blocked_failure(
                    stage="design_draft_blocked",
                    summary=f"[{self._dev.agent_role}] design draft failed (exit={response.exit_code})",
                    updated_by_agent_id=self._dev.agent_id,
                    from_role=self._dev.agent_role,
                    to_role="myswat",
                    event_type="proposal_failure",
                    title="Technical design draft failed",
                )
            return ""
        self._emit("agent_done", "Design draft submitted", stage="design", agent_role=self._dev.agent_role)
        self._console.print("[green]Dev submitted design.[/green]")
        self._persist_task_state(
            current_stage="design_draft",
            latest_summary=response.content[:4000],
            next_todos=["Run QA design review"],
            updated_by_agent_id=self._dev.agent_id,
        )
        self._append_process_event(
            event_type="design_draft",
            title="Technical design draft",
            summary=response.content,
            from_role=self._dev.agent_role,
            to_role="qa",
            updated_by_agent_id=self._dev.agent_id,
        )
        return response.content

    def _run_planning(self, design: str, requirement: str, abort_on_failure: bool = False) -> str:
        if self._cancelled():
            return ""
        self._emit("agent_working", "Creating implementation plan...", stage="planning", agent_role=self._dev.agent_role)
        self._console.print("[yellow]Dev creating implementation plan...[/yellow]")
        prompt = DEV_IMPLEMENTATION_PLAN.format(
            requirement=requirement[:4000],
            design=design[:8000],
        )
        response = self._send_agent(
            self._dev,
            prompt,
            task_description="Implementation planning",
            stage="planning",
        )
        if not response.success:
            self._console.print(f"[red]Dev agent failed (exit={response.exit_code})[/red]")
            if abort_on_failure:
                self._record_blocked_failure(
                    stage="plan_draft_blocked",
                    summary=f"[{self._dev.agent_role}] implementation plan failed (exit={response.exit_code})",
                    updated_by_agent_id=self._dev.agent_id,
                    from_role=self._dev.agent_role,
                    to_role="myswat",
                    event_type="proposal_failure",
                    title="Implementation plan draft failed",
                )
            return ""
        self._emit("agent_done", "Implementation plan submitted", stage="planning", agent_role=self._dev.agent_role)
        self._console.print("[green]Dev submitted implementation plan.[/green]")
        self._persist_task_state(
            current_stage="plan_draft",
            latest_summary=response.content[:4000],
            next_todos=["Run QA plan review"],
            updated_by_agent_id=self._dev.agent_id,
        )
        self._append_process_event(
            event_type="plan_draft",
            title="Implementation plan draft",
            summary=response.content,
            from_role=self._dev.agent_role,
            to_role="qa",
            updated_by_agent_id=self._dev.agent_id,
        )
        return response.content

    def _run_phase(
        self,
        phase_name: str,
        phase_index: int,
        total_phases: int,
        requirement: str,
        design: str,
        plan: str,
        completed_summaries: list[str],
        *,
        persist_result: bool = True,
    ) -> PhaseResult:
        if self._cancelled():
            return PhaseResult(name=phase_name, summary="Cancelled by user.", review_iterations=0)
        self._emit("phase_start", f"Phase {phase_index}/{total_phases}: {phase_name}", stage=f"phase_{phase_index}",
                    phase_index=phase_index, total_phases=total_phases, phase_name=phase_name)
        completed_ctx = "\n".join(completed_summaries) if completed_summaries else "None yet."

        # Step 1: Dev implements
        self._emit("agent_working", "Implementing...", stage=f"phase_{phase_index}", agent_role=self._dev.agent_role)
        self._console.print("[yellow]Dev implementing...[/yellow]")
        self._persist_task_state(
            current_stage=f"phase_{phase_index}_implementing",
            latest_summary=phase_name,
            next_todos=[f"Complete phase {phase_index}: {phase_name}", "Produce a handoff summary for QA"],
            updated_by_agent_id=self._dev.agent_id,
        )
        prompt = DEV_IMPLEMENT_PHASE.format(
            requirement=requirement[:2000],
            design=design[:3000],
            plan=plan[:3000],
            phase_name=phase_name,
            phase_index=phase_index,
            total_phases=total_phases,
            completed_phases=completed_ctx,
        )
        response = self._send_agent(
            self._dev,
            prompt,
            task_description=f"Phase {phase_index}: {phase_name}",
            stage=f"phase_{phase_index}",
        )
        if not response.success:
            self._emit("agent_error", "Implementation failed", stage=f"phase_{phase_index}", agent_role=self._dev.agent_role)
            return PhaseResult(name=phase_name, summary="Implementation failed.", review_iterations=0)
        if self._cancelled():
            return PhaseResult(name=phase_name, summary="Cancelled by user.", review_iterations=0)
        self._emit("agent_done", "Implementation complete", stage=f"phase_{phase_index}", agent_role=self._dev.agent_role)

        # Step 2: Dev summarizes
        self._emit("agent_working", "Summarizing changes...", stage=f"phase_{phase_index}", agent_role=self._dev.agent_role)
        self._console.print("[yellow]Dev summarizing changes...[/yellow]")
        summary_prompt = DEV_SUMMARIZE_PHASE.format(
            phase_name=phase_name,
            phase_index=phase_index,
        )
        summary_resp = self._send_agent(
            self._dev,
            summary_prompt,
            task_description=f"Summarize phase {phase_index}",
            stage=f"phase_{phase_index}",
        )
        summary = summary_resp.content if summary_resp.success else response.content
        self._emit("agent_done", "Summary submitted for review", stage=f"phase_{phase_index}", agent_role=self._dev.agent_role)
        self._persist_task_state(
            current_stage=f"phase_{phase_index}_under_review",
            latest_summary=summary[:4000],
            next_todos=["QA review this phase summary and inspect the codebase"],
            updated_by_agent_id=self._dev.agent_id,
        )
        self._append_process_event(
            event_type="phase_summary",
            title=f"Phase {phase_index} summary",
            summary=summary,
            from_role=self._dev.agent_role,
            to_role="qa",
            updated_by_agent_id=self._dev.agent_id,
        )

        # Step 3: QA review loop
        reviewed_summary, review_iters, review_passed = self._run_review_loop(
            artifact=summary,
            artifact_type="code",
            context=(
                f"Phase {phase_index}/{total_phases}: {phase_name}\n\n"
                f"Requirement (brief):\n{requirement[:1000]}"
            ),
            stage_prefix=f"phase_{phase_index}_code",
        )
        if self._cancelled():
            return PhaseResult(
                name=phase_name,
                summary="Cancelled by user.",
                review_iterations=review_iters,
                review_passed=review_passed,
            )

        # Step 4: Dev commits
        self._emit("agent_working", f"Committing phase {phase_index}...", stage=f"phase_{phase_index}", agent_role=self._dev.agent_role)
        self._console.print(f"[yellow]Dev committing phase {phase_index}...[/yellow]")
        commit_prompt = DEV_COMMIT_PHASE.format(
            phase_name=phase_name,
            phase_index=phase_index,
        )
        commit_resp = self._send_agent(
            self._dev,
            commit_prompt,
            task_description=f"Commit phase {phase_index}",
            stage=f"phase_{phase_index}",
        )
        committed = commit_resp.success

        if committed:
            self._emit("phase_done", f"Phase {phase_index} committed", stage=f"phase_{phase_index}",
                        status="committed", review_iterations=review_iters, review_passed=review_passed,
                        phase_index=phase_index, total_phases=total_phases, phase_name=phase_name)
            self._console.print(f"[bold green]Phase {phase_index} committed.[/bold green]")
        else:
            self._emit("phase_done", f"Phase {phase_index} commit failed", stage=f"phase_{phase_index}",
                        status="failed", review_iterations=review_iters, review_passed=review_passed,
                        phase_index=phase_index, total_phases=total_phases, phase_name=phase_name)
            self._console.print(f"[red]Phase {phase_index} commit failed.[/red]")

        self._persist_task_state(
            current_stage=f"phase_{phase_index}_committed" if committed else f"phase_{phase_index}_commit_failed",
            latest_summary=reviewed_summary[:4000],
            next_todos=[] if committed else [f"Revisit phase {phase_index} commit failure"],
            updated_by_agent_id=self._dev.agent_id,
        )

        phase_result = PhaseResult(
            name=phase_name,
            summary=reviewed_summary[:2000],
            review_iterations=review_iters,
            review_passed=review_passed,
            committed=committed,
        )

        # Persist phase result as artifact for workflow resume reconstruction.
        # Skipped for bug-fix subworkflow phases to avoid overwriting main phases.
        if persist_result and self._work_item_id:
            try:
                self._store.create_artifact(
                    work_item_id=self._work_item_id,
                    agent_id=self._dev.agent_id,
                    iteration=phase_index,
                    artifact_type="phase_result",
                    title=f"Phase {phase_index}: {phase_name}",
                    content=reviewed_summary[:65000],
                    metadata_json={
                        "name": phase_name,
                        "review_iterations": review_iters,
                        "review_passed": review_passed,
                        "committed": committed,
                    },
                )
            except Exception as e:
                self._console.print(f"[dim red]Warning: Failed to persist phase result: {e}[/dim red]")

        return phase_result

    # ════════════════════════════════════════════════════════════════
    # GA Test phase
    # ════════════════════════════════════════════════════════════════

    def _run_ga_test_phase(
        self,
        requirement: str,
        design: str,
        plan: str,
        dev_summary: str,
        *,
        allow_arch_fix: bool = True,
    ) -> GATestResult:
        result = GATestResult()
        if self._cancelled():
            result.aborted = True
            return result
        qa_lead = self._qas[0]
        self._persist_task_state(
            current_stage="ga_test_planning",
            latest_summary=dev_summary[:4000],
            next_todos=["Generate GA test plan"],
            updated_by_agent_id=qa_lead.agent_id,
        )

        # Step 1: QA generates test plan
        self._emit("agent_working", "Generating GA test plan...", stage="ga_test", agent_role=qa_lead.agent_role)
        self._console.print("[yellow]QA generating GA test plan...[/yellow]")
        prompt = QA_GA_TEST_PLAN.format(
            requirement=requirement[:2000],
            design=design[:3000],
            dev_summary=dev_summary[:4000],
        )
        response = self._send_agent(
            qa_lead,
            prompt,
            task_description="GA test plan",
            stage="ga_test",
        )
        if not response.success:
            self._emit("agent_error", "Failed to generate test plan", stage="ga_test", agent_role=qa_lead.agent_role)
            self._console.print("[red]QA failed to generate test plan.[/red]")
            return result
        if self._cancelled():
            result.aborted = True
            return result
        self._emit("agent_done", "GA test plan submitted", stage="ga_test", agent_role=qa_lead.agent_role)
        test_plan = response.content
        result.test_plan = test_plan
        self._persist_task_state(
            current_stage="ga_test_plan_draft",
            latest_summary=test_plan[:4000],
            next_todos=["Review test plan", "Approve and start testing"],
            updated_by_agent_id=qa_lead.agent_id,
        )
        self._append_process_event(
            event_type="ga_test_plan",
            title="GA test plan",
            summary=test_plan,
            from_role=qa_lead.agent_role,
            to_role=self._dev.agent_role,
            updated_by_agent_id=qa_lead.agent_id,
        )

        # Step 2: Dev + User review test plan
        self._emit("stage_start", "Test Plan Review", stage="ga_test_review")
        self._console.print(Panel("[bold]Test Plan Review[/bold]", border_style="cyan"))
        test_plan, iters, test_plan_review_passed = self._run_review_loop(
            artifact=test_plan,
            artifact_type="test_plan",
            context=f"Requirement:\n{requirement[:2000]}",
            proposer=qa_lead,
            reviewers=[self._dev],
        )
        result.test_plan = test_plan
        result.test_plan_review_iterations = iters
        result.test_plan_review_passed = test_plan_review_passed

        # User checkpoint on test plan
        self._print_markdown_panel(
            test_plan,
            title="Reviewed Test Plan",
            border_style="green",
            label="reviewed-test-plan",
        )
        test_plan = self._user_checkpoint(
            test_plan, "test_plan",
            "Test plan reviewed by Dev. Approve and start testing? [Y/n/or type feedback] ",
            proposer=qa_lead,
        )
        if test_plan is None:
            result.aborted = True
            return result
        result.test_plan = test_plan
        self._persist_task_state(
            current_stage="ga_test_executing",
            latest_summary=test_plan[:4000],
            next_todos=["Execute approved GA tests"],
            updated_by_agent_id=qa_lead.agent_id,
        )

        # Step 3: QA executes tests
        self._emit("agent_working", "Executing GA tests...", stage="ga_test", agent_role=qa_lead.agent_role)
        self._console.print("[yellow]QA executing GA tests...[/yellow]")
        exec_prompt = QA_EXECUTE_GA_TEST.format(test_plan=test_plan[:8000])
        exec_response = self._send_agent(
            qa_lead,
            exec_prompt,
            task_description="Execute GA tests",
            stage="ga_test",
        )
        if not exec_response.success:
            self._emit("agent_error", "Failed to execute tests", stage="ga_test", agent_role=qa_lead.agent_role)
            self._console.print("[red]QA failed to execute tests.[/red]")
            return result
        if self._cancelled():
            result.aborted = True
            return result
        self._emit("agent_done", "GA tests executed", stage="ga_test", agent_role=qa_lead.agent_role)

        test_output = exec_response.content
        bugs = self._parse_test_results(test_output)
        test_history = [f"Initial run: {test_output[:2000]}"]

        if not bugs:
            self._console.print("[bold green]All GA tests passed![/bold green]")
            result.test_report = test_output
            result.passed = True
            self._persist_task_state(
                current_stage="ga_test_passed",
                latest_summary=test_output[:4000],
                next_todos=["Generate final report"],
                updated_by_agent_id=qa_lead.agent_id,
            )
            return result

        # Step 4: Bug fix loop
        result.bugs_found = len(bugs)
        self._console.print(f"[bold red]{len(bugs)} bug(s) found in GA test.[/bold red]")

        ga_review_round = 0
        while bugs:
            if self._cancelled():
                result.aborted = True
                break
            ga_review_round += 1
            if ga_review_round > self._ga_test_review_limit:
                self._emit(
                    "warning",
                    "GA test review limit reached; stopping automatic bug-fix rounds.",
                    stage="ga_test",
                    agent_role=qa_lead.agent_role,
                    iteration=ga_review_round - 1,
                    max_iterations=self._ga_test_review_limit,
                    review_skipped=True,
                )
                self._console.print(
                    "[yellow]GA test review limit reached; stopping automatic bug-fix rounds.[/yellow]"
                )
                result.aborted = True
                self._persist_task_state(
                    current_stage="ga_test_review_limit_reached",
                    latest_summary=test_output[:4000],
                    next_todos=["Review unresolved GA findings and decide next action"],
                    open_issues=[bug.get("title", "Unknown bug") for bug in bugs[:10]],
                    updated_by_agent_id=qa_lead.agent_id,
                )
                break
            if result.bugs_found > MAX_GA_BUGS:
                self._console.print(
                    f"\n[bold red]More than {MAX_GA_BUGS} bugs found ({result.bugs_found} total). "
                    f"Stopping GA test phase.[/bold red]"
                )
                self._console.print(
                    "[yellow]Too many bugs indicate deeper issues. "
                    "Please review the situation and decide how to proceed.[/yellow]"
                )
                result.aborted = True
                self._persist_task_state(
                    current_stage="ga_test_aborted",
                    latest_summary=test_output[:4000],
                    next_todos=["Review too many bugs and decide next action"],
                    open_issues=[bug.get("title", "Unknown bug") for bug in bugs[:10]],
                    updated_by_agent_id=qa_lead.agent_id,
                )
                break

            for bug in bugs:
                self._console.print(
                    f"\n[bold red]Bug: {bug.get('title', 'Unknown')} "
                    f"[{bug.get('severity', '?')}][/bold red]"
                )
                if bug.get("description"):
                    self._console.print(f"[dim]{bug['description'][:200]}[/dim]")

                bug_fix = self._run_bug_fix(
                    bug,
                    requirement,
                    design,
                    allow_arch_fix=allow_arch_fix,
                )
                result.bug_fixes.append(bug_fix)
                if bug_fix.fixed:
                    result.bugs_fixed += 1
                    self._console.print(f"[green]Bug fixed: {bug_fix.title}[/green]")
                else:
                    self._console.print(f"[red]Bug fix failed: {bug_fix.title}[/red]")

            unresolved_arch_changes = [
                bf for bf in result.bug_fixes
                if bf.arch_change and not bf.fixed
            ]
            if unresolved_arch_changes and not allow_arch_fix:
                self._persist_task_state(
                    current_stage="ga_test_arch_change_required",
                    latest_summary=test_output[:4000],
                    next_todos=["Review architecture-change findings and plan follow-up work"],
                    open_issues=[bf.title for bf in unresolved_arch_changes[:10]],
                    updated_by_agent_id=qa_lead.agent_id,
                )
                self._console.print(
                    "[yellow]Architecture-change findings recorded for follow-up. "
                    "Skipping automatic re-test.[/yellow]"
                )
                break

            # QA re-tests after all fixes in this round
            self._console.print("\n[yellow]QA re-running tests after bug fixes...[/yellow]")
            fixed_summaries = "\n".join(
                f"- {bf.title}: {bf.summary[:200]}"
                for bf in result.bug_fixes if bf.fixed
            )
            continue_prompt = QA_CONTINUE_GA_TEST.format(
                test_plan=test_plan[:4000],
                fixed_bugs=fixed_summaries,
            )
            continue_response = self._send_agent(
                qa_lead,
                continue_prompt,
                task_description="Continue GA tests",
                stage="ga_test",
            )
            if not continue_response.success:
                self._console.print("[red]QA failed to continue tests.[/red]")
                break
            if self._cancelled():
                result.aborted = True
                break

            test_output = continue_response.content
            test_history.append(f"Re-test: {test_output[:2000]}")
            bugs = self._parse_test_results(test_output)

            if bugs:
                result.bugs_found += len(bugs)
                self._console.print(
                    f"[bold red]{len(bugs)} new bug(s) found "
                    f"({result.bugs_found} total).[/bold red]"
                )
                self._persist_task_state(
                    current_stage="ga_test_bug_fixing",
                    latest_summary=test_output[:4000],
                    next_todos=["Fix newly found bugs", "Re-run QA tests"],
                    open_issues=[bug.get("title", "Unknown bug") for bug in bugs[:10]],
                    updated_by_agent_id=qa_lead.agent_id,
                )
            else:
                self._console.print("[bold green]All tests pass after bug fixes![/bold green]")

        if not result.aborted and not bugs:
            result.passed = True

        # QA generates final test report
        self._console.print("[yellow]QA generating test report...[/yellow]")
        report_prompt = QA_GA_TEST_REPORT.format(
            test_plan=test_plan[:3000],
            test_history="\n\n".join(test_history)[:6000],
        )
        report_response = self._send_agent(
            qa_lead,
            report_prompt,
            task_description="GA test report",
            stage="ga_test",
        )
        if report_response.success:
            result.test_report = report_response.content
            self._persist_task_state(
                current_stage="ga_test_report_ready",
                latest_summary=report_response.content[:4000],
                next_todos=["Generate final delivery report"] if result.passed else ["Review GA test report"],
                open_issues=[] if result.passed else [bug.get("title", "Unknown bug") for bug in bugs[:10]],
                updated_by_agent_id=qa_lead.agent_id,
            )

        return result

    def _parse_test_results(self, output: str) -> list[dict]:
        """Parse QA test output into a list of bugs. Empty list = all passed."""
        data = _extract_json_block(output)
        if isinstance(data, dict):
            status = data.get("status", "").lower()
            if status == "pass":
                return []
            bugs = data.get("bugs", [])
            if isinstance(bugs, list):
                return [b for b in bugs if isinstance(b, dict)]
        # Fallback: check for keywords
        lower = output.lower()
        if "all tests pass" in lower or '"status": "pass"' in lower:
            return []
        if "fail" in lower or "bug" in lower:
            return [{"title": "Unparsed test failure", "description": output[:500], "severity": "major"}]
        return []

    # ════════════════════════════════════════════════════════════════
    # Bug fix workflow
    # ════════════════════════════════════════════════════════════════

    def _run_bug_fix(
        self,
        bug: dict,
        requirement: str,
        design: str,
        *,
        allow_arch_fix: bool = True,
    ) -> BugFixResult:
        title = bug.get("title", "Unknown bug")
        result = BugFixResult(title=title)
        if self._cancelled():
            return result

        # Step 1: Dev estimates the bug
        self._emit("agent_working", f"Estimating bug: {title}", stage="bug_fix", agent_role=self._dev.agent_role)
        self._console.print(f"[yellow]Dev estimating bug: {title}...[/yellow]")
        estimate_prompt = DEV_ESTIMATE_BUG.format(
            bug_title=title,
            bug_description=bug.get("description", ""),
            repro_steps=bug.get("repro_steps", "N/A"),
            severity=bug.get("severity", "unknown"),
            requirement=requirement[:1000],
            design=design[:2000],
        )
        est_response = self._send_agent(
            self._dev,
            estimate_prompt,
            task_description=f"Estimate bug: {title[:60]}",
            stage="bug_fix",
        )
        if not est_response.success:
            self._console.print("[red]Dev failed to estimate bug.[/red]")
            return result
        if self._cancelled():
            return result

        assessment = self._parse_bug_estimation(est_response.content)

        if assessment == "arch_change":
            result.arch_change = True
            if not allow_arch_fix:
                self._console.print(
                    "[bold yellow]Bug requires architecture change. "
                    "Recording follow-up work instead of launching a redesign workflow.[/bold yellow]"
                )
                result.fixed = False
                result.summary = "Requires architecture change follow-up; test-only mode does not launch redesign."
                return result
            self._console.print(
                f"[bold yellow]Bug requires architecture change. "
                f"Running full design->dev sub-workflow...[/bold yellow]"
            )
            sub_result = self._run_bug_fix_arch_change(bug, requirement, design)
            result.fixed = sub_result.success
            result.summary = sub_result.final_report[:500] if sub_result.final_report else "Sub-workflow completed"
        else:
            self._emit("agent_working", f"Fixing bug: {title}", stage="bug_fix", agent_role=self._dev.agent_role)
            self._console.print(f"[yellow]Simple fix. Dev fixing...[/yellow]")
            # Step 2: Dev fixes the bug
            fix_prompt = DEV_FIX_BUG_SIMPLE.format(
                bug_title=title,
                bug_description=bug.get("description", ""),
                repro_steps=bug.get("repro_steps", "N/A"),
            )
            fix_response = self._send_agent(
                self._dev,
                fix_prompt,
                task_description=f"Fix bug: {title[:60]}",
                stage="bug_fix",
            )
            if not fix_response.success:
                self._console.print("[red]Dev failed to fix bug.[/red]")
                return result
            if self._cancelled():
                return result

            # Step 3: Dev summarizes the fix
            summary_prompt = DEV_SUMMARIZE_BUG_FIX.format(bug_title=title)
            summary_response = self._send_agent(
                self._dev,
                summary_prompt,
                task_description=f"Summarize fix: {title[:60]}",
                stage="bug_fix",
            )
            result.summary = summary_response.content[:1000] if summary_response.success else fix_response.content[:1000]
            result.fixed = fix_response.success

        return result

    def _run_bug_fix_arch_change(self, bug: dict, requirement: str, design: str) -> WorkflowResult:
        """Run a full design->dev cycle for an architecture-level bug fix."""
        if self._cancelled():
            return WorkflowResult(requirement="cancelled")
        bug_req = (
            f"Bug fix requiring architecture change:\n\n"
            f"**Bug:** {bug.get('title', '')}\n"
            f"**Description:** {bug.get('description', '')}\n"
            f"**Repro steps:** {bug.get('repro_steps', 'N/A')}\n\n"
            f"**Original requirement:**\n{requirement[:2000]}\n\n"
            f"**Current design:**\n{design[:3000]}"
        )
        sub = WorkflowResult(requirement=bug_req)

        # Design
        self._emit("stage_start", "Bug Fix: Design", stage="bug_fix_design")
        self._console.print(Panel("[bold]Bug Fix: Design[/bold]", border_style="yellow"))
        sub_design = self._run_design(bug_req)
        if not sub_design:
            return sub
        sub.design = sub_design

        # Design review
        sub_design, iters, design_review_passed = self._run_review_loop(
            artifact=sub_design,
            artifact_type="design",
            context=f"Bug fix design:\n{bug_req[:2000]}",
            persist_artifacts=False,
        )
        sub.design = sub_design
        sub.design_review_iterations = iters
        sub.design_review_passed = design_review_passed

        # User checkpoint
        self._print_markdown_panel(
            sub_design,
            title="Bug Fix Design",
            border_style="yellow",
            label="bug-fix-design",
        )
        sub_design = self._user_checkpoint(
            sub_design, "design",
            "Bug fix design approved by QA. Proceed? [Y/n/or type feedback] ",
            persist_artifact=False,
        )
        if sub_design is None:
            return sub
        sub.design = sub_design

        # Planning
        self._emit("stage_start", "Bug Fix: Planning", stage="bug_fix_planning")
        self._console.print(Panel("[bold]Bug Fix: Planning[/bold]", border_style="yellow"))
        sub_plan = self._run_planning(sub_design, bug_req)
        if not sub_plan:
            return sub
        sub.plan = sub_plan

        # Plan review
        sub_plan, iters, plan_review_passed = self._run_review_loop(
            artifact=sub_plan,
            artifact_type="plan",
            context=f"Bug fix:\n{bug_req[:2000]}",
            persist_artifacts=False,
        )
        sub.plan = sub_plan
        sub.plan_review_iterations = iters
        sub.plan_review_passed = plan_review_passed

        # User checkpoint
        self._print_markdown_panel(
            sub_plan,
            title="Bug Fix Plan",
            border_style="yellow",
            label="bug-fix-plan",
        )
        sub_plan = self._user_checkpoint(
            sub_plan, "plan",
            "Bug fix plan approved. Start implementation? [Y/n/or type feedback] ",
            persist_artifact=False,
        )
        if sub_plan is None:
            return sub
        sub.plan = sub_plan

        # Phased dev
        self._emit("stage_start", "Bug Fix: Development", stage="bug_fix_development")
        self._console.print(Panel("[bold]Bug Fix: Development[/bold]", border_style="yellow"))
        phases = self._parse_phases(sub_plan)
        completed: list[str] = []
        for i, phase in enumerate(phases, 1):
            self._console.print(f"\n[bold yellow]Bug fix phase {i}/{len(phases)}: {phase}[/bold yellow]")
            phase_result = self._run_phase(
                phase_name=phase,
                phase_index=i,
                total_phases=len(phases),
                requirement=bug_req,
                design=sub_design,
                plan=sub_plan,
                completed_summaries=completed,
                persist_result=False,
            )
            sub.phases.append(phase_result)
            completed.append(f"Phase {i}: {phase_result.summary[:300]}")

        sub.success = all(p.committed for p in sub.phases) if sub.phases else False
        sub.final_report = (
            f"Bug fix for: {bug.get('title', '')}\n"
            f"Phases: {len(sub.phases)}, "
            f"Committed: {sum(1 for p in sub.phases if p.committed)}"
        )
        return sub

    def _review_limit_for(
        self,
        artifact_type: str,
        *,
        stage_prefix: str | None = None,
    ) -> int:
        if artifact_type in {"arch_design", "design"}:
            return self._design_plan_review_limit
        if artifact_type == "plan":
            return self._dev_plan_review_limit
        if artifact_type == "code":
            return self._dev_code_review_limit
        if artifact_type == "test_plan":
            return self._ga_plan_review_limit
        if artifact_type in {"ga_test", "test_report"}:
            return self._ga_test_review_limit
        if stage_prefix and stage_prefix.startswith("ga_test"):
            return self._ga_test_review_limit
        return self._max_review

    def _parse_bug_estimation(self, output: str) -> str:
        """Parse dev's bug estimation. Returns 'simple_fix' or 'arch_change'."""
        data = _extract_json_block(output)
        if isinstance(data, dict):
            assessment = data.get("assessment", "").lower()
            if assessment in ("simple_fix", "arch_change"):
                return assessment
        # Fallback: keyword check
        lower = output.lower()
        if "arch_change" in lower or "architecture change" in lower or "redesign" in lower:
            return "arch_change"
        return "simple_fix"

    # ════════════════════════════════════════════════════════════════
    # Generalized review loop (supports swapping proposer/reviewers)
    # ════════════════════════════════════════════════════════════════

    def _run_review_loop(
        self,
        artifact: str,
        artifact_type: str,
        context: str = "",
        proposer: "SessionManager | None" = None,
        reviewers: "list[SessionManager] | None" = None,
        abort_on_agent_failure: bool = False,
        stage_prefix: str | None = None,
        persist_artifacts: bool = True,
    ) -> tuple[str, int, bool]:
        """Run multi-reviewer loop. Returns (final_artifact, iterations, passed).

        By default: proposer=dev, reviewers=QA(s).
        For test plan review: proposer=QA, reviewers=[dev].

        stage_prefix overrides artifact_type in _persist_task_state stage names
        (e.g. "phase_2_code" instead of "code") to keep the phase context.
        """
        self._reset_review_loop_state()
        prop = proposer or self._dev
        revs = reviewers or self._qas
        current = artifact
        change_summary: str = ""  # empty on first iteration
        skipped_reviewers: set[int] = set()  # persists across iterations
        approved_reviewers: set[int] = set()  # reviewers who already LGTM'd this phase
        sp = stage_prefix or artifact_type  # prefix for current_stage values
        review_limit = self._review_limit_for(artifact_type, stage_prefix=stage_prefix)

        for iteration in range(1, review_limit + 1):
            if self._cancelled():
                self._console.print(f"[yellow]{artifact_type} review cancelled by user.[/yellow]")
                break
            self._emit(
                "review_start",
                f"Review iteration {iteration}/{review_limit}",
                stage=sp,
                iteration=iteration,
                max_iterations=review_limit,
            )
            self._console.print(f"\n[dim]-- Review iteration {iteration}/{review_limit} --[/dim]")

            artifact_id = None
            if persist_artifacts and self._work_item_id:
                try:
                    artifact_id = self._store.create_artifact(
                        work_item_id=self._work_item_id,
                        agent_id=prop.agent_id,
                        iteration=iteration,
                        artifact_type=self._review_artifact_type(artifact_type),
                        title=f"{artifact_type} review v{iteration}",
                        content=current[:65000],
                        metadata_json={
                            "source": "review_loop",
                            "workflow_artifact_type": artifact_type,
                        },
                    )
                except Exception as e:
                    self._console.print(f"[dim red]Warning: Failed to persist artifact: {e}[/dim red]")

            all_issues: list[str] = []
            all_lgtm = True

            # Keep prior approvals sticky within the current phase so only
            # unresolved reviewers see later revisions. This intentionally
            # favors shorter review loops over re-reviewing every revision
            # with reviewers who are already satisfied in the current phase.
            active_revs = [
                r for r in revs
                if r.agent_id not in skipped_reviewers and r.agent_id not in approved_reviewers
            ]

            if not active_revs:
                if approved_reviewers or skipped_reviewers:
                    completed_iteration = max(iteration - 1, 0)
                    summary = (
                        f"No reviewers remain pending for {artifact_type}; "
                        "advancing with the latest artifact."
                    )
                    self._console.print(f"[dim]{summary}[/dim]")
                    self._append_process_event(
                        event_type="reaction",
                        title="MySwat reaction",
                        summary=summary,
                        from_role="myswat",
                        to_role=prop.agent_role,
                        updated_by_agent_id=prop.agent_id,
                    )
                    self._persist_task_state(
                        current_stage=f"{sp}_approved",
                        latest_summary=current[:4000],
                        next_todos=["Proceed to the next workflow stage"],
                        open_issues=[],
                        last_artifact_id=artifact_id,
                        updated_by_agent_id=prop.agent_id,
                    )
                    return current, completed_iteration, True
                self._console.print(f"[dim]No reviewers configured — auto-approving {artifact_type}.[/dim]")
                self._append_process_event(
                    event_type="reaction",
                    title="MySwat reaction",
                    summary=f"No reviewers configured — auto-approved {artifact_type}.",
                    from_role="myswat",
                    to_role=prop.agent_role,
                    updated_by_agent_id=prop.agent_id,
                )
                self._persist_task_state(
                    current_stage=f"{sp}_approved",
                    latest_summary=current[:4000],
                    next_todos=["Proceed to the next workflow stage"],
                    open_issues=[],
                    last_artifact_id=artifact_id,
                    updated_by_agent_id=prop.agent_id,
                )
                return current, iteration, True

            # Build prompts and log review requests for all active reviewers upfront.
            reviewer_prompts: list[tuple[SessionManager, str]] = []
            for reviewer in active_revs:
                self._console.print(f"[yellow]{reviewer.agent_role} reviewing {artifact_type}...[/yellow]")
                self._append_process_event(
                    event_type="review_request",
                    title=f"{artifact_type} review iteration {iteration}",
                    summary=current,
                    from_role=prop.agent_role,
                    to_role=reviewer.agent_role,
                    updated_by_agent_id=prop.agent_id,
                )
                prompt = self._build_review_prompt(
                    artifact_type,
                    context,
                    current,
                    iteration,
                    reviewer=reviewer,
                    change_summary=change_summary,
                )
                reviewer_prompts.append((reviewer, prompt))

            # Send review requests to all reviewers concurrently.
            review_results: dict[str, tuple[SessionManager, AgentResponse]] = {}
            with ThreadPoolExecutor(max_workers=len(reviewer_prompts)) as executor:
                future_to_reviewer = {
                    executor.submit(
                        self._send_agent,
                        reviewer,
                        prompt,
                        task_description=f"Review {artifact_type} (iter {iteration})",
                        stage=sp,
                    ): reviewer
                    for reviewer, prompt in reviewer_prompts
                }
                for future in as_completed(future_to_reviewer):
                    reviewer = future_to_reviewer[future]
                    review_results[reviewer.agent_id] = (reviewer, future.result())

            # Process ALL results in stable reviewer order before deciding on abort.
            abort_failure: str | None = None
            abort_reviewer: SessionManager | None = None
            for reviewer in active_revs:
                _, response = review_results[reviewer.agent_id]

                if not response.success:
                    self._console.print(f"[red]{reviewer.agent_role} failed (exit={response.exit_code})[/red]")
                    if response.raw_stderr.strip():
                        self._console.print(f"[dim]  stderr: {response.raw_stderr.strip()}[/dim]")
                    failure_summary = f"[{reviewer.agent_role}] review failed (exit={response.exit_code})"
                    failure_detail = response.raw_stderr.strip() or response.content[:200]
                    all_lgtm = False

                    self._emit(
                        "review_verdict",
                        f"{reviewer.agent_role}: {'FAILED' if abort_on_agent_failure else 'SKIPPED'}",
                        stage=sp,
                        agent_role=reviewer.agent_role,
                        verdict="failed" if abort_on_agent_failure else "skipped",
                        issues=[failure_summary],
                        summary=failure_detail,
                    )

                    if abort_on_agent_failure and abort_failure is None:
                        # Mandatory reviewer — record first failure for abort after loop
                        abort_failure = failure_summary
                        abort_reviewer = reviewer
                        all_issues.append(failure_summary)
                    elif not abort_on_agent_failure:
                        # Optional reviewer — skip and continue with remaining reviewers
                        skipped_reviewers.add(reviewer.agent_id)
                        all_issues.append(f"[{reviewer.agent_role}] skipped: {response.content[:200]}")
                        self._console.print(
                            f"[yellow]{reviewer.agent_role} removed from review: "
                            f"{response.content[:100]}[/yellow]"
                        )
                    else:
                        all_issues.append(failure_summary)
                    continue

                verdict = _parse_verdict(response.content)
                self._emit("review_verdict", f"{reviewer.agent_role}: {verdict.verdict.upper()}",
                            stage=sp, agent_role=reviewer.agent_role,
                            verdict=verdict.verdict, issues=verdict.issues[:5],
                            summary=verdict.summary or "")
                self._console.print(f"  [bold]{reviewer.agent_role}: {verdict.verdict.upper()}[/bold]")

                if verdict.verdict != "lgtm":
                    all_lgtm = False
                    approved_reviewers.discard(reviewer.agent_id)
                    for issue in verdict.issues:
                        all_issues.append(f"[{reviewer.agent_role}] {issue}")
                        self._console.print(f"    [red]- {issue}[/red]")
                    if verdict.summary and not verdict.issues:
                        all_issues.append(f"[{reviewer.agent_role}] {verdict.summary}")
                else:
                    approved_reviewers.add(reviewer.agent_id)
                    if verdict.summary:
                        self._console.print(f"    [dim]{verdict.summary}[/dim]")

                verdict_summary = verdict.summary or ""
                if verdict.issues:
                    verdict_summary = (
                        (verdict_summary + " Issues: ") if verdict_summary else "Issues: "
                    ) + "; ".join(verdict.issues[:6])
                if not verdict_summary:
                    verdict_summary = verdict.verdict
                self._append_process_event(
                    event_type="review_response",
                    title=f"{artifact_type} review verdict: {verdict.verdict}",
                    summary=verdict_summary,
                    from_role=reviewer.agent_role,
                    to_role=prop.agent_role,
                    updated_by_agent_id=reviewer.agent_id,
                )

                # Store review cycle in DB
                if self._work_item_id and artifact_id:
                    try:
                        cycle_id = self._store.create_review_cycle(
                            work_item_id=self._work_item_id,
                            iteration=iteration,
                            proposer_agent_id=prop.agent_id,
                            reviewer_agent_id=reviewer.agent_id,
                            artifact_id=artifact_id,
                            proposal_session_id=prop.session.id if prop.session else None,
                        )
                        self._store.update_review_verdict(
                            cycle_id=cycle_id,
                            verdict=verdict.verdict,
                            verdict_json=verdict.model_dump(),
                            review_session_id=reviewer.session.id if reviewer.session else None,
                        )
                    except Exception as e:
                        self._console.print(f"[dim red]Warning: Failed to persist review cycle: {e}[/dim red]")

            # After processing all responses, abort if any reviewer failed fatally.
            if abort_failure is not None and abort_reviewer is not None:
                self._emit("stage_complete", abort_failure, stage=sp, failed=True)
                self._record_blocked_failure(
                    stage=f"{sp}_review_blocked",
                    summary=abort_failure,
                    updated_by_agent_id=abort_reviewer.agent_id,
                    from_role=abort_reviewer.agent_role,
                    to_role=prop.agent_role,
                    event_type="review_failure",
                    title=f"{artifact_type} review failed",
                )
                return current, iteration, False

            if all_lgtm:
                self._append_process_event(
                    event_type="reaction",
                    title="MySwat reaction",
                    summary=f"Accepted all reviewer LGTM responses and advanced the {artifact_type} stage.",
                    from_role="myswat",
                    to_role=prop.agent_role,
                    updated_by_agent_id=prop.agent_id,
                )
                self._emit("review_complete", f"All reviewers approved at iteration {iteration}", stage=sp, approved=True, iteration=iteration)
                self._console.print(f"\n[bold green]All reviewers gave LGTM at iteration {iteration}![/bold green]")
                self._persist_task_state(
                    current_stage=f"{sp}_approved",
                    latest_summary=current[:4000],
                    next_todos=["Proceed to the next workflow stage"],
                    open_issues=[],
                    last_artifact_id=artifact_id,
                    updated_by_agent_id=prop.agent_id,
                )
                return current, iteration, True

            if iteration >= review_limit:
                self._record_review_limit_reached(
                    artifact_type=artifact_type,
                    stage=sp,
                    iteration=iteration,
                    max_iterations=review_limit,
                    proposer=prop,
                    issues=all_issues,
                )
                self._console.print(
                    f"[yellow]Max iterations reached for {artifact_type} review; "
                    "continuing with the latest artifact.[/yellow]"
                )
                return current, iteration, False

            # Proposer addresses comments
            self._emit("revision_start", f"Addressing {len(all_issues)} comment(s)...", stage=sp, agent_role=prop.agent_role)
            self._console.print(f"\n[yellow]{prop.agent_role} addressing {len(all_issues)} comment(s)...[/yellow]")
            self._persist_task_state(
                current_stage=f"{sp}_review",
                latest_summary=current[:4000],
                next_todos=[f"{prop.agent_role} address {len(all_issues)} review comment(s)"],
                open_issues=all_issues,
                last_artifact_id=artifact_id,
                updated_by_agent_id=prop.agent_id,
            )
            self._append_process_event(
                event_type="reaction",
                title="MySwat reaction",
                summary=f"Collected {len(all_issues)} review comment(s) and asked {prop.agent_role} to revise the {artifact_type}.",
                from_role="myswat",
                to_role=prop.agent_role,
                updated_by_agent_id=prop.agent_id,
            )
            feedback = "\n".join(f"- {issue}" for issue in all_issues)

            address_prompt = self._build_address_prompt(artifact_type, current, feedback)
            response = self._send_agent(
                prop,
                address_prompt,
                task_description=f"Address {artifact_type} review",
                stage=sp,
            )
            if self._cancelled():
                self._console.print(f"[yellow]{artifact_type} review cancelled by user.[/yellow]")
                break

            if not response.success:
                self._console.print(f"[red]{prop.agent_role} failed to address comments.[/red]")
                if abort_on_agent_failure:
                    self._record_blocked_failure(
                        stage=f"{sp}_revision_blocked",
                        summary=f"[{prop.agent_role}] failed to address {artifact_type} comments (exit={response.exit_code})",
                        updated_by_agent_id=prop.agent_id,
                        from_role=prop.agent_role,
                        to_role="myswat",
                        event_type="revision_failure",
                        title=f"{artifact_type} revision failed",
                    )
                    return current, iteration, False
                break

            issue = self._validate_reviewable_design(artifact_type, response.content)
            if issue is not None:
                summary = f"[{prop.agent_role}] produced a non-reviewable {artifact_type} revision: {issue}"
                self._console.print(f"[red]{issue}[/red]")
                if abort_on_agent_failure:
                    self._record_blocked_failure(
                        stage=f"{sp}_revision_blocked",
                        summary=summary,
                        updated_by_agent_id=prop.agent_id,
                        from_role=prop.agent_role,
                        to_role="myswat",
                        event_type="revision_failure",
                        title=f"{artifact_type} revision invalid",
                    )
                    return current, iteration, False
                break

            current = response.content
            # Build change summary for the next review round so reviewers
            # know what was addressed since the previous iteration.
            change_summary = (
                f"Issues raised in iteration {iteration}:\n"
                + feedback
                + f"\n\nProposer's revision response:\n{response.content[:6000]}"
            )
            self._persist_task_state(
                current_stage=f"{sp}_revision_ready",
                latest_summary=current[:4000],
                next_todos=["Await another review round"],
                open_issues=all_issues,
                updated_by_agent_id=prop.agent_id,
            )

        self._console.print(f"[yellow]Max iterations reached for {artifact_type} review.[/yellow]")
        return current, review_limit, False

    def _review_artifact_type(self, artifact_type: str) -> str:
        if artifact_type in {"design", "arch_design"}:
            return "design_doc"
        if artifact_type == "test_plan":
            return "test_plan"
        if artifact_type == "code":
            return "diff"
        return "proposal"

    def _build_review_prompt(
        self,
        artifact_type: str,
        context: str,
        artifact: str,
        iteration: int,
        reviewer: "SessionManager | None" = None,
        change_summary: str = "",
    ) -> str:
        reviewer_role = getattr(reviewer, "agent_role", None)
        if not isinstance(reviewer_role, str):
            reviewer_role = None
        if artifact_type == "arch_design":
            base = DESIGN_REVIEW.format(context=context, design=artifact[:12000], iteration=iteration)
        elif artifact_type == "design":
            base = QA_DESIGN_REVIEW.format(context=context, design=artifact[:12000], iteration=iteration)
        elif artifact_type == "plan":
            base = QA_PLAN_REVIEW.format(context=context, plan=artifact[:12000], iteration=iteration)
        elif artifact_type == "test_plan" and reviewer_role == "architect":
            base = TEST_PLAN_REVIEW.format(context=context, test_plan=artifact[:12000], iteration=iteration)
        elif artifact_type == "test_plan":
            base = DEV_REVIEW_TEST_PLAN.format(context=context, test_plan=artifact[:12000], iteration=iteration)
        else:
            base = QA_CODE_REVIEW.format(context=context, summary=artifact[:12000], iteration=iteration)
        if change_summary:
            base += f"\n\n## Changes Since Last Review\n{change_summary}\n"
        return base

    def _build_address_prompt(
        self, artifact_type: str, artifact: str, feedback: str,
    ) -> str:
        if artifact_type == "arch_design":
            return ARCH_ADDRESS_DESIGN_COMMENTS.format(design=artifact[:8000], feedback=feedback)
        if artifact_type == "design":
            return DEV_ADDRESS_DESIGN_COMMENTS.format(design=artifact[:8000], feedback=feedback)
        if artifact_type == "plan":
            return DEV_ADDRESS_PLAN_COMMENTS.format(plan=artifact[:8000], feedback=feedback)
        if artifact_type == "test_plan":
            return QA_ADDRESS_TEST_PLAN_COMMENTS.format(test_plan=artifact[:8000], feedback=feedback)
        return DEV_ADDRESS_CODE_COMMENTS.format(summary=artifact[:8000], feedback=feedback)

    # ════════════════════════════════════════════════════════════════
    # User checkpoints
    # ════════════════════════════════════════════════════════════════

    def _user_checkpoint(
        self,
        artifact: str,
        artifact_type: str,
        prompt_text: str,
        proposer: "SessionManager | None" = None,
        persist_artifact: bool = True,
    ) -> str | None:
        """Let user approve, reject, or provide feedback.

        Feedback is sent to `proposer` (defaults to dev).
        Returns updated artifact or None to abort.
        """
        target = proposer or self._dev
        if self._auto_approve:
            self._console.print("[dim]Auto-approve enabled; continuing without user checkpoint.[/dim]")
            return artifact
        while True:
            response = self._ask(prompt_text)
            if response.lower() in ("", "y", "yes"):
                return artifact
            elif response.lower() in ("n", "no"):
                return None
            else:
                self._real_console.print(f"[yellow]Sending your feedback to {target.agent_role}...[/yellow]")
                address_prompt = self._build_address_prompt(artifact_type, artifact, response)
                agent_response = self._send_agent(
                    target,
                    address_prompt,
                    task_description=f"Address user feedback on {artifact_type}",
                    stage=artifact_type,
                )
                if agent_response.success:
                    artifact = agent_response.content
                    self._print_markdown_panel(
                        artifact,
                        title=f"Updated {artifact_type.title()}",
                        border_style="yellow",
                        label=f"updated-{artifact_type}",
                    )
                    # Persist the revised artifact so resume loads the
                    # post-feedback version, not the stale review-loop version.
                    if persist_artifact and self._work_item_id:
                        try:
                            self._store.create_artifact(
                                work_item_id=self._work_item_id,
                                agent_id=target.agent_id,
                                iteration=0,
                                artifact_type=self._review_artifact_type(artifact_type),
                                title=f"{artifact_type} (user-feedback revision)",
                                content=artifact[:65000],
                                metadata_json={
                                    "source": "user_checkpoint",
                                    "workflow_artifact_type": artifact_type,
                                },
                            )
                        except Exception as e:
                            self._console.print(f"[dim red]Warning: Failed to persist revised artifact: {e}[/dim red]")
                else:
                    self._console.print(f"[red]{target.agent_role} failed to address feedback.[/red]")

    # ════════════════════════════════════════════════════════════════
    # Phase parsing
    # ════════════════════════════════════════════════════════════════

    def _parse_phases(self, plan: str) -> list[str]:
        """Extract phase names from the plan text."""
        phases = []
        for line in plan.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()

            # "Phase N: ..." or "Step N: ..."
            for prefix in ("phase ", "step "):
                if lower.startswith(prefix):
                    rest = stripped[len(prefix):]
                    for i, ch in enumerate(rest):
                        if ch in (":", ".", ")") or (ch == " " and i > 0):
                            phases.append(rest[i + 1:].strip() if rest[i + 1:].strip() else rest.strip())
                            break
                    else:
                        phases.append(rest.strip())
                    break

            # "## Phase N: ..." (markdown header)
            if lower.startswith("## phase ") or lower.startswith("### phase "):
                parts = stripped.split(":", 1)
                if len(parts) > 1:
                    phases.append(parts[1].strip())
                else:
                    phases.append(stripped.lstrip("#").strip())

        if not phases:
            for line in plan.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if len(stripped) > 2 and stripped[0].isdigit():
                    rest = stripped.lstrip("0123456789")
                    if rest and rest[0] in (".", ")", "-"):
                        name = rest[1:].strip()
                        if name and len(name) > 3:
                            phases.append(name)

        if not phases:
            phases = ["Full implementation"]

        return phases

    # ════════════════════════════════════════════════════════════════
    # Report generation
    # ════════════════════════════════════════════════════════════════

    def _generate_architect_design_report(self, result: WorkflowResult) -> str:
        status = "Approved" if result.success else "Not approved"
        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
            f"## Design Review\n{status} after {result.design_review_iterations} review iteration(s).\n",
        ]
        if result.failure_summary:
            lines.append(f"## Failure\n{result.failure_summary}\n")
        if result.design:
            lines.append(f"## Final Design\n{result.design[:2000]}\n")
        return "\n".join(lines)

    def _generate_testplan_design_report(self, result: WorkflowResult) -> str:
        ga = result.ga_test or GATestResult()
        status = "Approved" if result.success else "Not approved"
        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
            f"## Test Plan Review\n{status} after {ga.test_plan_review_iterations} review iteration(s).\n",
        ]
        if result.failure_summary:
            lines.append(f"## Failure\n{result.failure_summary}\n")
        if ga.test_plan:
            lines.append(f"## Final Test Plan\n{ga.test_plan[:2000]}\n")
        return "\n".join(lines)

    def _generate_design_report(self, result: WorkflowResult) -> str:
        total_reviews = result.design_review_iterations + result.plan_review_iterations
        design_status = "Passed" if result.design_review_passed else "Not passed"
        plan_status = "Passed" if result.plan_review_passed else "Not passed"
        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
            f"## Design Review\n{design_status} after {result.design_review_iterations} review iteration(s).\n",
            f"## Plan Review\n{plan_status} after {result.plan_review_iterations} review iteration(s).\n",
            (
                "\n## Totals\n"
                f"- Total review cycles: {total_reviews}\n"
                f"- Design review: {design_status}\n"
                f"- Plan review: {plan_status}\n"
            ),
        ]
        return "\n".join(lines)

    def _generate_develop_report(self, result: WorkflowResult, completed_summaries: list[str]) -> str:
        dev_report = ""
        if completed_summaries:
            prompt = DEV_FINAL_REPORT.format(
                completed_phases="\n".join(completed_summaries),
            )
            response = self._send_agent(
                self._dev,
                prompt,
                task_description="Final report",
                stage="report",
            )
            if response.success:
                dev_report = response.content

        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
            "## Development Phases\n",
        ]
        for i, phase in enumerate(result.phases, 1):
            status = "Committed" if phase.committed else "Not committed"
            review_status = "Passed" if phase.review_passed else "Not passed"
            lines.append(
                f"### Phase {i}: {phase.name}\n"
                f"- Status: {status}\n"
                f"- Review: {review_status}\n"
                f"- Review iterations: {phase.review_iterations}\n"
                f"- Summary: {phase.summary[:300]}\n"
            )

        total_reviews = sum(p.review_iterations for p in result.phases)
        committed = sum(1 for p in result.phases if p.committed)
        lines.append(
            f"\n## Totals\n"
            f"- Phases: {len(result.phases)} ({committed} committed)\n"
            f"- Total review cycles: {total_reviews}\n"
        )
        if dev_report:
            lines.append(f"\n## Developer's Final Summary\n{dev_report}\n")
        return "\n".join(lines)

    def _generate_test_report(self, result: WorkflowResult) -> str:
        ga = result.ga_test
        test_plan_status = "Passed" if (ga and ga.test_plan_review_passed) else "Not passed"
        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
        ]
        if ga:
            lines.append(
                f"## Test Plan Review\n"
                f"{test_plan_status} after {ga.test_plan_review_iterations} review iteration(s).\n"
            )
            lines.append("## GA Test\n")
            if ga.aborted:
                lines.append(
                    f"**ABORTED** — {ga.bugs_found} bugs found (limit: {MAX_GA_BUGS}).\n"
                    f"Bugs fixed: {ga.bugs_fixed}. Manual intervention required.\n"
                )
            elif ga.passed:
                lines.append("**PASSED** — All tests passed.\n")
            else:
                lines.append(f"**INCOMPLETE** — Bugs found: {ga.bugs_found}, Fixed: {ga.bugs_fixed}\n")

            if ga.bug_fixes:
                lines.append("### Bug Fixes\n")
                for bf in ga.bug_fixes:
                    fix_type = "arch change" if bf.arch_change else "simple fix"
                    fix_status = "Fixed" if bf.fixed else "Not fixed"
                    lines.append(f"- **{bf.title}** ({fix_type}): {fix_status}\n")
                    if bf.summary:
                        lines.append(f"  {bf.summary[:200]}\n")

            if ga.test_report:
                lines.append(f"\n### QA Test Report\n{ga.test_report[:2000]}\n")

            lines.append(
                f"\n## Totals\n"
                f"- Total review cycles: {ga.test_plan_review_iterations}\n"
                f"- Test plan review: {test_plan_status}\n"
                f"- GA test: {'Passed' if ga.passed else 'Failed/Incomplete'}\n"
                f"- Bugs found: {ga.bugs_found}, fixed: {ga.bugs_fixed}\n"
            )
        return "\n".join(lines)

    def _generate_report(self, result: WorkflowResult, completed_summaries: list[str]) -> str:
        if self._mode == WorkMode.architect_design:
            return self._generate_architect_design_report(result)
        if self._mode == WorkMode.testplan_design:
            return self._generate_testplan_design_report(result)
        if self._mode == WorkMode.design:
            return self._generate_design_report(result)
        if self._mode == WorkMode.develop:
            return self._generate_develop_report(result, completed_summaries)
        if self._mode == WorkMode.test:
            return self._generate_test_report(result)

        # Ask dev for a narrative final report
        dev_report = ""
        if completed_summaries:
            prompt = DEV_FINAL_REPORT.format(
                completed_phases="\n".join(completed_summaries),
            )
            response = self._send_agent(
                self._dev,
                prompt,
                task_description="Final report",
                stage="report",
            )
            if response.success:
                dev_report = response.content

        # Build structured report
        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
            f"## Design Review\n"
            f"{'Approved' if result.design_review_passed else 'Not approved'}"
            f" after {result.design_review_iterations} review iteration(s).\n",
            f"## Plan Review\n"
            f"{'Approved' if result.plan_review_passed else 'Not approved'}"
            f" after {result.plan_review_iterations} review iteration(s).\n",
            "## Development Phases\n",
        ]

        for i, phase in enumerate(result.phases, 1):
            status = "Committed" if phase.committed else "Not committed"
            lines.append(
                f"### Phase {i}: {phase.name}\n"
                f"- Status: {status}\n"
                f"- Review iterations: {phase.review_iterations}\n"
                f"- Summary: {phase.summary[:300]}\n"
            )

        # GA Test results
        ga = result.ga_test
        if ga:
            lines.append("## GA Test\n")
            if ga.aborted:
                lines.append(
                    f"**ABORTED** — {ga.bugs_found} bugs found (limit: {MAX_GA_BUGS}).\n"
                    f"Bugs fixed: {ga.bugs_fixed}. Manual intervention required.\n"
                )
            elif ga.passed:
                lines.append(f"**PASSED** — All tests passed.\n")
            else:
                lines.append(f"**INCOMPLETE** — Bugs found: {ga.bugs_found}, Fixed: {ga.bugs_fixed}\n")

            if ga.bug_fixes:
                lines.append("### Bug Fixes\n")
                for bf in ga.bug_fixes:
                    fix_type = "arch change" if bf.arch_change else "simple fix"
                    fix_status = "Fixed" if bf.fixed else "Not fixed"
                    lines.append(f"- **{bf.title}** ({fix_type}): {fix_status}\n")
                    if bf.summary:
                        lines.append(f"  {bf.summary[:200]}\n")

            if ga.test_report:
                lines.append(f"\n### QA Test Report\n{ga.test_report[:2000]}\n")

        # Totals
        total_reviews = (
            result.design_review_iterations
            + result.plan_review_iterations
            + sum(p.review_iterations for p in result.phases)
            + (ga.test_plan_review_iterations if ga else 0)
        )
        committed = sum(1 for p in result.phases if p.committed)

        lines.append(
            f"\n## Totals\n"
            f"- Phases: {len(result.phases)} ({committed} committed)\n"
            f"- Total review cycles: {total_reviews}\n"
            f"- GA test: {'Passed' if (ga and ga.passed) else 'Failed/Incomplete' if ga else 'Skipped'}\n"
        )
        if ga:
            lines.append(f"- Bugs found: {ga.bugs_found}, fixed: {ga.bugs_fixed}\n")

        if dev_report:
            lines.append(f"\n## Developer's Final Summary\n{dev_report}\n")

        return "\n".join(lines)
