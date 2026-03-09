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

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from myswat.models.work_item import ReviewVerdict
from myswat.workflow.prompts import (
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
    DEV_SUMMARIZE_PHASE,
    DEV_TECH_DESIGN,
    QA_ADDRESS_TEST_PLAN_COMMENTS,
    QA_CODE_REVIEW,
    QA_CONTINUE_GA_TEST,
    QA_DESIGN_REVIEW,
    QA_EXECUTE_GA_TEST,
    QA_GA_TEST_PLAN,
    QA_GA_TEST_REPORT,
    QA_PLAN_REVIEW,
)

if TYPE_CHECKING:
    from myswat.agents.session_manager import SessionManager
    from myswat.memory.store import MemoryStore

console = Console()

# Maximum bugs before aborting GA test phase
MAX_GA_BUGS = 5


# ── Data classes ──

@dataclass
class PhaseResult:
    name: str
    summary: str
    review_iterations: int
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
    plan: str = ""
    plan_review_iterations: int = 0
    phases: list[PhaseResult] = field(default_factory=list)
    ga_test: GATestResult | None = None
    final_report: str = ""
    success: bool = False


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
        max_review_iterations: int = 5,
        ask_user: Callable[[str], str] | None = None,
        auto_approve: bool = False,
    ) -> None:
        self._store = store
        self._dev = dev_sm
        self._qas = qa_sms
        self._project_id = project_id
        self._work_item_id = work_item_id
        self._max_review = max_review_iterations
        self._ask = ask_user or _default_ask
        self._auto_approve = auto_approve

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
            console.print(f"[dim red]Warning: Failed to persist task state: {e}[/dim red]")

    @staticmethod
    def _first_lines(text: str, limit: int = 4) -> list[str]:
        lines = [line.strip("-* ") for line in text.splitlines() if line.strip()]
        return lines[:limit]

    # ════════════════════════════════════════════════════════════════
    # Main workflow
    # ════════════════════════════════════════════════════════════════

    def run(self, requirement: str) -> WorkflowResult:
        result = WorkflowResult(requirement=requirement)
        self._persist_task_state(
            current_stage="workflow_started",
            latest_summary=requirement,
            next_todos=["Produce technical design"],
            updated_by_agent_id=self._dev.agent_id,
        )

        # Stage 1: Tech design
        console.print(Panel("[bold]Stage 1: Technical Design[/bold]", border_style="blue"))
        design = self._run_design(requirement)
        if not design:
            console.print("[red]Dev failed to produce design. Aborting.[/red]")
            return result
        result.design = design

        # Stage 2: Design review (all QA must LGTM)
        console.print(Panel("[bold]Stage 2: Design Review[/bold]", border_style="blue"))
        design, iters = self._run_review_loop(
            artifact=design,
            artifact_type="design",
            context=f"Requirement:\n{requirement}",
        )
        result.design = design
        result.design_review_iterations = iters

        # User checkpoint: approved design
        console.print(Panel(Markdown(design), title="QA-Approved Design", border_style="green"))
        design = self._user_checkpoint(
            design, "design", "Design approved by QA. Proceed to planning? [Y/n/or type feedback] "
        )
        if design is None:
            console.print("[yellow]Workflow stopped by user.[/yellow]")
            return result
        result.design = design

        # Stage 3: Implementation planning
        console.print(Panel("[bold]Stage 3: Implementation Planning[/bold]", border_style="blue"))
        plan = self._run_planning(design, requirement)
        if not plan:
            console.print("[red]Dev failed to produce plan. Aborting.[/red]")
            return result
        result.plan = plan

        # Stage 4: Plan review (QA + user approval)
        console.print(Panel("[bold]Stage 4: Plan Review[/bold]", border_style="blue"))
        plan, iters = self._run_review_loop(
            artifact=plan,
            artifact_type="plan",
            context=f"Requirement:\n{requirement[:2000]}\n\nApproved Design:\n{design[:4000]}",
        )
        result.plan = plan
        result.plan_review_iterations = iters

        # User checkpoint: approved plan
        console.print(Panel(Markdown(plan), title="QA-Approved Plan", border_style="green"))
        plan = self._user_checkpoint(
            plan, "plan", "Plan approved by QA. Start development? [Y/n/or type feedback] "
        )
        if plan is None:
            console.print("[yellow]Workflow stopped by user.[/yellow]")
            return result
        result.plan = plan

        # Stage 5: Phased development
        console.print(Panel("[bold]Stage 5: Development[/bold]", border_style="blue"))
        phases = self._parse_phases(plan)
        console.print(f"[dim]Parsed {len(phases)} phase(s) from plan.[/dim]")
        completed_summaries: list[str] = []

        for i, phase in enumerate(phases, 1):
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Phase {i}/{len(phases)}: {phase}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")

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

        # Stage 6: GA Test
        console.print(Panel("[bold]Stage 6: GA Test[/bold]", border_style="blue"))
        dev_summary = "\n\n".join(completed_summaries)
        ga_result = self._run_ga_test_phase(requirement, design, plan, dev_summary)
        result.ga_test = ga_result

        if ga_result.aborted:
            console.print(Panel(
                f"[bold red]GA Test aborted.[/bold red]\n"
                f"Bugs found: {ga_result.bugs_found}, Fixed: {ga_result.bugs_fixed}\n"
                f"User intervention required.",
                title="GA Test Aborted",
                border_style="red",
            ))
            # Still generate report so user sees the situation
        elif ga_result.passed:
            console.print("[bold green]GA Test passed![/bold green]")

        # Stage 7: Final report
        console.print(Panel("[bold]Stage 7: Final Report[/bold]", border_style="blue"))
        report = self._generate_report(result, completed_summaries)
        result.final_report = report
        result.success = (
            all(p.committed for p in result.phases)
            and ga_result.passed
        ) if result.phases else False

        console.print(Panel(Markdown(report), title="E2E Workflow Report", border_style="green"))
        self._persist_task_state(
            current_stage="workflow_completed" if result.success else "workflow_finished_with_issues",
            latest_summary=report[:4000],
            next_todos=[] if result.success else ["Review final report and unresolved issues"],
            open_issues=[] if result.success else self._first_lines(report, limit=8),
            updated_by_agent_id=self._dev.agent_id,
        )
        return result

    # ════════════════════════════════════════════════════════════════
    # Stage implementations
    # ════════════════════════════════════════════════════════════════

    def _run_design(self, requirement: str) -> str:
        console.print("[yellow]Dev producing technical design...[/yellow]")
        prompt = DEV_TECH_DESIGN.format(requirement=requirement)
        response = self._dev.send(prompt, task_description=f"Tech design: {requirement[:100]}")
        if not response.success:
            console.print(f"[red]Dev agent failed (exit={response.exit_code})[/red]")
            return ""
        console.print("[green]Dev submitted design.[/green]")
        self._persist_task_state(
            current_stage="design_draft",
            latest_summary=response.content[:4000],
            next_todos=["Run QA design review"],
            updated_by_agent_id=self._dev.agent_id,
        )
        return response.content

    def _run_planning(self, design: str, requirement: str) -> str:
        console.print("[yellow]Dev creating implementation plan...[/yellow]")
        prompt = DEV_IMPLEMENTATION_PLAN.format(
            requirement=requirement[:4000],
            design=design[:8000],
        )
        response = self._dev.send(prompt, task_description="Implementation planning")
        if not response.success:
            console.print(f"[red]Dev agent failed (exit={response.exit_code})[/red]")
            return ""
        console.print("[green]Dev submitted implementation plan.[/green]")
        self._persist_task_state(
            current_stage="plan_draft",
            latest_summary=response.content[:4000],
            next_todos=["Run QA plan review"],
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
    ) -> PhaseResult:
        completed_ctx = "\n".join(completed_summaries) if completed_summaries else "None yet."

        # Step 1: Dev implements
        console.print("[yellow]Dev implementing...[/yellow]")
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
        response = self._dev.send(prompt, task_description=f"Phase {phase_index}: {phase_name}")
        if not response.success:
            return PhaseResult(name=phase_name, summary="Implementation failed.", review_iterations=0)

        # Step 2: Dev summarizes
        console.print("[yellow]Dev summarizing changes...[/yellow]")
        summary_prompt = DEV_SUMMARIZE_PHASE.format(
            phase_name=phase_name,
            phase_index=phase_index,
        )
        summary_resp = self._dev.send(summary_prompt, task_description=f"Summarize phase {phase_index}")
        summary = summary_resp.content if summary_resp.success else response.content
        self._persist_task_state(
            current_stage=f"phase_{phase_index}_under_review",
            latest_summary=summary[:4000],
            next_todos=["QA review this phase summary and inspect the codebase"],
            updated_by_agent_id=self._dev.agent_id,
        )

        # Step 3: QA review loop
        reviewed_summary, review_iters = self._run_review_loop(
            artifact=summary,
            artifact_type="code",
            context=(
                f"Phase {phase_index}/{total_phases}: {phase_name}\n\n"
                f"Requirement (brief):\n{requirement[:1000]}"
            ),
        )

        # Step 4: Dev commits
        console.print(f"[yellow]Dev committing phase {phase_index}...[/yellow]")
        commit_prompt = DEV_COMMIT_PHASE.format(
            phase_name=phase_name,
            phase_index=phase_index,
        )
        commit_resp = self._dev.send(commit_prompt, task_description=f"Commit phase {phase_index}")
        committed = commit_resp.success

        if committed:
            console.print(f"[bold green]Phase {phase_index} committed.[/bold green]")
        else:
            console.print(f"[red]Phase {phase_index} commit failed.[/red]")

        self._persist_task_state(
            current_stage=f"phase_{phase_index}_committed" if committed else f"phase_{phase_index}_commit_failed",
            latest_summary=reviewed_summary[:4000],
            next_todos=[] if committed else [f"Revisit phase {phase_index} commit failure"],
            updated_by_agent_id=self._dev.agent_id,
        )

        return PhaseResult(
            name=phase_name,
            summary=reviewed_summary[:2000],
            review_iterations=review_iters,
            committed=committed,
        )

    # ════════════════════════════════════════════════════════════════
    # GA Test phase
    # ════════════════════════════════════════════════════════════════

    def _run_ga_test_phase(
        self,
        requirement: str,
        design: str,
        plan: str,
        dev_summary: str,
    ) -> GATestResult:
        result = GATestResult()
        qa_lead = self._qas[0]
        self._persist_task_state(
            current_stage="ga_test_planning",
            latest_summary=dev_summary[:4000],
            next_todos=["Generate GA test plan"],
            updated_by_agent_id=qa_lead.agent_id,
        )

        # Step 1: QA generates test plan
        console.print("[yellow]QA generating GA test plan...[/yellow]")
        prompt = QA_GA_TEST_PLAN.format(
            requirement=requirement[:2000],
            design=design[:3000],
            dev_summary=dev_summary[:4000],
        )
        response = qa_lead.send(prompt, task_description="GA test plan")
        if not response.success:
            console.print("[red]QA failed to generate test plan.[/red]")
            return result
        test_plan = response.content
        result.test_plan = test_plan
        self._persist_task_state(
            current_stage="ga_test_plan_draft",
            latest_summary=test_plan[:4000],
            next_todos=["Review test plan", "Approve and start testing"],
            updated_by_agent_id=qa_lead.agent_id,
        )

        # Step 2: Dev + User review test plan
        console.print(Panel("[bold]Test Plan Review[/bold]", border_style="cyan"))
        test_plan, iters = self._run_review_loop(
            artifact=test_plan,
            artifact_type="test_plan",
            context=f"Requirement:\n{requirement[:2000]}",
            proposer=qa_lead,
            reviewers=[self._dev],
        )
        result.test_plan = test_plan
        result.test_plan_review_iterations = iters

        # User checkpoint on test plan
        console.print(Panel(Markdown(test_plan), title="Reviewed Test Plan", border_style="green"))
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
        console.print("[yellow]QA executing GA tests...[/yellow]")
        exec_prompt = QA_EXECUTE_GA_TEST.format(test_plan=test_plan[:8000])
        exec_response = qa_lead.send(exec_prompt, task_description="Execute GA tests")
        if not exec_response.success:
            console.print("[red]QA failed to execute tests.[/red]")
            return result

        test_output = exec_response.content
        bugs = self._parse_test_results(test_output)
        test_history = [f"Initial run: {test_output[:2000]}"]

        if not bugs:
            console.print("[bold green]All GA tests passed![/bold green]")
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
        console.print(f"[bold red]{len(bugs)} bug(s) found in GA test.[/bold red]")

        while bugs:
            if result.bugs_found > MAX_GA_BUGS:
                console.print(
                    f"\n[bold red]More than {MAX_GA_BUGS} bugs found ({result.bugs_found} total). "
                    f"Stopping GA test phase.[/bold red]"
                )
                console.print(
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
                console.print(
                    f"\n[bold red]Bug: {bug.get('title', 'Unknown')} "
                    f"[{bug.get('severity', '?')}][/bold red]"
                )
                if bug.get("description"):
                    console.print(f"[dim]{bug['description'][:200]}[/dim]")

                bug_fix = self._run_bug_fix(bug, requirement, design)
                result.bug_fixes.append(bug_fix)
                if bug_fix.fixed:
                    result.bugs_fixed += 1
                    console.print(f"[green]Bug fixed: {bug_fix.title}[/green]")
                else:
                    console.print(f"[red]Bug fix failed: {bug_fix.title}[/red]")

            # QA re-tests after all fixes in this round
            console.print("\n[yellow]QA re-running tests after bug fixes...[/yellow]")
            fixed_summaries = "\n".join(
                f"- {bf.title}: {bf.summary[:200]}"
                for bf in result.bug_fixes if bf.fixed
            )
            continue_prompt = QA_CONTINUE_GA_TEST.format(
                test_plan=test_plan[:4000],
                fixed_bugs=fixed_summaries,
            )
            continue_response = qa_lead.send(continue_prompt, task_description="Continue GA tests")
            if not continue_response.success:
                console.print("[red]QA failed to continue tests.[/red]")
                break

            test_output = continue_response.content
            test_history.append(f"Re-test: {test_output[:2000]}")
            bugs = self._parse_test_results(test_output)

            if bugs:
                result.bugs_found += len(bugs)
                console.print(
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
                console.print("[bold green]All tests pass after bug fixes![/bold green]")

        if not result.aborted and not bugs:
            result.passed = True

        # QA generates final test report
        console.print("[yellow]QA generating test report...[/yellow]")
        report_prompt = QA_GA_TEST_REPORT.format(
            test_plan=test_plan[:3000],
            test_history="\n\n".join(test_history)[:6000],
        )
        report_response = qa_lead.send(report_prompt, task_description="GA test report")
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

    def _run_bug_fix(self, bug: dict, requirement: str, design: str) -> BugFixResult:
        title = bug.get("title", "Unknown bug")
        result = BugFixResult(title=title)

        # Step 1: Dev estimates the bug
        console.print(f"[yellow]Dev estimating bug: {title}...[/yellow]")
        estimate_prompt = DEV_ESTIMATE_BUG.format(
            bug_title=title,
            bug_description=bug.get("description", ""),
            repro_steps=bug.get("repro_steps", "N/A"),
            severity=bug.get("severity", "unknown"),
            requirement=requirement[:1000],
            design=design[:2000],
        )
        est_response = self._dev.send(estimate_prompt, task_description=f"Estimate bug: {title[:60]}")
        if not est_response.success:
            console.print("[red]Dev failed to estimate bug.[/red]")
            return result

        assessment = self._parse_bug_estimation(est_response.content)

        if assessment == "arch_change":
            result.arch_change = True
            console.print(
                f"[bold yellow]Bug requires architecture change. "
                f"Running full design->dev sub-workflow...[/bold yellow]"
            )
            sub_result = self._run_bug_fix_arch_change(bug, requirement, design)
            result.fixed = sub_result.success
            result.summary = sub_result.final_report[:500] if sub_result.final_report else "Sub-workflow completed"
        else:
            console.print(f"[yellow]Simple fix. Dev fixing...[/yellow]")
            # Step 2: Dev fixes the bug
            fix_prompt = DEV_FIX_BUG_SIMPLE.format(
                bug_title=title,
                bug_description=bug.get("description", ""),
                repro_steps=bug.get("repro_steps", "N/A"),
            )
            fix_response = self._dev.send(fix_prompt, task_description=f"Fix bug: {title[:60]}")
            if not fix_response.success:
                console.print("[red]Dev failed to fix bug.[/red]")
                return result

            # Step 3: Dev summarizes the fix
            summary_prompt = DEV_SUMMARIZE_BUG_FIX.format(bug_title=title)
            summary_response = self._dev.send(summary_prompt, task_description=f"Summarize fix: {title[:60]}")
            result.summary = summary_response.content[:1000] if summary_response.success else fix_response.content[:1000]
            result.fixed = fix_response.success

        return result

    def _run_bug_fix_arch_change(self, bug: dict, requirement: str, design: str) -> WorkflowResult:
        """Run a full design->dev cycle for an architecture-level bug fix."""
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
        console.print(Panel("[bold]Bug Fix: Design[/bold]", border_style="yellow"))
        sub_design = self._run_design(bug_req)
        if not sub_design:
            return sub
        sub.design = sub_design

        # Design review
        sub_design, iters = self._run_review_loop(
            artifact=sub_design,
            artifact_type="design",
            context=f"Bug fix design:\n{bug_req[:2000]}",
        )
        sub.design = sub_design
        sub.design_review_iterations = iters

        # User checkpoint
        console.print(Panel(Markdown(sub_design), title="Bug Fix Design", border_style="yellow"))
        sub_design = self._user_checkpoint(
            sub_design, "design",
            "Bug fix design approved by QA. Proceed? [Y/n/or type feedback] ",
        )
        if sub_design is None:
            return sub
        sub.design = sub_design

        # Planning
        console.print(Panel("[bold]Bug Fix: Planning[/bold]", border_style="yellow"))
        sub_plan = self._run_planning(sub_design, bug_req)
        if not sub_plan:
            return sub
        sub.plan = sub_plan

        # Plan review
        sub_plan, iters = self._run_review_loop(
            artifact=sub_plan,
            artifact_type="plan",
            context=f"Bug fix:\n{bug_req[:2000]}",
        )
        sub.plan = sub_plan
        sub.plan_review_iterations = iters

        # User checkpoint
        console.print(Panel(Markdown(sub_plan), title="Bug Fix Plan", border_style="yellow"))
        sub_plan = self._user_checkpoint(
            sub_plan, "plan",
            "Bug fix plan approved. Start implementation? [Y/n/or type feedback] ",
        )
        if sub_plan is None:
            return sub
        sub.plan = sub_plan

        # Phased dev
        console.print(Panel("[bold]Bug Fix: Development[/bold]", border_style="yellow"))
        phases = self._parse_phases(sub_plan)
        completed: list[str] = []
        for i, phase in enumerate(phases, 1):
            console.print(f"\n[bold yellow]Bug fix phase {i}/{len(phases)}: {phase}[/bold yellow]")
            phase_result = self._run_phase(
                phase_name=phase,
                phase_index=i,
                total_phases=len(phases),
                requirement=bug_req,
                design=sub_design,
                plan=sub_plan,
                completed_summaries=completed,
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
    ) -> tuple[str, int]:
        """Run multi-reviewer loop. Returns (final_artifact, iterations).

        By default: proposer=dev, reviewers=QA(s).
        For test plan review: proposer=QA, reviewers=[dev].
        """
        prop = proposer or self._dev
        revs = reviewers or self._qas
        current = artifact

        for iteration in range(1, self._max_review + 1):
            console.print(f"\n[dim]-- Review iteration {iteration}/{self._max_review} --[/dim]")

            artifact_id = None
            if self._work_item_id:
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
                    console.print(f"[dim red]Warning: Failed to persist artifact: {e}[/dim red]")

            all_issues: list[str] = []
            all_lgtm = True

            for reviewer in revs:
                console.print(f"[yellow]{reviewer.agent_role} reviewing {artifact_type}...[/yellow]")

                prompt = self._build_review_prompt(artifact_type, context, current, iteration)
                response = reviewer.send(prompt, task_description=f"Review {artifact_type} (iter {iteration})")

                if not response.success:
                    console.print(f"[red]{reviewer.agent_role} failed (exit={response.exit_code})[/red]")
                    all_lgtm = False
                    all_issues.append(f"[{reviewer.agent_role}] review failed (exit={response.exit_code})")
                    continue

                verdict = _parse_verdict(response.content)
                console.print(f"  [bold]{reviewer.agent_role}: {verdict.verdict.upper()}[/bold]")

                if verdict.verdict != "lgtm":
                    all_lgtm = False
                    for issue in verdict.issues:
                        all_issues.append(f"[{reviewer.agent_role}] {issue}")
                        console.print(f"    [red]- {issue}[/red]")
                    if verdict.summary and not verdict.issues:
                        all_issues.append(f"[{reviewer.agent_role}] {verdict.summary}")
                elif verdict.summary:
                    console.print(f"    [dim]{verdict.summary}[/dim]")

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
                        console.print(f"[dim red]Warning: Failed to persist review cycle: {e}[/dim red]")

            if all_lgtm:
                console.print(f"\n[bold green]All reviewers gave LGTM at iteration {iteration}![/bold green]")
                self._persist_task_state(
                    current_stage=f"{artifact_type}_approved",
                    latest_summary=current[:4000],
                    next_todos=["Proceed to the next workflow stage"],
                    open_issues=[],
                    last_artifact_id=artifact_id,
                    updated_by_agent_id=prop.agent_id,
                )
                return current, iteration

            # Proposer addresses comments
            console.print(f"\n[yellow]{prop.agent_role} addressing {len(all_issues)} comment(s)...[/yellow]")
            self._persist_task_state(
                current_stage=f"{artifact_type}_review",
                latest_summary=current[:4000],
                next_todos=[f"{prop.agent_role} address {len(all_issues)} review comment(s)"],
                open_issues=all_issues,
                last_artifact_id=artifact_id,
                updated_by_agent_id=prop.agent_id,
            )
            feedback = "\n".join(f"- {issue}" for issue in all_issues)

            address_prompt = self._build_address_prompt(artifact_type, current, feedback)
            response = prop.send(address_prompt, task_description=f"Address {artifact_type} review")

            if not response.success:
                console.print(f"[red]{prop.agent_role} failed to address comments.[/red]")
                break

            current = response.content
            self._persist_task_state(
                current_stage=f"{artifact_type}_revision_ready",
                latest_summary=current[:4000],
                next_todos=["Await another review round"],
                open_issues=all_issues,
                updated_by_agent_id=prop.agent_id,
            )

            # Preserve the final unreviewed revision when the loop stops here.
            if self._work_item_id and iteration == self._max_review:
                try:
                    self._store.create_artifact(
                        work_item_id=self._work_item_id,
                        agent_id=prop.agent_id,
                        iteration=iteration + 1,
                        artifact_type=self._review_artifact_type(artifact_type),
                        title=f"{artifact_type} draft v{iteration + 1}",
                        content=current[:65000],
                        metadata_json={
                            "source": "review_loop",
                            "workflow_artifact_type": artifact_type,
                            "reviewed": False,
                        },
                    )
                except Exception as e:
                    console.print(f"[dim red]Warning: Failed to persist final artifact: {e}[/dim red]")

        console.print(f"[yellow]Max iterations reached for {artifact_type} review.[/yellow]")
        return current, self._max_review

    def _review_artifact_type(self, artifact_type: str) -> str:
        if artifact_type == "design":
            return "design_doc"
        if artifact_type == "test_plan":
            return "test_plan"
        if artifact_type == "code":
            return "diff"
        return "proposal"

    def _build_review_prompt(
        self, artifact_type: str, context: str, artifact: str, iteration: int,
    ) -> str:
        if artifact_type == "design":
            return QA_DESIGN_REVIEW.format(context=context, design=artifact[:12000], iteration=iteration)
        elif artifact_type == "plan":
            return QA_PLAN_REVIEW.format(context=context, plan=artifact[:12000], iteration=iteration)
        elif artifact_type == "test_plan":
            return DEV_REVIEW_TEST_PLAN.format(context=context, test_plan=artifact[:12000], iteration=iteration)
        else:
            return QA_CODE_REVIEW.format(context=context, summary=artifact[:12000], iteration=iteration)

    def _build_address_prompt(
        self, artifact_type: str, artifact: str, feedback: str,
    ) -> str:
        if artifact_type == "design":
            return DEV_ADDRESS_DESIGN_COMMENTS.format(design=artifact[:8000], feedback=feedback)
        elif artifact_type == "plan":
            return DEV_ADDRESS_PLAN_COMMENTS.format(plan=artifact[:8000], feedback=feedback)
        elif artifact_type == "test_plan":
            return QA_ADDRESS_TEST_PLAN_COMMENTS.format(test_plan=artifact[:8000], feedback=feedback)
        else:
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
    ) -> str | None:
        """Let user approve, reject, or provide feedback.

        Feedback is sent to `proposer` (defaults to dev).
        Returns updated artifact or None to abort.
        """
        target = proposer or self._dev
        if self._auto_approve:
            console.print("[dim]Auto-approve enabled; continuing without user checkpoint.[/dim]")
            return artifact
        while True:
            response = self._ask(prompt_text)
            if response.lower() in ("", "y", "yes"):
                return artifact
            elif response.lower() in ("n", "no"):
                return None
            else:
                console.print(f"[yellow]Sending your feedback to {target.agent_role}...[/yellow]")
                address_prompt = self._build_address_prompt(artifact_type, artifact, response)
                agent_response = target.send(
                    address_prompt, task_description=f"Address user feedback on {artifact_type}",
                )
                if agent_response.success:
                    artifact = agent_response.content
                    console.print(Panel(
                        Markdown(artifact), title=f"Updated {artifact_type.title()}", border_style="yellow",
                    ))
                else:
                    console.print(f"[red]{target.agent_role} failed to address feedback.[/red]")

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

    def _generate_report(self, result: WorkflowResult, completed_summaries: list[str]) -> str:
        # Ask dev for a narrative final report
        dev_report = ""
        if completed_summaries:
            prompt = DEV_FINAL_REPORT.format(
                completed_phases="\n".join(completed_summaries),
            )
            response = self._dev.send(prompt, task_description="Final report")
            if response.success:
                dev_report = response.content

        # Build structured report
        lines = [
            "# Workflow Report\n",
            f"## Requirement\n{result.requirement}\n",
            f"## Design Review\n"
            f"Approved after {result.design_review_iterations} review iteration(s).\n",
            f"## Plan Review\n"
            f"Approved after {result.plan_review_iterations} review iteration(s).\n",
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
