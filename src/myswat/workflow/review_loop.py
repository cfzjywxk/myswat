"""Review loop — alternates developer and reviewer until LGTM or max iterations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console

from myswat.cli.progress import _collapse_text
from myswat.large_payloads import maybe_externalize_list, maybe_externalize_summary
from myswat.models.work_item import ReviewVerdict
from myswat.workflow.prompts import DEVELOPER_INITIAL, DEVELOPER_REVISION, REVIEWER

if TYPE_CHECKING:
    from myswat.agents.session_manager import SessionManager
    from myswat.memory.store import MemoryStore

console = Console()


def _parse_verdict(raw: str) -> ReviewVerdict:
    """Parse reviewer output into a structured ReviewVerdict."""
    text = raw.strip()

    if not text:
        return ReviewVerdict(
            verdict="changes_requested",
            issues=["Reviewer returned empty output."],
            summary="Reviewer returned empty output; treating as changes_requested.",
        )

    # Extract JSON from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        # Try to find JSON block
        parts = text.split("```")
        for part in parts[1::2]:  # odd-indexed parts are inside code blocks
            part = part.strip()
            if part.startswith("{"):
                text = part
                break

    # Try direct JSON parse
    try:
        data = json.loads(text)
        return ReviewVerdict(
            verdict=data.get("verdict", "changes_requested"),
            issues=data.get("issues", []),
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: check for LGTM keywords in raw text
    lower = raw.lower()
    if "lgtm" in lower and "changes_requested" not in lower:
        return ReviewVerdict(verdict="lgtm", issues=[], summary=raw[:200])

    # Default: treat as changes_requested
    return ReviewVerdict(
        verdict="changes_requested",
        issues=[raw[:500]],
        summary="Could not parse structured verdict; treating as changes_requested.",
    )


def run_review_loop(
    store: MemoryStore,
    dev_sm: SessionManager,
    reviewer_sm: SessionManager,
    task: str,
    project_id: int,
    work_item_id: int,
    max_iterations: int = 5,
    context: str = "",
    should_cancel=None,
) -> ReviewVerdict:
    """Run the developer+reviewer feedback loop.

    Returns the final ReviewVerdict (either LGTM or last verdict at max iterations).
    """
    previous_artifact = ""
    final_verdict = ReviewVerdict(verdict="changes_requested", issues=["max iterations reached"])

    def _persist_state(
        *,
        stage: str,
        summary: str,
        next_todos: list[str] | None = None,
        open_issues: list[str] | None = None,
        last_artifact_id: int | None = None,
        updated_by_agent_id: int | None = None,
    ) -> None:
        try:
            store.update_work_item_state(
                work_item_id,
                current_stage=stage,
                latest_summary=maybe_externalize_summary(summary[:4000], label=f"{stage}-summary"),
                next_todos=maybe_externalize_list(next_todos, label=f"{stage}-todo"),
                open_issues=maybe_externalize_list(open_issues, label=f"{stage}-issue"),
                last_artifact_id=last_artifact_id,
                updated_by_agent_id=updated_by_agent_id,
            )
        except Exception:
            pass

    def _append_process_event(
        *,
        event_type: str,
        summary: str,
        from_role: str | None = None,
        to_role: str | None = None,
        title: str | None = None,
        updated_by_agent_id: int | None = None,
    ) -> None:
        try:
            store.append_work_item_process_event(
                work_item_id,
                event_type=event_type,
                title=title,
                summary=_collapse_text(
                    maybe_externalize_summary(summary, label=f"{event_type}-summary"),
                ),
                from_role=from_role,
                to_role=to_role,
                updated_by_agent_id=updated_by_agent_id,
            )
        except Exception:
            pass

    for iteration in range(1, max_iterations + 1):
        if should_cancel and should_cancel():
            final_verdict = ReviewVerdict(
                verdict="changes_requested",
                issues=["cancelled by user"],
                summary="Review loop cancelled by user.",
            )
            break
        console.print(f"\n[bold cyan]── Iteration {iteration}/{max_iterations} ──[/bold cyan]")

        # === Developer turn ===
        if iteration == 1:
            dev_prompt = DEVELOPER_INITIAL.format(context=context, task=task)
        else:
            feedback = final_verdict.summary
            if final_verdict.issues:
                feedback += "\n\nIssues:\n" + "\n".join(f"- {i}" for i in final_verdict.issues)
            dev_prompt = DEVELOPER_REVISION.format(
                context=context, task=task,
                previous_artifact=previous_artifact[:8000],
                feedback=feedback,
            )

        _persist_state(
            stage="review_loop_developing" if iteration == 1 else "review_loop_revising",
            summary=str(task) if iteration == 1 else (final_verdict.summary or str(task)),
            next_todos=(
                ["Developer prepare the initial proposal and implementation summary"]
                if iteration == 1
                else ["Developer address reviewer feedback and update the proposal"]
            ),
            open_issues=(
                []
                if iteration == 1
                else (final_verdict.issues or ([final_verdict.summary] if final_verdict.summary else []))
            ),
            updated_by_agent_id=dev_sm.agent_id,
        )
        console.print(f"[yellow]Developer working...[/yellow]")
        dev_response = dev_sm.send(dev_prompt, task_description=task)

        if not dev_response.success:
            console.print(f"[red]Developer agent failed (exit={dev_response.exit_code})[/red]")
            break

        artifact_content = dev_response.content
        previous_artifact = artifact_content

        # Store artifact in DB
        artifact_id = None
        try:
            artifact_id = store.create_artifact(
                work_item_id=work_item_id,
                agent_id=dev_sm.agent_id,
                iteration=iteration,
                artifact_type="proposal" if iteration == 1 else "diff",
                title=f"Iteration {iteration}",
                content=artifact_content,
            )
        except Exception as e:
            console.print(f"[dim red]Warning: Failed to persist artifact: {e}[/dim red]")

        console.print(f"[green]Developer submitted (artifact_id={artifact_id}, {len(artifact_content)} chars)[/green]")
        _persist_state(
            stage="review_loop_reviewing",
            summary=artifact_content,
            next_todos=["Reviewer inspect the latest implementation summary and codebase"],
            last_artifact_id=artifact_id,
            updated_by_agent_id=dev_sm.agent_id,
        )
        _append_process_event(
            event_type="review_request",
            title=f"Iteration {iteration} submission",
            summary=artifact_content,
            from_role=dev_sm.agent_role,
            to_role=reviewer_sm.agent_role,
            updated_by_agent_id=dev_sm.agent_id,
        )

        # === Reviewer turn ===
        review_prompt = REVIEWER.format(
            context=context, task=task,
            iteration=iteration, artifact=artifact_content[:12000],
        )

        console.print(f"[yellow]Reviewer analyzing...[/yellow]")
        review_response = reviewer_sm.send(review_prompt, task_description=f"Review: {task}")
        if should_cancel and should_cancel():
            final_verdict = ReviewVerdict(
                verdict="changes_requested",
                issues=["cancelled by user"],
                summary="Review loop cancelled by user.",
            )
            break

        if not review_response.success:
            console.print(f"[red]Reviewer agent failed (exit={review_response.exit_code})[/red]")
            break

        verdict = _parse_verdict(review_response.content)
        final_verdict = verdict

        # Store review cycle in DB
        if artifact_id is not None:
            try:
                cycle_id = store.create_review_cycle(
                    work_item_id=work_item_id,
                    iteration=iteration,
                    proposer_agent_id=dev_sm.agent_id,
                    reviewer_agent_id=reviewer_sm.agent_id,
                    artifact_id=artifact_id,
                    proposal_session_id=dev_sm.session.id if dev_sm.session else None,
                )
                store.update_review_verdict(
                    cycle_id=cycle_id,
                    verdict=verdict.verdict,
                    verdict_json=verdict.model_dump(),
                    review_session_id=reviewer_sm.session.id if reviewer_sm.session else None,
                )
            except Exception as e:
                console.print(f"[dim red]Warning: Failed to persist review cycle: {e}[/dim red]")

        console.print(f"[bold]Verdict: {verdict.verdict.upper()}[/bold]")
        if verdict.summary:
            console.print(f"[dim]{verdict.summary}[/dim]")
        if verdict.issues:
            for issue in verdict.issues:
                console.print(f"  [red]• {issue}[/red]")

        verdict_summary = verdict.summary or ""
        if verdict.issues:
            verdict_summary = (verdict_summary + " Issues: " if verdict_summary else "Issues: ") + "; ".join(verdict.issues[:6])
        if not verdict_summary:
            verdict_summary = verdict.verdict
        _append_process_event(
            event_type="review_response",
            title=f"Iteration {iteration} verdict: {verdict.verdict}",
            summary=verdict_summary,
            from_role=reviewer_sm.agent_role,
            to_role=dev_sm.agent_role,
            updated_by_agent_id=reviewer_sm.agent_id,
        )

        if verdict.verdict == "lgtm":
            _append_process_event(
                event_type="reaction",
                title="MySwat reaction",
                summary="Accepted QA LGTM and marked the review loop approved.",
                from_role="myswat",
                to_role=dev_sm.agent_role,
                updated_by_agent_id=reviewer_sm.agent_id,
            )
            console.print(f"\n[bold green]Review passed at iteration {iteration}![/bold green]")
            _persist_state(
                stage="review_loop_approved",
                summary=artifact_content,
                next_todos=[],
                open_issues=[],
                last_artifact_id=artifact_id,
                updated_by_agent_id=reviewer_sm.agent_id,
            )
            break
        _persist_state(
            stage="review_loop_changes_requested",
            summary=artifact_content,
            next_todos=["Developer address reviewer feedback"],
            open_issues=verdict.issues or ([verdict.summary] if verdict.summary else []),
            last_artifact_id=artifact_id,
            updated_by_agent_id=reviewer_sm.agent_id,
        )
        _append_process_event(
            event_type="reaction",
            title="MySwat reaction",
            summary="Collected QA feedback and asked developer to revise the submission.",
            from_role="myswat",
            to_role=dev_sm.agent_role,
            updated_by_agent_id=reviewer_sm.agent_id,
        )

    return final_verdict
