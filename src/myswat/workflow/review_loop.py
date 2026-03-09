"""Review loop — alternates developer and reviewer until LGTM or max iterations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console

from myswat.models.work_item import ReviewVerdict
from myswat.workflow.prompts import DEVELOPER_INITIAL, DEVELOPER_REVISION, REVIEWER

if TYPE_CHECKING:
    from myswat.agents.session_manager import SessionManager
    from myswat.memory.store import MemoryStore

console = Console()


def _parse_verdict(raw: str) -> ReviewVerdict:
    """Parse reviewer output into a structured ReviewVerdict."""
    text = raw.strip()

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
) -> ReviewVerdict:
    """Run the developer+reviewer feedback loop.

    Returns the final ReviewVerdict (either LGTM or last verdict at max iterations).
    """
    previous_artifact = ""
    final_verdict = ReviewVerdict(verdict="changes_requested", issues=["max iterations reached"])

    for iteration in range(1, max_iterations + 1):
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

        # === Reviewer turn ===
        review_prompt = REVIEWER.format(
            context=context, task=task,
            iteration=iteration, artifact=artifact_content[:12000],
        )

        console.print(f"[yellow]Reviewer analyzing...[/yellow]")
        review_response = reviewer_sm.send(review_prompt, task_description=f"Review: {task}")

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

        if verdict.verdict == "lgtm":
            console.print(f"\n[bold green]Review passed at iteration {iteration}![/bold green]")
            break

    return final_verdict
