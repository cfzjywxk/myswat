"""Workflow error handler — records errors, consults architect, reports to user."""

from __future__ import annotations

import json
import sys
import traceback as tb_module
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from myswat.memory.store import MemoryStore

console = Console()

ARCHITECT_DIAGNOSE_PROMPT = """\
A workflow error occurred in myswat. Analyze the error and suggest a concise fix.

**Error:** {error_type}: {message}
**Stage:** {stage}

**Context:**
```json
{context}
```

**Traceback (last 1500 chars):**
```
{traceback}
```

Reply with:
1. Root cause (1-2 sentences)
2. Suggested fix (concrete actionable steps)
3. Recoverable without user intervention? (yes/no)

Be concise. Focus on actionable advice.
"""


@dataclass
class WorkflowError:
    """Structured error context for recording and diagnosis."""

    error: Exception
    stage: str
    context: dict[str, Any] = field(default_factory=dict)
    traceback_str: str = ""

    def __post_init__(self) -> None:
        if not self.traceback_str:
            self.traceback_str = tb_module.format_exc()

    def summary(self) -> str:
        return f"[{self.stage}] {type(self.error).__name__}: {self.error}"

    def to_record(self) -> dict:
        return {
            "error_type": type(self.error).__name__,
            "message": str(self.error),
            "stage": self.stage,
            "context": {k: str(v)[:500] for k, v in self.context.items()},
            "traceback": self.traceback_str[:2000],
        }


def _build_runner(agent_row: dict):
    """Create a lightweight runner for the architect agent."""
    from myswat.agents.codex_runner import CodexRunner
    from myswat.agents.kimi_runner import KimiRunner

    backend = agent_row["cli_backend"]
    cli_path = agent_row["cli_path"]
    model = agent_row["model_name"]
    extra_flags = (
        json.loads(agent_row["cli_extra_args"])
        if agent_row.get("cli_extra_args")
        else []
    )

    if backend == "codex":
        return CodexRunner(cli_path=cli_path, model=model, extra_flags=extra_flags)
    elif backend == "kimi":
        return KimiRunner(cli_path=cli_path, model=model, extra_flags=extra_flags)
    return None


def _consult_architect(
    werr: WorkflowError,
    store: "MemoryStore",
    project_id: int,
) -> str | None:
    """Try to get diagnosis from the architect agent."""
    try:
        arch_agent = store.get_agent(project_id, "architect")
        if not arch_agent:
            return None

        runner = _build_runner(arch_agent)
        if not runner:
            return None

        record = werr.to_record()
        prompt = ARCHITECT_DIAGNOSE_PROMPT.format(
            error_type=type(werr.error).__name__,
            message=str(werr.error),
            stage=werr.stage,
            context=json.dumps(record.get("context", {}), indent=2)[:3000],
            traceback=werr.traceback_str[-1500:],
        )
        response = runner.invoke(prompt)
        if response.success:
            return response.content
    except Exception as e:
        print(f"[error_handler] Architect diagnosis failed: {e}", file=sys.stderr)
    return None


def handle_workflow_error(
    werr: WorkflowError,
    store: "MemoryStore | None" = None,
    project_id: int | None = None,
) -> str | None:
    """Record error, consult architect, and report to user.

    1. Persists error to knowledge table (searchable, auto-expires in 30 days)
    2. Asks architect agent for diagnosis (if available)
    3. Prints error + suggestion (or raw traceback) to the user

    Returns the architect's suggestion if available, None otherwise.
    Never raises — all internal failures are caught and logged to stderr.
    """
    record = werr.to_record()

    # ── 1. Persist error to knowledge (best-effort) ──
    if store and project_id:
        try:
            store.store_knowledge(
                project_id=project_id,
                category="error_log",
                title=f"Error: {werr.stage} — {type(werr.error).__name__}",
                content=json.dumps(record, indent=2),
                tags=["error", werr.stage],
                relevance_score=0.8,
                confidence=1.0,
                ttl_days=30,
                compute_embedding=False,
            )
        except Exception as rec_err:
            print(
                f"[error_handler] Failed to record error: {rec_err}",
                file=sys.stderr,
            )

    # ── 2. Consult architect (best-effort) ──
    suggestion = None
    if store and project_id:
        suggestion = _consult_architect(werr, store, project_id)

    # ── 3. Report to user ──
    console.print(f"\n[bold red]Workflow error in {werr.stage}:[/bold red]")
    console.print(f"[red]{type(werr.error).__name__}: {werr.error}[/red]")

    if suggestion:
        console.print(f"\n[bold yellow]Architect's analysis:[/bold yellow]")
        console.print(suggestion[:2000])
    else:
        console.print(
            "\n[dim]Error recorded. No automated diagnosis available.[/dim]"
        )
        # Show traceback for manual debugging
        tb = werr.traceback_str
        if tb and "NoneType: None" not in tb:
            console.print(f"[dim]{tb[-800:]}[/dim]")

    return suggestion
