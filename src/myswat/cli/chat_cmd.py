"""myswat chat — interactive conversation with agents."""

from __future__ import annotations

import os
import sys
import threading
from typing import Callable

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from myswat.agents.base import AgentRunner
from myswat.agents.factory import make_runner_from_row
from myswat.agents.session_manager import SessionManager
from myswat.cli.prompting import create_prompt_session, make_prompt_callback
from myswat.cli.progress import (
    _build_live_display,
    _check_esc,
    _describe_process_event,
    _fmt_duration,
    _run_with_task_monitor,
    _send_with_timer,
    _single_line_preview,
)
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.large_payloads import maybe_externalize_response, maybe_externalize_summary
from myswat.memory.embedder import preload_model
from myswat.memory.learn_triggers import submit_chat_learn_request
from myswat.memory.store import MemoryStore
from myswat.workflow.modes import (
    DEFAULT_DELEGATION_MODE,
    DELEGATION_MODE_SPECS,
    WorkMode,
    normalize_delegation_mode,
)

console = Console()

HELP_TEXT = """
[bold]Commands:[/bold]
  /help                 Show this help
  /status               Show project status
  /task <id>            Show detailed status for one work item
  /history [n]          Show recent turns from the active chat session
  /work <requirement>   Start full workflow: design -> plan -> develop -> test
  /role <role>          Switch agent role (developer, architect, qa_main)
  /reset                Reset AI session (fresh context reload, same TiDB session)
  /review <task>        Start the legacy dev+reviewer loop for a task
  /agents               List available agents
  /sessions             Show active sessions
  /new                  Start a new session (close current)
  /quit, /exit, Ctrl+D  Exit chat
"""


def _build_prompt_session(
    settings: MySwatSettings,
    history_name: str,
) -> PromptSession[str]:
    return create_prompt_session(
        config_path=settings.config_path,
        history_name=history_name,
        prompt_session_factory=PromptSession,
        key_bindings_factory=KeyBindings,
    )


def _extract_delegation(text: str) -> tuple[str, str] | None:
    """Extract a delegation task and mode from an agent response.

    Looks for a ```delegate block with TASK:/MODE: lines.
    Returns (task, mode) or None. Mode defaults to ``develop``.
    """
    import re

    match = re.search(r"```delegate\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        return None

    block = match.group(1)
    task = None
    mode = DEFAULT_DELEGATION_MODE
    for line in block.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("TASK:"):
            task = line[5:].strip()
        elif line.upper().startswith("MODE:"):
            mode = normalize_delegation_mode(line[5:])

    if not task:
        fallback_lines: list[str] = []
        for raw_line in block.strip().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.upper().startswith("MODE:"):
                continue
            if stripped.upper().startswith("TASK:"):
                continue
            fallback_lines.append(stripped)
        task = "\n".join(fallback_lines) or None


    return (task, mode) if task else None


def run_chat(
    project_slug: str,
    role: str = "developer",
    workdir: str | None = None,
) -> None:
    """Interactive chat session with an agent."""
    settings = MySwatSettings()

    # Preload embedding model in background so first query doesn't block
    threading.Thread(target=preload_model, daemon=True).start()

    pool = TiDBPool(settings.tidb)
    ensure_schema(pool)
    store = MemoryStore(
        pool,
        tidb_embedding_model=settings.embedding.tidb_model,
        embedding_backend=settings.embedding.backend,
    )

    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = workdir or proj.get("repo_path")
    # State
    current_role = role
    sm: SessionManager | None = None

    def _switch_agent(new_role: str, runner_settings: MySwatSettings) -> SessionManager:
        """Create a new SessionManager for the given role."""
        agent_row = store.get_agent(proj["id"], new_role)
        if not agent_row:
            console.print(f"[red]Agent role '{new_role}' not found.[/red]")
            return None
        runner = make_runner_from_row(agent_row, settings=runner_settings)
        runner.workdir = effective_workdir
        new_sm = SessionManager(
            store=store, runner=runner, agent_row=agent_row,
            project_id=proj["id"], settings=runner_settings,
        )
        with console.status("[dim]Loading session from TiDB...[/dim]", spinner="dots"):
            new_sm.create_or_resume(purpose=f"Interactive chat ({new_role})")
        console.print(
            f"[dim]Agent: {agent_row['display_name']} "
            f"({agent_row['cli_backend']}/{agent_row['model_name']}) | "
            f"Session: {new_sm.session.session_uuid[:8]}[/dim]"
        )
        return new_sm

    # Initialize
    console.print(Panel(
        f"[bold]MySwat Chat[/bold] — project: [cyan]{proj['name']}[/cyan]\n"
        f"Type a message to chat with the agent. Type [bold]/help[/bold] for commands.\n"
        f"[dim]Enter to submit · Alt+Enter for new line · Paste supported[/dim]",
        border_style="blue",
    ))
    sm = _switch_agent(current_role, settings)
    if sm is None:
        raise typer.Exit(1)

    prompt_session = _build_prompt_session(settings, f"chat-{project_slug}")

    # REPL
    while True:
        try:
            print()  # blank line before prompt
            user_input = prompt_session.prompt(
                f"you ({current_role})> ",
                multiline=False,
            ).strip()
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except KeyboardInterrupt:
            console.print("\n[dim]Type /quit to exit, or Ctrl+D.[/dim]")
            continue

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                break

            elif cmd == "/help":
                console.print(HELP_TEXT)

            elif cmd == "/status":
                _show_status(store, pool, proj)

            elif cmd == "/task":
                item_id = int(arg) if arg.isdigit() else None
                _show_task_details(store, proj, item_id)

            elif cmd == "/role":
                if not arg:
                    console.print(f"[dim]Current role: {current_role}. Usage: /role <role>[/dim]")
                    continue
                new_role = arg.strip()
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                new_sm = _switch_agent(new_role, settings)
                if new_sm:
                    sm = new_sm
                    current_role = new_role

            elif cmd == "/agents":
                agents = store.list_agents(proj["id"])
                for a in agents:
                    marker = " [bold cyan]<-[/bold cyan]" if a["role"] == current_role else ""
                    console.print(
                        f"  {a['role']:12s} {a['display_name']:20s} "
                        f"({a['cli_backend']}/{a['model_name']}){marker}"
                    )

            elif cmd == "/sessions":
                sessions = pool.fetch_all(
                    "SELECT s.*, a.role, a.display_name FROM sessions s "
                    "JOIN agents a ON s.agent_id = a.id "
                    "WHERE a.project_id = %s AND s.status = 'active' "
                    "ORDER BY s.created_at DESC LIMIT 10",
                    (proj["id"],),
                )
                if sessions:
                    for s in sessions:
                        turns = store.count_session_turns(s["id"])
                        console.print(
                            f"  {s['session_uuid'][:8]} [{s['role']}] "
                            f"{(s.get('purpose') or '')[:40]} ({turns} turns)"
                        )
                else:
                    console.print("[dim]No active sessions.[/dim]")

            elif cmd == "/history":
                if sm and sm.session:
                    n = int(arg) if arg.isdigit() else 10
                    turns = store.get_session_turns(sm.session.id)
                    for t in turns[-n:]:
                        label = "[green]you[/green]" if t.role == "user" else "[cyan]agent[/cyan]"
                        content_preview = t.content[:200]
                        if len(t.content) > 200:
                            content_preview += "..."
                        console.print(f"  {label}: {content_preview}")
                else:
                    console.print("[dim]No active session.[/dim]")

            elif cmd == "/reset":
                if sm:
                    sm.reset_ai_session()
                    console.print(
                        "[green]AI session reset.[/green] "
                        "[dim]Next message will start a fresh AI session with TiDB context reload.[/dim]"
                    )
                else:
                    console.print("[dim]No active session to reset.[/dim]")

            elif cmd == "/new":
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                sm = _switch_agent(current_role, settings)

            elif cmd == "/work":
                if not arg:
                    console.print("[dim]Usage: /work <requirement description>[/dim]")
                    continue
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                _run_workflow_interactive(
                    store, proj, effective_workdir, settings, arg, prompt_session,
                )
                sm = _switch_agent(current_role, settings)

            elif cmd == "/review":
                if not arg:
                    console.print("[dim]Usage: /review <task description>[/dim]")
                    continue
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                _run_inline_review_interactive(
                    store, proj, effective_workdir, settings, arg,
                )
                # Reopen chat session after review
                sm = _switch_agent(current_role, settings)

            else:
                console.print(f"[dim]Unknown command: {cmd}. Type /help for commands.[/dim]")

            continue

        # Regular message — send to agent
        if sm is None:
            sm = _switch_agent(current_role, settings)
            if sm is None:
                continue

        response, elapsed = _send_with_timer(console, sm, user_input)

        if response.cancelled:
            console.print(f"\n[yellow]Request cancelled.[/yellow] [dim]({_fmt_duration(elapsed)})[/dim]")
        elif response.success:
            console.print()
            rendered, _ = maybe_externalize_response(
                response.content,
                label=f"{current_role}-chat-response",
            )
            console.print(Markdown(rendered))
            console.print(f"\n[dim]({_fmt_duration(elapsed)})[/dim]")

            try:
                submit_chat_learn_request(
                    store=store,
                    settings=settings,
                    project_id=proj["id"],
                    user_message=user_input,
                    assistant_response=response.content,
                    workdir=effective_workdir,
                    source_session_id=getattr(getattr(sm, "session", None), "id", None),
                )
            except Exception as exc:
                print(f"[chat learn] Failed: {exc}", file=sys.stderr)

            # Check for agent delegation signal
            delegation = _extract_delegation(response.content)
            if delegation:
                delegation_task, delegation_mode = delegation
                delegation_spec = DELEGATION_MODE_SPECS.get(delegation_mode)
                if delegation_spec is None:
                    console.print(
                        f"[yellow]Delegation mode '{delegation_mode}' is not supported.[/yellow]"
                    )
                elif current_role not in delegation_spec.allowed_roles:
                    allowed_roles = ", ".join(sorted(delegation_spec.allowed_roles))
                    console.print(
                        f"[yellow]Delegation mode '{delegation_mode}' is only available for role(s): {allowed_roles}. Current role: '{current_role}'.[/yellow]"
                    )
                else:
                    delegation_handlers = {
                        "workflow": lambda: _run_workflow_interactive(
                            store,
                            proj,
                            effective_workdir,
                            settings,
                            delegation_task,
                            prompt_session=prompt_session,
                            mode=delegation_spec.engine_mode,
                        ),
                        "design_review": lambda: _run_design_review_interactive(
                            store,
                            proj,
                            sm,
                            effective_workdir,
                            settings,
                            delegation_task,
                            prompt_session=prompt_session,
                        ),
                        "full_workflow": lambda: _run_full_workflow_interactive(
                            store,
                            proj,
                            sm,
                            effective_workdir,
                            settings,
                            delegation_task,
                            prompt_session=prompt_session,
                        ),
                        "testplan_review": lambda: _run_testplan_review_interactive(
                            store,
                            proj,
                            sm,
                            effective_workdir,
                            settings,
                            delegation_task,
                            prompt_session=prompt_session,
                        ),
                    }
                    handler = delegation_handlers.get(delegation_spec.chat_handler)
                    if handler is None:
                        console.print(
                            f"[yellow]Delegation mode '{delegation_mode}' is misconfigured.[/yellow]"
                        )
                    else:
                        console.print(
                            f"\n[bold cyan]{delegation_spec.banner}:[/bold cyan] {delegation_task[:160]}"
                        )
                        if delegation_spec.detail:
                            console.print(f"[dim]{delegation_spec.detail}[/dim]")
                        if delegation_spec.save_session_before_run and sm:
                            with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                                sm.close()
                        handler()
                        if delegation_spec.reset_role_session:
                            sm = _switch_agent(current_role, settings)
        else:
            console.print(Panel(
                response.content,
                title=f"Agent Error ({_fmt_duration(elapsed)})",
                border_style="red",
            ))
            if response.raw_stderr:
                console.print(f"[dim red]{response.raw_stderr[:300]}[/dim red]")

    # Cleanup
    if sm:
        with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
            sm.close()
    console.print("[dim]Session closed. Turns persisted to TiDB.[/dim]")


def _show_status(store: MemoryStore, pool: TiDBPool, proj: dict) -> None:
    """Inline status display for chat mode."""
    items = list(store.list_work_items(proj["id"]))
    active = [i for i in items if i["status"] in ("in_progress", "review", "pending")]
    if active:
        active_with_state: list[tuple[dict, dict]] = []
        table = Table(title="Work Items")
        table.add_column("ID", style="cyan")
        table.add_column("Status")
        table.add_column("Stage")
        table.add_column("Title", max_width=50)
        for item in active[:10]:
            metadata = item.get("metadata_json") if isinstance(item, dict) else None
            task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
            if not isinstance(task_state, dict):
                task_state = {}
            active_with_state.append((item, task_state))
            stage = "-"
            stage = _single_line_preview(task_state.get("current_stage"), 30) or "-"
            table.add_row(str(item["id"]), item["status"], stage, item["title"][:50])
        console.print(table)
        for item, task_state in active_with_state[:3]:
            process_log = task_state.get("process_log")
            if not isinstance(process_log, list) or not process_log:
                continue
            console.print(f"\n[bold]Work Item #{item['id']} Flow[/bold] — {item['title'][:80]}")
            for event in process_log[-8:]:
                if isinstance(event, dict):
                    console.print(f"  - {_describe_process_event(event, 140)}")
    else:
        console.print("[dim]No active work items.[/dim]")


def _show_task_details(store: MemoryStore, proj: dict, work_item_id: int | None = None) -> None:
    items = list(store.list_work_items(proj["id"]))
    if work_item_id is None:
        active = [i for i in items if i["status"] in ("in_progress", "review", "pending")]
        if active:
            work_item_id = active[0]["id"]
        elif items:
            work_item_id = items[0]["id"]
        else:
            console.print("[dim]No work items yet.[/dim]")
            return

    item = store.get_work_item(work_item_id)
    if not item:
        console.print(f"[red]Work item {work_item_id} not found.[/red]")
        return

    state = store.get_work_item_state(work_item_id)
    if not isinstance(state, dict):
        state = {}
    console.print(
        Panel(
            f"[bold]Work Item #{item['id']}[/bold]\n"
            f"Status: {item.get('status', '?')}\n"
            f"Type: {item.get('item_type', '?')}\n"
            f"Title: {item.get('title', '')}",
            border_style="blue",
        )
    )

    if state:
        if state.get("current_stage"):
            console.print(f"[bold]Stage:[/bold] {state['current_stage']}")
        if state.get("latest_summary"):
            console.print(Panel(_single_line_preview(state["latest_summary"], 600), title="Latest Summary"))
        if state.get("next_todos"):
            console.print("[bold]Next TODOs:[/bold]")
            for todo in state["next_todos"][:10]:
                console.print(f"  - {todo}")
        if state.get("open_issues"):
            console.print("[bold]Open Issues:[/bold]")
            for issue in state["open_issues"][:10]:
                console.print(f"  - {issue}")
        process_log = state.get("process_log")
        if isinstance(process_log, list) and process_log:
            console.print("[bold]Process Log:[/bold]")
            for event in process_log[-20:]:
                if isinstance(event, dict):
                    console.print(f"  - {_describe_process_event(event, 160)}")

    artifacts = store.list_artifacts(work_item_id)
    if isinstance(artifacts, list) and artifacts:
        table = Table(title="Recent Artifacts")
        table.add_column("Iter")
        table.add_column("Type")
        table.add_column("Title", max_width=40)
        for art in artifacts[-5:]:
            table.add_row(str(art["iteration"]), art["artifact_type"], (art.get("title") or "")[:40])
        console.print(table)


def _run_inline_review(
    store: MemoryStore, proj: dict,
    workdir: str | None, settings: MySwatSettings, task: str,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[AgentRunner], None] | None = None,
    initial_process_events: list[dict] | None = None,
) -> None:
    """Run a dev+reviewer loop from within the chat REPL."""
    from myswat.workflow.review_loop import run_review_loop

    dev_agent = store.get_agent(proj["id"], "developer")
    reviewer_agent = store.get_agent(proj["id"], "qa_main")
    if not dev_agent or not reviewer_agent:
        console.print("[red]Missing developer or qa_main agent.[/red]")
        return

    dev_runner = make_runner_from_row(dev_agent, settings=settings)
    dev_runner.workdir = workdir
    if register_cancel_target:
        register_cancel_target(dev_runner)
    reviewer_runner = make_runner_from_row(reviewer_agent, settings=settings)
    reviewer_runner.workdir = workdir
    if register_cancel_target:
        register_cancel_target(reviewer_runner)

    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], settings=settings,
    )
    reviewer_sm = SessionManager(
        store=store, runner=reviewer_runner, agent_row=reviewer_agent,
        project_id=proj["id"], settings=settings,
    )

    work_item_id = store.create_work_item(
        project_id=proj["id"], title=task[:200],
        description=task, item_type="code_change",
        assigned_agent_id=dev_agent["id"],
    )
    if on_work_item_created:
        on_work_item_created(work_item_id)
    store.update_work_item_status(work_item_id, "in_progress")
    kickoff_events = initial_process_events or [
        {
            "event_type": "task_request",
            "title": "Review loop task",
            "summary": task,
            "from_role": "user",
            "to_role": "developer",
            "updated_by_agent_id": None,
        }
    ]
    for event in kickoff_events:
        try:
            store.append_work_item_process_event(
                work_item_id,
                event_type=event.get("event_type", "task_request"),
                title=event.get("title"),
                summary=maybe_externalize_summary(
                    event.get("summary", task),
                    label=f"{event.get('event_type', 'task_request')}-summary",
                ),
                from_role=event.get("from_role"),
                to_role=event.get("to_role"),
                updated_by_agent_id=event.get("updated_by_agent_id"),
            )
        except Exception:
            pass

    dev_sm.create_or_resume(purpose=f"Dev: {task[:100]}", work_item_id=work_item_id)
    reviewer_sm.create_or_resume(purpose=f"Review: {task[:100]}", work_item_id=work_item_id)

    console.print(f"\n[bold]Starting review loop for:[/bold] {task}")

    verdict = run_review_loop(
        store=store, dev_sm=dev_sm, reviewer_sm=reviewer_sm,
        task=task, project_id=proj["id"], work_item_id=work_item_id,
        max_iterations=settings.workflow.max_review_iterations,
        should_cancel=should_cancel,
    )

    if should_cancel and should_cancel():
        store.update_work_item_status(work_item_id, "blocked")
    elif verdict.verdict == "lgtm":
        store.update_work_item_status(work_item_id, "approved")
    else:
        store.update_work_item_status(work_item_id, "review")

    dev_sm.close()
    reviewer_sm.close()


def _make_prompt_callback(prompt_session: PromptSession | None = None) -> Callable[[str], str]:
    return make_prompt_callback(prompt_session)


def _run_design_review(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[AgentRunner], None] | None = None,
) -> None:
    if proposer_sm is None:
        console.print("[red]Missing architect session.[/red]")
        return
    return _run_workflow(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        prompt_session=prompt_session,
        should_cancel=should_cancel,
        on_work_item_created=on_work_item_created,
        register_cancel_target=register_cancel_target,
        mode=WorkMode.architect_design,
        proposer_sm=proposer_sm,
    )


def _run_testplan_review(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[AgentRunner], None] | None = None,
    on_event: Callable | None = None,
) -> None:
    from myswat.workflow.engine import WorkflowEngine, WorkMode

    if proposer_sm is None:
        console.print("[red]Missing QA session.[/red]")
        return

    arch_agent = store.get_agent(proj["id"], "architect")
    dev_agent = store.get_agent(proj["id"], "developer")
    if not arch_agent or not dev_agent:
        console.print("[red]Missing architect or developer agent.[/red]")
        return

    arch_runner = make_runner_from_row(arch_agent, settings=settings)
    arch_runner.workdir = workdir
    if register_cancel_target:
        register_cancel_target(arch_runner)
    dev_runner = make_runner_from_row(dev_agent, settings=settings)
    dev_runner.workdir = workdir
    if register_cancel_target:
        register_cancel_target(dev_runner)
    proposer_runner = getattr(proposer_sm, "_runner", None)
    if proposer_runner is not None and register_cancel_target:
        register_cancel_target(proposer_runner)

    arch_sm = SessionManager(
        store=store, runner=arch_runner, agent_row=arch_agent,
        project_id=proj["id"], settings=settings,
    )
    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], settings=settings,
    )

    work_item_id = store.create_work_item(
        project_id=proj["id"],
        title=task[:200],
        description=task,
        item_type="review",
        assigned_agent_id=proposer_sm.agent_id,
        metadata_json={"work_mode": WorkMode.testplan_design.value},
    )
    if on_work_item_created:
        on_work_item_created(work_item_id)
    store.update_work_item_status(work_item_id, "in_progress")

    qa_workflow_sm = proposer_sm.fork_for_work_item(
        work_item_id,
        purpose=f"QA test plan: {task[:80]}",
    )
    arch_sm.create_or_resume(purpose=f"Workflow architect review: {task[:80]}", work_item_id=work_item_id)
    dev_sm.create_or_resume(purpose=f"Workflow dev review: {task[:80]}", work_item_id=work_item_id)

    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=[qa_workflow_sm],
        arch_sm=arch_sm,
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_review_iterations=settings.workflow.max_review_iterations,
        mode=WorkMode.testplan_design,
        ask_user=_make_prompt_callback(prompt_session),
        auto_approve=False,
        should_cancel=should_cancel,
        on_event=on_event,
    )

    try:
        result = engine.run(task)
        if should_cancel and should_cancel():
            store.update_work_item_status(work_item_id, "blocked")
        elif getattr(result, "blocked", False):
            store.update_work_item_status(work_item_id, "blocked")
        elif result.success:
            store.update_work_item_status(work_item_id, "completed")
        else:
            store.update_work_item_status(work_item_id, "review")
    except Exception:
        store.update_work_item_status(work_item_id, "blocked")
        raise
    finally:
        qa_workflow_sm.close()
        arch_sm.close()
        dev_sm.close()


def _run_workflow(
    store: MemoryStore, proj: dict,
    workdir: str | None, settings: MySwatSettings, requirement: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[AgentRunner], None] | None = None,
    mode: WorkMode = WorkMode.full,
    proposer_sm: SessionManager | None = None,
    on_event: Callable | None = None,
) -> None:
    """Run a teamwork workflow, optionally reusing an architect chat session."""
    from myswat.workflow.engine import WorkflowEngine

    if proposer_sm is not None and mode not in {WorkMode.full, WorkMode.architect_design}:
        proposer_sm = None

    # Set up Dev
    dev_agent = store.get_agent(proj["id"], "developer")
    if not dev_agent:
        console.print("[red]Missing developer agent.[/red]")
        return

    arch_agent = None
    arch_sm: SessionManager | None = None
    if mode in {WorkMode.full, WorkMode.architect_design}:
        if proposer_sm is not None:
            proposed_row = getattr(proposer_sm, "_agent_row", None)
            if isinstance(proposed_row, dict):
                arch_agent = proposed_row
            else:
                arch_agent = store.get_agent(proj["id"], "architect") or {
                    "id": proposer_sm.agent_id,
                    "display_name": "Architect / PM",
                    "cli_backend": "",
                    "model_name": "",
                }
            proposer_runner = getattr(proposer_sm, "_runner", None)
            if proposer_runner is not None and register_cancel_target:
                register_cancel_target(proposer_runner)
        else:
            arch_agent = store.get_agent(proj["id"], "architect")
            if arch_agent:
                arch_runner = make_runner_from_row(arch_agent, settings=settings)
                arch_runner.workdir = workdir
                if register_cancel_target:
                    register_cancel_target(arch_runner)
                arch_sm = SessionManager(
                    store=store,
                    runner=arch_runner,
                    agent_row=arch_agent,
                    project_id=proj["id"],
                    settings=settings,
                )
    if mode == WorkMode.architect_design and arch_agent is None:
        console.print("[red]Missing architect agent.[/red]")
        return

    dev_runner = make_runner_from_row(dev_agent, settings=settings)
    dev_runner.workdir = workdir
    if register_cancel_target:
        register_cancel_target(dev_runner)
    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], settings=settings,
    )

    # Set up QA(s)
    qa_sms = []
    for qa_role in ("qa_main", "qa_vice"):
        qa_agent = store.get_agent(proj["id"], qa_role)
        if qa_agent:
            qa_runner = make_runner_from_row(qa_agent, settings=settings)
            qa_runner.workdir = workdir
            if register_cancel_target:
                register_cancel_target(qa_runner)
            qa_sm = SessionManager(
                store=store, runner=qa_runner, agent_row=qa_agent,
                project_id=proj["id"], settings=settings,
            )
            qa_sms.append(qa_sm)

    if not qa_sms:
        console.print("[red]No QA agents found.[/red]")
        return

    # Create work item
    work_item_id = store.create_work_item(
        project_id=proj["id"], title=requirement[:200],
        description=requirement,
        item_type="design" if mode in {WorkMode.design, WorkMode.architect_design} else "code_change",
        assigned_agent_id=arch_agent["id"] if isinstance(arch_agent, dict) else dev_agent["id"],
        metadata_json={"work_mode": mode.value},
    )
    if on_work_item_created:
        on_work_item_created(work_item_id)
    store.update_work_item_status(work_item_id, "in_progress")

    # Create sessions
    if proposer_sm is not None:
        arch_sm = proposer_sm.fork_for_work_item(
            work_item_id,
            purpose=("Arch design" if mode == WorkMode.architect_design else "Arch workflow") + f": {requirement[:80]}",
        )
    elif arch_sm is not None:
        arch_sm.create_or_resume(
            purpose=("Workflow architect design" if mode == WorkMode.architect_design else "Workflow architect") + f": {requirement[:80]}",
            work_item_id=work_item_id,
        )
    dev_sm.create_or_resume(purpose=f"Workflow dev: {requirement[:80]}", work_item_id=work_item_id)
    for qa_sm in qa_sms:
        qa_sm.create_or_resume(purpose=f"Workflow QA: {requirement[:80]}", work_item_id=work_item_id)

    console.print(f"\n[bold]Starting workflow for:[/bold] {requirement}")
    if isinstance(arch_agent, dict):
        console.print(
            f"[dim]Architect: {arch_agent['display_name']} ({arch_agent['cli_backend']}/{arch_agent['model_name']})[/dim]"
        )
    console.print(
        f"[dim]Dev: {dev_agent['display_name']} ({dev_agent['cli_backend']}/{dev_agent['model_name']})[/dim]"
    )
    for qa_sm in qa_sms:
        qa_row = qa_sm._agent_row
        console.print(
            f"[dim]QA:  {qa_row['display_name']} ({qa_row['cli_backend']}/{qa_row['model_name']})[/dim]"
        )
    console.print(f"[dim]Work item: {work_item_id}[/dim]\n")

    engine = WorkflowEngine(
        store=store,
        dev_sm=dev_sm,
        qa_sms=qa_sms,
        arch_sm=arch_sm,
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_review_iterations=settings.workflow.max_review_iterations,
        mode=mode,
        ask_user=_make_prompt_callback(prompt_session),
        auto_approve=False,
        should_cancel=should_cancel,
        on_event=on_event,
    )

    try:
        result = engine.run(requirement)

        if should_cancel and should_cancel():
            store.update_work_item_status(work_item_id, "blocked")
        elif getattr(result, "blocked", False):
            store.update_work_item_status(work_item_id, "blocked")
        elif result.success:
            store.update_work_item_status(work_item_id, "completed")
        else:
            store.update_work_item_status(work_item_id, "review")
    except Exception:
        store.update_work_item_status(work_item_id, "blocked")
        raise
    finally:
        if arch_sm is not None:
            arch_sm.close()
        dev_sm.close()
        for qa_sm in qa_sms:
            qa_sm.close()


def _run_inline_review_interactive(
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    initial_process_events: list[dict] | None = None,
) -> None:
    work_item_ref: dict[str, int | None] = {"id": None}
    cancel_targets: list[AgentRunner] = []
    cancel_event = threading.Event()

    def _worker():
        return _run_inline_review(
            store=store,
            proj=proj,
            workdir=workdir,
            settings=settings,
            task=task,
            should_cancel=cancel_event.is_set,
            on_work_item_created=lambda wid: work_item_ref.__setitem__("id", wid),
            register_cancel_target=cancel_targets.append,
            initial_process_events=initial_process_events,
        )

    _run_with_task_monitor(
        console=console,
        store=store,
        proj=proj,
        label="Running dev+QA review loop",
        worker_fn=_worker,
        work_item_ref=work_item_ref,
        cancel_targets=cancel_targets,
        cancel_event=cancel_event,
    )


def _run_design_review_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    _run_workflow_interactive(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        prompt_session=prompt_session,
        mode=WorkMode.architect_design,
        proposer_sm=proposer_sm,
        label="Running design workflow",
    )


def _run_testplan_review_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    from myswat.cli.workflow_display import WorkflowDisplay

    work_item_ref: dict[str, int | None] = {"id": None}
    cancel_targets: list[AgentRunner] = []
    cancel_event = threading.Event()
    display = WorkflowDisplay()

    def _worker():
        return _run_testplan_review(
            store=store,
            proj=proj,
            proposer_sm=proposer_sm,
            workdir=workdir,
            settings=settings,
            task=task,
            prompt_session=prompt_session,
            should_cancel=cancel_event.is_set,
            on_work_item_created=lambda wid: work_item_ref.__setitem__("id", wid),
            register_cancel_target=cancel_targets.append,
            on_event=display.handle_event,
        )

    _run_with_task_monitor(
        console=console,
        store=store,
        proj=proj,
        label="Running QA test-plan workflow",
        worker_fn=_worker,
        work_item_ref=work_item_ref,
        cancel_targets=cancel_targets,
        cancel_event=cancel_event,
        workflow_display=display,
    )


def _run_workflow_interactive(
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    requirement: str,
    prompt_session: PromptSession | None = None,
    mode: WorkMode = WorkMode.full,
    proposer_sm: SessionManager | None = None,
    label: str | None = None,
) -> None:
    from myswat.cli.workflow_display import WorkflowDisplay

    work_item_ref: dict[str, int | None] = {"id": None}
    cancel_targets: list[AgentRunner] = []
    cancel_event = threading.Event()
    display = WorkflowDisplay()

    def _worker():
        return _run_workflow(
            store=store,
            proj=proj,
            workdir=workdir,
            settings=settings,
            requirement=requirement,
            prompt_session=prompt_session,
            should_cancel=cancel_event.is_set,
            on_work_item_created=lambda wid: work_item_ref.__setitem__("id", wid),
            register_cancel_target=cancel_targets.append,
            mode=mode,
            proposer_sm=proposer_sm,
            on_event=display.handle_event,
        )

    _run_with_task_monitor(
        console=console,
        store=store,
        proj=proj,
        label=label or ("Running full teamwork workflow" if mode == WorkMode.full else f"Running {mode.value} teamwork workflow"),
        worker_fn=_worker,
        work_item_ref=work_item_ref,
        cancel_targets=cancel_targets,
        cancel_event=cancel_event,
        workflow_display=display,
    )


def _run_full_workflow(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    workdir: str | None,
    settings: MySwatSettings,
    requirement: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[AgentRunner], None] | None = None,
) -> None:
    return _run_workflow(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=requirement,
        prompt_session=prompt_session,
        should_cancel=should_cancel,
        on_work_item_created=on_work_item_created,
        register_cancel_target=register_cancel_target,
        mode=WorkMode.full,
        proposer_sm=proposer_sm,
    )


def _run_full_workflow_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    workdir: str | None,
    settings: MySwatSettings,
    requirement: str,
    prompt_session: PromptSession | None = None,
) -> None:
    _run_workflow_interactive(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=requirement,
        prompt_session=prompt_session,
        mode=WorkMode.full,
        proposer_sm=proposer_sm,
        label="Running architect-led full workflow",
    )
