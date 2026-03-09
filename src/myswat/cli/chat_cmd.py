"""myswat chat — interactive conversation with agents."""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import time
import tty

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from myswat.agents.base import AgentRunner
from myswat.agents.codex_runner import CodexRunner
from myswat.agents.kimi_runner import KimiRunner
from myswat.agents.session_manager import SessionManager
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import run_migrations
from myswat.memory.compactor import KnowledgeCompactor
from myswat.memory.embedder import preload_model
from myswat.memory.store import MemoryStore

console = Console()

HELP_TEXT = """
[bold]Commands:[/bold]
  /help                 Show this help
  /status               Show project status
  /work <requirement>   Start full workflow: design -> review -> plan -> dev -> commit
  /role <role>          Switch agent role (developer, architect, qa_main, qa_vice)
  /reset                Reset AI session (fresh context reload, same TiDB session)
  /review <task>        Start dev+reviewer loop for a task (legacy)
  /agents               List available agents
  /sessions             Show active sessions
  /history [n]          Show last n turns (default 10)
  /new                  Start a new session (close current)
  /quit, /exit, Ctrl+D  Exit chat
"""


def _make_compaction_runner(
    store: MemoryStore, proj: dict, settings: MySwatSettings,
) -> AgentRunner | None:
    """Create a runner for knowledge compaction. Uses the first available codex agent."""
    # Try to find a codex agent for compaction
    agents = store.list_agents(proj["id"])
    for a in agents:
        if a["cli_backend"] == settings.compaction.compaction_backend:
            return _make_runner(a)
    # Fallback: use any agent
    if agents:
        return _make_runner(agents[0])
    return None


def _make_runner(agent_row: dict) -> AgentRunner:
    backend = agent_row["cli_backend"]
    cli_path = agent_row["cli_path"]
    model = agent_row["model_name"]
    extra_flags = json.loads(agent_row["cli_extra_args"]) if agent_row.get("cli_extra_args") else []

    if backend == "codex":
        return CodexRunner(cli_path=cli_path, model=model, extra_flags=extra_flags)
    elif backend == "kimi":
        return KimiRunner(cli_path=cli_path, model=model, extra_flags=extra_flags)
    else:
        raise typer.BadParameter(f"Unknown CLI backend: {backend}")


_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _check_esc() -> bool:
    """Non-blocking check if ESC key was pressed. Returns True if ESC detected."""
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                return True
    except Exception:
        pass
    return False


_MAX_LIVE_LINES = 8  # show last N lines of live output


def _extract_delegation(text: str) -> str | None:
    """Extract a delegation task from architect's response.

    Looks for a ```delegate block with a TASK: line.
    Returns the task description or None.
    """
    import re
    match = re.search(r"```delegate\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        return None
    block = match.group(1)
    for line in block.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("TASK:"):
            return line[5:].strip()
    # Fallback: use the whole block content
    return block.strip() or None


def _build_live_display(
    frame_idx: int, elapsed: float, live_lines: list[str],
) -> Text:
    """Build the Rich Text renderable for the live display."""
    frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
    header = f" {frame}  Agent working... ({_fmt_duration(elapsed)})  ESC to cancel\n"

    # Show tail of live output
    tail = live_lines[-_MAX_LIVE_LINES:] if live_lines else []
    if len(live_lines) > _MAX_LIVE_LINES:
        tail = [f"  ... ({len(live_lines) - _MAX_LIVE_LINES} earlier lines)"] + tail

    body = "\n".join(f"  {line}" for line in tail)

    text = Text(header, style="bold cyan")
    if body:
        text.append(body + "\n", style="dim")
    return text


def _send_with_timer(console: Console, sm: SessionManager, prompt: str):
    """Send a message to the agent while showing live output + timer.

    Press ESC to cancel the running agent subprocess.
    Returns (response, elapsed_seconds).
    """
    result = [None]
    error = [None]
    start = time.monotonic()

    def _run():
        try:
            result[0] = sm.send(prompt)
        except Exception as e:
            error[0] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    cancelled = False
    frame_idx = 0
    try:
        tty.setcbreak(fd)
        with Live(console=console, refresh_per_second=4, transient=True) as live:
            while worker.is_alive():
                elapsed = time.monotonic() - start
                live_lines = sm._runner.live_output
                live.update(_build_live_display(frame_idx, elapsed, live_lines))
                frame_idx += 1

                if _check_esc():
                    cancelled = True
                    live.update(Text(" Cancelling...", style="bold yellow"))
                    sm._runner.cancel()
                    worker.join(timeout=5)
                    break

                worker.join(timeout=0.25)

            # Restore terminal BEFORE Live.__exit__ so Rich's transient
            # cleanup sees the correct terminal mode and cursor position.
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
    finally:
        # Safety: ensure terminal is always restored even on exceptions.
        try:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
        except Exception:
            pass

    elapsed = time.monotonic() - start

    if error[0] is not None:
        raise error[0]

    if cancelled and result[0] is None:
        from myswat.agents.base import AgentResponse
        return AgentResponse(content="Request cancelled.", exit_code=-1, cancelled=True), elapsed

    return result[0], elapsed


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
    run_migrations(pool)
    store = MemoryStore(pool)

    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found.[/red]")
        raise typer.Exit(1)

    effective_workdir = workdir or proj.get("repo_path")

    # Auto-learn if project hasn't been learned yet
    from myswat.cli.learn_cmd import ensure_learned
    ensure_learned(store, project_slug, proj["id"], effective_workdir)

    # Create a lightweight runner for compaction (uses codex by default)
    compaction_runner = _make_compaction_runner(store, proj, settings)
    compactor = KnowledgeCompactor(
        store=store,
        runner=compaction_runner,
        threshold_turns=settings.compaction.threshold_turns,
        threshold_tokens=settings.compaction.threshold_tokens,
    )

    # State
    current_role = role
    sm: SessionManager | None = None

    def _switch_agent(new_role: str) -> SessionManager:
        """Create a new SessionManager for the given role."""
        agent_row = store.get_agent(proj["id"], new_role)
        if not agent_row:
            console.print(f"[red]Agent role '{new_role}' not found.[/red]")
            return None
        runner = _make_runner(agent_row)
        runner.workdir = effective_workdir
        new_sm = SessionManager(
            store=store, runner=runner, agent_row=agent_row,
            project_id=proj["id"], compactor=compactor,
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
    sm = _switch_agent(current_role)
    if sm is None:
        raise typer.Exit(1)

    # Multi-line prompt: Enter submits, Alt+Enter inserts newline (like codex CLI)
    _bindings = KeyBindings()

    @_bindings.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    prompt_session: PromptSession[str] = PromptSession(
        multiline=False,
        key_bindings=_bindings,
        prompt_continuation="... ",
    )

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

            elif cmd == "/role":
                if not arg:
                    console.print(f"[dim]Current role: {current_role}. Usage: /role <role>[/dim]")
                    continue
                new_role = arg.strip()
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                new_sm = _switch_agent(new_role)
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
                sm = _switch_agent(current_role)

            elif cmd == "/work":
                if not arg:
                    console.print("[dim]Usage: /work <requirement description>[/dim]")
                    continue
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                _run_workflow(store, proj, compactor, effective_workdir, settings, arg, prompt_session)
                sm = _switch_agent(current_role)

            elif cmd == "/review":
                if not arg:
                    console.print("[dim]Usage: /review <task description>[/dim]")
                    continue
                if sm:
                    with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                        sm.close()
                _run_inline_review(store, proj, compactor, effective_workdir, settings, arg)
                # Reopen chat session after review
                sm = _switch_agent(current_role)

            else:
                console.print(f"[dim]Unknown command: {cmd}. Type /help for commands.[/dim]")

            continue

        # Regular message — send to agent
        if sm is None:
            sm = _switch_agent(current_role)
            if sm is None:
                continue

        response, elapsed = _send_with_timer(console, sm, user_input)

        if response.cancelled:
            console.print(f"\n[yellow]Request cancelled.[/yellow] [dim]({_fmt_duration(elapsed)})[/dim]")
        elif response.success:
            console.print()
            console.print(Markdown(response.content))
            console.print(f"\n[dim]({_fmt_duration(elapsed)})[/dim]")

            # Check for architect delegation signal
            delegation_task = _extract_delegation(response.content)
            if delegation_task and current_role == "architect":
                console.print(
                    f"\n[bold cyan]Architect wants to delegate:[/bold cyan] {delegation_task[:120]}"
                )
                try:
                    confirm = prompt_session.prompt(
                        "Start dev+review loop? [Y/n] > ",
                        multiline=False,
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "n"
                if confirm in ("", "y", "yes"):
                    if sm:
                        with console.status("[dim]Saving session to TiDB...[/dim]", spinner="dots"):
                            sm.close()
                    _run_inline_review(store, proj, compactor, effective_workdir, settings, delegation_task)
                    sm = _switch_agent(current_role)
                else:
                    console.print("[dim]Delegation skipped.[/dim]")
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
        table = Table(title="Work Items")
        table.add_column("ID", style="cyan")
        table.add_column("Status")
        table.add_column("Title", max_width=50)
        for item in active[:10]:
            table.add_row(str(item["id"]), item["status"], item["title"][:50])
        console.print(table)
    else:
        console.print("[dim]No active work items.[/dim]")


def _run_inline_review(
    store: MemoryStore, proj: dict, compactor: KnowledgeCompactor,
    workdir: str | None, settings: MySwatSettings, task: str,
) -> None:
    """Run a dev+reviewer loop from within the chat REPL."""
    from myswat.workflow.review_loop import run_review_loop

    dev_agent = store.get_agent(proj["id"], "developer")
    reviewer_agent = store.get_agent(proj["id"], "qa_main")
    if not dev_agent or not reviewer_agent:
        console.print("[red]Missing developer or qa_main agent.[/red]")
        return

    dev_runner = _make_runner(dev_agent)
    dev_runner.workdir = workdir
    reviewer_runner = _make_runner(reviewer_agent)
    reviewer_runner.workdir = workdir

    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], compactor=compactor,
    )
    reviewer_sm = SessionManager(
        store=store, runner=reviewer_runner, agent_row=reviewer_agent,
        project_id=proj["id"], compactor=compactor,
    )

    work_item_id = store.create_work_item(
        project_id=proj["id"], title=task[:200],
        description=task, item_type="code_change",
        assigned_agent_id=dev_agent["id"],
    )
    store.update_work_item_status(work_item_id, "in_progress")

    dev_sm.create_or_resume(purpose=f"Dev: {task[:100]}", work_item_id=work_item_id)
    reviewer_sm.create_or_resume(purpose=f"Review: {task[:100]}", work_item_id=work_item_id)

    console.print(f"\n[bold]Starting review loop for:[/bold] {task}")

    verdict = run_review_loop(
        store=store, dev_sm=dev_sm, reviewer_sm=reviewer_sm,
        task=task, project_id=proj["id"], work_item_id=work_item_id,
        max_iterations=settings.workflow.max_review_iterations,
    )

    if verdict.verdict == "lgtm":
        store.update_work_item_status(work_item_id, "approved")
    else:
        store.update_work_item_status(work_item_id, "review")

    dev_sm.close()
    reviewer_sm.close()


def _run_workflow(
    store: MemoryStore, proj: dict, compactor: KnowledgeCompactor,
    workdir: str | None, settings: MySwatSettings, requirement: str,
    prompt_session: PromptSession | None = None,
) -> None:
    """Run the full teamwork workflow: design -> review -> plan -> dev -> commit."""
    from myswat.workflow.engine import WorkflowEngine

    # Set up Dev
    dev_agent = store.get_agent(proj["id"], "developer")
    if not dev_agent:
        console.print("[red]Missing developer agent.[/red]")
        return

    dev_runner = _make_runner(dev_agent)
    dev_runner.workdir = workdir
    dev_sm = SessionManager(
        store=store, runner=dev_runner, agent_row=dev_agent,
        project_id=proj["id"], compactor=compactor,
    )

    # Set up QA(s)
    qa_sms = []
    for qa_role in ("qa_main", "qa_vice"):
        qa_agent = store.get_agent(proj["id"], qa_role)
        if qa_agent:
            qa_runner = _make_runner(qa_agent)
            qa_runner.workdir = workdir
            qa_sm = SessionManager(
                store=store, runner=qa_runner, agent_row=qa_agent,
                project_id=proj["id"], compactor=compactor,
            )
            qa_sms.append(qa_sm)

    if not qa_sms:
        console.print("[red]No QA agents found.[/red]")
        return

    # Create work item
    work_item_id = store.create_work_item(
        project_id=proj["id"], title=requirement[:200],
        description=requirement, item_type="code_change",
        assigned_agent_id=dev_agent["id"],
    )
    store.update_work_item_status(work_item_id, "in_progress")

    # Create sessions
    dev_sm.create_or_resume(purpose=f"Workflow dev: {requirement[:80]}", work_item_id=work_item_id)
    for qa_sm in qa_sms:
        qa_sm.create_or_resume(purpose=f"Workflow QA: {requirement[:80]}", work_item_id=work_item_id)

    # User input callback — use prompt_toolkit if available, else plain input
    def ask_user(prompt_text: str) -> str:
        if prompt_session:
            try:
                return prompt_session.prompt(prompt_text, multiline=False).strip()
            except (EOFError, KeyboardInterrupt):
                return "n"
        try:
            return input(f"\n{prompt_text}").strip()
        except (EOFError, KeyboardInterrupt):
            return "n"

    console.print(f"\n[bold]Starting workflow for:[/bold] {requirement}")
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
        project_id=proj["id"],
        work_item_id=work_item_id,
        max_review_iterations=settings.workflow.max_review_iterations,
        ask_user=ask_user,
    )

    result = engine.run(requirement)

    # Update work item
    if result.success:
        store.update_work_item_status(work_item_id, "completed")
    else:
        store.update_work_item_status(work_item_id, "review")

    # Close sessions
    dev_sm.close()
    for qa_sm in qa_sms:
        qa_sm.close()
