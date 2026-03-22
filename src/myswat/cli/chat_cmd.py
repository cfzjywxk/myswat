"""myswat chat — interactive conversation with agents."""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from myswat.agents.base import AgentResponse
from myswat.cli.prompting import create_prompt_session
from myswat.cli.progress import (
    _describe_process_event,
    _fmt_duration,
    _run_with_task_monitor,
    _single_line_preview,
)
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import ensure_schema
from myswat.large_payloads import maybe_externalize_response
from myswat.memory.embedder import preload_model
from myswat.memory.learn_triggers import submit_chat_learn_request
from myswat.memory.store import MemoryStore
from myswat.models.session import Session
from myswat.server.control_client import DaemonClient, DaemonClientError
from myswat.server.mcp_http_client import MCPHTTPClient, MCPHTTPClientError
from myswat.workflow.modes import (
    DEFAULT_DELEGATION_MODE,
    DELEGATION_MODE_SPECS,
    WorkMode,
    normalize_delegation_mode,
)
from myswat.workflow.prompts import ARCHITECT_PRD_WORKFLOW

console = Console()

HELP_TEXT = """
[bold]Commands:[/bold]
  /help                 Show this help
  /status               Show project status
  /task <id>            Show detailed status for one work item
  /history [n]          Show recent turns from the active chat session
  /prd <requirement>    Start an interactive PRD workflow in chat
  /work <requirement>   Start full workflow: design -> plan -> develop -> report (no GA test)
  /dev <task>           Start the development workflow for a task
  /ga-test <task>       Start the standalone GA test workflow for a task
  /role <role>          Switch agent role (developer, architect, qa_main)
  /reset                Reset AI session (fresh context reload, same TiDB session)
  /agents               List available agents
  /sessions             Show active sessions
  /new                  Start a new session (close current)
  /quit, /exit, Ctrl+D  Exit chat
"""

_REMOTE_WORKFLOW_ACTIVE_STATUSES = frozenset({"pending", "in_progress", "review"})
_SEND_POLL_INTERVAL_SECONDS = 0.05
_SEND_HEALTHCHECK_INTERVAL_SECONDS = 5.0


class _RemoteRunnerStub:
    """Minimal runner surface required by shared chat helpers."""

    def __init__(self) -> None:
        self.live_output: list[str] = []

    def clear_live_output(self) -> None:
        self.live_output.clear()

    def cancel(self) -> None:
        self.live_output.clear()


class SessionManager:
    """MCP-backed chat session manager used by the chat CLI."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        project_row: dict,
        agent_row: dict,
        settings: MySwatSettings | None = None,
        workdir: str | None = None,
        mcp: MCPHTTPClient | None = None,
    ) -> None:
        self._store = store
        self._project_row = project_row
        self._agent_row = agent_row
        self._settings = settings or MySwatSettings()
        self._workdir = workdir or project_row.get("repo_path")
        self._runner = _RemoteRunnerStub()
        self._session: Session | None = None
        self._mcp = mcp or MCPHTTPClient(
            f"http://{self._settings.server.host}:{self._settings.server.port}",
            timeout_seconds=None,
        )

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def agent_role(self) -> str:
        return str(self._agent_row.get("role") or "")

    @property
    def agent_id(self) -> int:
        return int(self._agent_row.get("id") or 0)

    def create_or_resume(
        self,
        purpose: str | None = None,
        work_item_id: int | None = None,
    ) -> Session:
        if work_item_id is not None:
            raise ValueError("MCP chat sessions do not support work-item-scoped sessions.")
        if self._session is not None:
            return self._session

        result = self._mcp.call_tool(
            "open_chat_session",
            {
                "project_id": int(self._project_row["id"]),
                "agent_role": self.agent_role,
                "purpose": purpose,
                "workdir": self._workdir,
            },
        )
        self._session = Session(
            id=int(result.get("session_id") or 0),
            agent_id=int(result.get("agent_id") or self.agent_id),
            session_uuid=str(result.get("session_uuid") or ""),
            purpose=purpose,
        )
        return self._session

    def send(
        self,
        prompt: str,
        task_description: str | None = None,
        status_callback: Callable[[str, dict[str, object]], None] | None = None,
    ) -> AgentResponse:
        if self._session is None:
            self.create_or_resume(purpose=task_description)

        result = self._mcp.call_tool(
            "send_chat_message",
            {
                "session_id": int(self._session.id or 0),
                "prompt": prompt,
                "task_description": task_description,
                "workdir": self._workdir,
            },
        )
        return AgentResponse(
            content=str(result.get("content") or ""),
            exit_code=int(result.get("exit_code") or 0),
            raw_stdout=str(result.get("raw_stdout") or ""),
            raw_stderr=str(result.get("raw_stderr") or ""),
            token_usage=result.get("token_usage") or {},
            cancelled=bool(result.get("cancelled")),
        )

    def reset_ai_session(self) -> None:
        if self._session is None:
            return
        self._mcp.call_tool(
            "reset_chat_session",
            {
                "session_id": int(self._session.id or 0),
                "workdir": self._workdir,
            },
        )

    def close(self) -> None:
        if self._session is None:
            return
        self._mcp.call_tool(
            "close_chat_session",
            {
                "session_id": int(self._session.id or 0),
                "workdir": self._workdir,
            },
        )
        self._session = None


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
    """Extract a delegation task and mode from an agent response."""
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


def _extract_prd_block(text: str) -> str | None:
    import re

    match = re.search(r"```prd\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _cancel_prd_work_item(store: MemoryStore, work_item_id: int, summary: str) -> None:
    store.update_work_item_status(work_item_id, "cancelled")
    store.update_work_item_state(
        work_item_id,
        current_stage="prd_cancelled",
        latest_summary=summary,
        next_todos=[],
        open_issues=[],
    )
    store.append_work_item_process_event(
        work_item_id,
        event_type="prd_cancelled",
        title="PRD workflow cancelled",
        summary=summary,
        from_role="user",
        to_role="architect",
    )


def _create_prd_work_item(
    *,
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    requirement: str,
    workdir: str | None,
) -> int:
    metadata_json = {
        "work_mode": WorkMode.prd.value,
        "execution_mode": "chat",
        "submitted_via": "chat_prd",
        "requested_workdir": workdir,
    }
    work_item_id = store.create_work_item(
        project_id=int(proj["id"]),
        title=requirement[:200],
        description=requirement,
        item_type="design",
        assigned_agent_id=proposer_sm.agent_id,
        metadata_json=metadata_json,
    )
    store.update_work_item_status(work_item_id, "in_progress")
    store.update_work_item_state(
        work_item_id,
        current_stage="prd",
        latest_summary="Interactive PRD workflow started in chat.",
        next_todos=["Answer architect clarifying questions until the PRD is approved."],
        open_issues=[],
        updated_by_agent_id=proposer_sm.agent_id,
    )
    store.append_work_item_process_event(
        work_item_id,
        event_type="prd_started",
        title="PRD workflow started",
        summary="Interactive PRD drafting started in chat.",
        from_role="user",
        to_role="architect",
        updated_by_agent_id=proposer_sm.agent_id,
    )
    return work_item_id


def _persist_approved_prd(
    *,
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    work_item_id: int,
    requirement: str,
    prd_markdown: str,
) -> int:
    title = requirement[:200]
    heading_lines = [line[2:].strip() for line in prd_markdown.splitlines() if line.startswith("# ")]
    if heading_lines:
        title = heading_lines[0][:200]

    artifact_id = store.create_artifact(
        work_item_id=work_item_id,
        agent_id=proposer_sm.agent_id,
        iteration=1,
        artifact_type="prd_doc",
        title=title,
        content=prd_markdown,
        metadata_json={
            "approved": True,
            "source": "chat_prd",
            "work_mode": WorkMode.prd.value,
        },
    )
    latest_summary = f"Approved PRD stored as artifact #{artifact_id}."
    store.update_work_item_status(work_item_id, "approved")
    store.update_work_item_state(
        work_item_id,
        current_stage="prd_approved",
        latest_summary=latest_summary,
        next_todos=[
            f"Use `/work PRD_ARTIFACT: {artifact_id}` in chat to start delivery from this PRD.",
            (
                f"Use `myswat work -p {proj['slug']} \"PRD_ARTIFACT: {artifact_id}\"` "
                "from the CLI to run the daemon workflow from this PRD."
            ),
        ],
        open_issues=[],
        last_artifact_id=artifact_id,
        updated_by_agent_id=proposer_sm.agent_id,
    )
    store.append_work_item_process_event(
        work_item_id,
        event_type="prd_approved",
        title="PRD approved",
        summary=latest_summary,
        from_role="architect",
        to_role="user",
        updated_by_agent_id=proposer_sm.agent_id,
    )
    return artifact_id


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


def _print_daemon_error(exc: Exception) -> None:
    message = str(exc)
    console.print(f"[red]{message}[/red]")
    lowered = message.lower()
    if "unavailable at" in lowered or "connection refused" in lowered:
        console.print("[dim]Start the daemon first: myswat server[/dim]")
        return
    if "timed out" in lowered:
        console.print(
            "[dim]The daemon is running, but the request is still in progress or blocked.[/dim]"
        )


def _public_chat_work_mode(mode: WorkMode) -> WorkMode:
    if mode == WorkMode.architect_design:
        return WorkMode.design
    if mode == WorkMode.testplan_design:
        return WorkMode.test
    return mode


def _workflow_poll_interval_seconds(settings: MySwatSettings) -> float:
    raw_value = getattr(settings.workflow, "assignment_poll_interval_seconds", 1.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 1.0
    return value if value > 0 else 1.0


def _run_remote_workflow(
    *,
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    requirement: str,
    mode: WorkMode,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
) -> int:
    client = DaemonClient(settings)
    result = client.submit_work(
        project=str(proj["slug"]),
        requirement=requirement,
        workdir=workdir,
        mode=_public_chat_work_mode(mode).value,
    )
    work_item_id = int(result.get("work_item_id") or 0)
    if work_item_id <= 0:
        raise RuntimeError("Daemon response did not include a valid work item ID.")
    if on_work_item_created is not None:
        on_work_item_created(work_item_id)

    poll_interval = _workflow_poll_interval_seconds(settings)
    cancel_requested = False

    while True:
        payload = client.get_work_item(
            project=str(proj["slug"]),
            work_item_id=work_item_id,
        )
        work_item = payload.get("work_item") or {}
        status = str(work_item.get("status") or "pending")

        if status not in _REMOTE_WORKFLOW_ACTIVE_STATUSES:
            return work_item_id

        if should_cancel is not None and should_cancel() and not cancel_requested:
            client.control_work(
                project=str(proj["slug"]),
                work_item_id=work_item_id,
                action="cancel",
            )
            cancel_requested = True

        time.sleep(poll_interval)


def _send_with_timer(
    console: Console,
    sm: SessionManager,
    prompt: str,
    task_description: str | None = None,
) -> tuple[AgentResponse, float]:
    """Send a message while showing a simple status spinner."""
    result: list[AgentResponse | None] = [None]
    error: list[BaseException | None] = [None]
    start = time.monotonic()
    last_healthcheck_at = start

    def _run() -> None:
        try:
            result[0] = sm.send(prompt, task_description=task_description)
        except BaseException as exc:  # pragma: no cover - exact exception asserted in tests
            error[0] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    try:
        with console.status("[dim]Waiting for agent response...[/dim]", spinner="dots"):
            while worker.is_alive():
                worker.join(timeout=_SEND_POLL_INTERVAL_SECONDS)
                now = time.monotonic()
                if (
                    worker.is_alive()
                    and now - last_healthcheck_at >= _SEND_HEALTHCHECK_INTERVAL_SECONDS
                ):
                    last_healthcheck_at = now
                    mcp = getattr(sm, "_mcp", None)
                    healthcheck = getattr(mcp, "healthcheck", None)
                    if callable(healthcheck):
                        try:
                            healthcheck()
                        except Exception:
                            pass
    except KeyboardInterrupt:
        runner = getattr(sm, "_runner", None)
        if runner is not None:
            runner.cancel()
        worker.join(timeout=1.0)
        elapsed = time.monotonic() - start
        if error[0] is not None:
            raise error[0]
        if result[0] is not None:
            return result[0], elapsed
        return AgentResponse(content="Request cancelled.", exit_code=-1, cancelled=True), elapsed

    elapsed = time.monotonic() - start
    if error[0] is not None:
        raise error[0]
    return result[0], elapsed


def run_chat(
    project_slug: str,
    role: str = "developer",
    workdir: str | None = None,
) -> None:
    """Interactive chat session with an agent."""
    settings = MySwatSettings()

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
    current_role = role
    sm: SessionManager | None = None

    def _switch_agent(new_role: str, runner_settings: MySwatSettings) -> SessionManager | None:
        agent_row = store.get_agent(proj["id"], new_role)
        if not agent_row:
            console.print(f"[red]Agent role '{new_role}' not found.[/red]")
            return None
        new_sm = SessionManager(
            store=store,
            project_row=proj,
            agent_row=agent_row,
            settings=runner_settings,
            workdir=effective_workdir,
        )
        try:
            with console.status("[dim]Opening chat session...[/dim]", spinner="dots"):
                new_sm.create_or_resume(purpose=f"Interactive chat ({new_role})")
        except (DaemonClientError, MCPHTTPClientError) as exc:
            _print_daemon_error(exc)
            return None
        session = new_sm.session
        session_uuid = session.session_uuid[:8] if session is not None else "?"
        console.print(
            f"[dim]Agent: {agent_row['display_name']} "
            f"({agent_row['cli_backend']}/{agent_row['model_name']}) | "
            f"Session: {session_uuid}[/dim]"
        )
        return new_sm

    def _close_session_best_effort(manager: SessionManager | None) -> None:
        if not manager:
            return
        try:
            with console.status("[dim]Closing chat session...[/dim]", spinner="dots"):
                manager.close()
        except (DaemonClientError, MCPHTTPClientError) as exc:
            _print_daemon_error(exc)

    console.print(
        Panel(
            f"[bold]MySwat Chat[/bold] — project: [cyan]{proj['name']}[/cyan]\n"
            f"Type a message to chat with the agent. Type [bold]/help[/bold] for commands.\n"
            f"[dim]Enter to submit · Alt+Enter for new line · Paste supported[/dim]",
            border_style="blue",
        )
    )
    sm = _switch_agent(current_role, settings)
    if sm is None:
        raise typer.Exit(1)

    prompt_session = _build_prompt_session(settings, f"chat-{project_slug}")

    def _handle_delegation(
        delegation_task: str,
        delegation_mode: str,
        banner_override: str | None = None,
        detail_override: str | None = None,
    ) -> None:
        nonlocal sm

        delegation_spec = DELEGATION_MODE_SPECS.get(delegation_mode)
        if delegation_spec is None:
            console.print(
                f"[yellow]Delegation mode '{delegation_mode}' is not supported.[/yellow]"
            )
            return

        if current_role not in delegation_spec.allowed_roles:
            allowed_roles = ", ".join(sorted(delegation_spec.allowed_roles))
            console.print(
                f"[yellow]Delegation mode '{delegation_mode}' is only available for role(s): "
                f"{allowed_roles}. Current role: '{current_role}'.[/yellow]"
            )
            return

        delegation_handlers = {
            "prd_workflow": lambda: _run_prd_workflow_interactive(
                store,
                proj,
                sm,
                effective_workdir,
                settings,
                delegation_task,
                prompt_session=prompt_session,
            ),
            "workflow": lambda: _run_workflow_interactive(
                store,
                proj,
                effective_workdir,
                settings,
                delegation_task,
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
            return

        console.print(
            f"\n[bold cyan]{banner_override or delegation_spec.banner}:[/bold cyan] "
            f"{delegation_task[:160]}"
        )
        detail = detail_override if detail_override is not None else delegation_spec.detail
        if detail:
            console.print(f"[dim]{detail}[/dim]")
        if delegation_spec.save_session_before_run:
            _close_session_best_effort(sm)
        handler()
        if delegation_spec.reset_role_session:
            sm = _switch_agent(current_role, settings)

    while True:
        try:
            print()
            user_input = prompt_session.prompt(
                f"you ({current_role})> ",
                multiline=False,
            )
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except KeyboardInterrupt:
            console.print("\n[dim]Type /quit to exit, or Ctrl+D.[/dim]")
            continue

        user_input = _strip_wrapping_quotes(user_input)
        if not user_input:
            continue

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
                    console.print(
                        f"[dim]Current role: {current_role}. Usage: /role <role>[/dim]"
                    )
                    continue
                new_role = arg.strip()
                _close_session_best_effort(sm)
                new_sm = _switch_agent(new_role, settings)
                if new_sm:
                    sm = new_sm
                    current_role = new_role

            elif cmd == "/agents":
                agents = store.list_agents(proj["id"])
                for agent in agents:
                    marker = (
                        " [bold cyan]<-[/bold cyan]"
                        if agent["role"] == current_role
                        else ""
                    )
                    console.print(
                        f"  {agent['role']:12s} {agent['display_name']:20s} "
                        f"({agent['cli_backend']}/{agent['model_name']}){marker}"
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
                    for session in sessions:
                        turns = store.count_session_turns(session["id"])
                        console.print(
                            f"  {session['session_uuid'][:8]} [{session['role']}] "
                            f"{(session.get('purpose') or '')[:40]} ({turns} turns)"
                        )
                else:
                    console.print("[dim]No active sessions.[/dim]")

            elif cmd == "/history":
                if sm and sm.session:
                    n = int(arg) if arg.isdigit() else 10
                    turns = store.get_session_turns(sm.session.id)
                    for turn in turns[-n:]:
                        label = "[green]you[/green]" if turn.role == "user" else "[cyan]agent[/cyan]"
                        content_preview = turn.content[:200]
                        if len(turn.content) > 200:
                            content_preview += "..."
                        console.print(f"  {label}: {content_preview}")
                else:
                    console.print("[dim]No active session.[/dim]")

            elif cmd == "/reset":
                if sm:
                    try:
                        sm.reset_ai_session()
                    except (DaemonClientError, MCPHTTPClientError) as exc:
                        _print_daemon_error(exc)
                        continue
                    console.print(
                        "[green]AI session reset.[/green] "
                        "[dim]Next message will start a fresh AI session with TiDB context reload.[/dim]"
                    )
                else:
                    console.print("[dim]No active session to reset.[/dim]")

            elif cmd == "/new":
                _close_session_best_effort(sm)
                sm = _switch_agent(current_role, settings)

            elif cmd == "/work":
                if not arg:
                    console.print("[dim]Usage: /work <requirement description>[/dim]")
                    continue
                _close_session_best_effort(sm)
                _run_workflow_interactive(
                    store,
                    proj,
                    effective_workdir,
                    settings,
                    arg,
                )
                sm = _switch_agent(current_role, settings)

            elif cmd == "/prd":
                if not arg:
                    console.print("[dim]Usage: /prd <requirement description>[/dim]")
                    continue
                original_role = current_role
                using_temp_architect = current_role != "architect"
                if using_temp_architect:
                    _close_session_best_effort(sm)
                    sm = _switch_agent("architect", settings)
                    if sm is None:
                        current_role = original_role
                        continue
                    current_role = "architect"
                _run_prd_workflow_interactive(
                    store,
                    proj,
                    sm,
                    effective_workdir,
                    settings,
                    arg,
                    prompt_session=prompt_session,
                )
                if using_temp_architect:
                    _close_session_best_effort(sm)
                    sm = _switch_agent(original_role, settings)
                    current_role = original_role

            elif cmd == "/dev":
                if not arg:
                    console.print("[dim]Usage: /dev <task description>[/dim]")
                    continue
                _close_session_best_effort(sm)
                _run_inline_review_interactive(
                    store,
                    proj,
                    effective_workdir,
                    settings,
                    arg,
                )
                sm = _switch_agent(current_role, settings)

            elif cmd == "/ga-test":
                if not arg:
                    console.print("[dim]Usage: /ga-test <task description>[/dim]")
                    continue
                _close_session_best_effort(sm)
                _run_ga_test_interactive(
                    store,
                    proj,
                    effective_workdir,
                    settings,
                    arg,
                )
                sm = _switch_agent(current_role, settings)

            else:
                console.print(
                    f"[dim]Unknown command: {cmd}. Type /help for commands.[/dim]"
                )

            continue

        if sm is None:
            sm = _switch_agent(current_role, settings)
            if sm is None:
                continue

        try:
            response, elapsed = _send_with_timer(console, sm, user_input)
        except (DaemonClientError, MCPHTTPClientError) as exc:
            _print_daemon_error(exc)
            continue

        if response.cancelled:
            console.print(
                f"\n[yellow]Request cancelled.[/yellow] [dim]({_fmt_duration(elapsed)})[/dim]"
            )
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

            delegation = _extract_delegation(response.content)
            if delegation:
                delegation_task, delegation_mode = delegation
                _handle_delegation(delegation_task, delegation_mode)
        else:
            console.print(
                Panel(
                    response.content,
                    title=f"Agent Error ({_fmt_duration(elapsed)})",
                    border_style="red",
                )
            )
            if response.raw_stderr:
                console.print(f"[dim red]{response.raw_stderr[:300]}[/dim red]")

    _close_session_best_effort(sm)
    console.print("[dim]Session closed. Turns persisted to TiDB.[/dim]")


def _show_status(store: MemoryStore, pool: TiDBPool, proj: dict) -> None:
    """Inline status display for chat mode."""
    items = list(store.list_work_items(proj["id"]))
    active = [item for item in items if item["status"] in ("in_progress", "review", "pending")]
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
        active = [item for item in items if item["status"] in ("in_progress", "review", "pending")]
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
            console.print(
                Panel(
                    _single_line_preview(state["latest_summary"], 600),
                    title="Latest Summary",
                )
            )
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
        for artifact in artifacts[-5:]:
            table.add_row(
                str(artifact["iteration"]),
                artifact["artifact_type"],
                (artifact.get("title") or "")[:40],
            )
        console.print(table)


def _run_inline_review(
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[object], None] | None = None,
    initial_process_events: list[dict] | None = None,
) -> None:
    return _run_workflow(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        should_cancel=should_cancel,
        on_work_item_created=on_work_item_created,
        mode=WorkMode.develop,
    )


def _run_design_review(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[object], None] | None = None,
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
        should_cancel=should_cancel,
        on_work_item_created=on_work_item_created,
        mode=WorkMode.architect_design,
        proposer_sm=proposer_sm,
    )


def _run_testplan_review(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[object], None] | None = None,
    on_event: Callable | None = None,
    ask_user: Callable[[str], str] | None = None,
    auto_approve: bool = True,
) -> None:
    if proposer_sm is None:
        console.print("[red]Missing QA session.[/red]")
        return
    return _run_workflow(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        should_cancel=should_cancel,
        on_work_item_created=on_work_item_created,
        mode=WorkMode.testplan_design,
        proposer_sm=proposer_sm,
    )


def _run_prd_workflow_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    if proposer_sm is None:
        console.print("[red]Missing architect session.[/red]")
        return

    max_prd_turns = 30
    active_prompt_session = prompt_session or _build_prompt_session(settings, f"prd-{proj['slug']}")
    work_item_id = _create_prd_work_item(
        store=store,
        proj=proj,
        proposer_sm=proposer_sm,
        requirement=task,
        workdir=workdir,
    )
    current_prompt = ARCHITECT_PRD_WORKFLOW.format(requirement=task)

    for _turn in range(max_prd_turns):
        try:
            response, elapsed = _send_with_timer(
                console,
                proposer_sm,
                current_prompt,
                task_description=f"PRD: {task[:120]}",
            )
        except (DaemonClientError, MCPHTTPClientError) as exc:
            store.update_work_item_status(work_item_id, "blocked")
            store.update_work_item_state(
                work_item_id,
                current_stage="prd_blocked",
                latest_summary=str(exc),
                next_todos=["Resume PRD drafting in chat after the architect session is available again."],
                updated_by_agent_id=proposer_sm.agent_id,
            )
            _print_daemon_error(exc)
            return

        if response.cancelled:
            _cancel_prd_work_item(store, work_item_id, "PRD workflow cancelled.")
            console.print(f"\n[yellow]PRD workflow cancelled.[/yellow] [dim]({_fmt_duration(elapsed)})[/dim]")
            return

        console.print()
        rendered, _ = maybe_externalize_response(
            response.content,
            label="prd-workflow-response",
        )
        console.print(Markdown(rendered))
        console.print(f"\n[dim]({_fmt_duration(elapsed)})[/dim]")

        prd_markdown = _extract_prd_block(response.content)
        if prd_markdown:
            store.update_work_item_state(
                work_item_id,
                current_stage="prd_review",
                latest_summary="Architect proposed a PRD draft for approval.",
                next_todos=["Approve the PRD, reject it, or provide revision feedback."],
                updated_by_agent_id=proposer_sm.agent_id,
            )
            approval = active_prompt_session.prompt("approve prd [Y/n/feedback]> ", multiline=False).strip()
            normalized = approval.lower()
            if normalized in {"", "y", "yes"}:
                artifact_id = _persist_approved_prd(
                    store=store,
                    proj=proj,
                    proposer_sm=proposer_sm,
                    work_item_id=work_item_id,
                    requirement=task,
                    prd_markdown=prd_markdown,
                )
                console.print(
                    f"[green]PRD approved.[/green] "
                    f"[dim]Artifact #{artifact_id} on work item {work_item_id}.[/dim]"
                )
                console.print(
                    f"[dim]Start delivery with `/work PRD_ARTIFACT: {artifact_id}` or "
                    f"`myswat work -p {proj['slug']} \"PRD_ARTIFACT: {artifact_id}\"`.[/dim]"
                )
                return
            if normalized in {"n", "no", "cancel", "/cancel", "quit", "/quit"}:
                _cancel_prd_work_item(store, work_item_id, "User declined the PRD draft.")
                console.print("[yellow]PRD workflow cancelled.[/yellow]")
                return
            store.update_work_item_state(
                work_item_id,
                current_stage="prd_revision",
                latest_summary="User requested PRD revisions.",
                next_todos=["Revise the PRD draft and return an updated `prd` block."],
                open_issues=[approval],
                updated_by_agent_id=proposer_sm.agent_id,
            )
            store.append_work_item_process_event(
                work_item_id,
                event_type="prd_feedback",
                title="PRD feedback requested",
                summary=approval,
                from_role="user",
                to_role="architect",
                updated_by_agent_id=proposer_sm.agent_id,
            )
            current_prompt = approval
            continue

        store.update_work_item_state(
            work_item_id,
            current_stage="prd_clarification",
            latest_summary="Architect requested clarification before finalizing the PRD.",
            next_todos=["Answer the architect's questions in chat."],
            updated_by_agent_id=proposer_sm.agent_id,
        )
        user_reply = active_prompt_session.prompt("prd> ", multiline=False).strip()
        if user_reply.lower() in {"cancel", "/cancel", "quit", "/quit"}:
            _cancel_prd_work_item(store, work_item_id, "User cancelled the PRD workflow.")
            console.print("[yellow]PRD workflow cancelled.[/yellow]")
            return
        if not user_reply:
            console.print("[dim]Reply required or type /cancel to stop the PRD workflow.[/dim]")
            continue
        store.append_work_item_process_event(
            work_item_id,
            event_type="prd_user_reply",
            title="PRD clarification received",
            summary=user_reply,
            from_role="user",
            to_role="architect",
            updated_by_agent_id=proposer_sm.agent_id,
        )
        current_prompt = user_reply
    else:
        _cancel_prd_work_item(
            store, work_item_id, f"PRD workflow stopped after {max_prd_turns} turns without approval."
        )
        console.print(
            f"[yellow]PRD workflow stopped after {max_prd_turns} turns without reaching approval.[/yellow]"
        )


def _run_workflow(
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    requirement: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[object], None] | None = None,
    mode: WorkMode = WorkMode.full,
    proposer_sm: SessionManager | None = None,
    on_event: Callable | None = None,
    ask_user: Callable[[str], str] | None = None,
    auto_approve: bool = True,
) -> None:
    dev_agent = store.get_agent(proj["id"], "developer")
    if not dev_agent:
        console.print("[red]Missing developer agent.[/red]")
        return

    if mode == WorkMode.testplan_design:
        arch_agent = store.get_agent(proj["id"], "architect")
        if not arch_agent or not dev_agent:
            console.print("[red]Missing architect or developer agent.[/red]")
            return
    else:
        qa_agents = [
            agent
            for agent in (
                store.get_agent(proj["id"], "qa_main"),
                store.get_agent(proj["id"], "qa_vice"),
            )
            if agent
        ]
        if not qa_agents:
            console.print("[red]No QA agents found.[/red]")
            return
        if mode == WorkMode.architect_design:
            arch_agent = store.get_agent(proj["id"], "architect")
            if not arch_agent:
                console.print("[red]Missing architect agent.[/red]")
                return

    try:
        _run_remote_workflow(
            store=store,
            proj=proj,
            workdir=workdir,
            settings=settings,
            requirement=requirement,
            mode=mode,
            should_cancel=should_cancel,
            on_work_item_created=on_work_item_created,
        )
    except (DaemonClientError, MCPHTTPClientError) as exc:
        _print_daemon_error(exc)


def _run_inline_review_interactive(
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    initial_process_events: list[dict] | None = None,
) -> None:
    _run_workflow_interactive(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        mode=WorkMode.develop,
        label="Running development workflow",
    )


def _run_ga_test_interactive(
    store: MemoryStore,
    proj: dict,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
) -> None:
    _run_workflow_interactive(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        mode=WorkMode.test,
        label="Running GA test workflow",
    )


def _run_design_review_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    if proposer_sm is None:
        console.print("[red]Missing architect session.[/red]")
        return
    _run_workflow_interactive(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        mode=WorkMode.architect_design,
        proposer_sm=proposer_sm,
        label="Running design workflow",
    )


def _run_testplan_review_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    if proposer_sm is None:
        console.print("[red]Missing QA session.[/red]")
        return
    _run_workflow_interactive(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=task,
        mode=WorkMode.testplan_design,
        proposer_sm=proposer_sm,
        label="Running QA test-plan workflow",
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
    work_item_ref: dict[str, int | None] = {"id": None}
    cancel_targets: list[object] = []
    cancel_event = threading.Event()

    def _worker():
        return _run_workflow(
            store=store,
            proj=proj,
            workdir=workdir,
            settings=settings,
            requirement=requirement,
            should_cancel=cancel_event.is_set,
            on_work_item_created=lambda wid: work_item_ref.__setitem__("id", wid),
            mode=mode,
            proposer_sm=proposer_sm,
        )

    _run_with_task_monitor(
        console=console,
        store=store,
        proj=proj,
        label=label or (
            "Running full teamwork workflow"
            if mode == WorkMode.full
            else f"Running {mode.value} teamwork workflow"
        ),
        worker_fn=_worker,
        work_item_ref=work_item_ref,
        cancel_targets=cancel_targets,
        cancel_event=cancel_event,
    )


def _run_full_workflow(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
    workdir: str | None,
    settings: MySwatSettings,
    requirement: str,
    prompt_session: PromptSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_work_item_created: Callable[[int], None] | None = None,
    register_cancel_target: Callable[[object], None] | None = None,
) -> None:
    return _run_workflow(
        store=store,
        proj=proj,
        workdir=workdir,
        settings=settings,
        requirement=requirement,
        should_cancel=should_cancel,
        on_work_item_created=on_work_item_created,
        mode=WorkMode.full,
        proposer_sm=proposer_sm,
    )


def _run_full_workflow_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager | None,
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
