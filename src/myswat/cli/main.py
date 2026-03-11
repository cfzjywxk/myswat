"""MySwat CLI — main entry point."""

import json

import typer

from myswat.cli.progress import _describe_process_event
from myswat.cli.memory_cmd import memory_app
from myswat.workflow.engine import WorkMode

app = typer.Typer(
    name="myswat",
    help="Multi-AI agent co-working system for code development.",
    no_args_is_help=True,
)

app.add_typer(memory_app, name="memory", help="Search and manage project knowledge")


@app.command()
def chat(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    role: str = typer.Option("architect", "--role", help="Initial agent role"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
):
    """Interactive chat session with an agent. Switch roles, trigger reviews, all from the REPL."""
    from myswat.cli.chat_cmd import run_chat
    run_chat(project, role=role, workdir=workdir)


@app.command()
def run(
    task: str = typer.Argument(None, help="Task description (omit for interactive mode)"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    single: bool = typer.Option(False, "--single", help="Single-agent mode (no review loop)"),
    role: str = typer.Option("architect", "--role", help="Agent role to use"),
    reviewer: str = typer.Option("qa_main", "--reviewer", help="Reviewer role (for review loop)"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
):
    """Run a task with AI agents. Omit task to enter interactive mode."""
    if task is None:
        from myswat.cli.chat_cmd import run_chat
        run_chat(project, role=role, workdir=workdir)
    elif single:
        from myswat.cli.run_cmd import run_single
        run_single(project, task, role=role, workdir=workdir)
    else:
        from myswat.cli.run_cmd import run_with_review
        run_with_review(project, task, developer_role=role, reviewer_role=reviewer, workdir=workdir)


def _resolve_work_mode(*, design: bool, development: bool, test: bool) -> WorkMode:
    selected = []
    if design:
        selected.append(WorkMode.design)
    if development:
        selected.append(WorkMode.development)
    if test:
        selected.append(WorkMode.test)
    if len(selected) > 1:
        raise typer.BadParameter(
            "Choose only one of --design/--plan, --development/--dev, or --test/--ga-test."
        )
    return selected[0] if selected else WorkMode.full


@app.command()
def work(
    requirement: str = typer.Argument(..., help="Requirement description"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
    background: bool = typer.Option(
        False,
        "--background",
        help="Detach the workflow and keep it running after this terminal exits.",
    ),
    design_mode: bool = typer.Option(False, "--design", "--plan", help="Run design + planning only."),
    development_mode: bool = typer.Option(False, "--development", "--dev", help="Run development only."),
    test_mode: bool = typer.Option(False, "--test", "--ga-test", help="Run GA testing only."),
):
    """Run the teamwork workflow in full or selected foreground mode."""
    from myswat.cli.work_cmd import run_work

    mode = _resolve_work_mode(
        design=design_mode,
        development=development_mode,
        test=test_mode,
    )
    if background and mode == WorkMode.design:
        raise typer.BadParameter("Design mode cannot be combined with --background.")
    run_work(project, requirement, workdir=workdir, background=background, mode=mode)


@app.command(name="work-background-worker", hidden=True)
def work_background_worker(
    requirement: str = typer.Argument(..., help="Requirement description"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    work_item_id: int = typer.Option(..., "--work-item-id", help="Existing work item ID"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
    mode: str = typer.Option(WorkMode.full.value, "--mode", help="Workflow mode for detached worker."),
):
    """Internal detached worker entry point for `myswat work --background`."""
    from myswat.cli.work_cmd import run_background_work_item

    run_background_work_item(
        project,
        requirement,
        work_item_id=work_item_id,
        workdir=workdir,
        mode=WorkMode(mode),
    )


@app.command()
def stop(
    work_item_id: int = typer.Argument(..., help="Work item ID"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Request cancellation of a background workflow."""
    from myswat.cli.work_cmd import stop_work_item

    stop_work_item(project, work_item_id)


@app.command()
def feed(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    glob_pattern: str = typer.Option("**/*.md", "--glob", "-g", help="Glob pattern (for directories)"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip AI distillation, store raw chunks"),
):
    """Feed documents into the project knowledge base."""
    from myswat.cli.feed_cmd import run_feed
    run_feed(path, project, glob_pattern, no_ai)


@app.command()
def learn(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    workdir: str = typer.Option(None, "--workdir", "-w", help="Working directory override"),
):
    """Learn a project's build system, test tiers, conventions, and invariants."""
    from myswat.cli.learn_cmd import run_learn
    run_learn(project, workdir=workdir)


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    repo_path: str = typer.Option(None, "--repo", "-r", help="Path to git repo"),
    description: str = typer.Option(None, "--desc", "-d", help="Project description"),
):
    """Initialize a new MySwat project with TiDB schema and default agents."""
    from myswat.cli.init_cmd import run_init
    run_init(name, repo_path, description)


def _parse_verdict_payload(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_teamwork_flow_entries(pool, item: dict) -> list[dict]:
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
    if isinstance(task_state, dict):
        process_log = task_state.get("process_log")
        if isinstance(process_log, list) and process_log:
            return [event for event in process_log if isinstance(event, dict)]

    rows = pool.fetch_all(
        "SELECT rc.iteration, rc.verdict, rc.verdict_json, "
        "a1.role AS proposer_role, a2.role AS reviewer_role, "
        "art.title AS artifact_title, art.artifact_type, art.content AS artifact_content "
        "FROM review_cycles rc "
        "JOIN agents a1 ON rc.proposer_agent_id = a1.id "
        "JOIN agents a2 ON rc.reviewer_agent_id = a2.id "
        "LEFT JOIN artifacts art ON rc.artifact_id = art.id "
        "WHERE rc.work_item_id = %s "
        "ORDER BY rc.created_at, rc.id",
        (item["id"],),
    )

    if not rows:
        return []

    events: list[dict] = []
    description = item.get("description")
    if description:
        events.append({
            "type": "task_request",
            "from_role": "request",
            "to_role": rows[0].get("proposer_role") or "developer",
            "title": "Task request",
            "summary": description,
        })

    for row in rows:
        artifact_summary = row.get("artifact_content") or row.get("artifact_title") or row.get("artifact_type") or ""
        events.append({
            "type": "review_request",
            "from_role": row.get("proposer_role"),
            "to_role": row.get("reviewer_role"),
            "title": row.get("artifact_title") or row.get("artifact_type") or f"Iteration {row.get('iteration')}",
            "summary": artifact_summary,
        })

        verdict_payload = _parse_verdict_payload(row.get("verdict_json"))
        verdict_summary = verdict_payload.get("summary") or row.get("verdict") or ""
        issues = verdict_payload.get("issues")
        if isinstance(issues, list) and issues:
            issue_text = "; ".join(str(issue) for issue in issues[:6])
            verdict_summary = f"{verdict_summary} Issues: {issue_text}" if verdict_summary else f"Issues: {issue_text}"
        events.append({
            "type": "review_response",
            "from_role": row.get("reviewer_role"),
            "to_role": row.get("proposer_role"),
            "title": f"Verdict: {row.get('verdict', '?')}",
            "summary": verdict_summary or row.get("verdict") or "",
        })

    return events


def _print_message_flow(tree, flow_entries: list[dict]) -> None:
    flow_branch = tree.add("[bold]Message Flow[/bold]")
    last_node = None
    for entry in flow_entries[-20:]:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "reaction" and entry.get("from_role") == "myswat" and last_node is not None:
            summary = str(entry.get("summary") or "").strip()
            if summary:
                last_node.add(f"MySwat next: {summary}")
            continue
        from_role = entry.get("from_role") or "event"
        to_role = entry.get("to_role")
        title = entry.get("title") or entry.get("type") or "Message"
        header = f"{from_role} -> {to_role}: {title}" if to_role else f"{from_role}: {title}"
        node = flow_branch.add(header)
        last_node = node

        summary = str(entry.get("summary") or "").strip()
        if summary:
            node.add(summary)


def _print_teamwork_details(pool, item, console, details: bool = False) -> None:
    """Print collaboration details for a teamwork work item."""
    from rich.table import Table
    from rich.tree import Tree

    item_id = item["id"]
    title = item["title"] if details else item["title"][:60]

    tree = Tree(
        f"[bold]Work Item #{item_id}:[/bold] {title} "
        f"[{'green' if item['status'] == 'completed' else 'yellow'}]"
        f"[{item['status']}][/{'green' if item['status'] == 'completed' else 'yellow'}]"
    )

    # ── Review cycles (chronological) ──
    all_cycles = pool.fetch_all(
        "SELECT rc.iteration, rc.verdict, rc.created_at, "
        "a1.role AS proposer_role, a1.display_name AS proposer_name, "
        "a2.role AS reviewer_role, a2.display_name AS reviewer_name "
        "FROM review_cycles rc "
        "JOIN agents a1 ON rc.proposer_agent_id = a1.id "
        "JOIN agents a2 ON rc.reviewer_agent_id = a2.id "
        "WHERE rc.work_item_id = %s "
        "ORDER BY rc.created_at",
        (item_id,),
    )

    if all_cycles:
        # Group consecutive cycles by (proposer_role, reviewer_role) into review rounds
        rounds: list[dict] = []
        current_round = None
        for cyc in all_cycles:
            key = (cyc["proposer_role"], cyc["reviewer_role"])
            if current_round is None or current_round["key"] != key:
                current_round = {
                    "key": key,
                    "proposer": cyc["proposer_name"],
                    "reviewer": cyc["reviewer_name"],
                    "proposer_role": cyc["proposer_role"],
                    "reviewer_role": cyc["reviewer_role"],
                    "verdicts": [],
                    "iterations": 0,
                }
                rounds.append(current_round)
            current_round["verdicts"].append(cyc["verdict"])
            current_round["iterations"] += 1

        # Infer stage names from the round patterns
        review_branch = tree.add("[bold]Review Rounds[/bold]")
        stage_labels = _infer_stage_labels(rounds)
        for i, (rd, label) in enumerate(zip(rounds, stage_labels)):
            final_verdict = rd["verdicts"][-1] if rd["verdicts"] else "?"
            verdict_style = "green" if final_verdict == "lgtm" else "red" if final_verdict == "changes_requested" else "dim"
            iters = rd["iterations"]
            iter_note = f"{iters} iter{'s' if iters > 1 else ''}"

            review_branch.add(
                f"[dim]{label}:[/dim] "
                f"{rd['proposer']} [dim]proposed →[/dim] {rd['reviewer']} "
                f"[dim]reviewed:[/dim] [{verdict_style}]{final_verdict}[/{verdict_style}] "
                f"[dim]({iter_note})[/dim]"
            )

    if details:
        flow_entries = _build_teamwork_flow_entries(pool, item)
        if flow_entries:
            _print_message_flow(tree, flow_entries)

    # ── Artifacts ──
    artifacts = pool.fetch_all(
        "SELECT a.artifact_type, a.title, a.iteration, a.created_at, "
        "ag.role AS agent_role, ag.display_name AS agent_name "
        "FROM artifacts a "
        "JOIN agents ag ON a.agent_id = ag.id "
        "WHERE a.work_item_id = %s "
        "ORDER BY a.created_at",
        (item_id,),
    )
    if artifacts:
        art_branch = tree.add("[bold]Artifacts[/bold]")
        for art in artifacts:
            art_branch.add(
                f"[dim]{art['artifact_type']}[/dim] "
                f"\"{art['title'] or '-'}\" "
                f"[dim]by[/dim] {art['agent_name']} "
                f"[dim](v{art['iteration']})[/dim]"
            )

    # ── Agent contributions (sessions + turns) ──
    agent_effort = pool.fetch_all(
        "SELECT a.role, a.display_name, "
        "COUNT(DISTINCT s.id) AS session_count, "
        "COUNT(st.id) AS turn_count, "
        "(SELECT COALESCE(SUM(s2.token_count_est), 0) "
        " FROM sessions s2 WHERE s2.agent_id = a.id AND s2.work_item_id = %s"
        ") AS total_tokens "
        "FROM sessions s "
        "JOIN agents a ON s.agent_id = a.id "
        "LEFT JOIN session_turns st ON st.session_id = s.id "
        "WHERE s.work_item_id = %s "
        "GROUP BY a.id, a.role, a.display_name "
        "ORDER BY turn_count DESC",
        (item_id, item_id),
    )
    if agent_effort:
        effort_branch = tree.add("[bold]Agent Contributions[/bold]")
        for ae in agent_effort:
            tokens = ae["total_tokens"]
            if tokens > 1000:
                token_str = f"{tokens // 1000}k tokens"
            else:
                token_str = f"{tokens} tokens"
            effort_branch.add(
                f"{ae['display_name']} [dim]({ae['role']})[/dim] — "
                f"{ae['turn_count']} turns, "
                f"{ae['session_count']} session{'s' if ae['session_count'] > 1 else ''}, "
                f"~{token_str}"
            )

    console.print(tree)


def _work_mode_value(item: dict) -> str | None:
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    work_mode = metadata.get("work_mode") if isinstance(metadata, dict) else None
    if isinstance(work_mode, str) and work_mode:
        return work_mode
    return None


def _display_mode(item: dict, fallback: str) -> str:
    work_mode = _work_mode_value(item)
    return work_mode if work_mode else fallback


def _print_task_state(console, item: dict) -> None:
    metadata = item.get("metadata_json") if isinstance(item, dict) else None
    background = metadata.get("background") if isinstance(metadata, dict) else {}
    task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
    work_mode = _work_mode_value(item)
    if work_mode:
        console.print(f"[bold]Workflow mode:[/bold] {work_mode}")
    if isinstance(background, dict) and background:
        console.print("[bold]Execution:[/bold]")
        if background.get("mode"):
            console.print(f"  Mode: {background['mode']}")
        if background.get("state"):
            console.print(f"  State: {background['state']}")
        if background.get("pid"):
            console.print(f"  PID: {background['pid']}")
        if background.get("log_path"):
            console.print(f"  Log: {background['log_path']}")
        if background.get("requested_at"):
            console.print(f"  Requested: {background['requested_at']}")
        if background.get("started_at"):
            console.print(f"  Started: {background['started_at']}")
        if background.get("finished_at"):
            console.print(f"  Finished: {background['finished_at']}")

    if not isinstance(task_state, dict) or not task_state:
        if isinstance(background, dict) and background:
            console.print()
        return

    if task_state.get("current_stage"):
        console.print(f"[bold]Stage:[/bold] {task_state['current_stage']}")
    if task_state.get("latest_summary"):
        console.print(f"[bold]Latest summary:[/bold] {str(task_state['latest_summary'])[:600]}")
    if task_state.get("next_todos"):
        console.print("[bold]Next TODOs:[/bold]")
        for todo in task_state["next_todos"][:10]:
            console.print(f"  - {todo}")
    if task_state.get("open_issues"):
        console.print("[bold]Open issues:[/bold]")
        for issue in task_state["open_issues"][:10]:
            console.print(f"  - {issue}")
    process_log = task_state.get("process_log")
    if isinstance(process_log, list) and process_log:
        console.print("[bold]Process Log:[/bold]")
        for event in process_log[-20:]:
            if not isinstance(event, dict):
                continue
            timestamp = event.get("at")
            prefix = f"[{timestamp}] " if timestamp else ""
            console.print(f"  - {prefix}{_describe_process_event(event, 160)}")
    console.print()


def _infer_stage_labels(rounds: list[dict]) -> list[str]:
    """Best-effort stage labels from review round patterns.

    The workflow runs review loops in a fixed order:
      1. Design review (dev → QA)
      2. Plan review (dev → QA)
      3. Code review per phase (dev → QA) — can repeat
      4. Test plan review (QA → dev)

    We use the proposer/reviewer direction + sequence to infer labels.
    """
    labels = []
    dev_to_qa_count = 0
    qa_to_dev_count = 0

    for rd in rounds:
        p_role = rd["proposer_role"]
        r_role = rd["reviewer_role"]

        if p_role == "developer" and r_role.startswith("qa"):
            dev_to_qa_count += 1
            if dev_to_qa_count == 1:
                labels.append("Design Review")
            elif dev_to_qa_count == 2:
                labels.append("Plan Review")
            else:
                labels.append(f"Code Review (phase {dev_to_qa_count - 2})")
        elif p_role.startswith("qa") and r_role == "developer":
            qa_to_dev_count += 1
            labels.append("Test Plan Review")
        else:
            labels.append(f"Review ({p_role} → {r_role})")

    return labels


@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    details: bool = typer.Option(False, "--details", help="Show detailed work-item flow and review data"),
):
    """Show project status: active work items, sessions, agents."""
    from rich.console import Console
    from rich.table import Table

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.memory.store import MemoryStore

    console = Console()
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Project:[/bold] {proj['name']} ({proj['slug']})")
    if proj.get("repo_path"):
        console.print(f"[bold]Repo:[/bold] {proj['repo_path']}")

    # Agents
    agents = store.list_agents(proj["id"])
    if agents:
        table = Table(title="Agents")
        table.add_column("Role")
        table.add_column("Model")
        table.add_column("Backend")
        for a in agents:
            table.add_row(a["role"], a["model_name"], a["cli_backend"])
        console.print(table)

    # Work items — with work mode (solo vs teamwork)
    items = store.list_work_items(proj["id"])
    if items:
        table = Table(title="Work Items")
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Stage")
        table.add_column("Mode")
        table.add_column("Type")
        table.add_column("Agents")
        table.add_column("Title", max_width=50)
        teamwork_items = []  # collect for detailed display below
        for item in items[:20]:
            # Determine work mode from review_cycles
            cycles = pool.fetch_all(
                "SELECT DISTINCT rc.proposer_agent_id, rc.reviewer_agent_id, "
                "a1.role AS proposer_role, a2.role AS reviewer_role "
                "FROM review_cycles rc "
                "JOIN agents a1 ON rc.proposer_agent_id = a1.id "
                "JOIN agents a2 ON rc.reviewer_agent_id = a2.id "
                "WHERE rc.work_item_id = %s",
                (item["id"],),
            )
            if cycles:
                inferred_mode = "[cyan]team[/cyan]"
                agent_names = set()
                for c in cycles:
                    agent_names.add(c["proposer_role"])
                    agent_names.add(c["reviewer_role"])
                agents_str = ", ".join(sorted(agent_names))
                teamwork_items.append(item)
            else:
                # Solo — find the assigned agent or session agent
                assigned = item.get("assigned_agent_id")
                if assigned:
                    agent_row = pool.fetch_one(
                        "SELECT role FROM agents WHERE id = %s", (assigned,),
                    )
                    agents_str = agent_row["role"] if agent_row else "?"
                else:
                    # Check sessions linked to this work item
                    sess_agents = pool.fetch_all(
                        "SELECT DISTINCT a.role FROM sessions s "
                        "JOIN agents a ON s.agent_id = a.id "
                        "WHERE s.work_item_id = %s",
                        (item["id"],),
                    )
                    agents_str = ", ".join(s["role"] for s in sess_agents) if sess_agents else "-"
                inferred_mode = "[dim]solo[/dim]"
            mode = _display_mode(item, inferred_mode)
            metadata = item.get("metadata_json") if isinstance(item, dict) else None
            task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
            stage = task_state.get("current_stage", "-") if isinstance(task_state, dict) else "-"
            table.add_row(
                str(item["id"]), item["status"], stage, mode,
                item["item_type"], agents_str, item["title"][:50],
            )
        console.print(table)

        if details:
            for item in items[:10]:
                console.print(f"\n[bold]Work Item #{item['id']} State[/bold] — {item['title'][:80]}")
                _print_task_state(console, item)

            # ── Teamwork details for recent work items ──
            for item in teamwork_items[:10]:
                _print_teamwork_details(pool, item, console, details=True)
    else:
        console.print("\n[dim]No work items yet.[/dim]")

    # Active sessions
    from myswat.models.session import Session
    active_sessions = pool.fetch_all(
        "SELECT s.*, a.role, a.display_name FROM sessions s "
        "JOIN agents a ON s.agent_id = a.id "
        "WHERE a.project_id = %s AND s.status = 'active' "
        "ORDER BY s.created_at DESC LIMIT 10",
        (proj["id"],),
    )
    if active_sessions:
        table = Table(title="Active Sessions")
        table.add_column("UUID", style="cyan", max_width=12)
        table.add_column("Agent")
        table.add_column("Turns", justify="right")
        table.add_column("Tokens (est)", justify="right")
        table.add_column("Progress", max_width=60)
        for sess in active_sessions:
            turn_count = store.count_session_turns(sess["id"])
            # Check if agent is currently thinking (last turn is user, no agent reply yet)
            last_turn = pool.fetch_one(
                "SELECT role FROM session_turns WHERE session_id = %s "
                "ORDER BY turn_index DESC LIMIT 1",
                (sess["id"],),
            )
            is_thinking = last_turn and last_turn["role"] == "user"
            status_icon = " [bold yellow]⏳ agent thinking[/bold yellow]" if is_thinking else ""
            progress = (sess.get("purpose") or "")[:60]

            table.add_row(
                sess["session_uuid"][:12],
                sess["display_name"],
                str(turn_count),
                str(sess.get("token_count_est", 0)),
                progress,
            )
            if is_thinking:
                # Show what the agent is working on
                pending_turn = pool.fetch_one(
                    "SELECT content FROM session_turns WHERE session_id = %s "
                    "AND role = 'user' ORDER BY turn_index DESC LIMIT 1",
                    (sess["id"],),
                )
                if pending_turn:
                    pending_text = pending_turn["content"][:200].replace("\n", " ")
                    if len(pending_turn["content"]) > 200:
                        pending_text += "..."
        console.print(table)

        # Show recent turns + thinking status from active sessions
        for sess in active_sessions[:3]:
            recent_turns = pool.fetch_all(
                "SELECT role, content, metadata_json, created_at FROM session_turns "
                "WHERE session_id = %s ORDER BY turn_index DESC LIMIT 6",
                (sess["id"],),
            )
            if not recent_turns:
                continue

            last_role = recent_turns[0]["role"]
            thinking = last_role == "user"
            recent_turns.reverse()  # chronological order

            header = (
                f"\n[bold]Recent activity[/bold] — "
                f"{sess['display_name']} ({sess['session_uuid'][:8]})"
            )
            if thinking:
                header += " [bold yellow]⏳ agent thinking...[/bold yellow]"
            console.print(header)

            for t in recent_turns:
                role_label = "[green]user[/green]" if t["role"] == "user" else "[cyan]agent[/cyan]"
                content = t["content"][:150].replace("\n", " ")
                if len(t["content"]) > 150:
                    content += "..."
                # Show elapsed time for agent turns
                time_tag = ""
                if t["role"] == "assistant" and t.get("metadata_json"):
                    try:
                        import json as _json
                        meta = t["metadata_json"]
                        if isinstance(meta, str):
                            meta = _json.loads(meta)
                        elapsed = meta.get("elapsed_seconds")
                        if elapsed is not None:
                            s = int(elapsed)
                            if s < 60:
                                time_tag = f" [dim]({s}s)[/dim]"
                            else:
                                m, s = divmod(s, 60)
                                time_tag = f" [dim]({m}m{s:02d}s)[/dim]"
                    except Exception:
                        pass
                console.print(f"  {role_label}{time_tag}: {content}")

    console.print("\n[dim]Use `myswat task <id> -p "
                  f"{proj['slug']}` for one work item, or `myswat status -p {proj['slug']}` to refresh.[/dim]")

    # Knowledge stats
    knowledge_row = pool.fetch_one(
        "SELECT COUNT(*) AS cnt FROM knowledge WHERE project_id = %s",
        (proj["id"],),
    )
    knowledge_count = knowledge_row["cnt"] if knowledge_row else 0
    compacted_row = pool.fetch_one(
        "SELECT COUNT(*) AS cnt FROM sessions s "
        "JOIN agents a ON s.agent_id = a.id "
        "WHERE a.project_id = %s AND s.status = 'compacted'",
        (proj["id"],),
    )
    compacted_count = compacted_row["cnt"] if compacted_row else 0
    console.print(
        f"\n[dim]Knowledge: {knowledge_count} entries | "
        f"Compacted sessions: {compacted_count}[/dim]"
    )


@app.command()
def task(
    work_item_id: int = typer.Argument(..., help="Work item ID"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Show detailed status for one work item."""
    from rich.console import Console

    from myswat.config.settings import MySwatSettings
    from myswat.db.connection import TiDBPool
    from myswat.memory.store import MemoryStore

    console = Console()
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    item = store.get_work_item(work_item_id)
    if not item or item.get("project_id") != proj["id"]:
        console.print(f"[red]Work item {work_item_id} not found in project '{project}'.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Work Item #{item['id']}[/bold] — {item['title']}")
    console.print(f"[bold]Status:[/bold] {item['status']}")
    console.print(f"[bold]Type:[/bold] {item['item_type']}")
    if item.get("description"):
        console.print(f"[bold]Description:[/bold] {item['description'][:500]}")

    _print_task_state(console, item)
    _print_teamwork_details(pool, item, console, details=True)


if __name__ == "__main__":
    app()
