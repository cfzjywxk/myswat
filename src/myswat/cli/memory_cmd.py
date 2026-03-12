"""myswat memory — search and manage project knowledge."""

from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.table import Table

from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.memory.store import MemoryStore

memory_app = typer.Typer()
console = Console()


@memory_app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    no_vector: bool = typer.Option(False, "--no-vector", help="Skip vector search (keyword only)"),
):
    """Search project knowledge base (hybrid: keyword + semantic)."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    if no_vector:
        results = store.search_knowledge_fulltext_only(
            project_id=proj["id"], query=query, limit=limit,
        )
    else:
        results = store.search_knowledge(
            project_id=proj["id"], query=query,
            category=category, limit=limit,
        )

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"Knowledge Search: '{query}'")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Title")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")

    for r in results:
        table.add_row(
            str(r["id"]),
            r["category"],
            r["title"][:60],
            f"{r.get('search_score', 0):.3f}",
            f"{r.get('confidence', 0):.2f}",
        )
    console.print(table)


@memory_app.command("add")
def add(
    title: str = typer.Argument(..., help="Knowledge title"),
    content: str = typer.Argument(..., help="Knowledge content"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    category: str = typer.Option("decision", "--category", "-c", help="Category"),
    tags: str = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
):
    """Manually add a knowledge entry."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    kid = store.store_knowledge(
        project_id=proj["id"],
        category=category,
        title=title,
        content=content,
        tags=tag_list,
    )
    console.print(f"[green]Knowledge entry created (id={kid})[/green]")


@memory_app.command("list")
def list_knowledge(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
):
    """List knowledge entries for a project."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    # Use search with empty query to list by relevance
    results = store.search_knowledge(
        project_id=proj["id"], query="",
        category=category, limit=limit,
        use_vector=False, use_fulltext=False,
    )

    if not results:
        console.print("[dim]No knowledge entries yet.[/dim]")
        return

    table = Table(title="Knowledge Entries")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Title")
    table.add_column("Relevance", justify="right")
    table.add_column("Tags")

    for r in results:
        tags_str = ", ".join(r.get("tags") or []) if r.get("tags") else ""
        if isinstance(r.get("tags"), str):
            import json
            try:
                tags_str = ", ".join(json.loads(r["tags"]))
            except Exception:
                tags_str = r["tags"]
        table.add_row(
            str(r["id"]),
            r["category"],
            r["title"][:50],
            f"{r.get('relevance_score', 0):.2f}",
            tags_str[:30],
        )
    console.print(table)


@memory_app.command("compact")
def compact(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
):
    """Compact all completed sessions into knowledge entries."""
    from myswat.agents.factory import make_runner_from_row
    from myswat.memory.compactor import KnowledgeCompactor

    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    # Find a runner for compaction
    agents = store.list_agents(proj["id"])
    runner = None
    for a in agents:
        if a["cli_backend"] == settings.compaction.compaction_backend:
            runner = make_runner_from_row(a, settings=settings)
            break
    if runner is None and agents:
        runner = make_runner_from_row(agents[0], settings=settings)

    if runner is None:
        console.print("[red]No agents available for compaction.[/red]")
        raise typer.Exit(1)

    compactor = KnowledgeCompactor(
        store=store,
        runner=runner,
        threshold_turns=settings.compaction.threshold_turns,
        threshold_tokens=settings.compaction.threshold_tokens,
    )

    console.print(f"[bold]Compacting sessions for project '{proj['name']}'...[/bold]")
    result = compactor.compact_all_pending(project_id=proj["id"])

    console.print(
        f"[green]Done.[/green] "
        f"Compacted: {result['compacted']}, "
        f"Knowledge created: {result['knowledge_created']}, "
        f"Skipped: {result['skipped']}"
    )


@memory_app.command("purge")
def purge(
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete all compacted sessions and their turns to free TiDB storage.

    Knowledge entries are preserved — only raw session turns are deleted.
    Sessions that haven't been compacted yet are left untouched.
    """
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    # Count what would be purged
    compacted = pool.fetch_all(
        "SELECT s.id FROM sessions s "
        "JOIN agents a ON s.agent_id = a.id "
        "WHERE a.project_id = %s AND s.status = 'compacted'",
        (proj["id"],),
    )
    if not compacted:
        console.print("[dim]No compacted sessions to purge.[/dim]")
        return

    total_turns = 0
    for sess in compacted:
        row = pool.fetch_one(
            "SELECT COUNT(*) AS cnt FROM session_turns WHERE session_id = %s",
            (sess["id"],),
        )
        total_turns += row["cnt"] if row else 0

    console.print(
        f"Will delete [bold]{len(compacted)}[/bold] compacted sessions "
        f"and [bold]{total_turns}[/bold] turns."
    )
    console.print("[dim]Knowledge entries are preserved.[/dim]")

    if not yes:
        confirm = console.input("[bold]Proceed? [y/N] [/bold]").strip().lower()
        if confirm != "y":
            console.print("[dim]Cancelled.[/dim]")
            return

    result = store.purge_compacted_sessions(proj["id"])
    console.print(
        f"[green]Purged {result['sessions_deleted']} sessions, "
        f"{result['turns_deleted']} turns deleted.[/green]"
    )
