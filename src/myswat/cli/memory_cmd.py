"""myswat memory — search and manage project knowledge."""

from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.table import Table

from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.memory.search_engine import KnowledgeSearchEngine, SearchPlanBuilder
from myswat.memory.store import MemoryStore

memory_app = typer.Typer()
console = Console()


@memory_app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(..., "--project", "-p", help="Project slug"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    source_type: str = typer.Option(None, "--source-type", help="Filter by source type"),
    mode: str = typer.Option("auto", "--mode", help="Search mode: auto, exact, concept, relation"),
    profile: str = typer.Option("standard", "--profile", help="Search profile: quick, standard, precise"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    no_vector: bool = typer.Option(False, "--no-vector", help="Skip vector search (keyword only)"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    """Search project knowledge base (hybrid: keyword + semantic)."""
    if not isinstance(source_type, str):
        source_type = None
    if not isinstance(category, str):
        category = None
    if not isinstance(mode, str):
        mode = "auto"
    if not isinstance(profile, str):
        profile = "standard"
    if not isinstance(limit, int):
        limit = 10
    if not isinstance(no_vector, bool):
        no_vector = False
    if not isinstance(json_output, bool):
        json_output = False

    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    engine = KnowledgeSearchEngine(store)
    plan = SearchPlanBuilder.build(
        project_id=proj["id"],
        query=query,
        category=category,
        source_type=source_type,
        limit=limit,
        mode=mode,
        profile=profile,
    )
    if no_vector:
        plan.use_vector = False
    results = engine.search_with_explanations(plan)

    if not results:
        if json_output:
            console.print(json.dumps({
                "query": query,
                "mode": plan.mode,
                "profile": plan.profile,
                "results": [],
            }, indent=2))
        else:
            console.print("[dim]No results found.[/dim]")
        return

    if json_output:
        payload = {
            "query": plan.query,
            "mode": plan.mode,
            "profile": plan.profile,
            "filters": {
                "project": project,
                "category": category,
                "source_type": source_type,
                "limit": plan.limit,
                "vector_enabled": plan.use_vector and not no_vector,
            },
            "results": [],
        }
        for r in results:
            metadata = r.get("search_metadata_json")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = None
            if not isinstance(metadata, dict):
                metadata = {}
            snippet = " ".join(str(r.get("content") or "").split())
            if len(snippet) > 220:
                snippet = snippet[:220] + "... [truncated]"
            payload["results"].append({
                "knowledge_id": r["id"],
                "category": r["category"],
                "title": r["title"],
                "score": r.get("search_score", 0),
                "confidence": r.get("confidence", 0),
                "why": r.get("why", []),
                "snippet": snippet,
                "provenance": {
                    "source_type": r.get("source_type"),
                    "source_file": r.get("source_file"),
                    "tags": r.get("tags"),
                    **metadata,
                },
            })
        console.print(json.dumps(payload, indent=2, default=str))
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
