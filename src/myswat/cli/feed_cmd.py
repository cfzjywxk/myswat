"""myswat feed — ingest documents into project knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.memory.store import MemoryStore

console = Console()


def _get_ingester(store: MemoryStore, proj: dict, settings: MySwatSettings, no_ai: bool):
    """Create a DocumentIngester with optional AI runner."""
    from myswat.agents.codex_runner import CodexRunner
    from myswat.agents.kimi_runner import KimiRunner
    from myswat.memory.ingester import DocumentIngester

    runner = None
    if not no_ai:
        agents = store.list_agents(proj["id"])
        for a in agents:
            if a["cli_backend"] == settings.compaction.compaction_backend:
                extra_flags = json.loads(a["cli_extra_args"]) if a.get("cli_extra_args") else []
                if a["cli_backend"] == "codex":
                    runner = CodexRunner(cli_path=a["cli_path"], model=a["model_name"], extra_flags=extra_flags)
                elif a["cli_backend"] == "kimi":
                    runner = KimiRunner(cli_path=a["cli_path"], model=a["model_name"], extra_flags=extra_flags)
                break

    return DocumentIngester(store=store, runner=runner)


def run_feed(path: str, project: str, glob_pattern: str, no_ai: bool) -> None:
    """Feed documents into the project knowledge base."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    target = Path(path).resolve()

    if target.is_file():
        ingester = _get_ingester(store, proj, settings, no_ai)
        console.print(f"[bold]Ingesting {target.name}...[/bold]")
        ids = ingester.ingest_file(file_path=str(target), project_id=proj["id"])
        console.print(f"[green]Done.[/green] Created {len(ids)} knowledge entries from {target.name}")

    elif target.is_dir():
        files = sorted(target.glob(glob_pattern))
        if not files:
            console.print(f"[dim]No files match pattern '{glob_pattern}' in {target}[/dim]")
            return

        ingester = _get_ingester(store, proj, settings, no_ai)
        total_ids = 0
        console.print(f"[bold]Ingesting {len(files)} files from {target}...[/bold]")

        for f in files:
            try:
                ids = ingester.ingest_file(file_path=str(f), project_id=proj["id"])
                total_ids += len(ids)
                console.print(f"  {f.name}: {len(ids)} entries")
            except Exception as e:
                console.print(f"  [red]{f.name}: {e}[/red]")

        console.print(f"\n[green]Done.[/green] Total: {total_ids} knowledge entries from {len(files)} files")

    else:
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)
