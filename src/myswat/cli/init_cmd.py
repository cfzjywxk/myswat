"""myswat init — initialize a new project."""

from __future__ import annotations

import re

import typer
from rich.console import Console

from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import run_migrations
from myswat.memory.store import MemoryStore

console = Console()


def _slugify(name: str) -> str:
    """Convert a project name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def run_init(name: str, repo_path: str | None, description: str | None) -> None:
    """Initialize a new MySwat project with TiDB schema and default agents."""
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)

    # Health check
    if not pool.health_check():
        console.print("[red]Cannot connect to TiDB. Check your config.[/red]")
        raise typer.Exit(1)

    # Run migrations
    console.print("[dim]Running schema migrations...[/dim]")
    applied = run_migrations(pool)
    if applied:
        console.print(f"[green]Applied migrations: {applied}[/green]")
    else:
        console.print("[dim]Schema up to date.[/dim]")

    # Create project
    slug = _slugify(name)
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    existing = store.get_project_by_slug(slug)
    if existing:
        console.print(f"[yellow]Project '{slug}' already exists (id={existing['id']}). Skipping creation.[/yellow]")
        project_id = existing["id"]
    else:
        project_id = store.create_project(
            slug=slug, name=name, description=description, repo_path=repo_path,
        )
        console.print(f"[green]Created project '{name}' (slug={slug}, id={project_id})[/green]")

    # Seed default agents
    _seed_default_agents(store, settings, project_id)

    console.print(f"\n[bold green]Project '{name}' is ready![/bold green]")
    console.print(f"  Use: myswat status -p {slug}")


ARCHITECT_SYSTEM_PROMPT = """\
You are the Architect / PM for this project. You handle two kinds of work:

## Self-handled (answer directly):
- Design discussions, architecture decisions, trade-off analysis
- Code review, explaining existing code, debugging guidance
- Project planning, task breakdown, priority decisions
- Quick questions, clarifications, documentation

## Delegate to Developer (requires implementation):
- Writing new features, modules, or substantial code changes
- Bug fixes that require code modification
- Refactoring, migrations, or infrastructure changes
- Any task where files need to be created or modified

When you decide a task needs delegation, end your response with a delegation block:

```delegate
TASK: <clear, actionable task description for the developer>
```

The system will automatically route this to the Developer + QA review loop.
If you handle it yourself, just respond normally without the delegate block.
"""


def _seed_default_agents(store: MemoryStore, settings: MySwatSettings, project_id: int) -> None:
    """Create the 4 default agent roles if they don't exist."""
    agent_defs = [
        {
            "role": "architect",
            "display_name": "Architect / PM",
            "cli_backend": "codex",
            "model_name": settings.agents.architect_model,
            "cli_path": settings.agents.codex_path,
            "cli_extra_args": settings.agents.codex_default_flags,
            "system_prompt": ARCHITECT_SYSTEM_PROMPT,
        },
        {
            "role": "developer",
            "display_name": "Developer",
            "cli_backend": "codex",
            "model_name": settings.agents.developer_model,
            "cli_path": settings.agents.codex_path,
            "cli_extra_args": settings.agents.codex_default_flags,
        },
        {
            "role": "qa_main",
            "display_name": "QA (Primary)",
            "cli_backend": "kimi",
            "model_name": settings.agents.qa_main_model,
            "cli_path": settings.agents.kimi_path,
            "cli_extra_args": settings.agents.kimi_default_flags,
        },
        {
            "role": "qa_vice",
            "display_name": "QA (Secondary)",
            "cli_backend": "kimi",
            "model_name": settings.agents.qa_vice_model,
            "cli_path": settings.agents.kimi_path,
            "cli_extra_args": settings.agents.kimi_default_flags,
        },
    ]

    for agent_def in agent_defs:
        existing = store.get_agent(project_id, agent_def["role"])
        if existing:
            console.print(f"  [dim]Agent '{agent_def['role']}' already exists.[/dim]")
            continue

        store.create_agent(project_id=project_id, **agent_def)
        console.print(f"  [green]Created agent: {agent_def['display_name']} ({agent_def['role']})[/green]")
