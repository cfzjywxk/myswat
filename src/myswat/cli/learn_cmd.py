"""myswat learn — architect agent learns a project's operational knowledge.

Deterministically discovers indicator files (Makefile, Cargo.toml, etc.),
reads CLAUDE.md/AGENTS.md for project conventions, and sends everything to
the architect agent for structured analysis. The resulting knowledge is
persisted as ``project_ops`` entries in TiDB so dev/QA agents always know
how to build, test, and work with the project.

Re-running ``myswat learn`` replaces previous project_ops knowledge.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from myswat.agents.factory import make_runner_from_row
from myswat.cli.progress import _SPINNER_FRAMES, _coerce_live_lines, _fmt_duration
from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import run_migrations
from myswat.memory.store import MemoryStore

console = Console()

# ── Indicator files to scan (category -> glob patterns) ──

INDICATOR_GLOBS: dict[str, list[str]] = {
    "build": [
        "Makefile",
        "CMakeLists.txt",
        "Cargo.toml",
        "go.mod",
        "package.json",
        "pyproject.toml",
        "build.gradle",
        "build.gradle.kts",
        "pom.xml",
        "Justfile",
        "meson.build",
    ],
    "test": [
        "pytest.ini",
        "setup.cfg",
        "jest.config.*",
        "vitest.config.*",
        ".mocharc.*",
        "phpunit.xml",
    ],
    "ci": [
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
        ".gitlab-ci.yml",
        "Jenkinsfile",
        ".circleci/config.yml",
    ],
    "style": [
        ".rustfmt.toml",
        "rustfmt.toml",
        ".eslintrc*",
        ".prettierrc*",
        "clippy.toml",
        ".editorconfig",
        "pyproject.toml",  # may contain [tool.ruff] etc.
        "biome.json",
        ".clang-format",
    ],
    "docs": [
        "README.md",
        "README.rst",
        "CONTRIBUTING.md",
        "DESIGN.md",
        "ARCHITECTURE.md",
        "DESIGN_REVIEW.md",
        "HACKING.md",
        "doc/*.md",
        "docs/*.md",
    ],
    "git": [
        ".gitignore",
        ".github/PULL_REQUEST_TEMPLATE*",
    ],
}

# Files that contain AI-agent instructions (conventions, not workflow)
AGENT_INSTRUCTION_FILES = [
    "CLAUDE.md",
    "AGENTS.md",
    ".cursorrules",
    ".github/copilot-instructions.md",
]

# Max bytes to read per file (avoid blowing up the prompt)
MAX_FILE_BYTES = 12_000

# Required top-level keys in the architect's output
REQUIRED_KEYS = {"build", "test", "structure"}
_LEARN_WAIT_LIVE_LINES = 6


def _stage_start(console: Console, stage_num: int, total_stages: int, label: str) -> float:
    """Print a stage header and return its start timestamp."""
    console.print(f"\n[bold cyan]Stage {stage_num}/{total_stages}:[/bold cyan] {label}")
    return time.monotonic()


def _stage_done(
    console: Console,
    stage_num: int,
    total_stages: int,
    started_at: float,
    detail: str | None = None,
) -> None:
    """Print a stage completion line with elapsed time."""
    message = (
        f"[green]Stage {stage_num}/{total_stages} complete.[/green] "
        f"[dim]({_fmt_duration(time.monotonic() - started_at)})[/dim]"
    )
    if detail:
        message += f" [dim]{detail}[/dim]"
    console.print(message)


def _build_wait_display(frame_idx: int, elapsed: float, live_lines: list[str]) -> Panel:
    """Build the live wait display shown while the architect agent is working."""
    frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
    whole_seconds = int(elapsed)
    seconds_label = "second" if whole_seconds == 1 else "seconds"
    text = Text()
    text.append(
        f"{frame} Waiting for AI agent... {whole_seconds} {seconds_label}\n",
        style="bold yellow",
    )

    if live_lines:
        tail = live_lines[-_LEARN_WAIT_LIVE_LINES:]
        if len(live_lines) > _LEARN_WAIT_LIVE_LINES:
            skipped = len(live_lines) - _LEARN_WAIT_LIVE_LINES
            text.append(f"{skipped} earlier update(s) hidden\n", style="dim")
        text.append("Latest agent updates:\n", style="bold")
        for line in tail:
            text.append(f"  {line}\n", style="dim")
    else:
        text.append(
            "Architect is analyzing discovered files and extracting project knowledge.\n",
            style="dim",
        )

    return Panel.fit(text, title="myswat learn", border_style="yellow")


def _invoke_with_wait_display(console: Console, runner, prompt: str):
    """Invoke the architect runner while showing a live wait display."""
    result = [None]
    error = [None]
    started_at = time.monotonic()

    def _run():
        try:
            result[0] = runner.invoke(prompt)
        except Exception as exc:
            error[0] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    frame_idx = 0
    with Live(console=console, refresh_per_second=6, transient=True) as live:
        while worker.is_alive():
            elapsed = time.monotonic() - started_at
            live_lines = _coerce_live_lines(getattr(runner, "live_output", []))
            live.update(_build_wait_display(frame_idx, elapsed, live_lines))
            frame_idx += 1
            worker.join(timeout=0.25)

    if error[0] is not None:
        raise error[0]

    return result[0], time.monotonic() - started_at


def _discover_files(repo_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Scan *repo_path* for indicator files.

    Returns ``{category: [(relative_path, content), ...]}``.
    """
    found: dict[str, list[tuple[str, str]]] = {}

    seen: set[Path] = set()

    for category, patterns in INDICATOR_GLOBS.items():
        entries: list[tuple[str, str]] = []
        for pattern in patterns:
            for match in sorted(repo_path.glob(pattern)):
                if not match.is_file() or match in seen:
                    continue
                seen.add(match)
                try:
                    text = match.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_BYTES]
                except OSError:
                    continue
                rel = str(match.relative_to(repo_path))
                entries.append((rel, text))
        if entries:
            found[category] = entries

    return found


def _read_agent_instructions(repo_path: Path) -> list[tuple[str, str]]:
    """Read CLAUDE.md, AGENTS.md, and similar files if they exist."""
    results: list[tuple[str, str]] = []
    for name in AGENT_INSTRUCTION_FILES:
        target = repo_path / name
        if target.is_file():
            try:
                text = target.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_BYTES]
                results.append((name, text))
            except OSError:
                continue
    return results


def _format_file_contents(discovered: dict[str, list[tuple[str, str]]]) -> str:
    """Format discovered files into a prompt section."""
    parts: list[str] = []
    for category, files in discovered.items():
        parts.append(f"### {category.upper()}")
        for rel_path, content in files:
            parts.append(f"#### {rel_path}\n```\n{content}\n```")
    return "\n\n".join(parts) if parts else "(no indicator files found)"


def _format_agent_instructions(instructions: list[tuple[str, str]]) -> str:
    """Format agent instruction files into a prompt section."""
    if not instructions:
        return "(no agent instruction files found)"
    parts: list[str] = []
    for name, content in instructions:
        parts.append(f"#### {name}\n```\n{content}\n```")
    return "\n\n".join(parts)

def _validate_learned(data: dict) -> list[str]:
    """Return a list of fatal validation errors (empty = OK)."""
    errors: list[str] = []

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    build = data.get("build", {})
    if not build.get("commands"):
        errors.append("build.commands is empty — cannot determine how to build this project")

    test = data.get("test", {})
    if not test.get("tiers"):
        errors.append("test.tiers is empty — cannot determine how to test this project")

    structure = data.get("structure", {})
    if not structure.get("entry_points"):
        errors.append("structure.entry_points is empty — cannot determine project entry points")

    return errors


def _store_learned_knowledge(
    store: MemoryStore,
    project_id: int,
    data: dict,
    agent_id: int | None = None,
) -> int:
    """Persist the learned knowledge as project_ops entries.

    Deletes existing project_ops entries first (idempotent re-learn).
    Returns the number of entries created.
    """
    store.delete_knowledge_by_category(project_id, "project_ops")

    count = 0

    # Map each top-level key to a knowledge entry
    knowledge_map = {
        "build": ("Build System", ["build", "compile"]),
        "test": ("Test System", ["test", "qa"]),
        "git": ("Git Workflow", ["git", "vcs"]),
        "structure": ("Project Structure", ["structure", "layout"]),
        "conventions": ("Code Conventions", ["style", "lint", "format"]),
        "security": ("Security Requirements", ["security"]),
        "ci": ("CI/CD Pipeline", ["ci", "cd", "pipeline"]),
        "invariants": ("Project Invariants", ["invariants", "correctness"]),
    }

    # Store project overview entry
    overview = {
        "project_type": data.get("project_type", "unknown"),
        "language": data.get("language", "unknown"),
    }
    store.store_knowledge(
        project_id=project_id,
        agent_id=None,
        category="project_ops",
        title="Project Overview",
        content=json.dumps(overview, indent=2),
        tags=["overview", "project_type"],
        relevance_score=1.0,
        confidence=1.0,
    )
    count += 1

    for key, (title, tags) in knowledge_map.items():
        value = data.get(key)
        if not value:
            continue

        # invariants is a list, everything else is a dict
        if isinstance(value, list):
            if not value:
                continue
            content = "\n".join(f"- {item}" for item in value)
        else:
            content = json.dumps(value, indent=2)

        store.store_knowledge(
            project_id=project_id,
            agent_id=None,
            category="project_ops",
            title=title,
            content=content,
            tags=tags,
            relevance_score=1.0,
            confidence=1.0,
        )
        count += 1

    # Re-insert team workflow knowledge (deleted above with all project_ops)
    from myswat.cli.init_cmd import TEAM_WORKFLOWS_KNOWLEDGE
    store.store_knowledge(
        project_id=project_id,
        agent_id=None,
        category="project_ops",
        title="Team Workflows",
        content=TEAM_WORKFLOWS_KNOWLEDGE,
        tags=["workflow", "delegation", "team"],
        relevance_score=1.0,
        confidence=1.0,
    )
    count += 1

    return count


MYSWAT_MD_HEADER = """\
<!-- Auto-generated by: myswat learn -p {slug}
     Re-run to refresh. Do not edit manually — changes will be overwritten.
     This file is the local cache of project_ops knowledge stored in TiDB. -->

"""


def _write_myswat_md(repo_path: Path, data: dict, slug: str) -> Path:
    """Write learned knowledge to ``myswat.md`` in the repo root.

    This file acts as a fast local cache so the retriever can skip the TiDB
    round-trip for project_ops on every context build.  The canonical source
    of truth remains TiDB.
    """
    knowledge_map = {
        "build": "Build System",
        "test": "Test System",
        "git": "Git Workflow",
        "structure": "Project Structure",
        "conventions": "Code Conventions",
        "security": "Security Requirements",
        "ci": "CI/CD Pipeline",
        "invariants": "Project Invariants",
    }

    lines: list[str] = [MYSWAT_MD_HEADER.format(slug=slug)]
    lines.append("## Project Operations Knowledge\n")

    # Overview
    overview = {
        "project_type": data.get("project_type", "unknown"),
        "language": data.get("language", "unknown"),
    }
    lines.append(f"### Project Overview\n{json.dumps(overview, indent=2)}\n")

    for key, title in knowledge_map.items():
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, list):
            if not value:
                continue
            content = "\n".join(f"- {item}" for item in value)
        else:
            content = json.dumps(value, indent=2)
        lines.append(f"### {title}\n{content}\n")

    # Append team workflow knowledge (always present, written by init)
    from myswat.cli.init_cmd import TEAM_WORKFLOWS_KNOWLEDGE
    lines.append(f"\n{TEAM_WORKFLOWS_KNOWLEDGE}")

    md_path = repo_path / "myswat.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def run_learn(project_slug: str, workdir: str | None = None) -> None:
    """Learn a project's operational knowledge via the architect agent."""
    from myswat.workflow.prompts import ARCHITECT_LEARN_PROJECT

    stage_labels = (
        "Scanning project files",
        "Resolving architect agent",
        "Architect analyzing project files",
        "Parsing and validating learned knowledge",
        "Persisting learned knowledge",
        "Reporting learned project knowledge",
    )
    total_stages = len(stage_labels)
    settings = MySwatSettings()
    pool = TiDBPool(settings.tidb)
    applied = run_migrations(pool)
    if applied:
        console.print(f"[dim]Applied schema migrations: {applied}[/dim]")
    store = MemoryStore(pool, tidb_embedding_model=settings.embedding.tidb_model)

    proj = store.get_project_by_slug(project_slug)
    if not proj:
        console.print(f"[red]Project '{project_slug}' not found. Run 'myswat init' first.[/red]")
        raise typer.Exit(1)

    repo_path = Path(workdir or proj.get("repo_path") or ".").resolve()
    if not repo_path.is_dir():
        console.print(f"[red]Repository path not found: {repo_path}[/red]")
        raise typer.Exit(1)

    # ── Step 1: Discover indicator files ──
    stage_started = _stage_start(console, 1, total_stages, stage_labels[0])
    console.print(f"[bold]Scanning project at {repo_path}...[/bold]")
    discovered = _discover_files(repo_path)
    agent_instructions = _read_agent_instructions(repo_path)

    total_files = sum(len(files) for files in discovered.values())
    if total_files == 0 and not agent_instructions:
        console.print(
            "[red]No indicator files found (no Makefile, Cargo.toml, package.json, etc.).\n"
            "Cannot learn this project. Add a build file or README.md and try again.[/red]"
        )
        raise typer.Exit(1)

    # Show what was found
    table = Table(title="Discovered Files")
    table.add_column("Category")
    table.add_column("Files")
    for cat, files in discovered.items():
        table.add_row(cat, ", ".join(rel for rel, _ in files))
    if agent_instructions:
        table.add_row("agent instructions", ", ".join(name for name, _ in agent_instructions))
    console.print(table)
    _stage_done(
        console,
        1,
        total_stages,
        stage_started,
        f"Found {total_files} indicator file(s) and {len(agent_instructions)} instruction file(s).",
    )

    # ── Step 2: Get architect agent ──
    stage_started = _stage_start(console, 2, total_stages, stage_labels[1])
    arch_agent = store.get_agent(proj["id"], "architect")
    if not arch_agent:
        console.print("[red]Architect agent not found. Run 'myswat init' first.[/red]")
        raise typer.Exit(1)

    runner = make_runner_from_row(arch_agent, settings=settings)
    runner.workdir = str(repo_path)
    _stage_done(
        console,
        2,
        total_stages,
        stage_started,
        f"Using {arch_agent['display_name']} ({arch_agent['cli_backend']}/{arch_agent['model_name']}).",
    )

    # ── Step 3: Send to architect for analysis ──
    stage_started = _stage_start(console, 3, total_stages, stage_labels[2])
    file_contents = _format_file_contents(discovered)
    agent_instr_text = _format_agent_instructions(agent_instructions)

    prompt = ARCHITECT_LEARN_PROJECT.format(
        file_contents=file_contents,
        agent_instructions=agent_instr_text,
    )

    response, analysis_elapsed = _invoke_with_wait_display(console, runner, prompt)

    if not response.success:
        console.print(
            f"[red]Architect agent failed (exit={response.exit_code}).\n"
            f"Cannot learn this project. Check agent configuration.[/red]"
        )
        raise typer.Exit(1)
    _stage_done(
        console,
        3,
        total_stages,
        stage_started,
        f"Architect response received in {_fmt_duration(analysis_elapsed)}.",
    )

    # ── Step 4: Parse and validate ──
    stage_started = _stage_start(console, 4, total_stages, stage_labels[3])
    from myswat.workflow.engine import _extract_json_block

    data = _extract_json_block(response.content)
    if not isinstance(data, dict):
        console.print(
            "[red]Architect returned unparseable output.\n"
            "Expected a JSON object. Raw output:[/red]"
        )
        console.print(response.content[:2000])
        raise typer.Exit(1)

    errors = _validate_learned(data)
    if errors:
        console.print("[red]Validation failed:[/red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        console.print(
            "\n[yellow]The architect could not extract enough information.\n"
            "Make sure the project has a README, Makefile, or equivalent build files.[/yellow]"
        )
        raise typer.Exit(1)
    _stage_done(console, 4, total_stages, stage_started, "Structured knowledge passed validation.")

    # ── Step 5: Persist to TiDB ──
    stage_started = _stage_start(console, 5, total_stages, stage_labels[4])
    count = _store_learned_knowledge(store, proj["id"], data, agent_id=arch_agent["id"])

    # ── Step 5b: Write local cache (myswat.md) ──
    md_path = _write_myswat_md(repo_path, data, project_slug)
    console.print(f"[dim]Written local cache: {md_path}[/dim]")
    _stage_done(
        console,
        5,
        total_stages,
        stage_started,
        f"Stored {count} knowledge entr{'y' if count == 1 else 'ies'} and refreshed myswat.md.",
    )

    # ── Step 6: Report ──
    stage_started = _stage_start(console, 6, total_stages, stage_labels[5])
    console.print(f"\n[bold green]Learned {count} knowledge entries for '{project_slug}'.[/bold green]")

    report = Table(title="Learned Knowledge Summary")
    report.add_column("Area")
    report.add_column("Summary")

    report.add_row("Type", f"{data.get('language', '?')} / {data.get('project_type', '?')}")

    build = data.get("build", {})
    if build.get("commands"):
        report.add_row("Build", ", ".join(build["commands"][:3]))

    test = data.get("test", {})
    if test.get("tiers"):
        tier_names = [t.get("name", "?") for t in test["tiers"][:5]]
        report.add_row("Test tiers", ", ".join(tier_names))
    if test.get("gate_command"):
        report.add_row("Gate command", test["gate_command"])

    conventions = data.get("conventions", {})
    if conventions.get("rules"):
        report.add_row("Key rules", str(len(conventions["rules"])) + " rules")

    security = data.get("security", {})
    if security.get("requirements"):
        report.add_row("Security", str(len(security["requirements"])) + " requirements")

    invariants = data.get("invariants", [])
    if invariants:
        report.add_row("Invariants", str(len(invariants)) + " documented")

    console.print(report)
    console.print(
        f"\n[dim]Dev/QA agents will automatically receive this knowledge.\n"
        f"Re-run 'myswat learn -p {project_slug}' to refresh.[/dim]"
    )
    _stage_done(console, 6, total_stages, stage_started, "Learn command finished.")


def ensure_learned(
    store: MemoryStore,
    project_slug: str,
    project_id: int,
    repo_path: str | None,
) -> None:
    """Auto-learn the project if no project_ops knowledge exists.

    Called by ``chat``, ``run``, and ``work`` commands before agents start.
    Checks local ``myswat.md`` first (fast), then TiDB.  If neither has
    project_ops knowledge, runs the full learn phase automatically.
    """
    # Fast path: local cache file exists
    if repo_path:
        md_file = Path(repo_path) / "myswat.md"
        if md_file.is_file():
            return

    # Check TiDB — look for learned content specifically (not just the
    # "Team Workflows" entry seeded by init, which doesn't mean the project
    # has been scanned).
    ops = store.list_knowledge(project_id, category="project_ops", limit=50)
    if any(e.get("title") != "Team Workflows" for e in ops):
        return

    # No knowledge found — auto-learn
    console.print(
        "\n[bold yellow]No project operations knowledge found. "
        "Running auto-learn...[/bold yellow]\n"
    )
    run_learn(project_slug, workdir=repo_path)
