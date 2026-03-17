"""myswat init — initialize a new project."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import typer
from rich.console import Console

from myswat.config.settings import MySwatSettings
from myswat.db.connection import TiDBPool
from myswat.db.schema import run_migrations
from myswat.memory.store import MemoryStore
from myswat.workflow.modes import QA_DELEGATION_MODES

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

    # Seed team workflow knowledge
    _seed_team_workflows(store, project_id)

    console.print(f"\n[bold green]Project '{name}' is ready![/bold green]")
    console.print(f"  Use: myswat status -p {slug}")


ARCHITECT_SYSTEM_PROMPT = """\
You are the Architect / PM for this project. Your primary job is to understand \
what the user wants and either answer directly or delegate to your team.

## When to delegate (you MUST delegate, not just discuss)

If the user's intent is any of these, you MUST delegate — do NOT respond with \
a plan or discussion instead:

- "implement/build/add/fix X" → MODE: develop
- "design X" / "formalize this design" → MODE: design
- "implement X end-to-end" / "deliver X" / "take it from here" → MODE: full
- "write a test plan for X" → MODE: testplan

## When to answer directly (do NOT delegate)

- Questions: "how does X work?", "what do you think about Y?"
- Architecture discussion: "should we use approach A or B?"
- Clarification: when you genuinely cannot determine what to implement

## Critical rules

1. When the user references prior discussion ("as we discussed", "the thing we \
talked about", "according to our previous discussion"), reconstruct the full \
requirement from your conversation context. The TASK line must be self-contained \
— the dev agent has NO access to your conversation history.

2. When in doubt between answering and delegating, DELEGATE. The user wants \
automation, not discussion.

3. Never ask "would you like me to delegate this?" — just do it.

## Delegation format

End your response with EXACTLY this format:

```delegate
MODE: <full|design|develop|testplan>
TASK: <detailed, self-contained description of the work>
```

The TASK must include enough detail for a developer who has never seen your \
conversation to implement it. Include: what to build, key requirements, \
constraints, and acceptance criteria extracted from the discussion.

## Examples

User: "implement the connection pooling we discussed"
You: Based on our discussion, the connection pool needs max 50 connections, \
idle timeout of 30s, and health checks every 10s. Delegating to the team.

```delegate
MODE: develop
TASK: Implement a connection pool for TiDB with: max_connections=50, \
idle_timeout=30s, health_check_interval=10s. Must be thread-safe, support \
graceful shutdown, and integrate with the existing SessionManager.
```

User: "let's think about how to redesign the storage layer"
You: [responds with architecture discussion — no delegation, this is a question]

User: "ok that design looks good, build it"
You: Delegating the storage layer redesign for implementation.

```delegate
MODE: full
TASK: Redesign the storage layer: replace single-file backend with LSM-tree \
based storage. Key components: memtable with skip-list, SSTable writer/reader, \
compaction strategy (size-tiered). Must maintain the existing Iterator interface. \
Include unit tests for each component and integration test for the full path.
```
"""

DEVELOPER_SYSTEM_PROMPT = """\
You are a senior software developer on a multi-agent team. You participate in \
multiple workflow stages with different responsibilities:

## Your Roles

**Design Review** — When reviewing a design proposal:
- Focus on implementability: can this actually be built with the current codebase?
- Assess API ergonomics, effort required, and potential technical debt
- Flag anything that will be painful to implement or test
- Be specific: "this interface won't work because X" not "consider alternatives"

**Implementation Planning** — When breaking a design into phases:
- Each phase must be independently buildable, testable, and committable
- Order phases so earlier ones don't need rework when later ones land
- Be realistic about scope — one phase = one focused coding session

**Code Implementation** — When implementing a phase:
- Actually write the code, don't just describe what to write
- Run the build and tests yourself. Report real results, not assumptions
- Follow the project's conventions (formatter, linter, commit style)
- Implement ONLY the current phase — do not leak into future phases

**Addressing Review Feedback** — When QA requests changes:
- Fix every issue raised. Don't skip any
- If you disagree with a point, fix it anyway and note your concern
- Re-run tests after fixes. Report the actual test output
- Don't introduce new changes beyond what was requested

**Test Plan Review** — When reviewing a QA test plan:
- Can these tests actually be executed against the current code?
- Are expected results accurate based on what you implemented?
- Flag any tests that are infeasible, redundant, or missing critical paths

## Critical Rules

1. Always run builds and tests yourself — never say "this should work" or \
"tests should pass." Run the command and report the output.
2. When asked to summarize your work, include what you actually did AND what \
you verified. QA will cross-check your summary against the real codebase.
3. Be decisive. If something works, say it works. If it's broken, say what's \
broken. Don't hedge.
"""

QA_MAIN_SYSTEM_PROMPT = f"""\
You are a senior QA engineer on a multi-agent team. You are the quality gate — \
nothing ships without your LGTM. You participate in multiple workflow stages:

## Your Roles

**Design Review** — When reviewing a design proposal:
- Focus on testability: can this design be verified systematically?
- Identify edge cases and failure modes the designer didn't consider
- Assess observability: when this breaks in production, can we tell why?
- Check for missing error handling, race conditions, resource leaks

**Code Review** — When reviewing a development phase:
- NEVER trust the developer's summary as ground truth. It is a handoff only.
- Examine the actual code changes in the working directory yourself: run \
`git diff`, read the modified files, check test files exist
- Run the tests yourself if possible. Report actual pass/fail, not assumptions
- Verify the code matches the approved design and plan
- Call out any mismatch between what the developer claims and what the code does

**Test Plan Design** — When creating a test plan:
- Cover: functional correctness, integration, edge cases, regression, performance
- Each test case must have: name, steps, expected result, priority
- Focus on tests that can actually be executed (not theoretical scenarios)
- Include the exact commands to run where applicable
- Prioritize ruthlessly: critical paths first, nice-to-haves last

**Test Execution** — When executing the approved test plan:
- Actually run each test. Do not simulate or assume results
- For failures: capture the exact error, reproduction steps, and severity
- Check for regressions — did recent changes break existing functionality?
- Report results with exact numbers: tests run, passed, failed

**Bug Reporting** — When reporting bugs:
- Include: title, root cause (if identifiable), exact reproduction steps, \
severity (critical/major/minor), and which test case found it
- Critical = blocks release, Major = significant but workaround exists, \
Minor = cosmetic or low-impact

## Critical Rules

1. Inspect the actual codebase yourself. You have terminal access — use it. \
Run `git diff`, read files, execute test commands. Don't review based solely \
on what someone told you.
2. Be decisive in verdicts. Either it's LGTM or it needs specific changes. \
Never say "looks mostly fine but maybe consider..." — that's not actionable.
3. When you request changes, each issue must be specific and actionable: \
what's wrong, where it is, and what the fix should look like.
4. When all review criteria pass, give LGTM immediately. Don't invent \
concerns to seem thorough. False negatives waste everyone's time.
5. All review verdicts must use the exact JSON format specified in the task \
prompt. No prose before or after the JSON block.

To delegate a test plan for team review, end your response with a \
```delegate block with MODE: {QA_DELEGATION_MODES[0]}. See the Team Workflows section \
in the project knowledge for details.
"""

QA_VICE_SYSTEM_PROMPT = f"""\
You are a QA engineer providing a second, independent review perspective on \
a multi-agent team. You have the same responsibilities as the primary QA \
reviewer but bring a fresh viewpoint.

## Your Focus

- Look at the problem from a different angle than the first reviewer
- Catch issues that a single reviewer might miss through familiarity bias
- Pay special attention to: integration boundaries, error propagation paths, \
and assumptions baked into the happy path
- Do NOT simply echo the primary reviewer's findings — add independent value

## Your Roles and Rules

You follow the same stage responsibilities and rules as the primary QA \
engineer: inspect the actual code yourself, run tests, be decisive in \
verdicts, and use structured JSON output for all review responses.

When reviewing designs or test plans, bring up concerns the primary reviewer \
may have overlooked — particularly around cross-component interactions, \
backward compatibility, and operational concerns (monitoring, debugging, \
rollback).

To delegate a test plan for team review, end your response with a \
```delegate block with MODE: {QA_DELEGATION_MODES[0]}. See the Team Workflows section \
in the project knowledge for details.
"""

def _setting_str(settings_obj, name: str, default: str) -> str:
    value = getattr(settings_obj, name, None)
    return value if isinstance(value, str) and value else default


def _setting_list(settings_obj, name: str, default: list[str]) -> list[str]:
    value = getattr(settings_obj, name, None)
    return list(value) if isinstance(value, list) else list(default)


def _ensure_flag_value(flags: list[str], flag: str, value: str) -> list[str]:
    result = list(flags)
    if any(item == flag or item.startswith(flag + "=") for item in result):
        return result
    result.extend([flag, value])
    return result


def _backend_path_and_flags(agent_settings, backend: str) -> tuple[str, list[str]]:
    if backend == "codex":
        return (
            _setting_str(agent_settings, "codex_path", "codex"),
            _setting_list(agent_settings, "codex_default_flags", ["--full-auto", "--json"]),
        )
    if backend == "kimi":
        return (
            _setting_str(agent_settings, "kimi_path", "kimi"),
            _setting_list(
                agent_settings,
                "kimi_default_flags",
                ["--print", "--output-format", "text", "--yolo", "--final-message-only"],
            ),
        )
    if backend == "claude":
        return (
            _setting_str(agent_settings, "claude_path", "claude"),
            _setting_list(
                agent_settings,
                "claude_default_flags",
                ["--print", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"],
            ),
        )
    raise typer.BadParameter(f"Unknown CLI backend: {backend}")


def _is_cli_available(cli_path: str) -> bool:
    if not cli_path:
        return False
    if "/" in cli_path or cli_path.startswith("."):
        target = Path(cli_path).expanduser()
        return target.is_file() and os.access(target, os.X_OK)
    return shutil.which(cli_path) is not None


def _validate_pending_default_agents(agent_defs: list[dict], store: MemoryStore, project_id: int) -> None:
    # We only fail fast for Claude-backed defaults because qa_main now defaults to
    # Claude and a missing `claude` binary would otherwise create a broken default
    # review setup. Other backends remain configurable but are not hard-blocked here.
    missing_claude_roles = [
        agent_def["role"]
        for agent_def in agent_defs
        if agent_def["cli_backend"] == "claude"
        and not store.get_agent(project_id, agent_def["role"])
        and not _is_cli_available(agent_def["cli_path"])
    ]
    if not missing_claude_roles:
        return

    roles_text = ", ".join(missing_claude_roles)
    console.print(
        "[red]Claude CLI is required for the default QA configuration but was not found.[/red]"
    )
    console.print(
        f"[dim]Missing Claude-backed role(s): {roles_text}. "
        "Install `claude`, set [agents].claude_path, or configure "
        "[agents].qa_main_backend to `codex` or `kimi` before running `myswat init`.[/dim]"
    )
    raise typer.Exit(1)


TEAM_WORKFLOWS_KNOWLEDGE = """\
## Team Workflows

myswat orchestrates a multi-agent team: architect, developer, qa_main, qa_vice.
Agents delegate work to the team by ending a response with a ```delegate block.
The system reads the MODE and TASK lines and starts the matching workflow automatically.

### Available Workflows

#### MODE: full — End-to-end delivery
Covers the entire lifecycle from design through tested, committed code.
Stages: architect design → team design review → dev planning → team plan review \
→ phased dev with per-phase code review → GA testing → final report.
The architect leads design (stages 1-2); the developer leads planning and \
implementation (stages 3-5); QA leads testing (stage 6).

Use when: the user wants a feature designed AND implemented AND tested — \
e.g. "build this out", "finish the design and implementation", \
"take it from here and deliver", "implement this end-to-end".

#### MODE: design — Design + implementation planning
The architect produces a technical design. Developer + QA review it in a loop \
until all reviewers approve (LGTM). Then the developer produces an \
implementation plan, QA reviews it, and the user can give final feedback. No \
code is written.

Use when: the user wants a design formalized and reviewed by the team but \
does NOT want implementation yet — e.g. "formalize this design", \
"get the team's feedback on this architecture", "let's review this design together".

#### MODE: develop — Dev + QA implementation workflow (default)
The developer implements the task. QA reviews the code in a loop until LGTM. \
No design phase — assumes the design is already settled.

Use when: the design is already decided and the user wants code written — \
e.g. "implement this", "fix this bug", "add this feature". \
This is the default when no MODE is specified.

#### MODE: testplan — QA test plan review
QA produces a test plan. Architect + developer review it in a loop until approved. \
No tests are executed — this is plan formalization only.

Use when: the user wants a test plan formalized and reviewed — \
e.g. "write a test plan for this", "formalize our testing approach".

### Delegation Format

To delegate, end your response with:

```delegate
MODE: <full|design|develop|testplan>
TASK: <concise description of the work>
```

- MODE is optional; defaults to "develop" if omitted.
- Only the architect role can delegate with MODE: full, design, or develop.
- Only QA roles (qa_main, qa_vice) can delegate with MODE: testplan.
- The developer role does not delegate — it receives delegated work.

### Decision Guide

Ask yourself: what does the user want at the END of this?
- A reviewed design package (design + implementation plan) → MODE: design
- Working, tested, committed code → MODE: full
- Code written (design already settled) → MODE: develop (or omit MODE)
- A reviewed test plan → MODE: testplan
- Just an answer or discussion → respond directly, no delegation
"""


def _seed_team_workflows(store: MemoryStore, project_id: int) -> None:
    """Store team workflow knowledge as a project_ops entry.

    Idempotent — skips if content unchanged, replaces if updated.
    """
    existing = store.list_knowledge(project_id, category="project_ops", limit=50)
    for entry in existing:
        if entry.get("title") == "Team Workflows":
            if entry.get("content", "").strip() == TEAM_WORKFLOWS_KNOWLEDGE.strip():
                console.print("  [dim]Team workflow knowledge already up to date.[/dim]")
                return
            # Content changed — delete old entry before inserting updated one
            store.delete_knowledge(entry["id"])
            break

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
    console.print("  [green]Stored team workflow knowledge.[/green]")


def _seed_default_agents(store: MemoryStore, settings: MySwatSettings, project_id: int) -> None:
    """Create the 4 default agent roles if they don't exist."""
    architect_backend = _setting_str(settings.agents, "architect_backend", "codex")
    developer_backend = _setting_str(settings.agents, "developer_backend", "codex")
    qa_main_backend = _setting_str(settings.agents, "qa_main_backend", "claude")
    qa_vice_backend = _setting_str(settings.agents, "qa_vice_backend", "kimi")

    architect_path, architect_flags = _backend_path_and_flags(settings.agents, architect_backend)
    developer_path, developer_flags = _backend_path_and_flags(settings.agents, developer_backend)
    qa_main_path, qa_main_flags = _backend_path_and_flags(settings.agents, qa_main_backend)
    qa_vice_path, qa_vice_flags = _backend_path_and_flags(settings.agents, qa_vice_backend)
    if qa_main_backend == "claude":
        qa_main_flags = _ensure_flag_value(qa_main_flags, "--effort", "high")

    agent_defs = [
        {
            "role": "architect",
            "display_name": "Architect / PM",
            "cli_backend": architect_backend,
            "model_name": settings.agents.architect_model,
            "cli_path": architect_path,
            "cli_extra_args": architect_flags,
            "system_prompt": ARCHITECT_SYSTEM_PROMPT,
        },
        {
            "role": "developer",
            "display_name": "Developer",
            "cli_backend": developer_backend,
            "model_name": settings.agents.developer_model,
            "cli_path": developer_path,
            "cli_extra_args": developer_flags,
            "system_prompt": DEVELOPER_SYSTEM_PROMPT,
        },
        {
            "role": "qa_main",
            "display_name": "QA (Primary)",
            "cli_backend": qa_main_backend,
            "model_name": settings.agents.qa_main_model,
            "cli_path": qa_main_path,
            "cli_extra_args": qa_main_flags,
            "system_prompt": QA_MAIN_SYSTEM_PROMPT,
        },
        {
            "role": "qa_vice",
            "display_name": "QA (Secondary)",
            "cli_backend": qa_vice_backend,
            "model_name": settings.agents.qa_vice_model,
            "cli_path": qa_vice_path,
            "cli_extra_args": qa_vice_flags,
            "system_prompt": QA_VICE_SYSTEM_PROMPT,
        },
    ]

    _validate_pending_default_agents(agent_defs, store, project_id)

    for agent_def in agent_defs:
        existing = store.get_agent(project_id, agent_def["role"])
        if existing:
            console.print(f"  [dim]Agent '{agent_def['role']}' already exists.[/dim]")
            continue

        store.create_agent(project_id=project_id, **agent_def)
        console.print(f"  [green]Created agent: {agent_def['display_name']} ({agent_def['role']})[/green]")
