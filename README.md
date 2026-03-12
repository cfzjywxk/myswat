# MySwat

Multi-AI agent co-working system for software development. Agents with persistent long-term memory collaborate through structured review workflows, building knowledge across sessions.

## Key Features

- **Persistent memory** — sessions are distilled into reusable knowledge entries stored in TiDB, not raw transcripts
- **Project-shared knowledge** — compacted session knowledge is shared across the whole project, not trapped inside one role
- **Project learning** — `myswat learn` teaches agents how to build, test, and follow conventions for any project
- **Multi-agent review** — developer proposes, QA reviews, iterate until LGTM
- **Full teamwork workflow** — auto-continue through design review, planning, phased implementation, and GA testing unless a critical failure stops it
- **Selective work modes** — `--design` for design+planning only, `--dev` for phased implementation only, `--test` for GA testing only, or default full workflow
- **Knowledge-first context** — agents receive relevant knowledge via vector search, not bloated history
- **Persistent task state** — work items keep current stage, summary, todos, and issues so newly started agent sessions can recover ongoing work
- **Interactive long-task monitor** — `/work` and `/review` show MySwat-level progress, stage, todos, and `ESC` cancel hints instead of appearing stuck
- **CLAUDE.md / AGENTS.md aware** — extracts project conventions from AI instruction files without adopting their workflow

## Prerequisites

- Python 3.12+
- [TiDB Cloud](https://tidbcloud.com) account (free tier works)
- At least one AI CLI tool: [Codex CLI](https://github.com/openai/codex) (`codex`), Claude Code (`claude`), or Kimi CLI (`kimi`)
- (Optional) `FlagEmbedding` for vector search with BGE-M3

## Installation

```bash
# Self-bootstrapping — auto-creates venv, installs deps on first run
./myswat --help
```

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install pymysql pydantic pydantic-settings "typer[all]" rich prompt_toolkit
.venv/bin/pip install --no-deps -e .
```

## Configuration

Create `~/.myswat/config.toml`:

```toml
[tidb]
host = "your-tidb-host.tidbcloud.com"
port = 4000
user = "your_user"
password = "your_password"
ssl_ca = "/etc/ssl/certs/ca-certificates.crt"

[agents]
codex_path = "codex"
claude_path = "claude"
kimi_path = "kimi"
claude_required_ip = "your_claude_cli_ip"
architect_backend = "codex"
developer_backend = "codex"
qa_main_backend = "claude"
qa_vice_backend = "kimi"
architect_model = "gpt-5.4"
developer_model = "gpt-5.4"
qa_main_model = "claude-opus-4-6"
qa_vice_model = "kimi-code/kimi-for-coding"
```

Or use environment variables: `MYSWAT_TIDB_HOST`, `MYSWAT_TIDB_PASSWORD`, etc.

To seed new projects against Claude instead of Codex/Kimi, set per-role backends before `myswat init`, for example:

```toml
[agents]
architect_backend = "claude"
developer_backend = "claude"
qa_main_backend = "claude"
qa_vice_backend = "claude"
claude_path = "claude"
architect_model = "claude-sonnet-4-6"
developer_model = "claude-sonnet-4-6"
qa_main_model = "claude-opus-4-6"
qa_vice_model = "claude-sonnet-4-6"
```

When using Claude, MySwat validates the launch environment before every `claude` subprocess start: both `http_proxy` and `https_proxy` must be set, and `curl ipinfo.io` must report `154.28.2.59`. If that check fails, the workflow aborts before Claude is started. During `myswat init`, if the default Claude-backed `qa_main` role cannot find the `claude` binary, initialization aborts and asks you to either install/configure Claude or change `qa_main_backend` to `codex` or `kimi`. The default `qa_main` seed uses `claude-opus-4-6` with `--effort high`.

By default, Claude runners also add `--dangerously-skip-permissions` for non-interactive automation. Override `claude_default_flags` or provide an explicit Claude permission flag if you want a different permission model.

## Quick Start

### 1. Initialize a project

```bash
myswat init "my-project" --desc "Project description" --repo /path/to/repo
```

Creates the database schema, project record, and seeds 4 default agent roles (architect, developer, qa_main, qa_vice).

### 2. Learn the project

```bash
myswat learn -p my-project
```

The architect agent scans indicator files (Makefile, Cargo.toml, package.json, CI configs, CLAUDE.md, etc.) and extracts structured knowledge about:
- Build commands and prerequisites
- Test tiers and gate commands
- Git workflow and conventions
- Code style rules and invariants
- Security requirements

This knowledge is persisted to TiDB and cached locally in `myswat.md`. All agents automatically receive it.

### 3. Interactive chat

```bash
myswat chat -p my-project
```

REPL with role switching (`/role developer`), inline review (`/review "task"`), and persistent sessions.

Long-running `/work` and `/review` tasks now show a live MySwat monitor with:
- Current work item ID and status
- Current workflow stage
- Latest persisted summary
- Next TODOs and open issues
- `ESC` to cancel the current agent step

Useful commands while a task is running:

```bash
myswat status -p my-project
myswat task 42 -p my-project
```

### 4. Run a task

```bash
# Single agent
myswat run --single -p my-project "Implement feature X"

# Developer + reviewer loop
myswat run -p my-project "Add error handling to the parser"
```

### 5. Full teamwork workflow

```bash
# Full workflow (default): design -> review -> plan -> dev -> GA test -> report
myswat work -p my-project "Implement bloom filter for compaction"

# Design + planning only (interactive checkpoints, no --background)
myswat work -p my-project --design "Implement bloom filter for compaction"

# Development only (phased implementation, skip design/plan)
myswat work -p my-project --dev "Implement bloom filter for compaction"

# GA testing only (no architecture-change escalation)
myswat work -p my-project --test "Validate bloom filter correctness"

# Detach and keep running after this terminal exits
myswat work -p my-project --background "Implement bloom filter for compaction"
myswat work -p my-project --background --dev "Implement bloom filter for compaction"

# Monitor or stop it later
myswat task 42 -p my-project
myswat stop 42 -p my-project
```

| Mode | Flags | Stages | Success criteria |
|------|-------|--------|------------------|
| `full` | _(default)_ | design, design review, plan, plan review, phased dev, GA test, report | all phases committed AND GA passed |
| `design` | `--design`, `--plan` | design, design review, plan, plan review, report | both reviews passed |
| `development` | `--development`, `--dev` | phased dev (with informational QA review), report | all phases committed |
| `test` | `--test`, `--ga-test` | GA test plan/review, execute tests, bug fixes, report | GA passed |

`myswat work` auto-continues by default instead of stopping for manual approvals. Design mode (`--design`) keeps interactive checkpoints and cannot be combined with `--background`. In chat mode, `/work` also shows the live task monitor described above.

### 6. Feed documents

```bash
myswat feed /path/to/docs -p my-project --glob "**/*.md"
myswat feed /path/to/src -p my-project --glob "**/*.rs"
```

### 7. Search knowledge

```bash
myswat memory search "transaction isolation" -p my-project
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `myswat init <name> [-r repo] [-d desc]` | Initialize a project |
| `myswat learn -p <slug> [-w workdir]` | Learn project build/test/conventions |
| `myswat chat -p <slug> [--role R]` | Interactive chat session |
| `myswat run <task> -p <slug> [--single] [--role R] [--reviewer R]` | Run agent task |
| `myswat work <requirement> -p <slug> [--background] [--design\|--dev\|--test]` | Full or selective teamwork workflow |
| `myswat feed <path> -p <slug> [--glob pattern]` | Ingest documents into knowledge |
| `myswat status -p <slug>` | Show project status |
| `myswat task <id> -p <slug>` | Show detailed status for one work item |
| `myswat stop <id> -p <slug>` | Stop a background workflow |
| `myswat memory search <query> -p <slug>` | Search knowledge base |
| `myswat memory add <title> <content> -p <slug> [-c cat]` | Add knowledge manually |
| `myswat memory list -p <slug> [-c category]` | List knowledge entries |
| `myswat memory compact -p <slug>` | Compact sessions into knowledge |
| `myswat memory purge -p <slug> [--yes]` | Delete compacted sessions (keeps knowledge) |

## Architecture

```
User --> CLI (Typer)
          |
          +--> myswat learn --> Architect agent --> project_ops knowledge --> TiDB + myswat.md
          |
          +--> myswat work/run/chat
          |      |
          |      +--> SessionManager --> AgentRunner (codex/claude/kimi subprocess)
          |      |                   --> Mid-session compaction (watermark-based)
          |      |
          |      +--> WorkflowEngine (mode dispatch: full | design | development | test)
          |      |      full:        design -> review -> plan -> dev -> GA test -> report
          |      |      design:      design -> review -> plan -> review -> report (interactive)
          |      |      development: phased dev -> report (no design/plan)
          |      |      test:        GA test -> report (no arch_change escalation)
          |      |
          |      +--> MemoryRetriever --> project_ops (myswat.md cache or TiDB fallback)
          |                           --> work_item task state
          |                           --> knowledge (vector search)
          |                           --> session context
          |
          +--> MemoryStore --> TiDB Cloud (SQL + VECTOR)
          |
          +--> Embedder (BGE-M3, optional)
          |
          +--> KnowledgeCompactor (session turns --> distilled knowledge)
```

### TiDB Schema

| Table | Purpose |
|-------|---------|
| `projects` | Project registry |
| `agents` | Role configs per project |
| `sessions` | Dialog sessions with compaction watermark |
| `session_turns` | Individual messages |
| `knowledge` | Compacted/ingested knowledge with VECTOR(1024) |
| `work_items` | Task tracking |
| `artifacts` | Proposals/diffs under review |
| `review_cycles` | Review iterations with structured verdicts |

## License

MIT
