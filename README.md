# MySwat

Multi-AI agent co-working system for software development. Agents with persistent long-term memory collaborate through structured review workflows, building knowledge across sessions.

## Key Features

- **Persistent memory** — sessions are distilled into reusable knowledge entries stored in TiDB, not raw transcripts
- **Project learning** — `myswat learn` teaches agents how to build, test, and follow conventions for any project
- **Multi-agent review** — developer proposes, QA reviews, iterate until LGTM
- **Full teamwork workflow** — design review, plan review, phased implementation, GA testing
- **Knowledge-first context** — agents receive relevant knowledge via vector search, not bloated history
- **CLAUDE.md / AGENTS.md aware** — extracts project conventions from AI instruction files without adopting their workflow

## Prerequisites

- Python 3.12+
- [TiDB Cloud](https://tidbcloud.com) account (free tier works)
- At least one AI CLI tool: [Codex CLI](https://github.com/openai/codex) (`codex`) or Kimi CLI (`kimi`)
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
kimi_path = "kimi"
developer_model = "gpt-5.4"
qa_main_model = "kimi-code/kimi-for-coding"
```

Or use environment variables: `MYSWAT_TIDB_HOST`, `MYSWAT_TIDB_PASSWORD`, etc.

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

### 4. Run a task

```bash
# Single agent
myswat run --single -p my-project "Implement feature X"

# Developer + reviewer loop
myswat run -p my-project "Add error handling to the parser"
```

### 5. Full teamwork workflow

```bash
myswat work -p my-project "Implement bloom filter for compaction"
```

Runs: tech design -> design review -> planning -> plan review -> phased implementation (per-phase code review + commit) -> GA testing -> final report.

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
| `myswat work <requirement> -p <slug>` | Full teamwork workflow |
| `myswat feed <path> -p <slug> [--glob pattern]` | Ingest documents into knowledge |
| `myswat status -p <slug>` | Show project status |
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
          |      +--> SessionManager --> AgentRunner (codex/kimi subprocess)
          |      |                   --> Mid-session compaction (watermark-based)
          |      |
          |      +--> WorkflowEngine (design -> review -> plan -> dev -> GA test)
          |      |
          |      +--> MemoryRetriever --> project_ops (myswat.md cache or TiDB fallback)
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
