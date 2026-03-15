<p align="center">
  <img src="assets/myswat.png" alt="myswat — ARCH, DEV, QA" width="480"/><br/>
  <strong>Multi-AI agent conversation orchestrator for software development.</strong><br/><br/>
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/backend-TiDB_Cloud-4479A1?logo=tidb&logoColor=white" alt="TiDB Cloud"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/agents-Codex_%7C_Claude_%7C_Kimi-orange" alt="Agent Backends"/>
</p>

---

You have a codebase. You have AI agents that can write code and review it. But you're still the one copying outputs between them, deciding who goes next, and re-explaining context every session.

**MySwat automates that.** It routes prompts between agents, persists everything to a shared knowledge base, and loops dev/QA review cycles until LGTM — while you stay in the architect seat.

<p align="center">
  <img src="assets/workflow.svg" alt="myswat workflow — design, review, plan, dev, QA test, report" width="800"/>
</p>

## Why MySwat

**You architect. Agents build and review. MySwat connects them.**

- **Automated review loops** — developer proposes, QA reviews, iterate until LGTM. No manual copy-paste.
- **Shared project memory** — every agent sees what others said and learned. Knowledge compounds across sessions, not just within one.
- **Mix any AI backend** — Claude Opus for QA, GPT for dev, Kimi for a second opinion. Per-role configuration.
- **Full workflow or pick stages** — run the whole pipeline (design → plan → dev → QA → report), or just `--dev` or `--test`.
- **Learns your project** — build commands, test tiers, invariants, conventions. Agents stop guessing.

## Quick Start

```bash
# Install (auto-creates venv on first run)
./myswat --help

# Set up a project
myswat init "my-project" --repo /path/to/repo
myswat learn -p my-project

# Run a task — full dev/QA loop
myswat work -p my-project "Implement bloom filter for compaction"

# Or just chat
myswat chat -p my-project
```

## How It Works

```
 You (architect)
  |
  |  "Implement bloom filter for compaction"
  v
 MySwat ──────────────────────────────────────────────
  |                                                    |
  |  1. Loads project knowledge from TiDB              |
  |  2. Sends to Developer agent (codex/claude/kimi)   |
  |  3. Captures output, feeds to QA agent             |
  |  4. QA reviews → issues found? loop back to Dev    |
  |  5. LGTM → done. Knowledge compacted & persisted.  |
  |                                                    |
 ──────────────────────────────────────────────────────
```

MySwat does **not** run builds or tests — the agents do that themselves via their terminal access. MySwat handles routing, context, and persistence.

## What Agents Remember

Every conversation is persisted to TiDB and searchable:

- **Session turns** — raw conversation history, cross-role visible
- **Compacted knowledge** — AI-distilled insights from past sessions (architecture decisions, bug fixes, patterns, failure modes)
- **Ingested documents** — your docs and source code, chunked and indexed
- **Project ops** — build commands, test tiers, conventions (via `myswat learn`)

Search across all of it:

```bash
myswat search "transaction isolation" -p my-project
```

## Prerequisites

- Python 3.12+
- [TiDB Cloud](https://tidbcloud.com) account (free tier works)
- At least one AI CLI: [Codex](https://github.com/openai/codex), [Claude Code](https://claude.com/claude-code), or [Kimi](https://www.kimi.com/code)

## Documentation

- [Configuration](docs/configuration.md) — config file, environment variables, per-backend setup
- [CLI Reference](docs/cli-reference.md) — all commands, work modes, chat commands
- [Architecture](docs/architecture.md) — components, memory tiers, knowledge pipeline, TiDB schema

## License

MIT
