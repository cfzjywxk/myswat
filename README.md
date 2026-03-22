<p align="center">
  <img src="assets/myswat.png" alt="myswat — ARCH, DEV, QA" width="480"/><br/>
  <strong>Multi-AI workflow orchestrator with an MCP coordination server.</strong><br/><br/>
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/backend-TiDB_Cloud-4479A1?logo=tidb&logoColor=white" alt="TiDB Cloud"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
  <img src="https://img.shields.io/badge/agents-Codex_%7C_Claude_%7C_Kimi-orange" alt="Agent Backends"/>
</p>

---

You architect. Agents build and review. **MySwat connects them.**

MySwat automates the copy-paste routing between AI agents. It persists shared workflow state to TiDB, exposes coordination through an MCP server, and loops dev/QA review cycles until LGTM — while you stay in the architect seat.

## Quick Start

```bash
# Start the daemon
myswat server

# Initialize a project (in another shell)
myswat init "my-project" --repo /path/to/repo

# Queue a task
myswat work -p my-project "Implement bloom filter for compaction"

# Follow progress live
myswat work -p my-project "Add caching layer" --follow

# Interactive chat
myswat chat -p my-project

# Inspect state
myswat status -p my-project --details
myswat search "bloom filter" -p my-project
```

## How It Works

```
 You (architect)
  |
  |  "Implement bloom filter for compaction"
  v
 MySwat daemon
  |  1. Queues stage assignments in TiDB
  |  2. Starts managed workers for each agent role
  |  3. Workers claim work through MCP, agents execute
  |  4. WorkflowKernel advances or loops review stages
  |  5. Final report + persisted team knowledge
  v
 Done
```

## Workflow Modes

| Mode | CLI flags | What runs |
|------|-----------|-----------|
| Full | _(default)_ | Design, review, plan, develop, QA test, report |
| Design | `--design` | Design + plan with reviews, no code |
| Develop | `--develop` | Phased implementation with QA review |
| Test | `--test` | GA test plan, execute, bug fixes, report |

## Prerequisites

- Python 3.12+
- [TiDB Cloud](https://tidbcloud.com) account (free tier works)
- At least one AI CLI: [Codex](https://github.com/openai/codex), [Claude Code](https://claude.com/claude-code), or [Kimi](https://www.kimi.com/code)

## Getting Started

New to MySwat? Follow the **[Installation Guide](docs/installation.md)** for step-by-step setup: cloning, TiDB Cloud configuration, agent CLI installation, and running your first workflow.

## Documentation

- [Installation Guide](docs/installation.md) — full setup walkthrough for new users
- [CLI Reference](docs/cli-reference.md) — all commands, work modes, chat commands
- [Configuration](docs/configuration.md) — config file, environment variables, per-backend setup
- [Architecture](docs/architecture.md) — components, memory tiers, knowledge pipeline, TiDB schema

## License

MIT
