# CLI Reference

All commands require a running daemon (`myswat server`) unless noted otherwise.

## Commands

### `myswat server`

Run the persistent local daemon. All other commands communicate with it.

```bash
myswat server
```

### `myswat init`

Initialize a project and seed the default agent roles (architect, developer, qa_main).

```bash
myswat init <name> [--repo PATH] [--desc TEXT]
```

### `myswat work`

Submit a workflow to the daemon. Detaches by default after queueing.

```bash
myswat work <requirement> -p <slug> [options]
```

| Flag | Description |
|------|-------------|
| `--follow` | Stay attached and stream live progress |
| `--design` / `--plan` | Design + planning only (no code) |
| `--develop` / `--dev` | Development only (phased with QA review) |
| `--test` / `--ga-test` | GA testing only |
| `--with-ga-test` | Add GA test stage to the default full workflow |
| `--auto-approve` | Auto-continue through checkpoints (default) |
| `--interactive-checkpoints` | Require manual approval at checkpoints |
| `--resume <id>` | Resume a blocked/failed work item |
| `-w DIR` | Working directory override |

Only one of `--design`, `--develop`, `--test` can be used at a time.

### `myswat chat`

Interactive REPL for conversing with agents.

```bash
myswat chat -p <slug> [--role ROLE] [-w DIR]
```

Default role is `architect`. Switch roles inside the session with `/role`.

### `myswat run`

Legacy single-task agent flow. Omit the task to enter interactive chat.

```bash
myswat run [task] -p <slug> [--single] [--role ROLE] [--reviewer ROLE] [-w DIR]
```

### `myswat status`

Show project status: agents, work items, active sessions, knowledge stats.

```bash
myswat status -p <slug> [--details]
```

`--details` adds alerts, worker health, full message flow, and artifact history.

### `myswat task`

Show detailed status for one work item: message flow, artifacts, review cycles, worker health.

```bash
myswat task <id> -p <slug>
```

### `myswat stop`

Request cancellation of a running workflow.

```bash
myswat stop <id> -p <slug>
```

### `myswat search`

Search project knowledge with hybrid lexical + semantic retrieval.

```bash
myswat search <query> -p <slug> [options]
```

| Flag | Description |
|------|-------------|
| `--category`, `-c` | Filter by knowledge category |
| `--source-type` | Filter by source type |
| `--mode` | Search mode: `auto`, `exact`, `concept`, `relation` |
| `--profile` | Search profile: `quick`, `standard`, `precise` |
| `--limit`, `-n` | Max results (default 10) |
| `--no-vector` | Skip vector search (keyword only) |
| `--json` | Machine-readable JSON output |

This is a root-level shortcut for `myswat memory search`.

### `myswat history`

Show recent raw project turns in chronological order.

```bash
myswat history -p <slug> [--turns N] [--role ROLE]
```

### `myswat gc`

Garbage-collect old raw turns from fully compacted sessions.

```bash
myswat gc -p <slug> [--grace-days N] [--keep-recent N] [--dry-run]
```

### `myswat cleanup`

Delete one project and all related TiDB state. Prompts for confirmation unless `--yes` is passed.

```bash
myswat cleanup -p <slug> [--yes]
```

### `myswat reset`

Drop and re-create the entire TiDB database. Destroys all data.

```bash
myswat reset [-p slug] [-r repo] [-d desc] [-y]
```

Pass `-p` to automatically re-init a project after the reset.

### `myswat memory`

Subcommand group for knowledge management.

```bash
myswat memory search <query> -p <slug> [...]
myswat memory list -p <slug> [--category C] [--limit N]
```

---

## Work Modes

| Mode | Flags | Stages | Success criteria |
|------|-------|--------|------------------|
| `full` | _(default)_ | Design, design review, plan, plan review, phased development, report | All phases committed |
| `full` + `--with-ga-test` | `--with-ga-test` | Same as full + GA test stage | All phases committed, GA passed |
| `design` | `--design`, `--plan` | Design, design review, plan, plan review, report | Both reviews passed |
| `develop` | `--develop`, `--dev` | Phased development with QA review, report | All phases committed |
| `test` | `--test`, `--ga-test` | GA test plan/review, execute tests, bug fixes, report | GA passed |

---

## Chat Commands

Inside `myswat chat`:

| Command | Description |
|---------|-------------|
| `/help` | Show chat help |
| `/status` | Show project status |
| `/task <id>` | Show work item detail |
| `/history [n]` | Show recent turns from active session |
| `/work <requirement>` | Start full workflow (design, plan, develop, report) |
| `/dev <task>` | Start development workflow |
| `/ga-test <task>` | Start standalone GA test workflow |
| `/role <role>` | Switch agent role |
| `/agents` | List available agents |
| `/sessions` | Show active sessions |
| `/reset` | Reset AI session (fresh context, same TiDB session) |
| `/new` | Close current session, start a new one |
| `/quit`, `/exit`, `Ctrl+D` | Exit chat |
| `ESC` | Cancel currently running operation |

---

## Delegation Mapping (from chat)

| Intent | Delegate mode | Engine mode | Participants |
|--------|---------------|-------------|--------------|
| Review architecture before coding | `MODE: design` | `architect_design` | architect, developer, QA |
| Implement settled work | `MODE: develop` | `develop` | developer, QA |
| Deliver end-to-end | `MODE: full` | `full` | architect, developer, QA |
| Formalize a test plan | `MODE: testplan` | `testplan_design` | QA, architect, developer |

`architect_design` and `testplan_design` are internal engine modes used by chat-led workflows, not valid CLI flags.
