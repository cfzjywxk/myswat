# CLI Reference

This file reflects the current public CLI surface from `myswat --help`.

## Top-Level Commands

| Command | Description |
|---------|-------------|
| `myswat init <name> [-r repo] [-d desc]` | Initialize a project and seed the default core agents |
| `myswat chat -p <slug> [--role ROLE] [-w DIR]` | Start the interactive chat REPL |
| `myswat run [task] -p <slug> [--single] [--role ROLE] [--reviewer ROLE] [-w DIR]` | Run the legacy single-task agent flow; omit `task` to enter chat |
| `myswat work [requirement] -p <slug> [--background] [--design\|--plan\|--develop\|--dev\|--test\|--ga-test] [--auto-approve] [--resume ID] [-w DIR]` | Run the teamwork workflow in full or a selected mode |
| `myswat status -p <slug> [--details]` | Show project status, active work items, sessions, and agents |
| `myswat task <id> -p <slug>` | Show detailed status for one work item |
| `myswat stop <id> -p <slug>` | Request cancellation of a background workflow |
| `myswat search <query> -p <slug> [--category C] [--source-type T] [--mode M] [--profile P] [--limit N] [--no-vector] [--json]` | Search project knowledge |
| `myswat history -p <slug> [--turns N] [--role ROLE]` | Show recent raw project turns in chronological order |
| `myswat gc -p <slug> [--grace-days N] [--keep-recent N] [--dry-run]` | Garbage-collect old raw turns from compacted sessions |
| `myswat memory search ...` | Search project knowledge through the memory subcommand |
| `myswat memory list -p <slug> [--category C] [--limit N]` | List stored knowledge entries |
| `myswat reset [-p slug] [-r repo] [-d desc] [-y]` | Drop and re-create the TiDB database |

Notes:

- `myswat search ...` is the root-level shortcut for `myswat memory search ...`.
- `myswat work --resume <id>` resumes a blocked or failed work item; do not pass a new requirement with it.
- `myswat work --background` is supported for `full`, `develop`, and `test`, but not `design`.

## Work Modes

| Mode | Flags | Stages | Success criteria |
|------|-------|--------|------------------|
| `full` | _(default)_ | architect-led design, design review, plan, plan review, phased development, GA test, report | all phases committed and GA passed |
| `design` | `--design`, `--plan` | design, design review, plan, plan review, report | both reviews passed |
| `develop` | `--develop`, `--dev` | phased development with QA review, report | all phases committed |
| `test` | `--test`, `--ga-test` | GA test plan/review, execute tests, bug fixes, report | GA passed |

Foreground `myswat work` is interactive by default. Use `--auto-approve` to skip user checkpoints in foreground workflows.

## Delegation Mapping

| User intent | Delegate block | Engine mode | Participants | User checkpoints |
|-------------|----------------|-------------|--------------|------------------|
| Review architecture before coding | `MODE: design` | `architect_design` | architect, developer, QA | design and plan |
| Implement settled work | `MODE: develop` | `develop` | developer, QA | phase and report checkpoints |
| Deliver end-to-end | `MODE: full` | `full` | architect, developer, QA | design, plan, test plan |
| Formalize a test plan | `MODE: testplan` | `testplan_design` | QA, architect, developer | final test plan |

Internal engine-only modes: `architect_design` and `testplan_design`. They are orchestration entry points used by chat-led workflows, not valid CLI flags or delegate-block values.

## Chat Commands

Inside `myswat chat`:

- `/help` - show chat help
- `/status` - show project status
- `/task <id>` - show detailed work-item status
- `/history [n]` - show recent turns from the active chat session
- `/agents` - list roles and configured models
- `/sessions` - list active sessions
- `/role <name>` - switch to another configured agent role
- `/reset` - reset the active AI session while keeping TiDB session history
- `/new` - close the current session and start a new one
- `/work <requirement>` - start the teamwork workflow from chat
- `/review <task>` - run the legacy inline review loop
- `/quit` or `/exit` - leave chat
- `ESC` - cancel the currently running operation

`qa_vice` appears in role lists only when it is enabled and seeded for the project.
