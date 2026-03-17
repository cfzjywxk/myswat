# CLI Reference

## Commands

| Command | Description |
|---------|-------------|
| `myswat init <name> [-r repo] [-d desc]` | Initialize a project |
| `myswat learn -p <slug> [-w workdir]` | Learn project build/test/conventions |
| `myswat chat -p <slug> [--role R]` | Interactive chat session |
| `myswat run <task> -p <slug> [--single] [--role R] [--reviewer R]` | Run agent task |
| `myswat work <req> -p <slug> [--background] [--design\|--develop\|--dev\|--test] [--auto-approve]` | Full or selective teamwork workflow |
| `myswat feed <path> -p <slug> [--glob pattern]` | Ingest documents into knowledge |
| `myswat status -p <slug>` | Show project status |
| `myswat task <id> -p <slug>` | Show detailed status for one work item |
| `myswat stop <id> -p <slug>` | Stop a background workflow |
| `myswat gc -p <slug> [--grace-days N]` | Garbage-collect compacted turns |
| `myswat history -p <slug> [--turns N]` | Show raw recent turns |
| `myswat search <query> -p <slug> [--profile P] [--mode M] [--json]` | Search knowledge base |
| `myswat memory add <title> <content> -p <slug> [-c cat]` | Add knowledge manually |
| `myswat memory list -p <slug> [-c category]` | List knowledge entries |
| `myswat memory compact -p <slug>` | Compact sessions into knowledge |

## Work Modes

| Mode | Flags | Stages | Success criteria |
|------|-------|--------|------------------|
| `full` | _(default)_ | architect-led design, design review, plan, plan review, phased development, GA test, report | all phases committed AND GA passed |
| `design` | `--design`, `--plan` | design, design review, plan, plan review, report | both reviews passed |
| `develop` | `--develop`, `--dev` | phased development with QA review, report | all phases committed |
| `test` | `--test`, `--ga-test` | GA test plan/review, execute tests, bug fixes, report | GA passed |

Foreground `myswat work` is interactive by default. Use `--auto-approve` to skip user checkpoints.

## Delegation Mapping

| User intent | Delegate block | Engine mode | Participants | User checkpoints |
|-------------|----------------|-------------|--------------|------------------|
| Review architecture before coding | `MODE: design` | `architect_design` | architect, developer, QA | design and plan |
| Implement settled work | `MODE: develop` | `develop` | developer, QA | phase / report checkpoints |
| Deliver end-to-end | `MODE: full` | `full` | architect, developer, QA | design, plan, test plan |
| Formalize a test plan | `MODE: testplan` | `testplan_design` | QA, architect, developer | final test plan |

Internal engine-only modes: `architect_design` and `testplan_design`. They are orchestration entry points used by chat-led workflows, not valid CLI flags or delegate block values.

## Chat Commands

Inside `myswat chat`:

- `/role <name>` — switch agent role
- `/work <task>` — start teamwork workflow
- `/review <task>` — inline code review
- `/status` — show active work items
- `/task <id>` — show work item details
- `/agents` — list roles/models
- `/history [n]` — show recent turns
- `ESC` — cancel long-running task
