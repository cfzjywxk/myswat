# CLI Reference

## Commands

| Command | Description |
|---------|-------------|
| `myswat init <name> [-r repo] [-d desc]` | Initialize a project |
| `myswat learn -p <slug> [-w workdir]` | Learn project build/test/conventions |
| `myswat chat -p <slug> [--role R]` | Interactive chat session |
| `myswat run <task> -p <slug> [--single] [--role R] [--reviewer R]` | Run agent task |
| `myswat work <req> -p <slug> [--background] [--design\|--dev\|--test]` | Full or selective teamwork workflow |
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
| `full` | _(default)_ | design, design review, plan, plan review, phased dev, GA test, report | all phases committed AND GA passed |
| `design` | `--design`, `--plan` | design, design review, plan, plan review, report | both reviews passed |
| `development` | `--development`, `--dev` | phased dev (with informational QA review), report | all phases committed |
| `test` | `--test`, `--ga-test` | GA test plan/review, execute tests, bug fixes, report | GA passed |

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
