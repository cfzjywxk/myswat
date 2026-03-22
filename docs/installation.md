# Installation Guide

Step-by-step guide for getting MySwat running from a fresh clone.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | `python3 --version` to check |
| Git | any | For cloning and diff tracking |
| TiDB Cloud account | free tier works | [Sign up](https://tidbcloud.com) |
| AI CLI (at least one) | — | See [Agent Backends](#2-install-an-ai-cli) below |

## 1. Clone and Install

```bash
git clone https://github.com/user/myswat.git
cd myswat
```

**Option A — Use the self-bootstrapping launcher** (recommended):

```bash
chmod +x ./myswat
./myswat --help
```

On first run this automatically creates a `.venv`, installs all dependencies, and runs the command. Subsequent runs skip the bootstrap step.

**Option B — Manual venv setup**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
myswat --help
```

**Option C — Using [uv](https://docs.astral.sh/uv/)**:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
myswat --help
```

## 2. Install an AI CLI

MySwat delegates code writing, testing, and reviewing to AI agents via their CLI tools. You need at least one installed and on your `PATH`.

| Backend | Install | Docs |
|---------|---------|------|
| OpenAI Codex | `npm install -g @openai/codex` | [github.com/openai/codex](https://github.com/openai/codex) |
| Claude Code | `npm install -g @anthropic-ai/claude-code` | [claude.com/claude-code](https://claude.com/claude-code) |
| Kimi | `npm install -g @anthropic-ai/kimi` | [kimi.com/code](https://www.kimi.com/code) |

Verify the CLI is reachable:

```bash
codex --version   # if using Codex
claude --version  # if using Claude Code
kimi --version    # if using Kimi
```

Each backend requires its own API key / auth — follow the respective docs to complete that setup before continuing.

## 3. Set Up TiDB Cloud

MySwat stores all project state, conversation history, and extracted knowledge in TiDB Cloud.

### Create a cluster

1. Go to [tidbcloud.com](https://tidbcloud.com) and sign in (or create a free account).
2. Create a **Serverless** cluster (free tier is sufficient).
3. Once provisioned, open **Connect** and note the connection details:
   - **Host** (e.g. `gateway01.us-east-1.prod.aws.tidbcloud.com`)
   - **Port** (`4000`)
   - **User** (e.g. `randomstring.root`)
   - **Password** (generated once — save it)

### Create the database

In the TiDB Cloud SQL Editor (or any MySQL-compatible client), run:

```sql
CREATE DATABASE IF NOT EXISTS myswat;
```

MySwat auto-creates all tables on first `myswat init`, so no manual schema setup is needed.

### SSL certificate

TiDB Cloud requires TLS. The default CA path is `/etc/ssl/certs/ca-certificates.crt` (works on most Linux distributions). On macOS you may need to set `ssl_ca` to your system CA bundle path (e.g. from `brew --prefix`/etc/openssl/cert.pem, or download the [ISRG Root X1](https://letsencrypt.org/certs/isrgrootx1.pem) CA).

## 4. Configure MySwat

Create the config file:

```bash
mkdir -p ~/.myswat
```

Write `~/.myswat/config.toml`:

```toml
[tidb]
host = "gateway01.us-east-1.prod.aws.tidbcloud.com"  # your TiDB host
port = 4000
user = "randomstring.root"       # your TiDB user
password = "your_password_here"  # your TiDB password
ssl_ca = "/etc/ssl/certs/ca-certificates.crt"
database = "myswat"

[agents]
# Paths to CLI binaries (defaults shown, change if installed elsewhere)
codex_path = "codex"
claude_path = "claude"
kimi_path = "kimi"

# Which backend runs each role
architect_backend = "codex"
developer_backend = "codex"
qa_main_backend = "claude"

# Model selection per role
architect_model = "gpt-5.4"
developer_model = "gpt-5.4"
qa_main_model = "claude-opus-4-6"
```

### Alternative: environment variables

Every setting can be set via environment variables instead of (or overriding) the TOML file. The naming convention is `MYSWAT_<SECTION>_<KEY>` in upper case:

```bash
export MYSWAT_TIDB_HOST="gateway01.us-east-1.prod.aws.tidbcloud.com"
export MYSWAT_TIDB_USER="randomstring.root"
export MYSWAT_TIDB_PASSWORD="your_password_here"
export MYSWAT_AGENTS_ARCHITECT_BACKEND="codex"
export MYSWAT_AGENTS_QA_MAIN_BACKEND="claude"
```

Environment variables take precedence over TOML values.

### All-Claude setup

If you only have Claude Code installed:

```toml
[agents]
architect_backend = "claude"
developer_backend = "claude"
qa_main_backend = "claude"
claude_path = "claude"
architect_model = "claude-sonnet-4-6"
developer_model = "claude-sonnet-4-6"
qa_main_model = "claude-opus-4-6"
```

### Claude-specific: IP validation

When using Claude as a backend, MySwat validates the launch environment's IP before each subprocess. If you use a proxy, set the expected IP:

```toml
[agents]
claude_required_ip = "your_expected_ip"
```

Both `http_proxy` and `https_proxy` must be set in the environment for the check to pass.

## 5. Verify the Setup

```bash
# Start the daemon (keep this running in a dedicated terminal or background it)
myswat server

# In another terminal, initialize a project
myswat init "hello" --repo /path/to/your/repo --desc "Test project"

# Check that everything connected
myswat status -p hello
```

If `myswat init` succeeds, your TiDB connection and agent configuration are working.

## 6. First Workflow

```bash
# Submit a task and follow progress live
myswat work -p hello "Add a health check endpoint" --follow

# Or use interactive chat
myswat chat -p hello
```

Inside chat, type `/help` for available commands.

## Troubleshooting

### `Connection refused` on `myswat init` or `myswat work`

The daemon isn't running. Start it with `myswat server` in a separate terminal.

### `Can't connect to MySQL server` / TiDB connection errors

- Double-check host, port, user, and password in `~/.myswat/config.toml`.
- Ensure the `myswat` database exists on TiDB Cloud.
- Verify SSL CA path exists: `ls /etc/ssl/certs/ca-certificates.crt`.
- On macOS, you may need to point `ssl_ca` to a different CA bundle.

### `codex: command not found` (or `claude` / `kimi`)

The AI CLI isn't on your PATH. Install it per [step 2](#2-install-an-ai-cli) and verify with `which codex`.

### Claude IP validation fails

Set `MYSWAT_AGENTS_CLAUDE_REQUIRED_IP` or `claude_required_ip` in config, and ensure `http_proxy` / `https_proxy` environment variables are set.

### Schema / table errors on first run

`myswat init` auto-creates all required tables. If you see errors, ensure your TiDB user has `CREATE TABLE` privileges on the `myswat` database.

## What's Next

- [CLI Reference](cli-reference.md) — all commands and flags
- [Configuration](configuration.md) — full settings reference and per-backend details
- [Architecture](architecture.md) — how MySwat orchestrates agents and manages knowledge
