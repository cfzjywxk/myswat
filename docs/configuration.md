# Configuration

## Config File

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
architect_backend = "codex"
developer_backend = "codex"
qa_main_backend = "claude"
architect_model = "gpt-5.4"
developer_model = "gpt-5.4"
qa_main_model = "claude-opus-4-6"
```

All settings can also be set via environment variables: `MYSWAT_TIDB_HOST`, `MYSWAT_TIDB_PASSWORD`, etc.

## Agent Roles

`myswat init` seeds three core roles: **architect**, **developer**, and **qa_main**.

To add a secondary QA reviewer (`qa_vice`), set before running `myswat init`:

```toml
[agents]
qa_vice_enabled = true
qa_vice_backend = "kimi"
qa_vice_model = "kimi-code/kimi-for-coding"
```

Or via environment: `MYSWAT_AGENTS_QA_VICE_ENABLED=true`.

## All-Claude Setup

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

## Claude-Specific Notes

When using Claude, MySwat validates the launch environment before every `claude` subprocess: both `http_proxy` and `https_proxy` must be set, and `curl ipinfo.io` must report the configured IP. Set the expected IP with:

```toml
[agents]
claude_required_ip = "your_claude_cli_ip"
```

Claude runners add `--dangerously-skip-permissions` by default for non-interactive automation. Override with `claude_default_flags` for a different permission model.
