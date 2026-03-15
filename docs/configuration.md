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

## All-Claude Setup

To use Claude for all roles:

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

## Claude-Specific Notes

When using Claude, MySwat validates the launch environment before every `claude` subprocess start: both `http_proxy` and `https_proxy` must be set, and `curl ipinfo.io` must report the configured IP. If that check fails, the workflow aborts before Claude is started.

By default, Claude runners add `--dangerously-skip-permissions` for non-interactive automation. Override `claude_default_flags` for a different permission model.
