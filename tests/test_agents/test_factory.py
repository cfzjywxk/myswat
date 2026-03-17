"""Tests for the shared agent runner factory."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer

from myswat.agents.factory import make_memory_worker_runner, make_runner_from_row


class TestMakeRunnerFromRow:
    @patch("myswat.agents.factory.CodexRunner")
    def test_codex_backend(self, MockCodex):
        row = {
            "cli_backend": "codex",
            "cli_path": "/usr/bin/codex",
            "model_name": "gpt-5",
            "cli_extra_args": None,
        }

        make_runner_from_row(row)

        MockCodex.assert_called_once_with(
            cli_path="/usr/bin/codex",
            model="gpt-5",
            workdir=None,
            extra_flags=[],
        )

    @patch("myswat.agents.factory.KimiRunner")
    def test_kimi_backend(self, MockKimi):
        row = {
            "cli_backend": "kimi",
            "cli_path": "/usr/bin/kimi",
            "model_name": "kimi-2",
            "cli_extra_args": None,
        }

        make_runner_from_row(row)

        MockKimi.assert_called_once_with(
            cli_path="/usr/bin/kimi",
            model="kimi-2",
            workdir=None,
            extra_flags=[],
        )

    @patch("myswat.agents.factory.ClaudeRunner")
    def test_claude_backend_uses_settings(self, MockClaude):
        row = {
            "cli_backend": "claude",
            "cli_path": "/usr/bin/claude",
            "model_name": "claude-sonnet-4-6",
            "cli_extra_args": None,
        }
        settings = MagicMock()
        settings.agents.claude_required_ip = "203.0.113.10"
        settings.agents.claude_ip_check_timeout_seconds = 7

        make_runner_from_row(row, settings=settings, workdir="/tmp/project")

        MockClaude.assert_called_once_with(
            cli_path="/usr/bin/claude",
            model="claude-sonnet-4-6",
            workdir="/tmp/project",
            extra_flags=[],
            required_ip="203.0.113.10",
            ip_check_timeout_seconds=7,
        )

    @patch("myswat.agents.factory.ClaudeRunner")
    def test_claude_backend_falls_back_to_env(self, MockClaude, monkeypatch):
        row = {
            "cli_backend": "claude",
            "cli_path": "claude",
            "model_name": "claude-sonnet-4-6",
            "cli_extra_args": None,
        }
        monkeypatch.setenv("MYSWAT_AGENTS_CLAUDE_REQUIRED_IP", "198.51.100.5")
        monkeypatch.setenv("MYSWAT_AGENTS_CLAUDE_IP_CHECK_TIMEOUT_SECONDS", "9")

        make_runner_from_row(row)

        MockClaude.assert_called_once_with(
            cli_path="claude",
            model="claude-sonnet-4-6",
            workdir=None,
            extra_flags=[],
            required_ip="198.51.100.5",
            ip_check_timeout_seconds=9,
        )

    def test_claude_backend_requires_configured_ip(self, monkeypatch):
        row = {
            "cli_backend": "claude",
            "cli_path": "claude",
            "model_name": "claude-opus-4-6",
            "cli_extra_args": None,
        }
        settings = MagicMock()
        settings.agents.claude_required_ip = ""
        settings.agents.claude_ip_check_timeout_seconds = 10
        monkeypatch.delenv("MYSWAT_AGENTS_CLAUDE_REQUIRED_IP", raising=False)

        with pytest.raises(typer.BadParameter, match="claude_required_ip"):
            make_runner_from_row(row, settings=settings)

    @patch("myswat.agents.factory.CodexRunner")
    def test_extra_args_parsed_from_json(self, MockCodex):
        row = {
            "cli_backend": "codex",
            "cli_path": "codex",
            "model_name": "gpt-5",
            "cli_extra_args": json.dumps(["--verbose", "--timeout=60"]),
        }

        make_runner_from_row(row)

        MockCodex.assert_called_once_with(
            cli_path="codex",
            model="gpt-5",
            workdir=None,
            extra_flags=["--verbose", "--timeout=60"],
        )

    @patch("myswat.agents.factory.KimiRunner")
    def test_extra_args_parsed_from_list(self, MockKimi):
        row = {
            "cli_backend": "kimi",
            "cli_path": "kimi",
            "model_name": "k2",
            "cli_extra_args": ["--flag", "value"],
        }

        make_runner_from_row(row)

        MockKimi.assert_called_once_with(
            cli_path="kimi",
            model="k2",
            workdir=None,
            extra_flags=["--flag", "value"],
        )

    def test_unknown_backend_raises(self):
        row = {
            "cli_backend": "unknown",
            "cli_path": "foo",
            "model_name": "bar",
            "cli_extra_args": None,
        }

        with pytest.raises(typer.BadParameter, match="Unknown CLI backend"):
            make_runner_from_row(row)


class TestMakeMemoryWorkerRunner:
    @patch("myswat.agents.factory.CodexRunner")
    def test_uses_memory_worker_backend_and_agent_path(self, MockCodex):
        settings = MagicMock()
        settings.memory_worker.backend = "codex"
        settings.memory_worker.model = "gpt-5.4"
        settings.agents.codex_path = "/usr/bin/codex"
        settings.agents.codex_default_flags = ["--json"]

        make_memory_worker_runner(settings, workdir="/tmp/project")

        MockCodex.assert_called_once_with(
            cli_path="/usr/bin/codex",
            model="gpt-5.4",
            workdir="/tmp/project",
            extra_flags=["--json"],
        )

    @patch("myswat.agents.factory.ClaudeRunner")
    def test_claude_worker_uses_claude_settings(self, MockClaude):
        settings = MagicMock()
        settings.memory_worker.backend = "claude"
        settings.memory_worker.model = "claude-opus-4-6"
        settings.agents.claude_path = "claude"
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.claude_required_ip = "203.0.113.10"
        settings.agents.claude_ip_check_timeout_seconds = 7

        make_memory_worker_runner(settings)

        MockClaude.assert_called_once_with(
            cli_path="claude",
            model="claude-opus-4-6",
            workdir=None,
            extra_flags=["--print"],
            required_ip="203.0.113.10",
            ip_check_timeout_seconds=7,
        )
