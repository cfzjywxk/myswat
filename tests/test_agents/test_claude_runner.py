"""Tests for the ClaudeRunner."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from myswat.agents.base import AgentResponse
from myswat.agents.claude_runner import ClaudeEnvironmentError, ClaudeRunner


@pytest.fixture
def runner():
    return ClaudeRunner(
        cli_path="/usr/bin/claude",
        model="claude-sonnet-4-6",
        extra_flags=[],
        required_ip="154.28.2.59",
    )


class TestBaseFlags:
    def test_default_flags_added(self, runner):
        flags = runner._base_flags()
        assert "--print" in flags
        assert "--output-format" in flags
        assert "stream-json" in flags
        assert "--verbose" in flags
        assert "--dangerously-skip-permissions" in flags

    def test_does_not_duplicate_overrides(self, runner):
        runner.extra_flags = ["-p", "--output-format", "json", "--permission-mode", "auto"]
        flags = runner._base_flags()
        assert "--print" not in flags
        assert flags.count("--output-format") == 1
        assert flags[flags.index("--output-format") + 1] == "json"
        assert "--dangerously-skip-permissions" not in flags

    def test_verbose_not_duplicated_when_in_extra_flags(self, runner):
        runner.extra_flags = ["--verbose"]
        flags = runner._base_flags()
        assert flags.count("--verbose") == 1


class TestBuildCommand:
    def test_build_command_sets_session_id(self, runner):
        runner.workdir = "/tmp/project"
        cmd = runner.build_command("fix it", system_context="You are helpful")
        assert cmd[0] == "/usr/bin/claude"
        assert "--session-id" in cmd
        assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-6"
        assert cmd[cmd.index("--add-dir") + 1] == "/tmp/project"
        assert cmd[cmd.index("--append-system-prompt") + 1] == "You are helpful"
        assert cmd[-1] == "fix it"
        assert runner._requested_session_id is not None

    def test_build_resume_command_uses_resume(self, runner):
        runner.restore_session("b6984f9a-78eb-4f15-9ed8-cde0fd9980b1")
        cmd = runner.build_resume_command("continue")
        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == "b6984f9a-78eb-4f15-9ed8-cde0fd9980b1"
        assert "--session-id" not in cmd
        assert cmd[-1] == "continue"

    def test_build_resume_command_includes_workdir(self, runner):
        runner.workdir = "/tmp/project"
        runner.restore_session("b6984f9a-78eb-4f15-9ed8-cde0fd9980b1")
        cmd = runner.build_resume_command("continue")
        assert "--add-dir" in cmd
        assert cmd[cmd.index("--add-dir") + 1] == "/tmp/project"


class TestOutputParsing:
    def test_format_live_line_assistant(self, runner):
        line = (
            '{"type":"assistant","message":{"content":['
            '{"type":"text","text":"Investigating"},'
            '{"type":"tool_use","name":"Bash"}]}}'
        )
        assert runner.format_live_line(line) == "Investigating\n[tool] Bash"

    def test_parse_output_prefers_result(self, runner):
        stdout = "\n".join(
            [
                '{"type":"assistant","message":{"content":[{"type":"text","text":"draft"}]}}',
                '{"type":"result","subtype":"success","result":"final answer"}',
            ]
        )
        assert runner.parse_output(stdout, "") == "final answer"

    def test_parse_output_falls_back_to_raw_stderr(self, runner):
        assert runner.parse_output("", "stderr only") == "stderr only"

    def test_extract_session_id_uses_requested_id(self, runner):
        runner._requested_session_id = "4cf7f2c8-2558-433c-a6fc-a29680e2b2a5"
        assert runner.extract_session_id("", "") == "4cf7f2c8-2558-433c-a6fc-a29680e2b2a5"


class TestEnvironmentValidation:
    def test_missing_proxies_raises(self, runner):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ClaudeEnvironmentError, match="http_proxy and https_proxy"):
                runner._validate_launch_environment()

    @patch("myswat.agents.claude_runner.subprocess.run")
    def test_wrong_ip_raises(self, mock_run, runner):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"ip":"1.2.3.4"}'
        mock_run.return_value.stderr = ""

        with patch.dict(
            "os.environ",
            {"http_proxy": "http://proxy", "https_proxy": "http://proxy"},
            clear=True,
        ):
            with pytest.raises(ClaudeEnvironmentError, match="1.2.3.4 instead of 154.28.2.59"):
                runner._validate_launch_environment()

    @patch("myswat.agents.claude_runner.subprocess.run")
    def test_accepts_ipinfo_json(self, mock_run, runner):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"ip":"154.28.2.59"}'
        mock_run.return_value.stderr = ""

        with patch.dict(
            "os.environ",
            {"HTTP_PROXY": "http://proxy", "HTTPS_PROXY": "http://proxy"},
            clear=True,
        ):
            runner._validate_launch_environment()

    @patch("myswat.agents.base.AgentRunner.invoke", autospec=True)
    def test_invoke_checks_environment_each_time(self, mock_super_invoke, runner):
        mock_super_invoke.return_value = AgentResponse(content="ok", exit_code=0)

        with patch.object(runner, "_validate_launch_environment") as mock_validate:
            runner.invoke("hello")
            runner.invoke("again")

        assert mock_validate.call_count == 2
