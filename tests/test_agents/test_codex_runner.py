"""Tests for CodexRunner agent."""

import json

import pytest

from myswat.agents.codex_runner import CodexRunner


@pytest.fixture
def runner():
    """Create a CodexRunner with default settings."""
    return CodexRunner(
        cli_path="/usr/bin/codex",
        model="o4-mini",
        extra_flags=[],
    )


@pytest.fixture
def runner_with_extra_flags():
    """Create a CodexRunner with extra flags that overlap defaults."""
    return CodexRunner(
        cli_path="/usr/bin/codex",
        model="o4-mini",
        extra_flags=["--full-auto", "--verbose"],
    )


# ---------------------------------------------------------------------------
# _base_flags
# ---------------------------------------------------------------------------


class TestBaseFlags:
    def test_default_flags_added(self, runner):
        flags = runner._base_flags()
        assert "--full-auto" in flags
        assert "--json" in flags
        assert "--skip-git-repo-check" in flags

    def test_no_duplicate_full_auto(self, runner_with_extra_flags):
        flags = runner_with_extra_flags._base_flags()
        assert flags.count("--full-auto") == 1

    def test_no_duplicate_json(self):
        r = CodexRunner(
            cli_path="/usr/bin/codex",
            model="o4-mini",
            extra_flags=["--json"],
        )
        flags = r._base_flags()
        assert flags.count("--json") == 1

    def test_no_duplicate_skip_git_repo_check(self):
        r = CodexRunner(
            cli_path="/usr/bin/codex",
            model="o4-mini",
            extra_flags=["--skip-git-repo-check"],
        )
        flags = r._base_flags()
        assert flags.count("--skip-git-repo-check") == 1

    def test_extra_flags_appended(self, runner_with_extra_flags):
        flags = runner_with_extra_flags._base_flags()
        assert "--verbose" in flags

    def test_extra_flags_preserved_in_order(self):
        r = CodexRunner(
            cli_path="/usr/bin/codex",
            model="o4-mini",
            extra_flags=["--aa", "--bb"],
        )
        flags = r._base_flags()
        aa_idx = flags.index("--aa")
        bb_idx = flags.index("--bb")
        assert aa_idx < bb_idx


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_basic_command(self, runner):
        cmd = runner.build_command("fix the bug")
        assert cmd[0] == "/usr/bin/codex"
        assert cmd[1] == "exec"
        assert "-m" in cmd
        model_idx = cmd.index("-m")
        assert cmd[model_idx + 1] == "o4-mini"
        assert "--" in cmd
        sep_idx = cmd.index("--")
        assert cmd[sep_idx + 1] == "fix the bug"

    def test_with_system_context(self, runner):
        cmd = runner.build_command("fix the bug", system_context="You are a coder")
        prompt = cmd[-1]
        assert "You are a coder" in prompt
        assert "fix the bug" in prompt
        assert "---" in prompt

    def test_without_system_context(self, runner):
        cmd = runner.build_command("fix the bug", system_context=None)
        assert cmd[-1] == "fix the bug"

    def test_empty_system_context(self, runner):
        cmd = runner.build_command("fix the bug", system_context="")
        # Empty string is falsy, so no context prepended
        assert cmd[-1] == "fix the bug"

    def test_base_flags_included(self, runner):
        cmd = runner.build_command("hello")
        assert "--full-auto" in cmd
        assert "--json" in cmd
        assert "--skip-git-repo-check" in cmd

    def test_extra_flags_included(self, runner_with_extra_flags):
        cmd = runner_with_extra_flags.build_command("hello")
        assert "--verbose" in cmd

    def test_system_context_format(self, runner):
        cmd = runner.build_command("do stuff", system_context="sys prompt")
        expected = "sys prompt\n\n---\n\ndo stuff"
        assert cmd[-1] == expected


# ---------------------------------------------------------------------------
# build_resume_command
# ---------------------------------------------------------------------------


class TestBuildResumeCommand:
    def test_resume_includes_session_id(self, runner):
        runner._cli_session_id = "thread_abc123"
        cmd = runner.build_resume_command("continue working")
        assert "resume" in cmd
        assert "thread_abc123" in cmd
        assert cmd[-1] == "continue working"

    def test_resume_structure(self, runner):
        runner._cli_session_id = "sess_xyz"
        cmd = runner.build_resume_command("next step")
        assert cmd[0] == "/usr/bin/codex"
        assert cmd[1] == "exec"
        assert cmd[2] == "resume"
        assert "--" in cmd
        sep_idx = cmd.index("--")
        assert cmd[sep_idx + 1] == "next step"
        # session_id should appear before the "--" separator
        sess_idx = cmd.index("sess_xyz")
        assert sess_idx < sep_idx

    def test_resume_includes_base_flags(self, runner):
        runner._cli_session_id = "sess_1"
        cmd = runner.build_resume_command("go")
        assert "--full-auto" in cmd
        assert "--json" in cmd
        assert "--skip-git-repo-check" in cmd


# ---------------------------------------------------------------------------
# format_live_line
# ---------------------------------------------------------------------------


class TestFormatLiveLine:
    def test_empty_string_returns_none(self, runner):
        assert runner.format_live_line("") is None

    def test_whitespace_returns_none(self, runner):
        assert runner.format_live_line("   ") is None
        assert runner.format_live_line("\n") is None
        assert runner.format_live_line("\t") is None

    def test_invalid_json_returns_none(self, runner):
        assert runner.format_live_line("not json at all") is None
        assert runner.format_live_line("{broken") is None

    def test_item_completed_agent_message(self, runner):
        event = {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "I fixed the bug.",
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "I fixed the bug." in result

    def test_item_completed_command_execution(self, runner):
        event = {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "grep -r 'TODO' src/",
                "exit_code": 0,
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "[cmd]" in result.lower() or "cmd" in result.lower()

    def test_item_completed_command_execution_truncation(self, runner):
        long_cmd = "x" * 500
        event = {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": long_cmd,
                "exit_code": 1,
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        # Result should be truncated (shorter than the full command)
        assert len(result) < len(long_cmd) + 50

    def test_item_completed_command_execution_exit_code(self, runner):
        event = {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "ls",
                "exit_code": 127,
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "127" in result

    def test_item_completed_reasoning(self, runner):
        event = {
            "type": "item.completed",
            "item": {
                "type": "reasoning",
                "text": "Let me think about this problem carefully...",
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "[thinking]" in result.lower() or "thinking" in result.lower()

    def test_item_completed_reasoning_truncation(self, runner):
        long_text = "thinking " * 200
        event = {
            "type": "item.completed",
            "item": {
                "type": "reasoning",
                "text": long_text,
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert len(result) < len(long_text) + 50

    def test_item_started_command_execution(self, runner):
        event = {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": "python test.py",
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "[running]" in result.lower() or "running" in result.lower()

    def test_turn_completed_with_usage(self, runner):
        event = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 300,
            },
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "[tokens]" in result.lower() or "tokens" in result.lower()

    def test_unknown_event_type_returns_none(self, runner):
        event = {"type": "unknown.event", "data": "stuff"}
        assert runner.format_live_line(json.dumps(event)) is None

    def test_event_without_type_returns_none(self, runner):
        event = {"data": "no type field"}
        assert runner.format_live_line(json.dumps(event)) is None


# ---------------------------------------------------------------------------
# extract_session_id
# ---------------------------------------------------------------------------


class TestExtractSessionId:
    def test_finds_thread_started_event(self, runner):
        lines = [
            json.dumps({"type": "item.started", "item": {"type": "agent_message"}}),
            json.dumps({"type": "thread.started", "thread_id": "thread_abc123"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
        ]
        stdout = "\n".join(lines)
        session_id = runner.extract_session_id(stdout, "")
        assert session_id == "thread_abc123"

    def test_no_thread_started_returns_none(self, runner):
        lines = [
            json.dumps({"type": "item.started", "item": {"type": "agent_message"}}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
        ]
        stdout = "\n".join(lines)
        session_id = runner.extract_session_id(stdout, "")
        assert session_id is None

    def test_invalid_json_lines_skipped(self, runner):
        lines = [
            "not valid json",
            "{also broken",
            json.dumps({"type": "thread.started", "thread_id": "thread_found"}),
        ]
        stdout = "\n".join(lines)
        session_id = runner.extract_session_id(stdout, "")
        assert session_id == "thread_found"

    def test_empty_stdout(self, runner):
        session_id = runner.extract_session_id("", "")
        assert session_id is None

    def test_uses_first_thread_started(self, runner):
        lines = [
            json.dumps({"type": "thread.started", "thread_id": "first_id"}),
            json.dumps({"type": "thread.started", "thread_id": "second_id"}),
        ]
        stdout = "\n".join(lines)
        session_id = runner.extract_session_id(stdout, "")
        assert session_id == "first_id"


# ---------------------------------------------------------------------------
# parse_output / _parse_jsonl
# ---------------------------------------------------------------------------


class TestParseJsonl:
    def test_extracts_last_agent_message(self, runner):
        lines = [
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "First message"},
            }),
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Second message"},
            }),
        ]
        stdout = "\n".join(lines)
        result = runner._parse_jsonl(stdout)
        assert result == "Second message"

    def test_parse_output_delegates_to_parse_jsonl(self, runner):
        lines = [
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "Hello"},
            }),
        ]
        stdout = "\n".join(lines)
        assert runner.parse_output(stdout, "") == runner._parse_jsonl(stdout)

    def test_message_type_with_list_content_and_output_text(self, runner):
        event = {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Result from list content"},
            ],
        }
        stdout = json.dumps(event)
        result = runner._parse_jsonl(stdout)
        assert "Result from list content" in result

    def test_message_type_with_string_content(self, runner):
        event = {
            "type": "message",
            "role": "assistant",
            "content": "Plain string content",
        }
        stdout = json.dumps(event)
        result = runner._parse_jsonl(stdout)
        assert "Plain string content" in result

    def test_no_agent_messages_returns_raw_stdout(self, runner):
        lines = [
            json.dumps({"type": "item.started", "item": {"type": "command_execution"}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100}}),
        ]
        stdout = "\n".join(lines)
        result = runner._parse_jsonl(stdout)
        assert result == stdout

    def test_empty_input(self, runner):
        result = runner._parse_jsonl("")
        assert result == ""

    def test_skips_non_agent_message_items(self, runner):
        lines = [
            json.dumps({
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "ls", "exit_code": 0},
            }),
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "The only message"},
            }),
            json.dumps({
                "type": "item.completed",
                "item": {"type": "reasoning", "text": "thinking..."},
            }),
        ]
        stdout = "\n".join(lines)
        result = runner._parse_jsonl(stdout)
        assert result == "The only message"

    def test_message_type_non_assistant_ignored(self, runner):
        lines = [
            json.dumps({
                "type": "message",
                "role": "user",
                "content": "User message should be ignored",
            }),
        ]
        stdout = "\n".join(lines)
        result = runner._parse_jsonl(stdout)
        # No assistant message found, should return raw stdout
        assert result == stdout

    def test_mixed_item_completed_and_message_types(self, runner):
        lines = [
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "From item.completed"},
            }),
            json.dumps({
                "type": "message",
                "role": "assistant",
                "content": "From message type",
            }),
        ]
        stdout = "\n".join(lines)
        result = runner._parse_jsonl(stdout)
        # Should return the last agent message found
        assert result is not None
        assert len(result) > 0

    # -- Additional coverage for edge cases in format_live_line --

    def test_item_started_non_command_returns_none(self, runner):
        """item.started event with non-command_execution type should return None."""
        event = {"type": "item.started", "item": {"type": "agent_message"}}
        assert runner.format_live_line(json.dumps(event)) is None

    def test_item_started_command_execution_long_command(self, runner):
        """item.started with command longer than 120 chars should truncate."""
        long_cmd = "x" * 200
        event = {
            "type": "item.started",
            "item": {"type": "command_execution", "command": long_cmd},
        }
        result = runner.format_live_line(json.dumps(event))
        assert result is not None
        assert "[running]" in result
        assert "..." in result
        assert len(result) < 200

    def test_item_started_command_execution_empty_command(self, runner):
        """item.started with empty command should return None."""
        event = {
            "type": "item.started",
            "item": {"type": "command_execution", "command": ""},
        }
        assert runner.format_live_line(json.dumps(event)) is None

    def test_turn_completed_no_usage(self, runner):
        """turn.completed with no usage dict should return None."""
        event = {"type": "turn.completed"}
        assert runner.format_live_line(json.dumps(event)) is None

    def test_turn_completed_empty_usage(self, runner):
        """turn.completed with empty usage dict should return None."""
        event = {"type": "turn.completed", "usage": {}}
        assert runner.format_live_line(json.dumps(event)) is None

    # -- Additional coverage for _parse_jsonl edge cases --

    def test_parse_jsonl_skips_blank_lines(self, runner):
        """Blank lines in JSONL should be skipped without error."""
        lines = [
            "",
            "  ",
            json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "hello",
                },
            }),
            "",
        ]
        result = runner._parse_jsonl("\n".join(lines))
        assert result == "hello"

    def test_parse_jsonl_skips_invalid_json(self, runner):
        """Invalid JSON lines should be skipped."""
        lines = [
            "{broken json",
            "not json at all",
            json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": "valid",
                },
            }),
        ]
        result = runner._parse_jsonl("\n".join(lines))
        assert result == "valid"

    def test_extract_session_id_skips_blank_lines(self, runner):
        """extract_session_id should skip blank lines."""
        stdout = "\n\n" + json.dumps({"type": "thread.started", "thread_id": "t-123"}) + "\n"
        assert runner.extract_session_id(stdout, "") == "t-123"
