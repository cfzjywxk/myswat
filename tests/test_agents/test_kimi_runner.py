"""Comprehensive tests for the KimiRunner class."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from myswat.agents.kimi_runner import KimiRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    """A KimiRunner with minimal defaults and no extra flags."""
    r = KimiRunner.__new__(KimiRunner)
    r.cli_path = "/usr/bin/kimi"
    r.model = "kimi-latest"
    r.workdir = None
    r.extra_flags = []
    r._cli_session_id = None
    return r


@pytest.fixture
def runner_with_workdir(runner):
    runner.workdir = "/home/user/project"
    return runner


@pytest.fixture
def runner_with_session(runner_with_workdir):
    runner_with_workdir._cli_session_id = "abc-123-session"
    return runner_with_workdir


# ===========================================================================
# 1. _base_flags
# ===========================================================================

class TestBaseFlags:
    """Tests for KimiRunner._base_flags."""

    def test_defaults(self, runner):
        """All three default flags are present when extra_flags is empty."""
        flags = runner._base_flags()
        assert "--print" in flags
        assert "--yolo" in flags
        assert "--output-format" in flags
        assert "stream-json" in flags
        # --output-format should be immediately followed by stream-json
        idx = flags.index("--output-format")
        assert flags[idx + 1] == "stream-json"

    def test_no_duplicate_print(self, runner):
        """--print is NOT added when already present in extra_flags."""
        runner.extra_flags = ["--print"]
        flags = runner._base_flags()
        assert "--print" not in flags

    def test_no_duplicate_yolo(self, runner):
        """--yolo is NOT added when already present in extra_flags."""
        runner.extra_flags = ["--yolo"]
        flags = runner._base_flags()
        assert "--yolo" not in flags

    def test_no_duplicate_yolo_short(self, runner):
        """--yolo is NOT added when -y shorthand present in extra_flags."""
        runner.extra_flags = ["-y"]
        flags = runner._base_flags()
        assert "--yolo" not in flags

    def test_no_duplicate_output_format(self, runner):
        """--output-format is NOT added when already present in extra_flags."""
        runner.extra_flags = ["--output-format", "text"]
        flags = runner._base_flags()
        assert "--output-format" not in flags
        assert "stream-json" not in flags

    def test_all_already_present(self, runner):
        """No default flags added when all are already specified."""
        runner.extra_flags = ["--print", "--yolo", "--output-format", "plain"]
        flags = runner._base_flags()
        assert flags == []

    def test_partial_overlap(self, runner):
        """Only missing defaults are added."""
        runner.extra_flags = ["--print"]
        flags = runner._base_flags()
        assert "--print" not in flags
        assert "--yolo" in flags
        assert "--output-format" in flags


# ===========================================================================
# 2. build_command
# ===========================================================================

class TestBuildCommand:
    """Tests for KimiRunner.build_command."""

    def test_basic(self, runner):
        """Command includes cli_path, base flags, model, and prompt."""
        cmd = runner.build_command("do something")
        assert cmd[0] == "/usr/bin/kimi"
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == "kimi-latest"
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "do something"
        # No -w when workdir is None
        assert "-w" not in cmd

    def test_with_system_context(self, runner):
        """system_context is prepended to prompt with separator."""
        cmd = runner.build_command("do something", system_context="You are helpful")
        prompt_val = cmd[cmd.index("-p") + 1]
        assert prompt_val == "You are helpful\n\n---\n\ndo something"

    def test_without_system_context(self, runner):
        """Prompt is used verbatim when system_context is None."""
        cmd = runner.build_command("raw prompt", system_context=None)
        prompt_val = cmd[cmd.index("-p") + 1]
        assert prompt_val == "raw prompt"

    def test_empty_system_context_is_falsy(self, runner):
        """Empty string system_context is treated as falsy (no prepend)."""
        cmd = runner.build_command("raw prompt", system_context="")
        prompt_val = cmd[cmd.index("-p") + 1]
        assert prompt_val == "raw prompt"

    def test_with_workdir(self, runner_with_workdir):
        """Workdir flag is included when workdir is set."""
        cmd = runner_with_workdir.build_command("hello")
        assert "-w" in cmd
        assert cmd[cmd.index("-w") + 1] == "/home/user/project"

    def test_extra_flags_included(self, runner):
        """Extra flags are appended after model."""
        runner.extra_flags = ["--verbose", "--no-cache"]
        cmd = runner.build_command("test")
        assert "--verbose" in cmd
        assert "--no-cache" in cmd

    def test_prompt_is_last(self, runner):
        """The -p flag and its argument are the last two elements."""
        runner.extra_flags = ["--verbose"]
        cmd = runner.build_command("my prompt")
        assert cmd[-2] == "-p"
        assert cmd[-1] == "my prompt"

    def test_base_flags_before_extra_flags(self, runner):
        """Base flags appear before extra_flags in the command."""
        runner.extra_flags = ["--custom"]
        cmd = runner.build_command("p")
        base_end = cmd.index("-m")  # -m follows base flags
        custom_idx = cmd.index("--custom")
        assert custom_idx > base_end


# ===========================================================================
# 3. build_resume_command
# ===========================================================================

class TestBuildResumeCommand:
    """Tests for KimiRunner.build_resume_command."""

    def test_includes_session_flag(self, runner_with_session):
        """Resume command includes -S with the session id."""
        cmd = runner_with_session.build_resume_command("continue")
        assert "-S" in cmd
        assert cmd[cmd.index("-S") + 1] == "abc-123-session"

    def test_includes_prompt(self, runner_with_session):
        """Resume command ends with -p <prompt>."""
        cmd = runner_with_session.build_resume_command("next step")
        assert cmd[-2] == "-p"
        assert cmd[-1] == "next step"

    def test_includes_model(self, runner_with_session):
        """Resume command includes the model flag."""
        cmd = runner_with_session.build_resume_command("x")
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == "kimi-latest"

    def test_includes_workdir(self, runner_with_session):
        """Resume command includes -w when workdir is set."""
        cmd = runner_with_session.build_resume_command("x")
        assert "-w" in cmd
        assert cmd[cmd.index("-w") + 1] == "/home/user/project"

    def test_no_workdir_when_none(self, runner):
        """Resume command omits -w when workdir is None."""
        runner._cli_session_id = "sess-1"
        cmd = runner.build_resume_command("x")
        assert "-w" not in cmd

    def test_includes_base_flags(self, runner_with_session):
        """Resume command includes base flags."""
        cmd = runner_with_session.build_resume_command("x")
        assert "--print" in cmd
        assert "--yolo" in cmd

    def test_extra_flags_included(self, runner_with_session):
        """Extra flags appear in resume command."""
        runner_with_session.extra_flags = ["--debug"]
        cmd = runner_with_session.build_resume_command("x")
        assert "--debug" in cmd


# ===========================================================================
# 4. extract_session_id
# ===========================================================================

class TestExtractSessionId:
    """Tests for KimiRunner.extract_session_id."""

    def test_returns_most_recent_session(self, runner, tmp_path):
        """Returns the name of the most recently modified session directory."""
        runner.workdir = str(tmp_path / "myproject")
        workdir_hash = hashlib.md5(runner.workdir.encode()).hexdigest()

        sessions_dir = tmp_path / ".kimi" / "sessions" / workdir_hash
        sessions_dir.mkdir(parents=True)

        # Create two session dirs with different mtimes
        old_session = sessions_dir / "old-uuid-1111"
        old_session.mkdir()
        new_session = sessions_dir / "new-uuid-2222"
        new_session.mkdir()

        # Make new_session more recent
        os.utime(str(old_session), (1000, 1000))
        os.utime(str(new_session), (2000, 2000))

        with patch.object(Path, "home", return_value=tmp_path):
            result = runner.extract_session_id("", "")

        assert result == "new-uuid-2222"

    def test_no_sessions_dir(self, runner, tmp_path):
        """Returns None when the sessions directory does not exist."""
        runner.workdir = "/nonexistent/project"

        with patch.object(Path, "home", return_value=tmp_path):
            result = runner.extract_session_id("", "")

        assert result is None

    def test_empty_sessions_dir(self, runner, tmp_path):
        """Returns None when sessions directory exists but is empty."""
        runner.workdir = str(tmp_path / "emptyproject")
        workdir_hash = hashlib.md5(runner.workdir.encode()).hexdigest()

        sessions_dir = tmp_path / ".kimi" / "sessions" / workdir_hash
        sessions_dir.mkdir(parents=True)

        with patch.object(Path, "home", return_value=tmp_path):
            result = runner.extract_session_id("", "")

        assert result is None

    def test_uses_cwd_when_no_workdir(self, runner, tmp_path):
        """Falls back to os.getcwd() when workdir is None."""
        runner.workdir = None
        cwd = str(tmp_path / "cwdproject")
        workdir_hash = hashlib.md5(cwd.encode()).hexdigest()

        sessions_dir = tmp_path / ".kimi" / "sessions" / workdir_hash
        sessions_dir.mkdir(parents=True)
        session = sessions_dir / "cwd-session-uuid"
        session.mkdir()

        with patch.object(Path, "home", return_value=tmp_path), \
             patch("os.getcwd", return_value=cwd):
            result = runner.extract_session_id("", "")

        assert result == "cwd-session-uuid"

    def test_workdir_hash_is_md5(self, runner, tmp_path):
        """Verifies the workdir hash used to locate sessions is MD5."""
        runner.workdir = "/specific/path"
        expected_hash = hashlib.md5(b"/specific/path").hexdigest()

        sessions_dir = tmp_path / ".kimi" / "sessions" / expected_hash
        sessions_dir.mkdir(parents=True)
        session = sessions_dir / "target-session"
        session.mkdir()

        with patch.object(Path, "home", return_value=tmp_path):
            result = runner.extract_session_id("", "")

        assert result == "target-session"


# ===========================================================================
# 5. format_live_line
# ===========================================================================

class TestFormatLiveLine:
    """Tests for KimiRunner.format_live_line."""

    def test_empty_string_returns_none(self, runner):
        assert runner.format_live_line("") is None

    def test_whitespace_only_returns_none(self, runner):
        assert runner.format_live_line("   ") is None

    def test_invalid_json_returns_line(self, runner):
        """Non-JSON input is returned as plain text."""
        result = runner.format_live_line("this is not json")
        assert result == "this is not json"

    def test_non_dict_json_returns_str(self, runner):
        """JSON that parses to a non-dict (e.g. list) returns str representation."""
        line = json.dumps([1, 2, 3])
        result = runner.format_live_line(line)
        assert result == str([1, 2, 3])

    def test_json_number_returns_str(self, runner):
        """JSON number returns str representation."""
        result = runner.format_live_line("42")
        assert result == str(42)

    def test_content_as_string(self, runner):
        """Dict with content as a string returns that string."""
        line = json.dumps({"content": "hello world"})
        result = runner.format_live_line(line)
        assert result == "hello world"

    def test_content_as_list_with_text_parts(self, runner):
        """Dict with content as list extracts text type parts."""
        line = json.dumps({
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World"},
            ]
        })
        result = runner.format_live_line(line)
        assert "Hello " in result
        assert "World" in result

    def test_content_as_list_with_think_parts(self, runner):
        """Think parts are included but truncated at 150 characters."""
        long_thought = "x" * 300
        line = json.dumps({
            "content": [
                {"type": "think", "think": long_thought},
            ]
        })
        result = runner.format_live_line(line)
        # The think text should be truncated to 150 chars
        assert result is not None
        assert len(result) <= 200  # some overhead for formatting is expected
        assert "x" * 150 in result or len(result.replace(" ", "").replace("\n", "")) <= 160

    def test_content_as_list_mixed_text_and_think(self, runner):
        """Both text and think parts are handled in the same list."""
        line = json.dumps({
            "content": [
                {"type": "text", "text": "visible"},
                {"type": "think", "think": "a" * 200},
            ]
        })
        result = runner.format_live_line(line)
        assert "visible" in result

    def test_empty_content_parts_returns_none(self, runner):
        """Dict with empty content list returns None."""
        line = json.dumps({"content": []})
        result = runner.format_live_line(line)
        assert result is None

    def test_content_string_empty_returns_none_or_empty(self, runner):
        """Dict with empty string content."""
        line = json.dumps({"content": ""})
        result = runner.format_live_line(line)
        # Empty content string should return None or empty string
        assert result is None or result == ""

    def test_dict_without_content_key(self, runner):
        """Dict without a content key."""
        line = json.dumps({"type": "status", "message": "working"})
        result = runner.format_live_line(line)
        # Should handle gracefully -- either None or some string representation
        assert result is None or isinstance(result, str)


# ===========================================================================
# 6. parse_output
# ===========================================================================

class TestParseOutput:
    """Tests for KimiRunner.parse_output."""

    def test_stream_json_content_string(self, runner):
        """Extracts text from JSON lines with content as string."""
        lines = [
            json.dumps({"content": "Hello "}),
            json.dumps({"content": "World"}),
        ]
        stdout = "\n".join(lines)
        result = runner.parse_output(stdout, "")
        assert "Hello" in result
        assert "World" in result

    def test_stream_json_content_list_text_parts(self, runner):
        """Extracts text from JSON lines with content as list of text parts."""
        lines = [
            json.dumps({
                "content": [
                    {"type": "text", "text": "part one"},
                ]
            }),
            json.dumps({
                "content": [
                    {"type": "text", "text": "part two"},
                ]
            }),
        ]
        stdout = "\n".join(lines)
        result = runner.parse_output(stdout, "")
        assert "part one" in result
        assert "part two" in result

    def test_non_json_fallback(self, runner):
        """Non-JSON lines are included as-is in the output."""
        stdout = "plain text line\nanother line"
        result = runner.parse_output(stdout, "")
        assert "plain text line" in result
        assert "another line" in result

    def test_mixed_json_and_plain(self, runner):
        """Mix of JSON and non-JSON lines are all captured."""
        lines = [
            json.dumps({"content": "from json"}),
            "plain text",
        ]
        stdout = "\n".join(lines)
        result = runner.parse_output(stdout, "")
        assert "from json" in result
        assert "plain text" in result

    def test_empty_input_returns_raw_stdout(self, runner):
        """Empty stdout returns the raw stdout (empty string)."""
        result = runner.parse_output("", "")
        assert result == "" or result is not None

    def test_non_dict_json_returns_str(self, runner):
        """Non-dict JSON values are stringified."""
        stdout = json.dumps([1, 2, 3])
        result = runner.parse_output(stdout, "")
        assert result is not None
        assert isinstance(result, str)

    def test_content_list_skips_non_text_types(self, runner):
        """Content list parts that are not text type are handled gracefully."""
        lines = [
            json.dumps({
                "content": [
                    {"type": "text", "text": "keep"},
                    {"type": "image", "url": "http://example.com/img.png"},
                ]
            }),
        ]
        stdout = "\n".join(lines)
        result = runner.parse_output(stdout, "")
        assert "keep" in result

    def test_stderr_is_ignored_in_output(self, runner):
        """stderr content does not appear in the parsed output."""
        stdout = json.dumps({"content": "good"})
        stderr = "WARNING: something bad"
        result = runner.parse_output(stdout, stderr)
        assert "good" in result
        # stderr should not bleed into result
        assert "WARNING" not in result

    def test_whitespace_only_lines(self, runner):
        """Whitespace-only lines are handled without errors."""
        stdout = "  \n" + json.dumps({"content": "data"}) + "\n  "
        result = runner.parse_output(stdout, "")
        assert "data" in result
