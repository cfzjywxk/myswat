"""Tests for myswat.agents.base — AgentResponse and AgentRunner."""

import signal
import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

from myswat.agents.base import AgentResponse, AgentRunner


# ---------------------------------------------------------------------------
# Concrete stub so we can instantiate the ABC
# ---------------------------------------------------------------------------


class _FakeRunner(AgentRunner):
    """Minimal concrete implementation of AgentRunner for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_build_command_args = None
        self.last_build_resume_command_args = None
        self._fake_extract_session_id = None  # set to a value to simulate

    def build_command(self, prompt, system_context=None):
        self.last_build_command_args = (prompt, system_context)
        return [self.cli_path, "--model", self.model, "--prompt", prompt]

    def build_resume_command(self, prompt):
        self.last_build_resume_command_args = (prompt,)
        return [self.cli_path, "--resume", self._cli_session_id, "--prompt", prompt]

    def parse_output(self, stdout, stderr):
        return stdout.strip()


# ---------------------------------------------------------------------------
# Helper to build a mock Popen that the real invoke() can iterate over
# ---------------------------------------------------------------------------

# We keep a reference to the real Popen class BEFORE any patching happens,
# so MagicMock(spec=...) never accidentally specs against a mock.
_REAL_POPEN = subprocess.Popen


def _make_mock_popen(stdout_text="output\n", stderr_text="", returncode=0):
    """Create a MagicMock that behaves enough like Popen for invoke()."""
    proc = MagicMock(spec=_REAL_POPEN)
    # invoke() iterates proc.stdout / proc.stderr line by line
    proc.stdout = iter(stdout_text.splitlines(keepends=True))
    proc.stderr = iter(stderr_text.splitlines(keepends=True))
    proc.wait.return_value = returncode
    proc.returncode = returncode
    proc.poll.return_value = returncode
    proc.pid = 12345
    return proc


# ===================================================================
# AgentResponse
# ===================================================================


class TestAgentResponse:
    """Tests for the AgentResponse dataclass."""

    # -- defaults -------------------------------------------------------

    def test_defaults(self):
        resp = AgentResponse(content="hello")
        assert resp.content == "hello"
        assert resp.exit_code == 0
        assert resp.raw_stdout == ""
        assert resp.raw_stderr == ""
        assert resp.token_usage == {}
        assert resp.cancelled is False

    def test_token_usage_default_is_independent_per_instance(self):
        """Each instance should get its own dict, not share a mutable default."""
        r1 = AgentResponse(content="a")
        r2 = AgentResponse(content="b")
        r1.token_usage["x"] = 1
        assert "x" not in r2.token_usage

    # -- success property -----------------------------------------------

    def test_success_true_when_exit_zero_and_not_cancelled(self):
        resp = AgentResponse(content="ok", exit_code=0, cancelled=False)
        assert resp.success is True

    def test_success_false_when_exit_nonzero(self):
        resp = AgentResponse(content="fail", exit_code=1)
        assert resp.success is False

    def test_success_false_when_cancelled(self):
        resp = AgentResponse(content="cancelled", exit_code=0, cancelled=True)
        assert resp.success is False

    def test_success_false_when_both_nonzero_and_cancelled(self):
        resp = AgentResponse(content="bad", exit_code=2, cancelled=True)
        assert resp.success is False

    def test_success_false_negative_exit_code(self):
        resp = AgentResponse(content="neg", exit_code=-1)
        assert resp.success is False


# ===================================================================
# AgentRunner — initialisation & properties
# ===================================================================


class TestAgentRunnerInit:
    """Tests for AgentRunner.__init__ default values."""

    def test_stores_required_args(self):
        runner = _FakeRunner(cli_path="/usr/bin/agent", model="gpt-4")
        assert runner.cli_path == "/usr/bin/agent"
        assert runner.model == "gpt-4"

    def test_workdir_defaults_to_none(self):
        runner = _FakeRunner(cli_path="agent", model="m")
        assert runner.workdir is None

    def test_extra_flags_none_becomes_empty_list(self):
        runner = _FakeRunner(cli_path="agent", model="m", extra_flags=None)
        assert runner.extra_flags == []

    def test_extra_flags_preserved_when_given(self):
        flags = ["--verbose", "--json"]
        runner = _FakeRunner(cli_path="agent", model="m", extra_flags=flags)
        assert runner.extra_flags is flags

    def test_timeout_defaults_to_none(self):
        runner = _FakeRunner(cli_path="agent", model="m")
        assert runner.timeout is None

    def test_timeout_stored(self):
        runner = _FakeRunner(cli_path="agent", model="m", timeout=30)
        assert runner.timeout == 30

    def test_internal_state_defaults(self):
        runner = _FakeRunner(cli_path="agent", model="m")
        assert runner._process is None
        assert runner._cli_session_id is None
        assert runner._turn_count == 0
        assert runner._live_lines == []
        assert isinstance(runner._live_lock, type(threading.Lock()))


# ===================================================================
# AgentRunner — properties
# ===================================================================


class TestAgentRunnerProperties:

    def test_cli_session_id_initially_none(self):
        runner = _FakeRunner("a", "m")
        assert runner.cli_session_id is None

    def test_is_session_started_false_initially(self):
        runner = _FakeRunner("a", "m")
        assert runner.is_session_started is False

    def test_is_session_started_true_after_setting_id(self):
        runner = _FakeRunner("a", "m")
        runner._cli_session_id = "sess-1"
        assert runner.is_session_started is True

    def test_cli_session_id_reflects_internal_value(self):
        runner = _FakeRunner("a", "m")
        runner._cli_session_id = "xyz"
        assert runner.cli_session_id == "xyz"

    def test_live_output_returns_copy(self):
        runner = _FakeRunner("a", "m")
        runner._live_lines = ["line1", "line2"]
        output = runner.live_output
        assert output == ["line1", "line2"]
        # Mutating the returned list must not affect internal state
        output.append("line3")
        assert runner._live_lines == ["line1", "line2"]

    def test_clear_live_output(self):
        runner = _FakeRunner("a", "m")
        runner._live_lines = ["line1", "line2"]
        runner.clear_live_output()
        assert runner._live_lines == []
        assert runner.live_output == []

    def test_live_output_thread_safety(self):
        """Verify live_output acquires the lock by trying to get it while held."""
        runner = _FakeRunner("a", "m")
        runner._live_lines = ["x"]
        results = []

        def try_read():
            # This will block if the lock is properly used, until we release it
            out = runner.live_output
            results.append(out)

        # Hold the lock, start a thread that calls live_output, verify it blocks
        with runner._live_lock:
            t = threading.Thread(target=try_read)
            t.start()
            t.join(timeout=0.1)
            # Thread should still be alive (blocked on the lock)
            assert t.is_alive(), "live_output should block while lock is held"

        # After releasing the lock, the thread should finish
        t.join(timeout=2)
        assert not t.is_alive()
        assert results == [["x"]]


# ===================================================================
# AgentRunner — reset_session
# ===================================================================


class TestAgentRunnerResetSession:

    def test_clears_session_id(self):
        runner = _FakeRunner("a", "m")
        runner._cli_session_id = "sess-42"
        runner.reset_session()
        assert runner._cli_session_id is None

    def test_clears_turn_count(self):
        runner = _FakeRunner("a", "m")
        runner._turn_count = 5
        runner.reset_session()
        assert runner._turn_count == 0


# ===================================================================
# AgentRunner — format_live_line / extract_session_id defaults
# ===================================================================


class TestAgentRunnerDefaultMethods:

    def test_format_live_line_returns_line_unchanged(self):
        runner = _FakeRunner("a", "m")
        assert runner.format_live_line("hello world") == "hello world"
        assert runner.format_live_line("") == ""

    def test_extract_session_id_returns_none(self):
        runner = _FakeRunner("a", "m")
        assert runner.extract_session_id("stdout text", "stderr text") is None


# ===================================================================
# AgentRunner — cancel
# ===================================================================


class TestAgentRunnerCancel:

    def test_cancel_no_process(self):
        """cancel() should be a no-op when _process is None."""
        runner = _FakeRunner("a", "m")
        runner._process = None
        runner.cancel()  # should not raise

    def test_cancel_process_already_terminated(self):
        """cancel() should be a no-op when process has already exited."""
        runner = _FakeRunner("a", "m")
        proc = MagicMock(spec=_REAL_POPEN)
        proc.poll.return_value = 0  # already finished
        runner._process = proc
        runner.cancel()
        proc.send_signal.assert_not_called()

    def test_cancel_kills_immediately(self):
        runner = _FakeRunner("a", "m")
        proc = MagicMock(spec=_REAL_POPEN)
        proc.poll.return_value = None  # still running
        runner._process = proc
        runner.cancel()
        proc.kill.assert_called_once()

    def test_cancel_ignores_oserror(self):
        runner = _FakeRunner("a", "m")
        proc = MagicMock(spec=_REAL_POPEN)
        proc.poll.return_value = None
        proc.kill.side_effect = OSError("no such process")
        runner._process = proc
        runner.cancel()  # should not raise


# ===================================================================
# AgentRunner — invoke
# ===================================================================


class TestAgentRunnerInvoke:

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_successful_run(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="result text\n", returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "model-1")
        resp = runner.invoke("do something")

        assert isinstance(resp, AgentResponse)
        assert resp.exit_code == 0
        assert resp.content == "result text"
        assert resp.cancelled is False
        assert resp.success is True

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_increments_turn_count(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        assert runner._turn_count == 0
        runner.invoke("prompt")
        assert runner._turn_count == 1

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_uses_build_command_on_first_call(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        assert runner._cli_session_id is None
        runner.invoke("hello", system_context="ctx")
        assert runner.last_build_command_args == ("hello", "ctx")
        assert runner.last_build_resume_command_args is None

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_uses_resume_command_when_session_exists(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner._cli_session_id = "existing-session"
        runner.invoke("follow up")
        assert runner.last_build_resume_command_args == ("follow up",)
        # build_command should NOT have been called
        assert runner.last_build_command_args is None

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_clears_live_lines(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="new line\n", returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner._live_lines = ["old line"]
        runner.invoke("prompt")
        # Old lines should be gone; only new lines (or empty) should remain
        assert "old line" not in runner._live_lines

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_populates_live_output(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="line1\nline2\n", returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner.invoke("prompt")
        assert runner.live_output == ["line1", "line2"]

    @patch("myswat.agents.base.time.monotonic")
    @patch("myswat.agents.base.time.sleep")
    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_timeout_returns_exit_neg1(self, mock_popen_cls, mock_sleep, mock_monotonic):
        """Watchdog kills stalled process; invoke returns exit_code=-1."""
        killed = threading.Event()

        # Stdout yields one line then blocks, simulating a stalled process.
        # When the watchdog calls proc.kill(), the event unblocks the iterator.
        def blocking_stdout():
            yield "partial\n"
            killed.wait(timeout=5)

        proc = _make_mock_popen(stdout_text="partial\n", returncode=-9)
        proc.stdout = blocking_stdout()
        # Use function-based side effects to avoid race with thread scheduling
        proc.poll.side_effect = lambda: -9 if killed.is_set() else None
        proc.wait.return_value = -9
        proc.kill.side_effect = lambda: killed.set()
        mock_popen_cls.return_value = proc

        # monotonic: returns 0 for first 3 calls, then 999 to trigger stall
        mono_calls = 0

        def monotonic_fn():
            nonlocal mono_calls
            mono_calls += 1
            return 999 if mono_calls > 3 else 0

        mock_monotonic.side_effect = monotonic_fn

        runner = _FakeRunner("agent", "m", timeout=10)
        resp = runner.invoke("slow prompt")

        assert resp.exit_code == -1
        assert "stall" in resp.content.lower()
        proc.kill.assert_called_once()

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_file_not_found(self, mock_popen_cls):
        mock_popen_cls.side_effect = FileNotFoundError("No such file")

        runner = _FakeRunner("/nonexistent/agent", "m")
        resp = runner.invoke("prompt")

        assert resp.exit_code == -2
        assert "not found" in resp.content.lower()
        assert "/nonexistent/agent" in resp.raw_stderr

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_negative_returncode_sets_cancelled(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="", stderr_text="", returncode=-15)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        resp = runner.invoke("prompt")

        assert resp.cancelled is True
        assert resp.exit_code == -15
        assert resp.success is False

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_negative_returncode_does_not_increment_turn_count(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=-9)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner.invoke("prompt")
        assert runner._turn_count == 0

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_extracts_session_id_on_first_invoke(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="session: abc-123\nresult\n", returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner.extract_session_id = MagicMock(return_value="abc-123")

        runner.invoke("first prompt")

        runner.extract_session_id.assert_called_once()
        assert runner._cli_session_id == "abc-123"

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_does_not_overwrite_session_id_on_subsequent(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner._cli_session_id = "already-set"
        runner.extract_session_id = MagicMock(return_value="new-id")

        runner.invoke("second prompt")

        # Session ID should remain unchanged because it was already set
        assert runner._cli_session_id == "already-set"
        # extract_session_id should NOT be called when session already exists
        runner.extract_session_id.assert_not_called()

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_passes_workdir_as_cwd_to_popen(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m", workdir="/tmp/project")
        runner.invoke("prompt")

        mock_popen_cls.assert_called_once()
        call_kwargs = mock_popen_cls.call_args
        assert call_kwargs.kwargs.get("cwd") == "/tmp/project" or \
            (len(call_kwargs) > 1 and call_kwargs[1].get("cwd") == "/tmp/project")

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_opens_popen_with_pipe_and_text(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner.invoke("prompt")

        call_kwargs = mock_popen_cls.call_args.kwargs
        assert call_kwargs.get("stdin") == subprocess.DEVNULL
        assert call_kwargs.get("stdout") == subprocess.PIPE
        assert call_kwargs.get("stderr") == subprocess.PIPE
        assert call_kwargs.get("text") is True
        assert call_kwargs.get("encoding") == "utf-8"
        assert call_kwargs.get("errors") == "replace"
        assert call_kwargs.get("bufsize") == 1

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_clears_process_after_completion(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner.invoke("prompt")
        # _process should be None after invoke completes (set in finally block)
        assert runner._process is None

    @patch("myswat.agents.base.time.monotonic")
    @patch("myswat.agents.base.time.sleep")
    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_clears_process_after_timeout(self, mock_popen_cls, mock_sleep, mock_monotonic):
        killed = threading.Event()

        def blocking_stdout():
            yield "partial\n"
            killed.wait(timeout=5)

        proc = _make_mock_popen(stdout_text="partial\n", returncode=-9)
        proc.stdout = blocking_stdout()
        proc.poll.side_effect = lambda: -9 if killed.is_set() else None
        proc.wait.return_value = -9
        proc.kill.side_effect = lambda: killed.set()
        mock_popen_cls.return_value = proc

        mono_calls = 0

        def monotonic_fn():
            nonlocal mono_calls
            mono_calls += 1
            return 999 if mono_calls > 3 else 0

        mock_monotonic.side_effect = monotonic_fn

        runner = _FakeRunner("agent", "m", timeout=5)
        runner.invoke("prompt")
        assert runner._process is None

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_clears_process_after_file_not_found(self, mock_popen_cls):
        mock_popen_cls.side_effect = FileNotFoundError("nope")

        runner = _FakeRunner("/bad", "m")
        runner.invoke("prompt")
        assert runner._process is None

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_passes_raw_stdout_and_stderr(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="out1\nout2\n", stderr_text="err\n", returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        resp = runner.invoke("prompt")

        assert resp.raw_stdout == "out1\nout2\n"
        assert resp.raw_stderr == "err\n"

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_nonzero_exit_code(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="error output\n", returncode=1)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        resp = runner.invoke("prompt")

        assert resp.exit_code == 1
        assert resp.cancelled is False
        assert resp.success is False

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_session_id_not_extracted_on_nonzero_exit(self, mock_popen_cls):
        proc = _make_mock_popen(returncode=1)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        runner.extract_session_id = MagicMock(return_value="should-not-set")

        runner.invoke("prompt")

        # extract_session_id should not be called when returncode != 0
        runner.extract_session_id.assert_not_called()
        assert runner._cli_session_id is None

    @patch("myswat.agents.base.subprocess.Popen")
    def test_invoke_format_live_line_filters_none(self, mock_popen_cls):
        proc = _make_mock_popen(stdout_text="keep\nskip\nkeep2\n", returncode=0)
        mock_popen_cls.return_value = proc

        runner = _FakeRunner("agent", "m")
        # Override format_live_line to filter "skip" lines
        original_format = runner.format_live_line
        runner.format_live_line = lambda line: None if "skip" in line else line

        runner.invoke("prompt")

        assert runner.live_output == ["keep", "keep2"]


# ===================================================================
# Integration-style: full round-trip through _FakeRunner
# ===================================================================


class TestFakeRunnerRoundTrip:
    """Verify _FakeRunner's abstract method implementations work together."""

    def test_build_command_includes_prompt(self):
        runner = _FakeRunner("cli", "model-x")
        cmd = runner.build_command("summarize this")
        assert "summarize this" in cmd

    def test_build_resume_command_includes_session_id(self):
        runner = _FakeRunner("cli", "model-x")
        runner._cli_session_id = "sess-99"
        cmd = runner.build_resume_command("continue")
        assert "sess-99" in cmd
        assert "continue" in cmd

    def test_parse_output_creates_stripped_content(self):
        runner = _FakeRunner("cli", "model-x")
        result = runner.parse_output("  hello  ", "warn")
        assert result == "hello"
