"""AgentRunner ABC and AgentResponse dataclass."""

from __future__ import annotations

import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentResponse:
    content: str
    exit_code: int = 0
    raw_stdout: str = ""
    raw_stderr: str = ""
    token_usage: dict = field(default_factory=dict)
    cancelled: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.cancelled


class AgentRunner(ABC):
    """Base class for CLI-based agent runners.

    Supports persistent AI sessions and live output streaming.
    The first invoke() starts a new CLI session, subsequent calls resume
    the same session via CLI-specific mechanisms (codex exec resume /
    kimi -S / claude --resume).

    Live output: stdout lines are accumulated in `live_output` as the subprocess
    runs. Callers can read this from another thread to display progress.

    Call reset_session() to force a fresh AI session (e.g., /reset command).
    """

    # Streaming output formats that produce JSONL events during work.
    _STREAMING_FORMATS = frozenset({"stream-json", "json"})

    def __init__(
        self,
        cli_path: str,
        model: str,
        workdir: str | None = None,
        extra_flags: list[str] | None = None,
        timeout: int | None = None,
    ) -> None:
        self.cli_path = cli_path
        self.model = model
        self.workdir = workdir
        self.extra_flags = extra_flags or []
        self.timeout = timeout  # None = no limit
        self._process: subprocess.Popen | None = None
        self._cli_session_id: str | None = None
        self._turn_count: int = 0
        # Live output buffer — read from display thread, written from reader thread
        self._live_lines: list[str] = []
        self._live_lock = threading.Lock()

    @property
    def cli_session_id(self) -> str | None:
        return self._cli_session_id

    @property
    def is_session_started(self) -> bool:
        return self._cli_session_id is not None

    @property
    def live_output(self) -> list[str]:
        """Get a snapshot of live output lines accumulated so far."""
        with self._live_lock:
            return list(self._live_lines)

    def clear_live_output(self) -> None:
        """Clear the live output buffer.

        Called before starting a new Live display to prevent stale lines
        from the previous invocation from being rendered.
        """
        with self._live_lock:
            self._live_lines.clear()

    @property
    def supports_activity_monitoring(self) -> bool:
        """Whether this runner produces streaming stdout suitable for stall detection.

        Returns False if extra_flags include --final-message-only or set
        --output-format to a non-streaming format (e.g. text).
        """
        if "--final-message-only" in self.extra_flags:
            return False
        # CodexRunner uses --json (always streaming)
        if "--json" in self.extra_flags:
            return True
        try:
            idx = self.extra_flags.index("--output-format")
            fmt = self.extra_flags[idx + 1]
            return fmt in self._STREAMING_FORMATS
        except (ValueError, IndexError):
            # No explicit --output-format in extra_flags.
            # All current runners add a streaming format in _base_flags().
            return True

    def reset_session(self) -> None:
        """Clear the CLI session ID, forcing a fresh AI session on next invoke()."""
        self._cli_session_id = None
        self._turn_count = 0

    def restore_session(self, session_id: str | None) -> None:
        """Restore a previously persisted CLI session ID."""
        self._cli_session_id = session_id

    @abstractmethod
    def build_command(self, prompt: str, system_context: str | None = None) -> list[str]:
        """Build the CLI command for the FIRST turn (new session)."""

    @abstractmethod
    def build_resume_command(self, prompt: str) -> list[str]:
        """Build the CLI command for subsequent turns (resume session)."""

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> str:
        """Extract the assistant's response from raw CLI output."""

    def format_live_line(self, line: str) -> str | None:
        """Convert a raw stdout line into a human-readable live output line.

        Override in subclasses to parse JSONL events, filter noise, etc.
        Return None to skip the line (not displayed).
        Default: return the line as-is.
        """
        return line

    def extract_session_id(self, stdout: str, stderr: str) -> str | None:
        """Extract the CLI session ID from the first invoke's output."""
        return None

    def cancel(self) -> None:
        """Kill the running subprocess immediately (called on ESC / Ctrl+C)."""
        proc = self._process
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def invoke(self, prompt: str, system_context: str | None = None) -> AgentResponse:
        """Run the agent CLI subprocess and return the parsed response.

        Stdout is read line-by-line and accumulated in live_output for
        real-time display by the caller.

        When a timeout is configured, a watchdog daemon thread monitors
        activity and kills the process if it stalls:
        - Streaming runners: killed if no stdout/stderr line for `timeout` seconds.
        - Non-streaming runners: killed if total wall-clock time exceeds `timeout`.
        """
        if self._cli_session_id is not None:
            cmd = self.build_resume_command(prompt)
        else:
            cmd = self.build_command(prompt, system_context)

        # Clear live output buffer
        with self._live_lock:
            self._live_lines.clear()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stalled = False
        proc: subprocess.Popen | None = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.workdir,
            )
            self._process = proc

            # Watchdog thread — monitors stdout/stderr activity or wall-clock time
            last_activity = time.monotonic()
            start_time = last_activity

            # Read stderr in a background thread so it doesn't block.
            # Stderr output also counts as activity for stall detection.
            def _read_stderr():
                nonlocal last_activity
                if proc is None or proc.stderr is None:
                    return
                for line in proc.stderr:
                    last_activity = time.monotonic()
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()
            use_activity_monitoring = self.supports_activity_monitoring

            if self.timeout:
                def _watchdog():
                    nonlocal stalled
                    if proc is None:
                        return
                    while proc.poll() is None:
                        now = time.monotonic()
                        if use_activity_monitoring:
                            elapsed_since_activity = now - last_activity
                        else:
                            elapsed_since_activity = now - start_time
                        if elapsed_since_activity > self.timeout:
                            stalled = True
                            try:
                                proc.kill()
                            except OSError:
                                pass
                            return
                        time.sleep(10)

                watcher = threading.Thread(target=_watchdog, daemon=True)
                watcher.start()
            else:
                watcher = None

            # Read stdout line by line — feeds live_output
            if proc.stdout is not None:
                for line in proc.stdout:
                    last_activity = time.monotonic()
                    stdout_lines.append(line)
                    formatted = self.format_live_line(line.rstrip("\n"))
                    if formatted is not None:
                        with self._live_lock:
                            self._live_lines.append(formatted)

            proc.wait()
            stderr_thread.join(timeout=5)
            if watcher is not None:
                watcher.join(timeout=2)
            returncode = proc.returncode

        except FileNotFoundError:
            return AgentResponse(
                content=f"CLI not found: {self.cli_path}",
                exit_code=-2,
                raw_stderr=f"FileNotFoundError: {self.cli_path}",
            )
        finally:
            self._process = None

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        if stalled:
            monitor_type = "no output" if use_activity_monitoring else "wall-clock"
            return AgentResponse(
                content=f"Agent stalled ({monitor_type} timeout): no progress for {self.timeout}s",
                exit_code=-1,
                raw_stdout=stdout,
                raw_stderr=stderr,
            )

        if returncode < 0:
            return AgentResponse(
                content="Request cancelled.",
                exit_code=returncode,
                raw_stdout=stdout,
                raw_stderr=stderr,
                cancelled=True,
            )

        if self._cli_session_id is None and returncode == 0:
            sid = self.extract_session_id(stdout, stderr)
            if sid:
                self._cli_session_id = sid

        self._turn_count += 1

        content = self.parse_output(stdout, stderr)
        return AgentResponse(
            content=content,
            exit_code=returncode,
            raw_stdout=stdout,
            raw_stderr=stderr,
        )
