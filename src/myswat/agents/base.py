"""AgentRunner ABC and AgentResponse dataclass."""

from __future__ import annotations

import signal
import subprocess
import threading
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
    the same session via CLI-specific mechanisms (codex exec resume / kimi -S).

    Live output: stdout lines are accumulated in `live_output` as the subprocess
    runs. Callers can read this from another thread to display progress.

    Call reset_session() to force a fresh AI session (e.g., /reset command).
    """

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

    def reset_session(self) -> None:
        """Clear the CLI session ID, forcing a fresh AI session on next invoke()."""
        self._cli_session_id = None
        self._turn_count = 0

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
        """Kill the running subprocess (called on ESC)."""
        proc = self._process
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except OSError:
                pass

    def invoke(self, prompt: str, system_context: str | None = None) -> AgentResponse:
        """Run the agent CLI subprocess and return the parsed response.

        Stdout is read line-by-line and accumulated in live_output for
        real-time display by the caller.
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

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.workdir,
            )

            # Read stderr in a background thread so it doesn't block
            def _read_stderr():
                for line in self._process.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            # Read stdout line by line — feeds live_output
            for line in self._process.stdout:
                stdout_lines.append(line)
                formatted = self.format_live_line(line.rstrip("\n"))
                if formatted is not None:
                    with self._live_lock:
                        self._live_lines.append(formatted)

            self._process.wait(timeout=self.timeout)
            stderr_thread.join(timeout=5)
            returncode = self._process.returncode

        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            return AgentResponse(
                content=f"Agent timed out after {self.timeout}s",
                exit_code=-1,
                raw_stdout="".join(stdout_lines),
                raw_stderr="".join(stderr_lines),
            )
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
