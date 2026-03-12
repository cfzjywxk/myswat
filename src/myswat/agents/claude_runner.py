"""Claude CLI subprocess wrapper with resume support and environment checks."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from collections.abc import Iterable

from myswat.agents.base import AgentRunner

_DEFAULT_REQUIRED_IP = "154.28.2.59"


class ClaudeEnvironmentError(RuntimeError):
    """Raised when the required Claude proxy/IP environment is not available."""


class ClaudeRunner(AgentRunner):
    """Wraps `claude -p` with explicit session IDs and `--resume`.

    First invoke():  claude --print --output-format stream-json --session-id <uuid> ...
    Subsequent:      claude --print --output-format stream-json --resume <uuid> ...

    Every subprocess launch validates that both HTTP and HTTPS proxies are set and
    that `curl ipinfo.io` reports the required outbound IP. If validation fails,
    Claude is not launched.
    """

    def __init__(
        self,
        cli_path: str,
        model: str,
        workdir: str | None = None,
        extra_flags: list[str] | None = None,
        timeout: int | None = None,
        required_ip: str = _DEFAULT_REQUIRED_IP,
        ip_check_timeout_seconds: int = 10,
    ) -> None:
        super().__init__(
            cli_path=cli_path,
            model=model,
            workdir=workdir,
            extra_flags=extra_flags,
            timeout=timeout,
        )
        self._requested_session_id: str | None = None
        self._required_ip = required_ip
        self._ip_check_timeout_seconds = ip_check_timeout_seconds

    def _base_flags(self) -> list[str]:
        flags: list[str] = []
        extra = self.extra_flags

        if "--print" not in extra and "-p" not in extra:
            flags.append("--print")
        if "--output-format" not in extra:
            flags.extend(["--output-format", "stream-json"])
        # MySwat runs Claude in non-interactive automation mode. The default
        # is to bypass Claude's permission prompts and rely on the outer
        # sandbox/proxy policy instead. Callers can override this explicitly
        # with --permission-mode or by supplying their own permission flag.
        if (
            "--dangerously-skip-permissions" not in extra
            and "--permission-mode" not in extra
        ):
            flags.append("--dangerously-skip-permissions")

        flags.extend(extra)
        return flags

    def reset_session(self) -> None:
        super().reset_session()
        self._requested_session_id = None

    def restore_session(self, session_id: str | None) -> None:
        super().restore_session(session_id)
        self._requested_session_id = None

    def invoke(self, prompt: str, system_context: str | None = None):
        self._validate_launch_environment()
        if self._cli_session_id is None and self._requested_session_id is None:
            self._requested_session_id = str(uuid.uuid4())

        response = super().invoke(prompt, system_context=system_context)
        if response.success and self._cli_session_id is not None:
            self._requested_session_id = None
        return response

    def build_command(self, prompt: str, system_context: str | None = None) -> list[str]:
        session_id = self._requested_session_id or str(uuid.uuid4())
        self._requested_session_id = session_id

        cmd = [self.cli_path]
        cmd.extend(self._base_flags())
        cmd.extend(["--model", self.model, "--session-id", session_id])

        if self.workdir:
            cmd.extend(["--add-dir", self.workdir])
        if system_context:
            cmd.extend(["--append-system-prompt", system_context])

        cmd.append(prompt)
        return cmd

    def build_resume_command(self, prompt: str) -> list[str]:
        cmd = [self.cli_path]
        cmd.extend(self._base_flags())
        cmd.extend(["--model", self.model, "--resume", self._cli_session_id])

        if self.workdir:
            cmd.extend(["--add-dir", self.workdir])

        cmd.append(prompt)
        return cmd

    def extract_session_id(self, stdout: str, stderr: str) -> str | None:
        if self._requested_session_id:
            return self._requested_session_id
        return self._cli_session_id

    def format_live_line(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

        if not isinstance(event, dict):
            return str(event).strip() or None

        etype = event.get("type")
        if etype == "assistant":
            content = self._extract_event_text(event)
            return content or None
        if etype == "result":
            subtype = event.get("subtype")
            if subtype and subtype != "success":
                return f"[result] {subtype}"
            return None
        if etype == "system":
            subtype = event.get("subtype")
            if subtype:
                return f"[system] {subtype}"
        return None

    def parse_output(self, stdout: str, stderr: str) -> str:
        result_texts: list[str] = []
        assistant_texts: list[str] = []

        for raw_line in stdout.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                assistant_texts.append(line)
                continue

            if not isinstance(event, dict):
                text = str(event).strip()
                if text:
                    assistant_texts.append(text)
                continue

            etype = event.get("type")
            if etype == "result":
                text = str(event.get("result", "")).strip()
                if text:
                    result_texts.append(text)
            elif etype == "assistant":
                text = self._extract_event_text(event)
                if text:
                    assistant_texts.append(text)

        if result_texts:
            return result_texts[-1]
        if assistant_texts:
            return assistant_texts[-1]
        return stdout.strip() or stderr.strip()

    @staticmethod
    def _extract_event_text(event: dict) -> str:
        message = event.get("message")
        if isinstance(message, dict):
            text = ClaudeRunner._extract_content_text(message.get("content"))
            if text:
                return text
            if isinstance(message.get("text"), str):
                return message["text"].strip()

        return ClaudeRunner._extract_content_text(event.get("content"))

    @staticmethod
    def _extract_content_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            if content.get("type") == "text":
                return str(content.get("text", "")).strip()
            return ""
        if isinstance(content, Iterable):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        text = str(part.get("text", "")).strip()
                        if text:
                            parts.append(text)
                    elif ptype == "tool_use":
                        name = str(part.get("name", "")).strip()
                        if name:
                            parts.append(f"[tool] {name}")
                elif isinstance(part, str):
                    text = part.strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    def _validate_launch_environment(self) -> None:
        http_proxy = self._first_env("http_proxy", "HTTP_PROXY")
        https_proxy = self._first_env("https_proxy", "HTTPS_PROXY")
        if not http_proxy or not https_proxy:
            raise ClaudeEnvironmentError(
                "Claude launch aborted: both http_proxy and https_proxy must be set."
            )

        try:
            result = subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--connect-timeout",
                    "5",
                    "--max-time",
                    str(self._ip_check_timeout_seconds),
                    "ipinfo.io",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            raise ClaudeEnvironmentError(
                "Claude launch aborted: curl is required for the outbound IP check."
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}"
            raise ClaudeEnvironmentError(
                f"Claude launch aborted: failed to verify outbound IP via curl ipinfo.io ({detail})."
            )

        actual_ip = self._parse_ipinfo_output(result.stdout)
        if actual_ip != self._required_ip:
            raise ClaudeEnvironmentError(
                "Claude launch aborted: curl ipinfo.io reported "
                f"{actual_ip or 'unknown'} instead of {self._required_ip}."
            )

    @staticmethod
    def _first_env(*names: str) -> str | None:
        for name in names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    @staticmethod
    def _parse_ipinfo_output(output: str) -> str | None:
        text = output.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text.splitlines()[0].strip()
        if isinstance(payload, dict):
            ip = payload.get("ip")
            if isinstance(ip, str):
                return ip.strip()
        return None
