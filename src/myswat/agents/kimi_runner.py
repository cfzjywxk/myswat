"""Kimi CLI subprocess wrapper with persistent session support."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from myswat.agents.base import AgentRunner


class KimiRunner(AgentRunner):
    """Wraps `kimi` CLI with session persistence via `kimi -S <session_id>`.

    First invoke():  kimi --print --yolo --final-message-only -m <model> -p <prompt>
                     → discovers session_id from ~/.kimi/sessions/<workdir_hash>/
    Subsequent:      kimi --print --yolo --final-message-only -m <model> -S <session_id> -p <prompt>
                     → resumes the same AI conversation
    """

    def _base_flags(self) -> list[str]:
        """Common flags for non-interactive mode.

        Uses stream-json output (not final-message-only) so we get live output.
        """
        flags = []
        if "--print" not in self.extra_flags:
            flags.append("--print")
        if "--yolo" not in self.extra_flags and "-y" not in self.extra_flags:
            flags.append("--yolo")
        # No --final-message-only: we want streaming output for live display
        if "--output-format" not in self.extra_flags:
            flags.extend(["--output-format", "stream-json"])
        return flags

    def build_command(self, prompt: str, system_context: str | None = None) -> list[str]:
        full_prompt = prompt
        if system_context:
            full_prompt = f"{system_context}\n\n---\n\n{prompt}"

        cmd = [self.cli_path]
        cmd.extend(self._base_flags())
        cmd.extend(["-m", self.model])

        if self.workdir:
            cmd.extend(["-w", self.workdir])

        cmd.extend(self.extra_flags)
        cmd.extend(["-p", full_prompt])
        return cmd

    def build_resume_command(self, prompt: str) -> list[str]:
        cmd = [self.cli_path]
        cmd.extend(self._base_flags())
        cmd.extend(["-m", self.model])

        if self.workdir:
            cmd.extend(["-w", self.workdir])

        # Resume the existing session
        cmd.extend(["-S", self._cli_session_id])

        cmd.extend(self.extra_flags)
        cmd.extend(["-p", prompt])
        return cmd

    def extract_session_id(self, stdout: str, stderr: str) -> str | None:
        """Discover the session ID from kimi's session directory.

        Kimi stores sessions at ~/.kimi/sessions/<md5(workdir)>/<session_uuid>/.
        We find the most recently modified session directory after the invocation.
        """
        workdir = self.workdir or os.getcwd()
        workdir_hash = hashlib.md5(workdir.encode()).hexdigest()
        sessions_dir = Path.home() / ".kimi" / "sessions" / workdir_hash

        if not sessions_dir.exists():
            return None

        # Find the most recently modified session
        sessions = sorted(
            sessions_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if sessions:
            return sessions[0].name
        return None

    def format_live_line(self, line: str) -> str | None:
        """Parse kimi stream-json events for live display."""
        if not line.strip():
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Plain text fallback
            return line if line.strip() else None

        if not isinstance(event, dict):
            return str(event).strip() or None

        # stream-json format: {"role":"assistant","content":[...]} or {"role":"assistant","content":"text"}
        content = event.get("content", "")
        if isinstance(content, str):
            return content if content.strip() else None
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "think":
                        think_text = part.get("think", "")
                        if think_text:
                            short = think_text[:150]
                            if len(think_text) > 150:
                                short += "..."
                            parts.append(f"[thinking] {short}")
            return "\n".join(parts) if parts else None
        return None

    def parse_output(self, stdout: str, stderr: str) -> str:
        """Extract the final assistant text from stream-json output."""
        final_text = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                final_text.append(line)
                continue

            if not isinstance(event, dict):
                s = str(event).strip()
                if s:
                    final_text.append(s)
                continue

            content = event.get("content", "")
            if isinstance(content, str):
                if content.strip():
                    final_text.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text.strip():
                            final_text.append(text)

        return "\n".join(final_text).strip() if final_text else stdout.strip()
