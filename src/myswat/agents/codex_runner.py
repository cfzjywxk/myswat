"""Codex CLI subprocess wrapper with persistent session support."""

from __future__ import annotations

import json

from myswat.agents.base import AgentRunner


class CodexRunner(AgentRunner):
    """Wraps `codex exec` with session persistence via `codex exec resume`.

    First invoke():  codex exec -m <model> --json --full-auto -- <prompt>
                     → extracts thread_id from {"type":"thread.started"} event
    Subsequent:      codex exec resume --json --full-auto <thread_id> -- <prompt>
                     → resumes the same AI conversation
    """

    def _base_flags(self) -> list[str]:
        """Common flags for both new and resume commands."""
        flags = []
        if "--full-auto" not in self.extra_flags:
            flags.append("--full-auto")
        if "--json" not in self.extra_flags:
            flags.append("--json")
        if "--skip-git-repo-check" not in self.extra_flags:
            flags.append("--skip-git-repo-check")
        flags.extend(self.extra_flags)
        return flags

    def build_command(self, prompt: str, system_context: str | None = None) -> list[str]:
        full_prompt = prompt
        if system_context:
            full_prompt = f"{system_context}\n\n---\n\n{prompt}"

        cmd = [self.cli_path, "exec", "-m", self.model]
        cmd.extend(self._base_flags())
        cmd.append("--")
        cmd.append(full_prompt)
        return cmd

    def build_resume_command(self, prompt: str) -> list[str]:
        cmd = [self.cli_path, "exec", "resume"]
        cmd.extend(self._base_flags())
        cmd.append(self._cli_session_id)
        cmd.append("--")
        cmd.append(prompt)
        return cmd

    def format_live_line(self, line: str) -> str | None:
        """Parse JSONL events and return human-readable live output."""
        if not line.strip():
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None

        etype = event.get("type", "")

        if etype == "item.completed":
            item = event.get("item", {})
            itype = item.get("type", "")
            if itype == "agent_message":
                return item.get("text", "")
            if itype == "command_execution":
                cmd_str = item.get("command", "")
                status = item.get("status", "")
                exit_code = item.get("exit_code")
                if cmd_str:
                    # Show abbreviated command
                    short_cmd = cmd_str[:120]
                    if len(cmd_str) > 120:
                        short_cmd += "..."
                    result = f"[cmd] {short_cmd}"
                    if exit_code is not None and exit_code != 0:
                        result += f" (exit={exit_code})"
                    return result
            if itype == "reasoning":
                text = item.get("text", "")
                if text:
                    short = text[:150]
                    if len(text) > 150:
                        short += "..."
                    return f"[thinking] {short}"
            return None

        if etype == "item.started":
            item = event.get("item", {})
            if item.get("type") == "command_execution":
                cmd_str = item.get("command", "")
                if cmd_str:
                    short_cmd = cmd_str[:120]
                    if len(cmd_str) > 120:
                        short_cmd += "..."
                    return f"[running] {short_cmd}"
            return None

        if etype == "turn.completed":
            usage = event.get("usage", {})
            if usage:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                cached = usage.get("cached_input_tokens", 0)
                return f"[tokens] input={inp} (cached={cached}) output={out}"
            return None

        # Skip thread.started, turn.started, etc.
        return None

    def extract_session_id(self, stdout: str, stderr: str) -> str | None:
        """Extract thread_id from the first JSONL event."""
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "thread.started":
                    return event.get("thread_id")
            except json.JSONDecodeError:
                continue
        return None

    def parse_output(self, stdout: str, stderr: str) -> str:
        return self._parse_jsonl(stdout)

    def _parse_jsonl(self, stdout: str) -> str:
        """Extract assistant message content from codex JSONL output."""
        agent_messages = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        agent_messages.append(text)

            elif etype == "message" and event.get("role") == "assistant":
                content = event.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            agent_messages.append(part.get("text", ""))
                elif isinstance(content, str) and content:
                    agent_messages.append(content)

        if agent_messages:
            return agent_messages[-1].strip()
        return stdout.strip()
