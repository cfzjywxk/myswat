"""Session lifecycle manager with TiDB persistence and persistent AI sessions."""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING, Callable

import pymysql.err

from myswat.agents.base import AgentResponse
from myswat.config.settings import MySwatSettings
from myswat.large_payloads import (
    AGENT_FILE_PROMPT,
    maybe_externalize_prompt,
    maybe_externalize_response,
    maybe_externalize_system_context,
)
from myswat.memory.learn_triggers import submit_session_summary_learn_request
from myswat.memory.retriever import MemoryRetriever

if TYPE_CHECKING:
    from myswat.agents.base import AgentRunner
    from myswat.memory.store import MemoryStore
    from myswat.models.session import Session

logger = logging.getLogger(__name__)
_MEMORY_LAYER_ERRORS = (pymysql.err.Error, OSError)


class SessionManager:
    """Manages the lifecycle of a single agent session with TiDB persistence.

    Architecture:
    - The underlying AI CLI (codex/kimi/claude) maintains its OWN persistent session.
    - The first send() builds TiDB context and injects it as system context.
    - Subsequent send() calls resume the SAME AI session — no context rebuild.
    - The AI remembers all prior conversation naturally via its own session state.
    - TiDB is used for: persistence, cross-session knowledge, progress tracking,
      compaction — NOT for rebuilding context every turn.

    Lifecycle: create_or_resume() -> send() [repeated] -> close()
    Reset:     reset_ai_session() clears the CLI session, next send() starts fresh.
    """

    def __init__(
        self,
        store: MemoryStore,
        runner: AgentRunner,
        agent_row: dict,
        project_id: int,
        settings: MySwatSettings | None = None,
    ) -> None:
        self._store = store
        self._runner = runner
        self._agent_row = agent_row
        self._project_id = project_id
        self._settings = settings or MySwatSettings()
        self._retriever = MemoryRetriever(store)
        self._session: Session | None = None
        self._memory_revision_warned = False

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def agent_role(self) -> str:
        return self._agent_row["role"]

    @property
    def agent_id(self) -> int:
        return self._agent_row["id"]

    def create_or_resume(
        self, purpose: str | None = None, work_item_id: int | None = None,
    ) -> Session:
        """Create a new session or resume an active one for this agent."""
        existing = self._store.get_active_session(self._agent_row["id"], work_item_id=work_item_id)
        if existing:
            self._session = existing
            self._memory_revision_warned = False
            self._restore_cli_session(existing.id)
            return existing

        session = self._store.create_session(
            agent_id=self._agent_row["id"],
            purpose=purpose,
            work_item_id=work_item_id,
        )
        self._session = session
        self._memory_revision_warned = False
        return session

    def fork_for_work_item(
        self, work_item_id: int, purpose: str | None = None,
    ) -> "SessionManager":
        """Create a fresh work-item-scoped TiDB session while reusing the runner.

        The original SessionManager and its chat session stay untouched. The
        forked manager always creates a new TiDB session linked to
        ``work_item_id`` instead of resuming an existing one, but it shares the
        same underlying AI runner so in-process conversation context is
        preserved.
        """
        if isinstance(work_item_id, bool) or not isinstance(work_item_id, int):
            raise TypeError("work_item_id must be an integer")
        if work_item_id <= 0:
            raise ValueError("work_item_id must be a positive integer")

        forked = SessionManager(
            store=self._store,
            runner=self._runner,
            agent_row=self._agent_row,
            project_id=self._project_id,
            settings=self._settings,
        )
        parent_session_id = self._session.id if self._session is not None else None
        forked._session = self._store.create_session(
            agent_id=self._agent_row["id"],
            purpose=purpose,
            work_item_id=work_item_id,
            parent_session_id=parent_session_id,
        )
        return forked

    def _restore_cli_session(self, session_id: int) -> None:
        """Restore the last persisted CLI session ID for a resumed TiDB session."""
        try:
            turns = self._store.get_session_turns(session_id)
        except Exception:
            return

        for turn in reversed(turns):
            if turn.role != "assistant":
                continue
            metadata = turn.metadata_json
            if not isinstance(metadata, dict):
                continue
            cli_session_id = metadata.get("cli_session_id")
            if isinstance(cli_session_id, str) and cli_session_id:
                self._runner.restore_session(cli_session_id)
                return

    def _maybe_prefix_memory_revision_hint(self, prompt: str) -> str:
        if self._session is None or self._memory_revision_warned:
            return prompt
        built_revision = getattr(self._session, "memory_revision_at_context_build", None)
        if built_revision is None:
            return prompt
        try:
            current_revision = self._store.get_project_memory_revision(self._project_id)
        except _MEMORY_LAYER_ERRORS as exc:
            logger.warning("[session memory] Revision check failed: %s", exc)
            return prompt
        if current_revision <= built_revision:
            return prompt
        self._memory_revision_warned = True
        note = (
            "[myswat note: project knowledge changed since this session started. "
            "Start a fresh chat or workflow run if you need fully refreshed recall.]\n\n"
        )
        return note + prompt

    def reset_ai_session(self) -> None:
        """Reset the underlying AI CLI session.

        Next send() will start a fresh AI session with full TiDB context reload.
        The TiDB session (turns, knowledge) is preserved — only the AI's internal
        conversation state is reset.
        """
        self._runner.reset_session()

    # Maximum stall retries before giving up.
    _MAX_STALL_RETRIES = 3

    def _build_system_context_and_track_revision(
        self, task_description: str | None, prompt: str,
    ) -> str | None:
        """Build system context from agent prompt + TiDB knowledge.

        Also updates memory revision bookkeeping on the TiDB session so
        stale-knowledge warnings work correctly after a retry.
        """
        parts: list[str] = []
        parts.append(AGENT_FILE_PROMPT)
        agent_system_prompt = self._agent_row.get("system_prompt")
        if agent_system_prompt:
            parts.append(agent_system_prompt)
        context = None
        try:
            context = self._retriever.build_context_for_agent(
                project_id=self._project_id,
                agent_id=self._agent_row["id"],
                agent_role=self._agent_row.get("role"),
                task_description=task_description or prompt,
                current_session_id=self._session.id,
                repo_path=self._runner.workdir,
            )
        except _MEMORY_LAYER_ERRORS as exc:
            logger.warning("[session memory] Context build failed: %s", exc)

        try:
            current_memory_revision = self._store.get_project_memory_revision(self._project_id)
            self._store.set_session_memory_revision(self._session.id, current_memory_revision)
            if self._session is not None:
                self._session.memory_revision_at_context_build = current_memory_revision
        except _MEMORY_LAYER_ERRORS as exc:
            logger.warning("[session memory] Revision tracking failed: %s", exc)

        self._memory_revision_warned = False
        if context:
            parts.append(context)
        return "\n\n---\n\n".join(parts) if parts else None

    def send(
        self,
        prompt: str,
        task_description: str | None = None,
        status_callback: Callable[[str, dict[str, object]], None] | None = None,
    ) -> AgentResponse:
        """Send a prompt to the agent, persisting turns to TiDB.

        First turn (no AI session yet):
            1. Build context from TiDB (knowledge + history)
            2. Invoke AI with context as system prompt → starts AI session
            3. AI remembers this context for all subsequent turns

        Subsequent turns (AI session active):
            1. Just send the prompt — AI already has all context
            2. No TiDB context rebuild (the AI remembers naturally)

        Stall handling:
            If the agent stalls (exit_code == -1, not cancelled), retry up to
            _MAX_STALL_RETRIES times with increasing timeout. Each retry resets
            the AI session and rebuilds system context from TiDB.

        Always:
            - Save user + assistant turns to TiDB
            - Update progress note
        """
        if self._session is None:
            self.create_or_resume(purpose=task_description)

        # Build system context ONLY for the first turn of an AI session.
        system_context = None
        if not self._runner.is_session_started:
            system_context = self._build_system_context_and_track_revision(
                task_description, prompt,
            )

        prompt_to_send = (
            self._maybe_prefix_memory_revision_hint(prompt)
            if self._runner.is_session_started
            else prompt
        )

        stored_prompt, stored_prompt_path = maybe_externalize_prompt(
            prompt_to_send,
            label=f"{self.agent_role}-request",
        )
        user_metadata = {"externalized_prompt_path": stored_prompt_path} if stored_prompt_path else None

        # Save user turn
        token_est = len(prompt) // 4
        self._store.append_turn(
            session_id=self._session.id,
            role="user",
            content=stored_prompt,
            token_count_est=token_est,
            metadata=user_metadata,
        )

        # Invoke agent with stall retry loop
        original_timeout = self._runner.timeout
        t0 = time.monotonic()

        for attempt in range(1, self._MAX_STALL_RETRIES + 1):
            sent_prompt, _ = maybe_externalize_prompt(
                prompt_to_send,
                label=f"{self.agent_role}-request",
            )
            sent_system_context = None
            if system_context is not None:
                sent_system_context, _ = maybe_externalize_system_context(
                    system_context,
                    label=f"{self.agent_role}-context",
                )

            response = self._runner.invoke(
                sent_prompt,
                system_context=sent_system_context,
            )

            is_stall = response.exit_code == -1 and not response.cancelled
            if not is_stall:
                break  # success, explicit error, or user cancel

            timeout_seconds = self._runner.timeout or original_timeout
            if status_callback is not None:
                try:
                    status_callback(
                        "agent_stalled",
                        {
                            "attempt": attempt,
                            "max_attempts": self._MAX_STALL_RETRIES,
                            "timeout": timeout_seconds,
                        },
                    )
                except Exception:
                    pass

            print(
                f"[myswat] Agent stalled (attempt {attempt}/{self._MAX_STALL_RETRIES})",
                file=sys.stderr,
            )

            if attempt >= self._MAX_STALL_RETRIES:
                break  # exhausted retries

            # Reset AI session and rebuild context for fresh retry
            self._runner.reset_session()
            system_context = self._build_system_context_and_track_revision(
                task_description, prompt,
            )
            prompt_to_send = prompt  # fresh session, no revision hint needed

            # Backoff: increase stall timeout for next attempt
            # attempt 1 → 1.5x, attempt 2 → 2x of base timeout
            next_timeout = None
            if original_timeout:
                next_timeout = int(original_timeout * (1 + attempt * 0.5))
                self._runner.timeout = next_timeout
            if status_callback is not None:
                try:
                    status_callback(
                        "agent_retry",
                        {
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "max_attempts": self._MAX_STALL_RETRIES,
                            "next_timeout": next_timeout,
                        },
                    )
                except Exception:
                    pass

        # Restore original timeout after retries
        if original_timeout:
            self._runner.timeout = original_timeout

        elapsed = time.monotonic() - t0

        # Save assistant turn with timing
        assistant_token_est = len(response.content) // 4
        stored_response, stored_response_path = maybe_externalize_response(
            response.content,
            label=f"{self.agent_role}-response",
        )
        assistant_metadata = {
            "exit_code": response.exit_code,
            "token_usage": response.token_usage,
            "elapsed_seconds": round(elapsed, 1),
            "cancelled": response.cancelled,
            "cli_session_id": self._runner.cli_session_id,
        }
        if stored_response_path:
            assistant_metadata["externalized_response_path"] = stored_response_path
        self._store.append_turn(
            session_id=self._session.id,
            role="assistant",
            content=stored_response,
            token_count_est=assistant_token_est,
            metadata=assistant_metadata,
        )

        # Update progress note
        self._update_progress(prompt, response, elapsed)

        return response

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    def _update_progress(self, prompt: str, response: AgentResponse, elapsed: float) -> None:
        """Write a brief progress note to the session (visible via `myswat status`)."""
        if self._session is None:
            return
        try:
            turn_count = self._store.count_session_turns(self._session.id)
            dur = self._fmt_duration(elapsed)
            first_line = prompt.strip().split("\n")[0][:100]
            if response.cancelled:
                note = f"[turn {turn_count}, {dur}] CANCELLED: {first_line}"
            elif response.success:
                agent_summary = ""
                for line in response.content.strip().split("\n"):
                    line = line.strip().lstrip("#*->• ")
                    if len(line) > 10:
                        agent_summary = line[:100]
                        break
                note = f"[turn {turn_count}, {dur}] {first_line}"
                if agent_summary:
                    note += f" → {agent_summary}"
            else:
                note = f"[turn {turn_count}, {dur}] ERROR: {first_line}"
            self._store.update_session_progress(self._session.id, note[:512])
        except Exception:
            pass  # progress update is best-effort

    def _submit_session_summary(self) -> None:
        if self._session is None:
            return
        try:
            turn_count = self._store.count_session_turns(self._session.id)
        except Exception as exc:
            print(f"[session learn] Failed: {exc}", file=sys.stderr)
            return
        if turn_count < 2:
            return

        try:
            submit_session_summary_learn_request(
                store=self._store,
                settings=self._settings,
                project_id=self._project_id,
                source_session_id=self._session.id,
                source_work_item_id=getattr(self._session, "work_item_id", None),
                agent_role=self.agent_role,
                purpose=getattr(self._session, "purpose", None),
                workdir=self._runner.workdir,
                payload_json={"turn_count": turn_count},
                asynchronous=False,
            )
        except Exception as e:
            print(f"[session learn] Failed: {e}", file=sys.stderr)

    def close(self) -> None:
        """Close the session and submit a best-effort unified learn summary."""
        if self._session is None:
            return

        self._store.close_session(self._session.id)
        self._submit_session_summary()
