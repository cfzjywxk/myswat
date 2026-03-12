"""Session lifecycle manager with TiDB persistence and persistent AI sessions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from myswat.agents.base import AgentResponse
from myswat.memory.retriever import MemoryRetriever

if TYPE_CHECKING:
    from myswat.agents.base import AgentRunner
    from myswat.memory.compactor import KnowledgeCompactor
    from myswat.memory.store import MemoryStore
    from myswat.models.session import Session


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
        compactor: KnowledgeCompactor | None = None,
    ) -> None:
        self._store = store
        self._runner = runner
        self._agent_row = agent_row
        self._project_id = project_id
        self._compactor = compactor
        self._retriever = MemoryRetriever(store)
        self._session: Session | None = None

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
            self._restore_cli_session(existing.id)
            return existing

        session = self._store.create_session(
            agent_id=self._agent_row["id"],
            purpose=purpose,
            work_item_id=work_item_id,
        )
        self._session = session
        return session

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

    def reset_ai_session(self) -> None:
        """Reset the underlying AI CLI session.

        Next send() will start a fresh AI session with full TiDB context reload.
        The TiDB session (turns, knowledge) is preserved — only the AI's internal
        conversation state is reset.
        """
        self._runner.reset_session()

    def send(self, prompt: str, task_description: str | None = None) -> AgentResponse:
        """Send a prompt to the agent, persisting turns to TiDB.

        First turn (no AI session yet):
            1. Build context from TiDB (knowledge + history)
            2. Invoke AI with context as system prompt → starts AI session
            3. AI remembers this context for all subsequent turns

        Subsequent turns (AI session active):
            1. Just send the prompt — AI already has all context
            2. No TiDB context rebuild (the AI remembers naturally)

        Always:
            - Save user + assistant turns to TiDB
            - Update progress note
            - Check mid-session compaction
        """
        if self._session is None:
            self.create_or_resume(purpose=task_description)

        # Mid-session compaction: distill old turns into knowledge
        self._check_mid_session_compaction()

        # Build system context ONLY for the first turn of an AI session.
        # Once the AI session is started, it remembers everything — no need
        # to rebuild context from TiDB every turn.
        system_context = None
        if not self._runner.is_session_started:
            parts = []
            # Agent's own system prompt (role instructions, delegation rules, etc.)
            agent_system_prompt = self._agent_row.get("system_prompt")
            if agent_system_prompt:
                parts.append(agent_system_prompt)
            # TiDB knowledge + session context
            context = self._retriever.build_context_for_agent(
                project_id=self._project_id,
                agent_id=self._agent_row["id"],
                task_description=task_description or prompt,
                current_session_id=self._session.id,
                repo_path=self._runner.workdir,
            )
            if context:
                parts.append(context)
            system_context = "\n\n---\n\n".join(parts) if parts else None

        # Save user turn
        token_est = len(prompt) // 4
        self._store.append_turn(
            session_id=self._session.id,
            role="user",
            content=prompt,
            token_count_est=token_est,
        )

        # Invoke agent (first call starts session, subsequent calls resume)
        t0 = time.monotonic()
        response = self._runner.invoke(prompt, system_context=system_context)
        elapsed = time.monotonic() - t0

        # Save assistant turn with timing
        assistant_token_est = len(response.content) // 4
        self._store.append_turn(
            session_id=self._session.id,
            role="assistant",
            content=response.content,
            token_count_est=assistant_token_est,
            metadata={
                "exit_code": response.exit_code,
                "token_usage": response.token_usage,
                "elapsed_seconds": round(elapsed, 1),
                "cancelled": response.cancelled,
                "cli_session_id": self._runner.cli_session_id,
            },
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

    def _check_mid_session_compaction(self) -> None:
        """Compact the current session mid-flight if it exceeds thresholds."""
        if self._session is None or self._compactor is None:
            return
        if not self._compactor.should_compact(self._session.id):
            return

        try:
            ids = self._compactor.compact_session(
                session_id=self._session.id,
                project_id=self._project_id,
                agent_id=self._agent_row["id"],
                mark_compacted=False,
            )
            deleted = self._store.delete_compacted_turns(self._session.id)
            self._store.reset_session_token_count(self._session.id)

            if ids or deleted:
                import sys
                print(
                    f"[mid-session compaction] {len(ids)} knowledge entries created, "
                    f"{deleted} old turns deleted "
                    f"(session {self._session.session_uuid[:8]})",
                    file=sys.stderr,
                )
        except Exception as e:
            import sys
            print(f"[mid-session compaction] Failed: {e}", file=sys.stderr)

    def close(self) -> None:
        """Close the session, run final compaction, and clean up turns."""
        if self._session is None:
            return

        self._store.close_session(self._session.id)

        if self._compactor and self._compactor.should_compact(self._session.id):
            self._compact()

        if self._store.get_session(self._session.id):
            session = self._store.get_session(self._session.id)
            if session and session.get("status") == "compacted":
                self._store.delete_archived_session(self._session.id)

    def _compact(self) -> None:
        """Run knowledge compaction on the closed session, then delete turns."""
        if self._session is None or self._compactor is None:
            return
        try:
            ids = self._compactor.compact_session(
                session_id=self._session.id,
                project_id=self._project_id,
                agent_id=self._agent_row["id"],
                mark_compacted=True,
            )
            if ids:
                import sys
                print(f"[compaction] Created {len(ids)} knowledge entries, session archived", file=sys.stderr)
        except Exception as e:
            import sys
            print(f"[compaction] Failed: {e}", file=sys.stderr)
