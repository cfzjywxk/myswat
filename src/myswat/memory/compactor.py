"""KnowledgeCompactor — distills raw session logs into structured knowledge."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from myswat.memory.store import MemoryStore

if TYPE_CHECKING:
    from myswat.agents.base import AgentRunner

COMPACTION_PROMPT = """You are a knowledge extraction specialist. Given the following
conversation transcript between a user and an AI agent, extract the key knowledge items.

For each item, output a JSON object with:
- "category": one of "decision", "architecture", "pattern", "protocol", "invariant",
  "failure_mode", "bug_fix", "configuration", "api_reference", "performance_tuning",
  "review_feedback", "progress", "lesson_learned"
- "title": concise title (max 100 chars)
- "content": detailed knowledge content (1-3 paragraphs)
- "tags": list of relevant tags (max 5)
- "relevance_score": float 0.0-1.0 (how important/reusable is this)
- "confidence": float 0.0-1.0 (how reliable is this knowledge)

Output ONLY a JSON array of these objects. No other text before or after.
Only include genuinely useful knowledge — skip pleasantries, clarification questions,
and routine back-and-forth. If the conversation has no useful knowledge, output [].
Prefer stable, reusable technical knowledge over chatty summaries. When possible, anchor
items to concrete file paths, symbols, subsystems, config keys, or operational invariants.

TRANSCRIPT:
{transcript}
"""


def parse_compaction_output(raw_output: str) -> tuple[list[dict[str, Any]], bool]:
    """Parse the AI agent's JSON output into knowledge items.

    Returns ``(items, ok)`` where ``ok`` distinguishes a valid empty array
    from malformed or missing JSON output.
    """
    text = raw_output.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("["):
                text = part
                break

    # Try to find a JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        items = json.loads(text)
        if isinstance(items, list):
            return items, True
    except json.JSONDecodeError:
        pass

    return [], False


class KnowledgeCompactor:
    """Distills raw session transcripts into structured knowledge entries.

    Flow:
    1. Build transcript from session_turns
    2. Invoke an AI agent with the compaction prompt
    3. Parse structured JSON output
    4. Store entries in knowledge table with embeddings
    5. Mark session as 'compacted'
    """

    def __init__(
        self,
        store: MemoryStore,
        runner: AgentRunner | None = None,
        threshold_turns: int = 50,
        threshold_tokens: int = 5000,
    ) -> None:
        self._store = store
        self._runner = runner
        self._threshold_turns = threshold_turns
        self._threshold_tokens = threshold_tokens

    def should_compact(self, session_id: int) -> bool:
        """Check if uncompacted turns have exceeded the turn threshold."""
        session = self._store.get_session(session_id)
        if not session:
            return False
        if session.get("status") == "compacted":
            return False
        uncompacted = self._store.count_uncompacted_turns(session_id)
        return uncompacted >= self._threshold_turns

    @staticmethod
    def _build_transcript_from_turns(
        turns: list[Any], max_chars: int = 100000,
    ) -> tuple[str, list[Any]]:
        """Build a transcript and return the turns actually included."""
        lines = []
        included_turns = []
        total = 0
        for turn in turns:
            line = f"[{turn.role.upper()}]: {turn.content}"
            if total + len(line) > max_chars:
                lines.append("[... transcript truncated ...]")
                break
            lines.append(line)
            included_turns.append(turn)
            total += len(line)
        return "\n\n".join(lines), included_turns

    def build_transcript(self, session_id: int, max_chars: int = 100000) -> str:
        """Build a text transcript from session turns."""
        turns = self._store.get_session_turns(session_id)
        transcript, _included_turns = self._build_transcript_from_turns(turns, max_chars=max_chars)
        return transcript

    def compact_session(
        self, session_id: int, project_id: int, agent_id: int | None = None,
        mark_compacted: bool = True,
    ) -> list[int]:
        """Compact a session into knowledge entries.

        Args:
            mark_compacted: If True, attempt to mark session as fully compacted
                after successful processing of all remaining turns.
                If False (mid-session compaction), session stays 'active' and
                turns are preserved — they just get "covered" by knowledge entries
                so the retriever prefers compacted knowledge over raw turns.

        Returns list of knowledge entry IDs created.
        """
        if self._runner is None:
            return []

        # Check if already fully compacted
        session = self._store.get_session(session_id)
        if not session or session.get("status") == "compacted":
            return []

        # For mid-session compaction, only compact turns AFTER the watermark
        # (turns <= watermark were already compacted in a previous round)
        watermark = session.get("compacted_through_turn_index", -1) or -1

        turns = self._store.get_session_turns(session_id)
        # Filter to only uncompacted turns
        new_turns = [t for t in turns if t.turn_index > watermark]
        if len(new_turns) < 2:
            return []

        transcript, included_turns = self._build_transcript_from_turns(new_turns)
        if not included_turns:
            return []

        prompt = COMPACTION_PROMPT.format(transcript=transcript)

        # Invoke agent
        response = self._runner.invoke(prompt)
        if not response.success:
            return []

        # Parse output
        items, ok = parse_compaction_output(response.content)
        if not ok:
            return []

        last_included_turn_index = included_turns[-1].turn_index
        last_uncompacted_turn_index = new_turns[-1].turn_index
        turn_ids = [t.id for t in included_turns if t.id is not None]

        self._store.advance_compaction_watermark(session_id, last_included_turn_index)

        # Store knowledge entries
        created_ids = []
        for item in items:
            category = item.get("category", "progress")
            title = item.get("title", "Untitled")[:512]
            content = item.get("content", "")
            if not content:
                continue
            tags = item.get("tags", [])
            relevance = min(max(float(item.get("relevance_score", 0.8)), 0.0), 1.0)
            confidence = min(max(float(item.get("confidence", 0.8)), 0.0), 1.0)

            kid, _action = self._store.upsert_knowledge(
                project_id=project_id,
                source_type="session",
                agent_id=None,
                source_session_id=session_id,
                source_turn_ids=turn_ids,
                category=category,
                title=title,
                content=content,
                tags=tags,
                relevance_score=relevance,
                confidence=confidence,
                merge_runner=self._runner,
            )
            created_ids.append(kid)

        if mark_compacted and last_included_turn_index == last_uncompacted_turn_index:
            self._store.mark_session_fully_compacted(session_id)
        return created_ids

    def compact_all_pending(self, project_id: int, agent_id: int | None = None) -> dict:
        """Compact all completed sessions that haven't been compacted yet.

        Returns summary: {compacted: N, knowledge_created: N, skipped: N}
        """
        sessions = self._store.get_compactable_sessions(project_id)
        result = {"compacted": 0, "knowledge_created": 0, "skipped": 0}

        for sess in sessions:
            turn_count = self._store.count_session_turns(sess["id"])
            if turn_count < 2:
                self._store.mark_session_fully_compacted(sess["id"])
                result["skipped"] += 1
                continue

            ids = self.compact_session(
                session_id=sess["id"],
                project_id=project_id,
                agent_id=agent_id or sess.get("agent_id"),
            )
            if ids:
                result["compacted"] += 1
                result["knowledge_created"] += len(ids)
            else:
                result["skipped"] += 1

        return result
