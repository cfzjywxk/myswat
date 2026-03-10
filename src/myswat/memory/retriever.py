"""MemoryRetriever — builds context for agent prompts from TiDB-persisted memory."""

from __future__ import annotations

from pathlib import Path

from myswat.cli.progress import _describe_process_event, _preview_text
from myswat.memory.store import MemoryStore

# Minimum number of knowledge results to consider the knowledge base "trained".
# Below this, we supplement with raw session history as fallback.
KNOWLEDGE_SUFFICIENCY_THRESHOLD = 3


class MemoryRetriever:
    """Builds context for an agent by searching its TiDB knowledge base first.

    Priority order:
    1. Knowledge base (vector search) — PRIMARY, always loaded
    2. Current session turns — within-session continuity
    3. Active work items + recent artifacts — lightweight metadata
    4. Raw session history — FALLBACK, only when knowledge base is thin

    A "trained" agent (one with enough compacted/fed knowledge) gets context
    entirely from vector-searched knowledge + current conversation. Raw session
    history is only loaded when the knowledge base has fewer than
    KNOWLEDGE_SUFFICIENCY_THRESHOLD results — i.e., early in a project's life.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def _build_myswat_cli_context(self, project: dict | None, repo_path: str | None) -> str:
        if not isinstance(project, dict):
            return ""

        slug = project.get("slug") or "<project-slug>"
        repo_label = repo_path or project.get("repo_path") or "."
        launcher = "./myswat"

        return (
            "## MySwat Project Access\n\n"
            f"**Project slug**: {slug}\n"
            f"**Repo root**: {repo_label}\n\n"
            "Use MySwat itself to inspect persisted project state before claiming information is unavailable.\n\n"
            "Useful commands from the repo root:\n"
            f"- `{launcher} status -p {slug}` — show project work items, stages, process logs, review rounds, sessions\n"
            f"- `{launcher} task <id> -p {slug}` — show one work item with process log, artifacts, and teamwork details\n"
            f"- `{launcher} memory search \"<query>\" -p {slug}` — search project knowledge\n"
            f"- `{launcher} memory list -p {slug}` — inspect stored knowledge entries\n"
            f"- `{launcher} learn -p {slug}` — refresh project operations knowledge\n\n"
            "Useful commands inside `myswat chat`:\n"
            "- `/status` — show active work items and recent process flow\n"
            "- `/task <id>` — show detailed work item state and process log\n"
            "- `/agents` — list roles/models\n"
            "- `/history [n]` — show recent turns\n"
        )

    def _load_project_ops(
        self, project_id: int, repo_path: str | None,
    ) -> str:
        """Load project_ops knowledge, preferring local ``myswat.md`` cache.

        Falls back to TiDB query if the file doesn't exist or is empty.
        The canonical data always lives in TiDB; the file is just a cache
        written by ``myswat learn`` to avoid a round-trip on every context build.
        """
        # Try local file first
        if repo_path:
            md_file = Path(repo_path) / "myswat.md"
            if md_file.is_file():
                try:
                    text = md_file.read_text(encoding="utf-8")
                    # Strip the HTML comment header — everything after it is content
                    marker = "## Project Operations Knowledge"
                    idx = text.find(marker)
                    if idx != -1:
                        return text[idx:]
                    # File exists but has unexpected format — use as-is
                    if text.strip():
                        return text
                except OSError:
                    pass  # fall through to TiDB

        # Fall back to TiDB
        ops_entries = self._store.list_knowledge(
            project_id, category="project_ops", limit=20,
        )
        if not ops_entries:
            return ""
        ops_lines = ["## Project Operations Knowledge\n"]
        for entry in ops_entries:
            ops_lines.append(f"### {entry['title']}\n{entry['content']}\n")
        return "\n".join(ops_lines)

    def _build_work_item_state_context(self, work_item_id: int, budget_tokens: int = 600) -> str:
        item = self._store.get_work_item(work_item_id)
        if not item:
            return ""

        state = self._store.get_work_item_state(work_item_id)
        if not state:
            return ""

        lines = ["## Current Task State\n"]
        title = item.get("title") or f"Work item {work_item_id}"
        status = item.get("status") or "unknown"
        lines.append(f"**Work Item**: {title} [{status}]\n")

        if state.get("current_stage"):
            lines.append(f"**Stage**: {state['current_stage']}\n")
        if state.get("latest_summary"):
            summary = str(state["latest_summary"])[:2000]
            lines.append(f"### Latest Summary\n{summary}\n")
        if state.get("open_issues"):
            lines.append("### Open Issues")
            for issue in state["open_issues"][:10]:
                lines.append(f"- {issue}")
            lines.append("")
        if state.get("next_todos"):
            lines.append("### Next TODOs")
            for todo in state["next_todos"][:10]:
                lines.append(f"- {todo}")
            lines.append("")
        process_log = state.get("process_log")
        if isinstance(process_log, list) and process_log:
            lines.append("### Process Log")
            for event in process_log[-8:]:
                if isinstance(event, dict):
                    lines.append(f"- {_describe_process_event(event, 140)}")
            lines.append("")

        text = "\n".join(lines).strip()
        if len(text) // 4 > budget_tokens:
            return text[:budget_tokens * 4] + "\n... [truncated]"
        return text

    def build_context_for_agent(
        self,
        project_id: int,
        agent_id: int,
        task_description: str | None = None,
        current_session_id: int | None = None,
        max_tokens: int = 8000,
        repo_path: str | None = None,
    ) -> str:
        """Build a context string from TiDB-persisted memory.

        Knowledge-first budget:
        - 50% knowledge entries via vector search (PRIMARY)
        - 25% current session turns (within-session continuity)
        -  5% work items + artifacts (lightweight metadata)
        - 20% raw session history (FALLBACK — only when knowledge < 3 results)
                if knowledge is sufficient, this 20% is redistributed to knowledge
        """
        sections: list[str] = []
        project = self._store.get_project(project_id)
        if not isinstance(project, dict):
            project = {}

        cli_context = self._build_myswat_cli_context(project, repo_path)
        if cli_context:
            sections.append(cli_context)

        # ── 0. Project ops (ALWAYS loaded — build, test, conventions) ──
        # Try local ``myswat.md`` first (fast file read, no TiDB round-trip).
        # Fall back to TiDB if the file doesn't exist or is unreadable.
        ops_text = self._load_project_ops(project_id, repo_path)
        ops_tokens_used = len(ops_text) // 4 if ops_text else 0
        if ops_text:
            sections.append(ops_text)

        # ── 0b. Current task state (ALWAYS loaded for work item sessions) ──
        if current_session_id is not None:
            session = self._store.get_session(current_session_id)
            work_item_id = session.get("work_item_id") if session else None
            if work_item_id:
                task_state = self._build_work_item_state_context(work_item_id)
                if task_state:
                    sections.append(task_state)

        # ── 1. Knowledge base search (PRIMARY — vector + keyword) ──
        # This is the core: fed documents + compacted session knowledge,
        # searched by semantic similarity (VEC_COSINE_DISTANCE) when the
        # embedder is available, falling back to keyword LIKE matching
        # otherwise. The search_knowledge() method handles both paths
        # internally — the retriever just uses the results.
        knowledge_budget = int(max_tokens * 0.5) - ops_tokens_used

        if task_description:
            results = self._store.search_knowledge(
                project_id=project_id,
                query=task_description,
                agent_id=agent_id,
                limit=15,
            )
        else:
            results = self._store.search_knowledge(
                project_id=project_id,
                query="",
                agent_id=agent_id,
                limit=5,
                use_vector=False,
                use_fulltext=False,
            )

        # knowledge_count is the number of SEARCH HITS returned by TiDB,
        # not how many fit in the budget. This drives the sufficiency
        # check: if the knowledge base returned >= 3 results for this
        # query, we consider the agent "trained" and skip raw history.
        knowledge_count = len(results)

        if results:
            knowledge_lines = ["## Relevant Knowledge\n"]
            knowledge_tokens_used = 0
            for entry in results:
                line = f"### [{entry['category']}] {entry['title']}\n{entry['content']}\n"
                line_tokens = len(line) // 4
                if knowledge_tokens_used + line_tokens > knowledge_budget:
                    break
                knowledge_tokens_used += line_tokens
                knowledge_lines.append(line)
            if len(knowledge_lines) > 1:
                sections.append("\n".join(knowledge_lines))

        # ── 2. Current session turns (within-session continuity) ──
        if current_session_id is not None:
            current_budget = int(max_tokens * 0.25)
            current_section = self._build_current_session_context(
                current_session_id, current_budget,
            )
            if current_section:
                sections.append(current_section)

        # ── 3. Active work items + recent artifacts (5% combined) ──
        metadata_budget = int(max_tokens * 0.05)
        metadata_tokens_used = 0

        active_items = list(self._store.list_work_items(project_id, status="in_progress"))
        review_items = list(self._store.list_work_items(project_id, status="review"))
        all_active = active_items + review_items

        if all_active:
            work_lines = ["## Active Work Items\n"]
            for item in all_active[:5]:
                metadata = item.get("metadata_json") if isinstance(item, dict) else None
                task_state = metadata.get("task_state") if isinstance(metadata, dict) else {}
                if not isinstance(task_state, dict):
                    task_state = {}
                stage = _preview_text(task_state.get("current_stage"), 40)
                process_log = task_state.get("process_log")
                latest_flow = ""
                if isinstance(process_log, list):
                    for event in reversed(process_log):
                        if isinstance(event, dict):
                            latest_flow = _describe_process_event(event, 80)
                            break
                extras = []
                if stage:
                    extras.append(f"stage: {stage}")
                if latest_flow:
                    extras.append(f"flow: {latest_flow}")
                extra_suffix = f", {'; '.join(extras)}" if extras else ""
                line = (
                    f"- [{item['status']}] {item['title']} "
                    f"(priority: {item['priority']}{extra_suffix})\n"
                )
                line_tokens = len(line) // 4
                if metadata_tokens_used + line_tokens > metadata_budget:
                    break
                metadata_tokens_used += line_tokens
                work_lines.append(line)
            if len(work_lines) > 1:
                sections.append("\n".join(work_lines))

        artifact_budget = metadata_budget - metadata_tokens_used
        if artifact_budget < 50:
            artifact_budget = 50  # minimum to show at least one artifact title
        artifacts = self._store.get_recent_artifacts_for_project(project_id, limit=2)
        if artifacts:
            artifact_lines = ["## Recent Artifacts\n"]
            artifact_tokens_used = 0
            for art in artifacts:
                content_preview = art["content"][:400]
                if len(art["content"]) > 400:
                    content_preview += "... [truncated]"
                line = (
                    f"### [{art['artifact_type']}] {art.get('work_item_title', '')}"
                    f" ({art.get('work_item_status', '?')})\n"
                    f"{content_preview}\n"
                )
                line_tokens = len(line) // 4
                if artifact_tokens_used + line_tokens > artifact_budget:
                    break
                artifact_tokens_used += line_tokens
                artifact_lines.append(line)
            if len(artifact_lines) > 1:
                sections.append("\n".join(artifact_lines))

        # ── 4. Raw session history (FALLBACK — only when knowledge is thin) ──
        # Once the knowledge base is sufficiently trained (>= 3 relevant results),
        # raw session history is redundant — the compacted knowledge already
        # captures what matters. Only load raw turns when the agent is "new"
        # and hasn't built up enough knowledge yet.
        if knowledge_count < KNOWLEDGE_SUFFICIENCY_THRESHOLD:
            history_budget = int(max_tokens * 0.2)
            history = self._store.get_recent_history_for_agent(
                agent_id=agent_id,
                exclude_session_id=current_session_id,
                max_turns=20,
                max_sessions=3,
            )
            if history:
                history_section = self._build_history_context(history, history_budget)
                if history_section:
                    sections.append(history_section)

        if not sections:
            return ""

        return "\n---\n\n".join(sections)

    def _build_current_session_context(
        self, session_id: int, budget_tokens: int,
    ) -> str:
        """Load recent UNcompacted turns from the current active session.

        Turns at or below the compaction watermark are excluded — they've been
        distilled into knowledge entries and would be redundant here.
        """
        session = self._store.get_session(session_id)
        watermark = -1
        if session:
            watermark = session.get("compacted_through_turn_index", -1) or -1

        turns = self._store.get_session_turns(session_id)
        # Only include turns after the watermark (uncompacted)
        turns = [t for t in turns if t.turn_index > watermark]
        if not turns:
            return ""

        lines = ["## Current Conversation\n"]
        tokens_used = 0

        # Walk backwards from most recent, then reverse for chronological order
        selected = []
        for turn in reversed(turns):
            role_label = "User" if turn.role == "user" else "Agent"
            content = turn.content
            if len(content) > 800:
                content = content[:800] + "... [truncated]"
            line = f"**{role_label}**: {content}\n"
            line_tokens = len(line) // 4
            if tokens_used + line_tokens > budget_tokens:
                break
            tokens_used += line_tokens
            selected.append(line)

        if not selected:
            return ""

        selected.reverse()
        if len(selected) < len(turns):
            lines.append(f"*[{len(turns) - len(selected)} earlier turns omitted]*\n")
        lines.extend(selected)
        return "\n".join(lines)

    def _build_history_context(
        self, history: list[dict], budget_tokens: int,
    ) -> str:
        """Build context from previous sessions of the same role+project."""
        lines = ["## Previous Session Context\n"]
        tokens_used = 0

        for sess_block in history:
            purpose = sess_block.get("purpose") or "unnamed session"
            header = f"### Session: {purpose}\n"
            header_tokens = len(header) // 4
            if tokens_used + header_tokens > budget_tokens:
                break
            tokens_used += header_tokens
            lines.append(header)

            for turn in sess_block["turns"]:
                role_label = "User" if turn["role"] == "user" else "Agent"
                content = turn["content"]
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                line = f"**{role_label}**: {content}\n"
                line_tokens = len(line) // 4
                if tokens_used + line_tokens > budget_tokens:
                    lines.append("*[earlier turns omitted]*\n")
                    break
                tokens_used += line_tokens
                lines.append(line)

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def search(
        self,
        project_id: int,
        query: str,
        agent_id: int | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Direct search pass-through for CLI usage."""
        return self._store.search_knowledge(
            project_id=project_id,
            query=query,
            agent_id=agent_id,
            category=category,
            limit=limit,
        )
