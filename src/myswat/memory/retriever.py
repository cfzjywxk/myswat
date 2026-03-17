"""MemoryRetriever — builds context for agent prompts from TiDB-persisted memory."""

from __future__ import annotations

import logging
from pathlib import Path

from myswat.cli.progress import _describe_process_event, _preview_text
from myswat.memory.search_engine import KnowledgeSearchEngine, SearchPlanBuilder
from myswat.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Builds context for an agent from deterministic state, recent raw turns,
    searched knowledge, and current-session continuity.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._search = KnowledgeSearchEngine(store)

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
            f"- `{launcher} status -p {slug} --details` — show detailed handoffs, process logs, and review breakdowns\n"
            f"- `{launcher} task <id> -p {slug}` — show one work item with process log, artifacts, and teamwork details\n"
            f"- `{launcher} search \"<query>\" -p {slug}` — search project knowledge\n"
            f"- `{launcher} memory list -p {slug}` — inspect stored knowledge entries\n"
            f"- `{launcher} learn -p {slug}` — refresh project operations knowledge\n\n"
            "Useful commands inside `myswat chat`:\n"
            "- `/status` — show active work items and recent process flow\n"
            "- `/task <id>` — show detailed work item state and process log\n"
            "- `/agents` — list roles/models\n"
            "- `/history [n]` — show recent turns\n\n"
            "## History Access\n\n"
            "Recent turns are persisted for this project. You are seeing the 10 most "
            "recent per role. If you need more context:\n"
            f"- `{launcher} status -p {slug}` — project overview\n"
            f"- `{launcher} history -p {slug} --turns 50` — raw recent turns\n"
        )

    def _build_cross_role_history(
        self, recent_turns: list[dict[str, object]], budget_tokens: int,
    ) -> str:
        """Build grouped recent project history across roles."""
        if not recent_turns:
            return ""

        role_count = max(len(recent_turns), 1)
        role_budget = max(budget_tokens // role_count, 1)

        lines = ["## Recent Project Conversation\n"]
        total_tokens_used = len(lines[0]) // 4

        for role_block in recent_turns:
            agent_role = str(role_block.get("agent_role") or "unknown")
            turns = role_block.get("turns")
            if not isinstance(turns, list) or not turns:
                continue

            header = f"### [{agent_role}] Recent Turns\n"
            header_tokens = len(header) // 4
            if total_tokens_used + header_tokens > budget_tokens:
                break

            selected: list[str] = []
            role_tokens_used = header_tokens
            for turn in reversed(turns):
                if not isinstance(turn, dict):
                    continue
                role_label = "User" if turn.get("role") == "user" else "Agent"
                content = str(turn.get("content") or "")
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                line = f"**{role_label}**: {content}\n"
                line_tokens = len(line) // 4
                if role_tokens_used + line_tokens > role_budget and selected:
                    break
                if total_tokens_used + role_tokens_used + line_tokens > budget_tokens and selected:
                    break
                role_tokens_used += line_tokens
                selected.append(line)

            if not selected:
                continue

            selected.reverse()
            lines.append(header)
            total_tokens_used += header_tokens
            omitted = len(turns) - len(selected)
            if omitted > 0:
                omitted_line = f"*[{omitted} earlier turns omitted]*\n"
                omitted_tokens = len(omitted_line) // 4
                if total_tokens_used + omitted_tokens <= budget_tokens:
                    lines.append(omitted_line)
                    total_tokens_used += omitted_tokens

            for line in selected:
                line_tokens = len(line) // 4
                if total_tokens_used + line_tokens > budget_tokens:
                    break
                lines.append(line)
                total_tokens_used += line_tokens

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _load_project_ops(
        self, project_id: int, repo_path: str | None,
    ) -> str:
        """Load project_ops knowledge, preferring local ``myswat.md`` cache.

        Falls back to TiDB query if the file doesn't exist or is empty.
        The canonical data always lives in TiDB; the file is just a cache
        written by ``myswat learn`` to avoid a round-trip on every context build.
        """
        file_text = ""
        file_titles: set[str] = set()

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
                        file_text = text[idx:].strip()
                    elif text.strip():
                        file_text = text.strip()
                    # File exists but has unexpected format — use as-is
                except OSError:
                    pass  # fall through to TiDB

        if file_text:
            for line in file_text.splitlines():
                if line.startswith("### "):
                    title = line[4:].strip()
                    if title:
                        file_titles.add(title)

        # Merge with TiDB so local cache doesn't hide newer project_ops entries
        try:
            ops_entries = self._store.list_knowledge(
                project_id, category="project_ops", limit=20,
            )
        except Exception:
            logger.warning(
                "Failed to load project_ops from TiDB for project %s; using local myswat.md cache only.",
                project_id,
                exc_info=True,
            )
            return file_text
        extra_lines: list[str] = []
        for entry in ops_entries:
            title = entry.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            if title in file_titles:
                continue
            extra_lines.append(f"### {title}\n{entry['content']}\n")

        if file_text and not extra_lines:
            return file_text

        if not file_text and not extra_lines:
            return ""

        if not file_text:
            return "\n".join(["## Project Operations Knowledge\n", *extra_lines])

        if "## Project Operations Knowledge" in file_text:
            return file_text.rstrip() + "\n\n" + "\n".join(extra_lines)

        return file_text.rstrip() + "\n\n" + "\n".join(["## Project Operations Knowledge\n", *extra_lines])

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
        agent_role: str | None = None,
        task_description: str | None = None,
        current_session_id: int | None = None,
        max_tokens: int = 8000,
        repo_path: str | None = None,
    ) -> str:
        """Build a context string from TiDB-persisted memory.

        Budget targets:
        - deterministic project/task context
        - 25% recent cross-role raw turns
        - 50% knowledge search results
        - 25% current session turns
        -  5% work items + artifacts
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
        current_stage: str | None = None
        if current_session_id is not None:
            session = self._store.get_session(current_session_id)
            work_item_id = session.get("work_item_id") if session else None
            if work_item_id:
                task_state = self._build_work_item_state_context(work_item_id)
                if task_state:
                    sections.append(task_state)
                work_item_state = self._store.get_work_item_state(work_item_id)
                if isinstance(work_item_state, dict):
                    stage = work_item_state.get("current_stage")
                    if isinstance(stage, str) and stage.strip():
                        current_stage = stage.strip()

        # ── 1. Recent project turns (ALWAYS loaded, project-scoped) ──
        history_budget = int(max_tokens * 0.25)
        recent_turns = self._store.get_recent_turns_by_project(
            project_id=project_id,
            per_role_limit=10,
            exclude_session_id=current_session_id,
        )
        if recent_turns:
            history_section = self._build_cross_role_history(recent_turns, history_budget)
            if history_section:
                sections.append(history_section)

        # ── 2. Knowledge base search (PRIMARY — vector + lexical) ──
        knowledge_budget = int(max_tokens * 0.5) - ops_tokens_used

        if task_description:
            plan = SearchPlanBuilder.build(
                project_id=project_id,
                query=task_description,
                agent_id=agent_id,
                agent_role=agent_role,
                current_stage=current_stage,
                limit=15,
                mode="auto",
                profile="standard",
            )
            results = self._search.search(plan)
        else:
            plan = SearchPlanBuilder.build(
                project_id=project_id,
                query="",
                agent_id=agent_id,
                agent_role=agent_role,
                current_stage=current_stage,
                limit=5,
            )
            results = self._search.search(plan)

        if results:
            knowledge_section = self._search.render_for_context(results, knowledge_budget)
            if knowledge_section:
                sections.append(knowledge_section)

        # ── 3. Current session turns (within-session continuity) ──
        if current_session_id is not None:
            current_budget = int(max_tokens * 0.25)
            current_section = self._build_current_session_context(
                current_session_id, current_budget,
            )
            if current_section:
                sections.append(current_section)

        # ── 4. Active work items + recent artifacts (5% combined) ──
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

        if not sections:
            return ""

        return "\n---\n\n".join(sections)

    def _build_current_session_context(
        self, session_id: int, budget_tokens: int,
    ) -> str:
        """Load physically present turns from the current active session."""
        turns = self._store.get_session_turns(session_id)
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
        agent_role: str | None = None,
        current_stage: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search using the Phase 1c search engine."""
        plan = SearchPlanBuilder.build(
            project_id=project_id,
            query=query,
            agent_id=agent_id,
            agent_role=agent_role,
            current_stage=current_stage,
            category=category,
            limit=limit,
        )
        return self._search.search(plan)
