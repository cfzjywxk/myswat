"""MemoryStore — core CRUD for all memory types in MySwat."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import pymysql.err

from myswat.db.connection import TiDBPool
from myswat.memory import embedder
from myswat.models.knowledge import KnowledgeEntry
from myswat.models.session import Session, SessionTurn
from myswat.models.work_item import Artifact, ReviewCycle, WorkItem


class MemoryStore:
    """Core CRUD interface for sessions, knowledge, work items, artifacts, and review cycles."""

    _PROCESS_LOG_LIMIT = 50

    def __init__(self, pool: TiDBPool, tidb_embedding_model: str = "") -> None:
        self._pool = pool
        self._tidb_embedding_model = tidb_embedding_model

    @staticmethod
    def _parse_json_field(value: Any) -> dict[str, Any] | list[Any] | None:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value

    # ──────────────────────────── Projects ────────────────────────────

    def create_project(
        self, slug: str, name: str, description: str | None = None,
        repo_path: str | None = None, config_json: dict | None = None,
    ) -> int:
        """Create a project (idempotent — returns existing ID on duplicate slug)."""
        try:
            return self._pool.insert_returning_id(
                "INSERT INTO projects (slug, name, description, repo_path, config_json) "
                "VALUES (%s, %s, %s, %s, %s)",
                (slug, name, description, repo_path,
                 json.dumps(config_json) if config_json else None),
            )
        except pymysql.err.IntegrityError as e:
            if e.args and e.args[0] == 1062:
                existing = self.get_project_by_slug(slug)
                if existing:
                    return existing["id"]
            raise

    def get_project_by_slug(self, slug: str) -> dict | None:
        return self._pool.fetch_one("SELECT * FROM projects WHERE slug = %s", (slug,))

    def get_project(self, project_id: int) -> dict | None:
        return self._pool.fetch_one("SELECT * FROM projects WHERE id = %s", (project_id,))

    # ──────────────────────────── Agents ────────────────────────────

    def create_agent(
        self, project_id: int, role: str, display_name: str,
        cli_backend: str, model_name: str, cli_path: str,
        cli_extra_args: list[str] | None = None, system_prompt: str | None = None,
    ) -> int:
        """Create an agent (idempotent — returns existing ID on duplicate project+role)."""
        try:
            return self._pool.insert_returning_id(
                "INSERT INTO agents (project_id, role, display_name, cli_backend, model_name, "
                "cli_path, cli_extra_args, system_prompt) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (project_id, role, display_name, cli_backend, model_name, cli_path,
                 json.dumps(cli_extra_args) if cli_extra_args else None, system_prompt),
            )
        except pymysql.err.IntegrityError as e:
            if e.args and e.args[0] == 1062:
                existing = self.get_agent(project_id, role)
                if existing:
                    return existing["id"]
            raise

    def get_agent(self, project_id: int, role: str) -> dict | None:
        return self._pool.fetch_one(
            "SELECT * FROM agents WHERE project_id = %s AND role = %s",
            (project_id, role),
        )

    def list_agents(self, project_id: int) -> list[dict]:
        return self._pool.fetch_all(
            "SELECT * FROM agents WHERE project_id = %s ORDER BY role", (project_id,),
        )

    # ──────────────────────────── Sessions ────────────────────────────

    def create_session(
        self, agent_id: int, purpose: str | None = None,
        work_item_id: int | None = None, parent_session_id: int | None = None,
    ) -> Session:
        session_uuid = str(uuid.uuid4())
        sid = self._pool.insert_returning_id(
            "INSERT INTO sessions (agent_id, session_uuid, parent_session_id, purpose, work_item_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (agent_id, session_uuid, parent_session_id, purpose, work_item_id),
        )
        return Session(id=sid, agent_id=agent_id, session_uuid=session_uuid,
                        parent_session_id=parent_session_id, purpose=purpose,
                        work_item_id=work_item_id)

    def get_active_session(self, agent_id: int, work_item_id: int | None = None) -> Session | None:
        sql = (
            "SELECT * FROM sessions WHERE agent_id = %s AND status = 'active' "
        )
        args: list[Any] = [agent_id]
        if work_item_id is None:
            sql += "AND work_item_id IS NULL "
        else:
            sql += "AND work_item_id = %s "
            args.append(work_item_id)
        sql += "ORDER BY updated_at DESC, created_at DESC LIMIT 1"
        row = self._pool.fetch_one(sql, tuple(args))
        return Session(**row) if row else None

    def close_session(self, session_id: int) -> None:
        self._pool.execute(
            "UPDATE sessions SET status = 'completed' WHERE id = %s", (session_id,),
        )

    def mark_session_compacted(self, session_id: int) -> None:
        self._pool.execute(
            "UPDATE sessions SET status = 'compacted' WHERE id = %s", (session_id,),
        )

    def reset_session_token_count(self, session_id: int) -> None:
        """Reset token counter after mid-session compaction to avoid re-triggering."""
        self._pool.execute(
            "UPDATE sessions SET token_count_est = 0 WHERE id = %s", (session_id,),
        )

    def advance_compaction_watermark(self, session_id: int, turn_index: int) -> None:
        """Advance the compaction watermark. Turns at or below this index are
        considered 'recycled' — covered by knowledge entries, excluded from
        raw context loading."""
        self._pool.execute(
            "UPDATE sessions SET compacted_through_turn_index = %s WHERE id = %s",
            (turn_index, session_id),
        )

    def update_session_progress(self, session_id: int, progress_note: str) -> None:
        """Update session purpose with a progress note (visible via `myswat status`)."""
        self._pool.execute(
            "UPDATE sessions SET purpose = %s WHERE id = %s",
            (progress_note[:512], session_id),
        )

    def get_session(self, session_id: int) -> dict | None:
        return self._pool.fetch_one("SELECT * FROM sessions WHERE id = %s", (session_id,))

    def get_compactable_sessions(self, project_id: int) -> list[dict]:
        """Get completed (not yet compacted) sessions for a project."""
        return self._pool.fetch_all(
            "SELECT s.* FROM sessions s "
            "JOIN agents a ON s.agent_id = a.id "
            "WHERE a.project_id = %s AND s.status = 'completed' "
            "ORDER BY s.created_at",
            (project_id,),
        )

    # ──────────────────────────── Session Turns ────────────────────────────

    def append_turn(
        self, session_id: int, role: str, content: str,
        token_count_est: int = 0, metadata: dict | None = None,
    ) -> SessionTurn:
        meta_json = json.dumps(metadata) if metadata else None

        # Retry loop handles race condition: if two concurrent appends both
        # compute the same turn_index, one will get a duplicate key error.
        # Retrying re-reads MAX(turn_index) and gets the correct next value.
        tid = None
        turn_index = -1
        for _attempt in range(3):
            row = self._pool.fetch_one(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
                "FROM session_turns WHERE session_id = %s", (session_id,),
            )
            turn_index = row["next_idx"]
            try:
                tid = self._pool.insert_returning_id(
                    "INSERT INTO session_turns (session_id, turn_index, role, content, "
                    "token_count_est, metadata_json) VALUES (%s, %s, %s, %s, %s, %s)",
                    (session_id, turn_index, role, content, token_count_est, meta_json),
                )
                break
            except pymysql.err.IntegrityError as e:
                if e.args and e.args[0] == 1062:  # Duplicate entry
                    continue
                raise
        else:
            raise RuntimeError(
                f"Failed to append turn to session {session_id} after 3 attempts "
                f"(persistent duplicate turn_index collision)"
            )

        # Update session token count
        self._pool.execute(
            "UPDATE sessions SET token_count_est = token_count_est + %s WHERE id = %s",
            (token_count_est, session_id),
        )

        return SessionTurn(id=tid, session_id=session_id, turn_index=turn_index,
                           role=role, content=content, token_count_est=token_count_est,
                           metadata_json=metadata)

    def get_session_turns(
        self, session_id: int, limit: int | None = None, offset: int = 0,
    ) -> list[SessionTurn]:
        sql = "SELECT * FROM session_turns WHERE session_id = %s ORDER BY turn_index"
        args: list[Any] = [session_id]
        if limit is not None:
            sql += " LIMIT %s OFFSET %s"
            args.extend([limit, offset])
        rows = self._pool.fetch_all(sql, tuple(args))
        return [SessionTurn(**r) for r in rows]

    def count_session_turns(self, session_id: int) -> int:
        row = self._pool.fetch_one(
            "SELECT COUNT(*) AS cnt FROM session_turns WHERE session_id = %s", (session_id,),
        )
        return row["cnt"] if row else 0

    def count_uncompacted_turns(self, session_id: int) -> int:
        """Count only turns after the compaction watermark."""
        session = self.get_session(session_id)
        watermark = -1
        if session:
            watermark = session.get("compacted_through_turn_index", -1) or -1
        row = self._pool.fetch_one(
            "SELECT COUNT(*) AS cnt FROM session_turns "
            "WHERE session_id = %s AND turn_index > %s",
            (session_id, watermark),
        )
        return row["cnt"] if row else 0

    def delete_compacted_turns(self, session_id: int) -> int:
        """Delete turns at or below the compaction watermark.

        These turns have been distilled into knowledge entries and are no
        longer needed for context building. Frees storage in TiDB.
        Returns the number of rows deleted.
        """
        session = self.get_session(session_id)
        if not session:
            return 0
        watermark = session.get("compacted_through_turn_index", -1) or -1
        if watermark < 0:
            return 0
        return self._pool.execute(
            "DELETE FROM session_turns WHERE session_id = %s AND turn_index <= %s",
            (session_id, watermark),
        )

    def delete_archived_session(self, session_id: int) -> dict:
        """Delete a fully-compacted session and all its remaining turns.

        Only works on sessions with status 'compacted'. Returns counts of
        deleted rows.
        """
        session = self.get_session(session_id)
        if not session or session.get("status") != "compacted":
            return {"turns": 0, "session": 0}

        turns_deleted = self._pool.execute(
            "DELETE FROM session_turns WHERE session_id = %s", (session_id,),
        )
        self._pool.execute(
            "DELETE FROM sessions WHERE id = %s", (session_id,),
        )
        return {"turns": turns_deleted, "session": 1}

    def purge_compacted_sessions(self, project_id: int) -> dict:
        """Delete all fully-compacted sessions and their turns for a project.

        Knowledge entries are preserved — they reference the session ID for
        provenance but don't require the session row to exist.
        Returns summary: {sessions_deleted: N, turns_deleted: N}
        """
        sessions = self._pool.fetch_all(
            "SELECT s.id FROM sessions s "
            "JOIN agents a ON s.agent_id = a.id "
            "WHERE a.project_id = %s AND s.status = 'compacted'",
            (project_id,),
        )
        total_turns = 0
        total_sessions = 0
        for sess in sessions:
            result = self.delete_archived_session(sess["id"])
            total_turns += result["turns"]
            total_sessions += result["session"]
        return {"sessions_deleted": total_sessions, "turns_deleted": total_turns}

    def get_recent_history_for_agent(
        self, agent_id: int, exclude_session_id: int | None = None,
        max_turns: int = 40, max_sessions: int = 5,
    ) -> list[dict]:
        """Fetch recent UNcompacted turns from this agent's previous sessions.

        Only loads turns AFTER the compaction watermark — turns at or below
        the watermark have been distilled into knowledge entries and are
        excluded (recycled). This prevents loading stale raw turns when
        compacted knowledge already covers them.

        Returns turns from the most recent sessions, oldest-session-first,
        turns within each session in chronological order.
        """
        session_sql = (
            "SELECT id, purpose, compacted_through_turn_index FROM sessions "
            "WHERE agent_id = %s AND status IN ('completed', 'compacted') "
        )
        args: list = [agent_id]
        if exclude_session_id is not None:
            session_sql += "AND id != %s "
            args.append(exclude_session_id)
        session_sql += "ORDER BY created_at DESC LIMIT %s"
        args.append(max_sessions)

        sessions = self._pool.fetch_all(session_sql, tuple(args))
        if not sessions:
            return []

        per_session_limit = max(max_turns // len(sessions), 4)
        all_turns = []
        for sess in reversed(sessions):  # oldest session first
            watermark = sess.get("compacted_through_turn_index", -1) or -1
            # Only load turns AFTER the watermark (uncompacted turns)
            turns = self._pool.fetch_all(
                "SELECT role, content, session_id FROM session_turns "
                "WHERE session_id = %s AND turn_index > %s "
                "ORDER BY turn_index DESC LIMIT %s",
                (sess["id"], watermark, per_session_limit),
            )
            if not turns:
                continue  # all turns recycled, skip this session
            turns = list(reversed(turns))
            all_turns.append({
                "session_id": sess["id"],
                "purpose": sess.get("purpose"),
                "turns": turns,
            })

        return all_turns

    def get_recent_artifacts_for_project(
        self, project_id: int, limit: int = 3,
    ) -> list[dict]:
        """Fetch the most recent artifacts (proposals/diffs) for a project."""
        return self._pool.fetch_all(
            "SELECT a.title, a.artifact_type, a.iteration, a.content, "
            "w.title AS work_item_title, w.status AS work_item_status "
            "FROM artifacts a "
            "JOIN work_items w ON a.work_item_id = w.id "
            "WHERE w.project_id = %s "
            "ORDER BY a.created_at DESC LIMIT %s",
            (project_id, limit),
        )

    # ──────────────────────────── Knowledge ────────────────────────────

    def store_knowledge(
        self, project_id: int, category: str, title: str, content: str,
        agent_id: int | None = None, source_session_id: int | None = None,
        source_turn_ids: list[int] | None = None, source_file: str | None = None,
        tags: list[str] | None = None, relevance_score: float = 1.0,
        confidence: float = 1.0, ttl_days: int | None = None,
        compute_embedding: bool = True,
    ) -> int:
        vec_sql = "NULL"
        embed_args: list[Any] = []
        if compute_embedding:
            vec_sql, embed_args = embedder.resolve_embed_sql(
                f"{title}\n{content}", self._tidb_embedding_model,
            )

        expires_at = None
        if ttl_days is not None:
            expires_at = datetime.now() + timedelta(days=ttl_days)

        sql = (
            "INSERT INTO knowledge (project_id, agent_id, source_session_id, "
            "source_turn_ids, source_file, category, title, content, embedding, tags, "
            "relevance_score, confidence, ttl_days, expires_at) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, {vec_sql}, %s, %s, %s, %s, %s)"
        )
        args: list[Any] = [
            project_id, agent_id, source_session_id,
            json.dumps(source_turn_ids) if source_turn_ids else None,
            source_file, category, title, content,
        ]
        args.extend(embed_args)
        args.extend([
            json.dumps(tags) if tags else None,
            relevance_score, confidence, ttl_days, expires_at,
        ])

        return self._pool.insert_returning_id(sql, tuple(args))

    def search_knowledge(
        self, project_id: int, query: str,
        agent_id: int | None = None, category: str | None = None,
        tags: list[str] | None = None, limit: int = 10,
        use_vector: bool = True, use_fulltext: bool = True,
    ) -> list[dict]:
        """Hybrid search across project knowledge.

        Knowledge in MySwat is project-scoped. The ``agent_id`` parameter is
        retained for call-site compatibility but does not narrow the search.
        """
        # Args are split by SQL position: SELECT (score) ... WHERE ... LIMIT
        score_args: list[Any] = []
        where_args: list[Any] = []

        conditions = ["k.project_id = %s"]
        where_args.append(project_id)
        if category:
            conditions.append("k.category = %s")
            where_args.append(category)

        # Build score components
        score_parts = ["k.relevance_score"]

        if use_fulltext and query:
            # TiDB Serverless doesn't support FULLTEXT; use LIKE for keyword matching
            score_parts.append(
                "CASE WHEN k.title LIKE %s OR k.content LIKE %s THEN 1.0 ELSE 0.0 END"
            )
            like_pattern = f"%{query}%"
            score_args.extend([like_pattern, like_pattern])

        if use_vector and query:
            vec_sql, vec_args = embedder.resolve_embed_sql(
                query, self._tidb_embedding_model,
            )
            if vec_sql != "NULL":
                score_parts.append(
                    f"COALESCE(1.0 - VEC_COSINE_DISTANCE(k.embedding, {vec_sql}), 0)"
                )
                score_args.extend(vec_args)

        score_expr = " + ".join(score_parts)
        where_clause = " AND ".join(conditions)

        sql = (
            f"SELECT k.*, ({score_expr}) AS search_score "
            f"FROM knowledge k "
            f"WHERE {where_clause} "
            f"AND (k.expires_at IS NULL OR k.expires_at > NOW()) "
            f"ORDER BY search_score DESC "
            f"LIMIT %s"
        )
        # Combine args in SQL order: score args, where args, limit
        args = score_args + where_args + [limit]

        return self._pool.fetch_all(sql, tuple(args))

    def search_knowledge_fulltext_only(
        self, project_id: int, query: str, limit: int = 10,
    ) -> list[dict]:
        """Keyword-only search (no embedding model needed)."""
        return self.search_knowledge(
            project_id=project_id, query=query, limit=limit,
            use_vector=False, use_fulltext=True,
        )

    def list_knowledge(
        self, project_id: int, category: str | None = None, limit: int = 50,
    ) -> list[dict]:
        """List knowledge entries, optionally filtered by category."""
        sql = "SELECT * FROM knowledge WHERE project_id = %s"
        args: list[Any] = [project_id]
        if category:
            sql += " AND category = %s"
            args.append(category)
        sql += " ORDER BY relevance_score DESC, created_at DESC LIMIT %s"
        args.append(limit)
        return self._pool.fetch_all(sql, tuple(args))

    def delete_knowledge_by_category(self, project_id: int, category: str) -> int:
        """Delete all knowledge entries for a project in a given category.

        Used by ``myswat learn`` to replace stale project_ops knowledge on
        re-learn without affecting other categories.
        Returns the number of rows deleted.
        """
        return self._pool.execute(
            "DELETE FROM knowledge WHERE project_id = %s AND category = %s",
            (project_id, category),
        )

    def decay_relevance(self, decay_factor: float = 0.95) -> int:
        """Reduce relevance scores for aging knowledge entries."""
        return self._pool.execute(
            "UPDATE knowledge SET relevance_score = relevance_score * %s "
            "WHERE relevance_score > 0.1", (decay_factor,),
        )

    def expire_stale_knowledge(self) -> int:
        """Delete knowledge entries past their TTL."""
        return self._pool.execute(
            "DELETE FROM knowledge WHERE expires_at IS NOT NULL AND expires_at < NOW()",
        )

    # ──────────────────────────── Work Items ────────────────────────────

    def create_work_item(
        self, project_id: int, title: str, item_type: str,
        description: str | None = None, assigned_agent_id: int | None = None,
        parent_item_id: int | None = None, priority: int = 3,
        metadata_json: dict[str, Any] | None = None,
    ) -> int:
        return self._pool.insert_returning_id(
            "INSERT INTO work_items (project_id, title, description, item_type, "
            "assigned_agent_id, parent_item_id, priority, metadata_json) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (project_id, title, description, item_type,
             assigned_agent_id, parent_item_id, priority,
             json.dumps(metadata_json) if metadata_json else None),
        )

    def update_work_item_status(self, item_id: int, status: str) -> None:
        self._pool.execute(
            "UPDATE work_items SET status = %s WHERE id = %s", (status, item_id),
        )

    def get_work_item(self, item_id: int) -> dict | None:
        row = self._pool.fetch_one("SELECT * FROM work_items WHERE id = %s", (item_id,))
        if row and row.get("metadata_json") is not None:
            row["metadata_json"] = self._parse_json_field(row["metadata_json"])
        return row

    def list_work_items(self, project_id: int, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM work_items WHERE project_id = %s"
        args: list[Any] = [project_id]
        if status:
            sql += " AND status = %s"
            args.append(status)
        sql += " ORDER BY priority, created_at"
        rows = self._pool.fetch_all(sql, tuple(args))
        for row in rows:
            if row.get("metadata_json") is not None:
                row["metadata_json"] = self._parse_json_field(row["metadata_json"])
        return rows

    def update_work_item_metadata(self, item_id: int, metadata_json: dict[str, Any] | None) -> None:
        self._pool.execute(
            "UPDATE work_items SET metadata_json = %s WHERE id = %s",
            (json.dumps(metadata_json) if metadata_json else None, item_id),
        )

    def get_work_item_state(self, item_id: int) -> dict[str, Any]:
        row = self.get_work_item(item_id)
        if not row:
            return {}
        metadata = row.get("metadata_json") or {}
        if not isinstance(metadata, dict):
            return {}
        task_state = metadata.get("task_state") or {}
        return task_state if isinstance(task_state, dict) else {}

    def get_work_item_process_log(self, item_id: int) -> list[dict[str, Any]]:
        task_state = self.get_work_item_state(item_id)
        process_log = task_state.get("process_log") if isinstance(task_state, dict) else None
        return process_log if isinstance(process_log, list) else []

    def update_work_item_state(
        self,
        item_id: int,
        *,
        current_stage: str | None = None,
        latest_summary: str | None = None,
        next_todos: list[str] | None = None,
        open_issues: list[str] | None = None,
        last_artifact_id: int | None = None,
        updated_by_agent_id: int | None = None,
    ) -> None:
        row = self.get_work_item(item_id)
        metadata = row.get("metadata_json") if row else None
        if not isinstance(metadata, dict):
            metadata = {}

        task_state = metadata.get("task_state")
        if not isinstance(task_state, dict):
            task_state = {}

        if current_stage is not None:
            task_state["current_stage"] = current_stage[:128]
        if latest_summary is not None:
            task_state["latest_summary"] = latest_summary[:4000]
        if next_todos is not None:
            task_state["next_todos"] = [str(item)[:300] for item in next_todos[:20]]
        if open_issues is not None:
            task_state["open_issues"] = [str(item)[:500] for item in open_issues[:20]]
        if last_artifact_id is not None:
            task_state["last_artifact_id"] = last_artifact_id
        if updated_by_agent_id is not None:
            task_state["updated_by_agent_id"] = updated_by_agent_id
        task_state["updated_at"] = datetime.now().isoformat(timespec="seconds")

        metadata["task_state"] = task_state
        self.update_work_item_metadata(item_id, metadata)

    def append_work_item_process_event(
        self,
        item_id: int,
        *,
        event_type: str,
        summary: str,
        from_role: str | None = None,
        to_role: str | None = None,
        title: str | None = None,
        updated_by_agent_id: int | None = None,
    ) -> dict[str, Any]:
        row = self.get_work_item(item_id)
        metadata = row.get("metadata_json") if row else None
        if not isinstance(metadata, dict):
            metadata = {}

        task_state = metadata.get("task_state")
        if not isinstance(task_state, dict):
            task_state = {}

        process_log = task_state.get("process_log")
        if not isinstance(process_log, list):
            process_log = []

        event = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "type": event_type[:64],
            "summary": summary[:4000],
        }
        if from_role:
            event["from_role"] = from_role[:64]
        if to_role:
            event["to_role"] = to_role[:64]
        if title:
            event["title"] = title[:300]

        process_log.append(event)
        task_state["process_log"] = process_log[-self._PROCESS_LOG_LIMIT:]
        if updated_by_agent_id is not None:
            task_state["updated_by_agent_id"] = updated_by_agent_id
        task_state["updated_at"] = datetime.now().isoformat(timespec="seconds")

        metadata["task_state"] = task_state
        self.update_work_item_metadata(item_id, metadata)
        return event

    # ──────────────────────────── Artifacts ────────────────────────────

    def create_artifact(
        self, work_item_id: int, agent_id: int, iteration: int,
        artifact_type: str, content: str, title: str | None = None,
        metadata_json: dict | None = None,
    ) -> int:
        """Create or update an artifact (idempotent on retry).

        Checks for an existing artifact with the same (work_item_id, agent_id,
        iteration, artifact_type). If found, updates its content instead of
        creating a duplicate row.
        """
        meta_str = json.dumps(metadata_json) if metadata_json else None

        existing = self._pool.fetch_one(
            "SELECT id FROM artifacts WHERE work_item_id = %s AND agent_id = %s "
            "AND iteration = %s AND artifact_type = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (work_item_id, agent_id, iteration, artifact_type),
        )
        if existing:
            self._pool.execute(
                "UPDATE artifacts SET content = %s, title = %s, metadata_json = %s "
                "WHERE id = %s",
                (content, title, meta_str, existing["id"]),
            )
            return existing["id"]

        return self._pool.insert_returning_id(
            "INSERT INTO artifacts (work_item_id, agent_id, iteration, artifact_type, "
            "title, content, metadata_json) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (work_item_id, agent_id, iteration, artifact_type, title, content, meta_str),
        )

    def get_artifact(self, artifact_id: int) -> dict | None:
        return self._pool.fetch_one("SELECT * FROM artifacts WHERE id = %s", (artifact_id,))

    def list_artifacts(self, work_item_id: int) -> list[dict]:
        return self._pool.fetch_all(
            "SELECT * FROM artifacts WHERE work_item_id = %s ORDER BY iteration",
            (work_item_id,),
        )

    # ──────────────────────────── Review Cycles ────────────────────────────

    def create_review_cycle(
        self, work_item_id: int, iteration: int,
        proposer_agent_id: int, reviewer_agent_id: int,
        artifact_id: int,
        proposal_session_id: int | None = None,
    ) -> int:
        """Create a review cycle (idempotent on duplicate artifact+reviewer).

        The UNIQUE constraint on (artifact_id, reviewer_agent_id) prevents the
        same reviewer from reviewing the same artifact twice. On retry, returns
        the existing cycle ID instead of crashing.
        """
        try:
            return self._pool.insert_returning_id(
                "INSERT INTO review_cycles (work_item_id, artifact_id, iteration, "
                "proposer_agent_id, reviewer_agent_id, proposal_session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (work_item_id, artifact_id, iteration,
                 proposer_agent_id, reviewer_agent_id, proposal_session_id),
            )
        except pymysql.err.IntegrityError as e:
            if e.args and e.args[0] == 1062:  # Duplicate entry
                existing = self._pool.fetch_one(
                    "SELECT id FROM review_cycles "
                    "WHERE artifact_id = %s AND reviewer_agent_id = %s",
                    (artifact_id, reviewer_agent_id),
                )
                if existing:
                    return existing["id"]
            raise

    def update_review_verdict(
        self, cycle_id: int, verdict: str, verdict_json: dict | None = None,
        review_session_id: int | None = None,
    ) -> None:
        self._pool.execute(
            "UPDATE review_cycles SET verdict = %s, verdict_json = %s, "
            "review_session_id = %s WHERE id = %s",
            (verdict, json.dumps(verdict_json) if verdict_json else None,
             review_session_id, cycle_id),
        )

    def get_review_cycles(self, work_item_id: int) -> list[dict]:
        return self._pool.fetch_all(
            "SELECT * FROM review_cycles WHERE work_item_id = %s ORDER BY created_at, id",
            (work_item_id,),
        )
