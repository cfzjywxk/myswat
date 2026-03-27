"""MemoryStore — core CRUD for all memory types in MySwat."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

import pymysql.err

from myswat.db.connection import TiDBPool
from myswat.large_payloads import (
    AGENT_FILE_PROMPT,
    maybe_externalize_prompt,
    resolve_externalized_text,
)
from myswat.memory import embedder
from myswat.models.knowledge import KnowledgeEntry
from myswat.models.session import Session, SessionTurn
from myswat.models.work_item import Artifact, ReviewCycle, WorkItem
from myswat.models.workflow_runtime import CoordinationEvent, RuntimeRegistration, StageRun

if TYPE_CHECKING:
    from myswat.agents.base import AgentRunner

TRANSIENT_KNOWLEDGE_CATEGORIES = frozenset({"progress", "review_feedback"})
LLM_MERGE_THRESHOLD = 0.60
LEXICAL_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
})
FIELD_WEIGHTS: dict[str, float] = {
    "title": 4.0,
    "entity": 4.0,
    "symbol": 4.0,
    "tag": 3.0,
    "source_file": 2.0,
    "content": 1.0,
}
RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?P<src>[A-Za-z][A-Za-z0-9_-]+)\s+depends on\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+)", re.IGNORECASE), "depends_on"),
    (re.compile(r"\b(?P<src>[A-Za-z][A-Za-z0-9_-]+)\s+uses\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+)", re.IGNORECASE), "uses"),
    (re.compile(r"\b(?P<src>[A-Za-z][A-Za-z0-9_-]+)\s+handles\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+)", re.IGNORECASE), "handles"),
    (re.compile(r"\b(?P<src>[A-Za-z][A-Za-z0-9_-]+)\s+executes\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+)", re.IGNORECASE), "executes"),
    (re.compile(r"\b(?P<src>[A-Za-z][A-Za-z0-9_-]+)\s+triggers\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+)", re.IGNORECASE), "triggers"),
    (re.compile(r"\b(?P<src>[A-Za-z][A-Za-z0-9_-]+)\s+requires\s+(?P<tgt>[A-Za-z][A-Za-z0-9 _-]+)", re.IGNORECASE), "requires"),
)

_UNSET = object()  # sentinel for "parameter not provided"

MERGE_PROMPT = """You are merging two project knowledge entries that refer to the same topic.

Rules:
- Preserve concrete technical facts.
- Prefer the more specific and more up-to-date details.
- Remove duplicates and contradictions when one side clearly supersedes the other.
- Output only the merged knowledge content, no markdown fences or commentary.

Existing title: {title}

Existing content:
{existing_content}

New content:
{new_content}
"""


class MemoryStore:
    """Core CRUD interface for sessions, knowledge, work items, artifacts, and review cycles."""

    _PROCESS_LOG_LIMIT = 50

    def __init__(
        self,
        pool: TiDBPool,
        tidb_embedding_model: str = "",
        embedding_backend: str = "auto",
    ) -> None:
        self._pool = pool
        self._tidb_embedding_model = tidb_embedding_model
        self._embedding_backend = (embedding_backend or "auto").strip().lower()

    @staticmethod
    def _parse_json_field(value: Any) -> dict[str, Any] | list[Any] | None:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value

    @staticmethod
    def _is_missing_embedding_function(exc: Exception) -> bool:
        if not isinstance(exc, pymysql.err.OperationalError):
            return False
        code = exc.args[0] if exc.args else None
        message = str(exc).lower()
        return code == 1305 and "embedding" in message

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for dedup and conservative merge checks."""
        text = re.sub(r"\s+", " ", value.strip().casefold())
        return text

    @classmethod
    def _compute_content_hash(cls, content: str) -> str:
        normalized = cls._normalize_text(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_raw_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def _token_set(cls, content: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", cls._normalize_text(content)))

    @classmethod
    def _token_overlap_ratio(cls, left: str, right: str) -> float:
        left_tokens = cls._token_set(left)
        right_tokens = cls._token_set(right)
        if not left_tokens or not right_tokens:
            return 0.0
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(left_tokens & right_tokens) / len(union)

    @staticmethod
    def _merge_tags(*tag_lists: list[str] | None) -> list[str] | None:
        merged: list[str] = []
        seen: set[str] = set()
        for tag_list in tag_lists:
            if not tag_list:
                continue
            for tag in tag_list:
                key = str(tag).strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(key)
        return merged or None

    @staticmethod
    def _ensure_dict(value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    @staticmethod
    def _ensure_list_of_dicts(value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, dict)] or None

    @staticmethod
    def _infer_source_type(
        *,
        source_type: str | None,
        category: str,
        source_file: str | None,
    ) -> str:
        if source_type:
            return source_type
        if category == "project_ops":
            return "manual"
        if source_file:
            return "document"
        return "session"

    def _bump_project_memory_revision(self, project_id: int) -> None:
        self._pool.execute(
            "UPDATE projects SET memory_revision = memory_revision + 1 WHERE id = %s",
            (project_id,),
        )

    def _scope_sql_and_args(
        self,
        *,
        project_id: int,
        category: str,
        source_type: str,
        source_file: str | None,
        alias: str = "k",
    ) -> tuple[str, list[Any]]:
        sql = (
            f"{alias}.project_id = %s AND {alias}.category = %s AND "
            f"{alias}.source_type = %s"
        )
        args: list[Any] = [project_id, category, source_type]
        if source_type == "document":
            sql += f" AND {alias}.source_file = %s"
            args.append(source_file)
        return sql, args

    def _parse_knowledge_row(self, row: dict | None) -> dict | None:
        if not row:
            return None
        parsed = dict(row)
        for field in ("source_turn_ids", "tags", "search_metadata_json", "merged_from"):
            if parsed.get(field) is not None:
                parsed[field] = self._parse_json_field(parsed[field])
        return parsed

    @staticmethod
    def _infer_language_from_source_file(source_file: str | None) -> str | None:
        if not source_file:
            return None
        lower = source_file.casefold()
        if lower.endswith(".rs"):
            return "rust"
        if lower.endswith(".go"):
            return "go"
        if lower.endswith(".py"):
            return "python"
        if lower.endswith(".md"):
            return "markdown"
        return None

    @classmethod
    def _build_default_search_metadata(
        cls,
        *,
        source_type: str,
        source_file: str | None,
        tags: list[str] | None,
        search_metadata_json: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        metadata = dict(search_metadata_json or {})
        metadata.setdefault("source_type", source_type)
        if source_file:
            metadata.setdefault("source_file", source_file)
        language = cls._infer_language_from_source_file(source_file)
        if language:
            metadata.setdefault("language", language)
        if tags:
            metadata.setdefault("tags", tags)
            for tag in tags:
                if ":" not in str(tag):
                    continue
                key, value = str(tag).split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in {"repo", "subsystem", "language"} and value:
                    metadata.setdefault(key, value)
        return metadata or None

    @staticmethod
    def _split_camel(token: str) -> list[str]:
        parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
            token,
        )
        return [part.casefold() for part in parts if part]

    @staticmethod
    def _add_term(term_map: dict[tuple[str, str], float], field: str, term: str) -> None:
        normalized = term.strip().casefold()
        if not normalized:
            return
        if not re.search(r"[a-z0-9]", normalized):
            return
        if normalized in LEXICAL_STOP_WORDS:
            return
        key = (field, normalized)
        term_map[key] = max(term_map.get(key, 0.0), FIELD_WEIGHTS[field])

    @classmethod
    def _tokenize_structured_token(
        cls,
        token: str,
        *,
        field: str,
        include_phrase_terms: bool,
        term_map: dict[tuple[str, str], float],
    ) -> None:
        raw = token.strip()
        if not raw:
            return
        cls._add_term(term_map, field, raw)

        if "::" in raw:
            for part in raw.split("::"):
                cls._tokenize_structured_token(
                    part, field=field, include_phrase_terms=False, term_map=term_map,
                )

        if "/" in raw:
            for part in [seg for seg in raw.split("/") if seg]:
                cls._tokenize_structured_token(
                    part, field=field, include_phrase_terms=False, term_map=term_map,
                )

        if "." in raw:
            pieces = [seg for seg in raw.split(".") if seg]
            if len(pieces) >= 2:
                cls._add_term(term_map, field, ".".join(pieces[:2]))
            for part in pieces:
                cls._tokenize_structured_token(
                    part, field=field, include_phrase_terms=False, term_map=term_map,
                )
            if raw.endswith(".go") or raw.endswith(".rs"):
                base = raw.rsplit(".", 1)[0]
                cls._add_term(term_map, field, base)
                cls._tokenize_structured_token(
                    base, field=field, include_phrase_terms=False, term_map=term_map,
                )

        split_tokens: list[str] = []
        for kebab_part in [seg for seg in re.split(r"[-_]", raw) if seg]:
            cls._add_term(term_map, field, kebab_part)
            camel_parts = cls._split_camel(kebab_part)
            if camel_parts:
                split_tokens.extend(camel_parts)
                for part in camel_parts:
                    cls._add_term(term_map, field, part)
            else:
                split_tokens.append(kebab_part.casefold())
        if include_phrase_terms and len(split_tokens) >= 2:
            for idx in range(len(split_tokens) - 1):
                cls._add_term(term_map, field, f"{split_tokens[idx]} {split_tokens[idx + 1]}")

    @classmethod
    def _extract_terms_for_field(
        cls,
        *,
        field: str,
        text: str,
        structured: bool,
        include_phrase_terms: bool,
    ) -> list[tuple[str, str, float]]:
        term_map: dict[tuple[str, str], float] = {}
        if not text.strip():
            return []

        if structured:
            candidates = re.findall(r"[A-Za-z0-9_:/.\-]+", text)
            for candidate in candidates:
                cls._tokenize_structured_token(
                    candidate,
                    field=field,
                    include_phrase_terms=include_phrase_terms,
                    term_map=term_map,
                )
            if include_phrase_terms:
                words = re.findall(r"[A-Za-z0-9]+", text.casefold())
                words = [word for word in words if word not in LEXICAL_STOP_WORDS]
                for idx in range(len(words) - 1):
                    cls._add_term(term_map, field, f"{words[idx]} {words[idx + 1]}")
        else:
            for token in re.findall(r"[A-Za-z0-9_:/.\-]+", text):
                cls._tokenize_structured_token(
                    token,
                    field=field,
                    include_phrase_terms=False,
                    term_map=term_map,
                )

        return [(field, term, weight) for (field, term), weight in term_map.items()]

    @classmethod
    def _build_knowledge_terms(
        cls,
        *,
        title: str,
        content: str,
        tags: list[str] | None,
        source_file: str | None,
        entities: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        terms: dict[tuple[str, str], float] = {}
        for field_name, text, structured, include_phrases in [
            ("title", title, True, True),
            ("content", content, False, False),
        ]:
            for field, term, weight in cls._extract_terms_for_field(
                field=field_name,
                text=text,
                structured=structured,
                include_phrase_terms=include_phrases,
            ):
                terms[(field, term)] = max(terms.get((field, term), 0.0), weight)

        for tag in tags or []:
            for field, term, weight in cls._extract_terms_for_field(
                field="tag",
                text=str(tag),
                structured=True,
                include_phrase_terms=True,
            ):
                terms[(field, term)] = max(terms.get((field, term), 0.0), weight)

        if source_file:
            for field, term, weight in cls._extract_terms_for_field(
                field="source_file",
                text=source_file,
                structured=True,
                include_phrase_terms=False,
            ):
                terms[(field, term)] = max(terms.get((field, term), 0.0), weight)

        for entity in entities or []:
            for field, term, weight in cls._extract_terms_for_field(
                field="entity",
                text=entity,
                structured=True,
                include_phrase_terms=True,
            ):
                terms[(field, term)] = max(terms.get((field, term), 0.0), weight)

        return [(field, term, weight) for (field, term), weight in terms.items()]

    @classmethod
    def _extract_entities(
        cls,
        *,
        title: str,
        content: str,
        tags: list[str] | None,
        source_file: str | None,
    ) -> list[str]:
        candidates: dict[str, str] = {}

        def add_entity(raw: str) -> None:
            cleaned = raw.strip(" .,:;()[]{}")
            if not cleaned or len(cleaned) < 2:
                return
            normalized = cleaned.casefold()
            if normalized in LEXICAL_STOP_WORDS:
                return
            candidates.setdefault(normalized, cleaned)

        for text in [title, *(tags or [])]:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_:\-./]+", text):
                if any(ch.isupper() for ch in token) or any(sep in token for sep in ("-", "_", "::", ".", "/")):
                    add_entity(token)

        if source_file:
            path = source_file.replace("\\", "/")
            add_entity(path)
            for segment in [part for part in path.split("/") if part]:
                add_entity(segment)
            basename = path.rsplit("/", 1)[-1]
            add_entity(basename)
            if "." in basename:
                add_entity(basename.rsplit(".", 1)[0])

        for token in re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", content):
            add_entity(token)

        return list(candidates.values())

    @classmethod
    def _extract_relations(
        cls,
        *,
        title: str,
        content: str,
    ) -> list[tuple[str, str, str]]:
        text = f"{title}\n{content[:500]}"
        relations: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for pattern, relation in RELATION_PATTERNS:
            for match in pattern.finditer(text):
                source = " ".join(match.group("src").split()).strip(" .,:;")
                target = " ".join(match.group("tgt").split()).strip(" .,:;")
                if not source or not target or source.casefold() == target.casefold():
                    continue
                key = (source.casefold(), relation, target.casefold())
                if key in seen:
                    continue
                seen.add(key)
                relations.append((source, relation, target))
        return relations

    def _replace_knowledge_terms(
        self,
        *,
        project_id: int,
        knowledge_id: int,
        title: str,
        content: str,
        tags: list[str] | None,
        source_file: str | None,
        entities: list[str] | None = None,
    ) -> None:
        self._pool.execute(
            "DELETE FROM knowledge_terms WHERE knowledge_id = %s",
            (knowledge_id,),
        )
        terms = self._build_knowledge_terms(
            title=title,
            content=content,
            tags=tags,
            source_file=source_file,
            entities=entities,
        )
        if not terms:
            return
        rows = [
            (project_id, knowledge_id, term, field, weight)
            for field, term, weight in terms
        ]
        values_sql = ", ".join(["(%s, %s, %s, %s, %s)"] * len(rows))
        args: list[Any] = []
        for row in rows:
            args.extend(row)
        self._pool.execute(
            "INSERT INTO knowledge_terms "
            "(project_id, knowledge_id, term, field, weight) "
            f"VALUES {values_sql}",
            tuple(args),
        )

    def _replace_knowledge_graph(
        self,
        *,
        project_id: int,
        knowledge_id: int,
        title: str,
        content: str,
        tags: list[str] | None,
        source_file: str | None,
    ) -> list[str]:
        self._pool.execute(
            "DELETE FROM knowledge_entities WHERE knowledge_id = %s",
            (knowledge_id,),
        )
        self._pool.execute(
            "DELETE FROM knowledge_relations WHERE knowledge_id = %s",
            (knowledge_id,),
        )
        entities = self._extract_entities(
            title=title,
            content=content,
            tags=tags,
            source_file=source_file,
        )
        if entities:
            entity_rows = [
                (project_id, knowledge_id, entity, 0.8)
                for entity in entities
            ]
            values_sql = ", ".join(["(%s, %s, %s, %s)"] * len(entity_rows))
            args: list[Any] = []
            for row in entity_rows:
                args.extend(row)
            self._pool.execute(
                "INSERT INTO knowledge_entities "
                "(project_id, knowledge_id, entity_name, confidence) "
                f"VALUES {values_sql}",
                tuple(args),
            )

        relations = self._extract_relations(title=title, content=content)
        if relations:
            relation_rows = [
                (project_id, knowledge_id, source, relation, target, 0.7)
                for source, relation, target in relations
            ]
            values_sql = ", ".join(["(%s, %s, %s, %s, %s, %s)"] * len(relation_rows))
            args = []
            for row in relation_rows:
                args.extend(row)
            self._pool.execute(
                "INSERT INTO knowledge_relations "
                "(project_id, knowledge_id, source_entity, relation, target_entity, confidence) "
                f"VALUES {values_sql}",
                tuple(args),
            )
        return entities

    def _query_terms(self, query: str) -> list[str]:
        term_map: dict[tuple[str, str], float] = {}
        for field, term, _weight in self._extract_terms_for_field(
            field="content",
            text=query,
            structured=True,
            include_phrase_terms=True,
        ):
            term_map[(field, term)] = 1.0
        return [term for (_field, term) in term_map.keys()]

    def _find_exact_knowledge_in_scope(
        self,
        *,
        project_id: int,
        category: str,
        source_type: str,
        source_file: str | None,
        content_hash: str,
    ) -> dict | None:
        scope_sql, args = self._scope_sql_and_args(
            project_id=project_id,
            category=category,
            source_type=source_type,
            source_file=source_file,
        )
        sql = (
            "SELECT * FROM knowledge k "
            f"WHERE {scope_sql} AND k.content_hash = %s "
            "AND (k.expires_at IS NULL OR k.expires_at > NOW()) "
            "ORDER BY k.updated_at DESC LIMIT 1"
        )
        args.append(content_hash)
        return self._parse_knowledge_row(self._pool.fetch_one(sql, tuple(args)))

    def _find_merge_candidates(
        self,
        *,
        project_id: int,
        category: str,
        source_type: str,
        source_file: str | None,
        normalized_title: str,
        limit: int = 5,
    ) -> list[dict]:
        scope_sql, args = self._scope_sql_and_args(
            project_id=project_id,
            category=category,
            source_type=source_type,
            source_file=source_file,
        )
        sql = (
            "SELECT * FROM knowledge k "
            f"WHERE {scope_sql} "
            "AND (k.expires_at IS NULL OR k.expires_at > NOW()) "
            "ORDER BY k.version DESC, k.updated_at DESC LIMIT %s"
        )
        args.append(max(limit * 5, limit))
        rows = self._pool.fetch_all(sql, tuple(args))
        filtered: list[dict] = []
        for row in rows:
            parsed = self._parse_knowledge_row(row)
            if not parsed:
                continue
            if self._normalize_text(str(parsed.get("title") or "")) == normalized_title:
                filtered.append(parsed)
            if len(filtered) >= limit:
                break
        return filtered

    def _compute_merge_score(
        self,
        *,
        title: str,
        content: str,
        candidate: dict,
        source_type: str,
        source_file: str | None,
    ) -> float:
        score = 0.0
        if self._normalize_text(candidate.get("title") or "") == self._normalize_text(title):
            score += 0.6
        score += 0.4 * self._token_overlap_ratio(content, candidate.get("content") or "")
        if source_type == "document" and candidate.get("source_file") == source_file:
            score += 0.05
        return min(score, 1.0)

    def _update_knowledge_after_merge(
        self,
        *,
        knowledge_id: int,
        project_id: int,
        title: str,
        content: str,
        source_file: str | None,
        tags: list[str] | None,
        relevance_score: float,
        confidence: float,
        search_metadata_json: dict[str, Any] | None,
        previous_row: dict,
        compute_embedding: bool,
        content_hash: str,
        merged_from_entry: dict[str, Any] | None = None,
    ) -> None:
        vec_sql = "NULL"
        embed_args: list[Any] = []
        if compute_embedding:
            vec_sql, embed_args = embedder.resolve_embed_sql(
                f"{title}\n{content}",
                self._tidb_embedding_model,
                backend=self._embedding_backend,
            )

        merged_from = self._ensure_list_of_dicts(previous_row.get("merged_from")) or []
        if merged_from_entry is not None:
            merged_from.append(merged_from_entry)

        sql = (
            "UPDATE knowledge SET "
            "title = %s, content = %s, embedding = "
            f"{vec_sql}, tags = %s, relevance_score = %s, confidence = %s, "
            "content_hash = %s, version = %s, search_metadata_json = %s, "
            "merged_from = %s "
            "WHERE id = %s"
        )
        args: list[Any] = [title, content]
        args.extend(embed_args)
        args.extend([
            json.dumps(tags) if tags else None,
            relevance_score,
            confidence,
            content_hash,
            int(previous_row.get("version") or 1) + 1,
            json.dumps(search_metadata_json) if search_metadata_json else None,
            json.dumps(merged_from) if merged_from else None,
            knowledge_id,
        ])
        self._pool.execute(sql, tuple(args))
        entities = self._replace_knowledge_graph(
            project_id=project_id,
            knowledge_id=knowledge_id,
            title=title,
            content=content,
            tags=tags,
            source_file=source_file,
        )
        self._replace_knowledge_terms(
            project_id=project_id,
            knowledge_id=knowledge_id,
            title=title,
            content=content,
            tags=tags,
            source_file=source_file,
            entities=entities,
        )
        self._bump_project_memory_revision(project_id)

    def _merge_with_runner(
        self,
        *,
        runner: "AgentRunner",
        title: str,
        existing_content: str,
        new_content: str,
    ) -> str | None:
        try:
            prompt = MERGE_PROMPT.format(
                title=title,
                existing_content=existing_content,
                new_content=new_content,
            )
            sent_prompt, _ = maybe_externalize_prompt(
                prompt,
                label="memory-merge-request",
            )
            response = runner.invoke(
                sent_prompt,
                system_context=AGENT_FILE_PROMPT,
            )
        except Exception:
            return None
        if not getattr(response, "success", False):
            return None
        merged = resolve_externalized_text(getattr(response, "content", "")).strip()
        return merged or None

    def _supersede_knowledge(
        self,
        *,
        knowledge_id: int,
        project_id: int,
    ) -> None:
        self._pool.execute(
            "UPDATE knowledge SET expires_at = NOW(), relevance_score = 0.0 WHERE id = %s",
            (knowledge_id,),
        )
        self._bump_project_memory_revision(project_id)

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

    def get_project_memory_revision(self, project_id: int) -> int:
        row = self._pool.fetch_one(
            "SELECT memory_revision FROM projects WHERE id = %s",
            (project_id,),
        )
        if not row:
            return 0
        return int(row.get("memory_revision") or 0)

    def delete_project(self, project_id: int) -> dict[str, int]:
        """Delete one project and all related state in a single transaction."""
        counts = {
            "session_turns": 0,
            "coordination_events": 0,
            "review_cycles": 0,
            "stage_runs": 0,
            "artifacts": 0,
            "learn_runs": 0,
            "learn_requests": 0,
            "runtime_registrations": 0,
            "document_sources": 0,
            "knowledge_terms": 0,
            "knowledge_entities": 0,
            "knowledge_relations": 0,
            "knowledge": 0,
            "sessions": 0,
            "work_items": 0,
            "agents": 0,
            "projects": 0,
        }
        with self._pool.connection() as conn:
            conn.autocommit(False)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE work_items SET parent_item_id = NULL WHERE project_id = %s",
                        (project_id,),
                    )
                    cur.execute(
                        "DELETE st FROM session_turns st "
                        "JOIN sessions s ON st.session_id = s.id "
                        "JOIN agents a ON s.agent_id = a.id "
                        "WHERE a.project_id = %s",
                        (project_id,),
                    )
                    counts["session_turns"] = cur.rowcount

                    cur.execute(
                        "DELETE ce FROM coordination_events ce "
                        "JOIN work_items wi ON ce.work_item_id = wi.id "
                        "WHERE wi.project_id = %s",
                        (project_id,),
                    )
                    counts["coordination_events"] = cur.rowcount

                    cur.execute(
                        "DELETE rc FROM review_cycles rc "
                        "JOIN work_items wi ON rc.work_item_id = wi.id "
                        "WHERE wi.project_id = %s",
                        (project_id,),
                    )
                    counts["review_cycles"] = cur.rowcount

                    cur.execute(
                        "DELETE sr FROM stage_runs sr "
                        "JOIN work_items wi ON sr.work_item_id = wi.id "
                        "WHERE wi.project_id = %s",
                        (project_id,),
                    )
                    counts["stage_runs"] = cur.rowcount

                    cur.execute(
                        "DELETE ar FROM artifacts ar "
                        "JOIN work_items wi ON ar.work_item_id = wi.id "
                        "WHERE wi.project_id = %s",
                        (project_id,),
                    )
                    counts["artifacts"] = cur.rowcount

                    cur.execute(
                        "DELETE lr FROM learn_runs lr "
                        "JOIN learn_requests lq ON lr.learn_request_id = lq.id "
                        "WHERE lq.project_id = %s",
                        (project_id,),
                    )
                    counts["learn_runs"] = cur.rowcount

                    cur.execute("DELETE FROM learn_requests WHERE project_id = %s", (project_id,))
                    counts["learn_requests"] = cur.rowcount

                    cur.execute(
                        "DELETE FROM runtime_registrations WHERE project_id = %s",
                        (project_id,),
                    )
                    counts["runtime_registrations"] = cur.rowcount

                    cur.execute("DELETE FROM document_sources WHERE project_id = %s", (project_id,))
                    counts["document_sources"] = cur.rowcount

                    cur.execute("DELETE FROM knowledge_terms WHERE project_id = %s", (project_id,))
                    counts["knowledge_terms"] = cur.rowcount

                    cur.execute("DELETE FROM knowledge_entities WHERE project_id = %s", (project_id,))
                    counts["knowledge_entities"] = cur.rowcount

                    cur.execute("DELETE FROM knowledge_relations WHERE project_id = %s", (project_id,))
                    counts["knowledge_relations"] = cur.rowcount

                    cur.execute("DELETE FROM knowledge WHERE project_id = %s", (project_id,))
                    counts["knowledge"] = cur.rowcount

                    cur.execute(
                        "DELETE s FROM sessions s "
                        "JOIN agents a ON s.agent_id = a.id "
                        "WHERE a.project_id = %s",
                        (project_id,),
                    )
                    counts["sessions"] = cur.rowcount

                    cur.execute("DELETE FROM work_items WHERE project_id = %s", (project_id,))
                    counts["work_items"] = cur.rowcount

                    cur.execute("DELETE FROM agents WHERE project_id = %s", (project_id,))
                    counts["agents"] = cur.rowcount

                    cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
                    counts["projects"] = cur.rowcount
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.autocommit(True)
        return counts

    # ──────────────────────────── Learn History ────────────────────────────

    def create_learn_request(
        self,
        *,
        project_id: int,
        source_kind: str,
        trigger_kind: str,
        payload_json: dict[str, Any] | None = None,
        source_session_id: int | None = None,
        source_work_item_id: int | None = None,
        status: str = "pending",
    ) -> int:
        return self._pool.insert_returning_id(
            "INSERT INTO learn_requests "
            "(project_id, source_kind, trigger_kind, source_session_id, source_work_item_id, "
            "payload_json, status) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                project_id,
                source_kind,
                trigger_kind,
                source_session_id,
                source_work_item_id,
                json.dumps(payload_json or {}, default=str),
                status,
            ),
        )

    def get_learn_request(self, request_id: int) -> dict | None:
        row = self._pool.fetch_one(
            "SELECT * FROM learn_requests WHERE id = %s",
            (request_id,),
        )
        if not row:
            return None
        parsed = dict(row)
        parsed["payload_json"] = self._parse_json_field(parsed.get("payload_json")) or {}
        return parsed

    def update_learn_request_status(self, request_id: int, status: str) -> None:
        self._pool.execute(
            "UPDATE learn_requests SET status = %s WHERE id = %s",
            (status, request_id),
        )

    def create_learn_run(
        self,
        *,
        learn_request_id: int,
        worker_backend: str,
        worker_model: str,
        input_context_json: dict[str, Any],
        status: str = "started",
    ) -> int:
        return self._pool.insert_returning_id(
            "INSERT INTO learn_runs "
            "(learn_request_id, worker_backend, worker_model, input_context_json, status) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                learn_request_id,
                worker_backend,
                worker_model,
                json.dumps(input_context_json, default=str),
                status,
            ),
        )

    def get_learn_run(self, run_id: int) -> dict | None:
        row = self._pool.fetch_one(
            "SELECT * FROM learn_runs WHERE id = %s",
            (run_id,),
        )
        if not row:
            return None
        parsed = dict(row)
        parsed["input_context_json"] = (
            self._parse_json_field(parsed.get("input_context_json")) or {}
        )
        parsed["output_envelope_json"] = self._parse_json_field(
            parsed.get("output_envelope_json"),
        )
        return parsed

    def complete_learn_run(
        self,
        run_id: int,
        *,
        output_envelope_json: dict[str, Any],
    ) -> None:
        self._pool.execute(
            "UPDATE learn_runs SET status = 'completed', output_envelope_json = %s, error_text = NULL "
            "WHERE id = %s",
            (json.dumps(output_envelope_json, default=str), run_id),
        )

    def fail_learn_run(self, run_id: int, *, error_text: str) -> None:
        self._pool.execute(
            "UPDATE learn_runs SET status = 'failed', error_text = %s WHERE id = %s",
            (error_text[:65535], run_id),
        )

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

    def get_agent_by_id(self, agent_id: int) -> dict | None:
        return self._pool.fetch_one(
            "SELECT * FROM agents WHERE id = %s",
            (agent_id,),
        )

    def update_agent_system_prompt(self, agent_id: int, system_prompt: str | None) -> None:
        self._pool.execute(
            "UPDATE agents SET system_prompt = %s WHERE id = %s",
            (system_prompt, agent_id),
        )

    def list_agents(self, project_id: int) -> list[dict]:
        return self._pool.fetch_all(
            "SELECT * FROM agents WHERE project_id = %s ORDER BY role", (project_id,),
        )

    # ─────────────────────── Runtime Registrations ───────────────────────

    def register_runtime(
        self,
        *,
        project_id: int,
        runtime_name: str,
        runtime_kind: str,
        agent_role: str,
        agent_id: int | None = None,
        endpoint: str | None = None,
        capabilities_json: dict[str, Any] | None = None,
        metadata_json: dict[str, Any] | None = None,
        lease_seconds: int = 300,
    ) -> int:
        lease_expires_at = datetime.now() + timedelta(seconds=max(lease_seconds, 30))
        return self._pool.insert_returning_id(
            "INSERT INTO runtime_registrations "
            "(project_id, agent_id, agent_role, runtime_name, runtime_kind, endpoint, "
            "status, capabilities_json, metadata_json, last_heartbeat_at, lease_expires_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, 'online', %s, %s, NOW(), %s)",
            (
                project_id,
                agent_id,
                agent_role,
                runtime_name,
                runtime_kind,
                endpoint,
                json.dumps(capabilities_json) if capabilities_json else None,
                json.dumps(metadata_json) if metadata_json else None,
                lease_expires_at,
            ),
        )

    def get_runtime_registration(self, runtime_registration_id: int) -> RuntimeRegistration | None:
        row = self._pool.fetch_one(
            "SELECT * FROM runtime_registrations WHERE id = %s",
            (runtime_registration_id,),
        )
        return RuntimeRegistration(**row) if row else None

    def heartbeat_runtime(
        self,
        runtime_registration_id: int,
        *,
        lease_seconds: int = 300,
        metadata_json: dict[str, Any] | None = None,
    ) -> None:
        runtime = self._pool.fetch_one(
            "SELECT metadata_json FROM runtime_registrations WHERE id = %s",
            (runtime_registration_id,),
        )
        merged_metadata = self._parse_json_field(runtime.get("metadata_json")) if runtime else None
        metadata = dict(merged_metadata) if isinstance(merged_metadata, dict) else {}
        if metadata_json:
            metadata.update(metadata_json)
        lease_expires_at = datetime.now() + timedelta(seconds=max(lease_seconds, 30))
        self._pool.execute(
            "UPDATE runtime_registrations SET status = 'online', last_heartbeat_at = NOW(), "
            "lease_expires_at = %s, metadata_json = %s WHERE id = %s",
            (
                lease_expires_at,
                json.dumps(metadata) if metadata else None,
                runtime_registration_id,
            ),
        )

    def list_runtime_registrations(
        self,
        project_id: int,
        *,
        agent_role: str | None = None,
        status: str | None = None,
    ) -> list[RuntimeRegistration]:
        sql = "SELECT * FROM runtime_registrations WHERE project_id = %s"
        args: list[Any] = [project_id]
        if agent_role is not None:
            sql += " AND agent_role = %s"
            args.append(agent_role)
        if status is not None:
            sql += " AND status = %s"
            args.append(status)
        sql += " ORDER BY updated_at DESC, id DESC"
        rows = self._pool.fetch_all(sql, tuple(args))
        return [RuntimeRegistration(**row) for row in rows]

    def update_runtime_status(
        self,
        runtime_registration_id: int,
        *,
        status: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> None:
        runtime = self._pool.fetch_one(
            "SELECT metadata_json FROM runtime_registrations WHERE id = %s",
            (runtime_registration_id,),
        )
        merged_metadata = self._parse_json_field(runtime.get("metadata_json")) if runtime else None
        metadata = dict(merged_metadata) if isinstance(merged_metadata, dict) else {}
        if metadata_json:
            metadata.update(metadata_json)
        self._pool.execute(
            "UPDATE runtime_registrations SET status = %s, metadata_json = %s WHERE id = %s",
            (
                status,
                json.dumps(metadata) if metadata else None,
                runtime_registration_id,
            ),
        )

    # ──────────────────────────── Sessions ────────────────────────────

    def create_session(
        self, agent_id: int, purpose: str | None = None,
        work_item_id: int | None = None, parent_session_id: int | None = None,
    ) -> Session:
        session_uuid = str(uuid.uuid4())
        sid = self._pool.insert_returning_id(
            "INSERT INTO sessions (agent_id, session_uuid, parent_session_id, purpose, work_item_id, "
            "memory_revision_at_context_build) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (agent_id, session_uuid, parent_session_id, purpose, work_item_id, None),
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

    def mark_session_fully_compacted(self, session_id: int) -> None:
        """Mark a session as fully compacted with a GC timestamp."""
        self._pool.execute(
            "UPDATE sessions SET status = 'compacted', compacted_at = NOW() WHERE id = %s",
            (session_id,),
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

    def set_session_memory_revision(self, session_id: int, memory_revision: int) -> None:
        self._pool.execute(
            "UPDATE sessions SET memory_revision_at_context_build = %s WHERE id = %s",
            (memory_revision, session_id),
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

    def get_recent_turns_by_project(
        self,
        project_id: int,
        per_role_limit: int = 10,
        exclude_session_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch recent raw turns across all project roles, grouped by role."""
        sql = (
            "SELECT role, content, created_at, agent_role, turn_id, session_id FROM ("
            "SELECT st.role, st.content, st.created_at, st.id AS turn_id, st.session_id, "
            "a.role AS agent_role, "
            "ROW_NUMBER() OVER ("
            "PARTITION BY a.role ORDER BY st.created_at DESC, st.id DESC"
            ") AS rn "
            "FROM session_turns st "
            "JOIN sessions s ON st.session_id = s.id "
            "JOIN agents a ON s.agent_id = a.id "
            "WHERE a.project_id = %s"
        )
        args: list[Any] = [project_id]
        if exclude_session_id is not None:
            sql += " AND s.id != %s"
            args.append(exclude_session_id)
        sql += (
            ") ranked "
            "WHERE rn <= %s "
            "ORDER BY agent_role, created_at ASC, turn_id ASC"
        )
        args.append(per_role_limit)

        rows = self._pool.fetch_all(sql, tuple(args))
        if not rows:
            return []

        grouped: list[dict[str, Any]] = []
        current_role: str | None = None
        current_group: dict[str, Any] | None = None

        for row in rows:
            agent_role = row["agent_role"]
            if agent_role != current_role:
                current_group = {"agent_role": agent_role, "turns": []}
                grouped.append(current_group)
                current_role = agent_role

            assert current_group is not None
            current_group["turns"].append({
                "turn_id": row.get("turn_id"),
                "session_id": row.get("session_id"),
                "role": row["role"],
                "content": row["content"],
                "created_at": row.get("created_at"),
            })

        return grouped

    def get_recent_turns_global(
        self,
        project_id: int,
        limit: int = 50,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch recent raw turns across a project in chronological order."""
        sql = (
            "SELECT st.id AS turn_id, st.session_id, st.turn_index, st.role, st.content, "
            "st.created_at, a.role AS agent_role "
            "FROM session_turns st "
            "JOIN sessions s ON st.session_id = s.id "
            "JOIN agents a ON s.agent_id = a.id "
            "WHERE a.project_id = %s"
        )
        args: list[Any] = [project_id]
        if role is not None:
            sql += " AND a.role = %s"
            args.append(role)
        sql += " ORDER BY st.created_at DESC, st.id DESC LIMIT %s"
        args.append(limit)

        rows = self._pool.fetch_all(sql, tuple(args))
        return list(reversed(rows))

    def gc_compacted_turns(
        self,
        project_id: int,
        grace_days: int = 7,
        keep_recent: int = 50,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Delete old raw turns from fully compacted sessions while preserving recent history."""
        cutoff = self._pool.fetch_one(
            "SELECT created_at, id FROM ("
            "SELECT st.created_at, st.id "
            "FROM session_turns st "
            "JOIN sessions s ON st.session_id = s.id "
            "JOIN agents a ON s.agent_id = a.id "
            "WHERE a.project_id = %s "
            "ORDER BY st.created_at DESC, st.id DESC "
            "LIMIT 1 OFFSET %s"
            ") AS cutoff_row",
            (project_id, keep_recent - 1),
        )
        if not cutoff:
            return {"turns_deleted": 0, "sessions_affected": 0}

        where_sql = (
            "FROM session_turns st "
            "JOIN sessions s ON st.session_id = s.id "
            "JOIN agents a ON s.agent_id = a.id "
            "WHERE a.project_id = %s "
            "AND s.status = 'compacted' "
            "AND s.compacted_at < NOW() - INTERVAL %s DAY "
            "AND (st.created_at < %s OR (st.created_at = %s AND st.id < %s))"
        )
        where_args = (
            project_id,
            grace_days,
            cutoff["created_at"],
            cutoff["created_at"],
            cutoff["id"],
        )

        counts = self._pool.fetch_one(
            "SELECT COUNT(*) AS turns_deleted, COUNT(DISTINCT s.id) AS sessions_affected "
            + where_sql,
            where_args,
        )
        turns_deleted = counts["turns_deleted"] if counts else 0
        sessions_affected = counts["sessions_affected"] if counts else 0

        if dry_run or turns_deleted == 0:
            return {
                "turns_deleted": turns_deleted,
                "sessions_affected": sessions_affected,
            }

        self._pool.execute(
            "DELETE st " + where_sql,
            where_args,
        )
        return {
            "turns_deleted": turns_deleted,
            "sessions_affected": sessions_affected,
        }

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

    def get_knowledge(self, knowledge_id: int) -> dict | None:
        return self._parse_knowledge_row(
            self._pool.fetch_one("SELECT * FROM knowledge WHERE id = %s", (knowledge_id,)),
        )

    def find_active_knowledge(
        self,
        *,
        project_id: int,
        category: str,
        title: str,
        source_type: str | None = None,
        source_file: str | None = None,
    ) -> dict | None:
        sql = (
            "SELECT * FROM knowledge WHERE project_id = %s AND category = %s AND title = %s "
            "AND (expires_at IS NULL OR expires_at > NOW())"
        )
        args: list[Any] = [project_id, category, title]
        if source_type is not None:
            sql += " AND source_type = %s"
            args.append(source_type)
        if source_file is not None:
            sql += " AND source_file = %s"
            args.append(source_file)
        sql += " ORDER BY updated_at DESC, id DESC LIMIT 1"
        return self._parse_knowledge_row(self._pool.fetch_one(sql, tuple(args)))

    def add_knowledge_relation(
        self,
        *,
        project_id: int,
        knowledge_id: int,
        source_entity: str,
        relation: str,
        target_entity: str,
        confidence: float,
    ) -> None:
        self._pool.execute(
            "INSERT INTO knowledge_relations "
            "(project_id, knowledge_id, source_entity, relation, target_entity, confidence) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                project_id,
                knowledge_id,
                source_entity,
                relation,
                target_entity,
                confidence,
            ),
        )

    def delete_knowledge_relation(
        self,
        *,
        knowledge_id: int,
        source_entity: str,
        relation: str,
        target_entity: str,
    ) -> None:
        self._pool.execute(
            "DELETE FROM knowledge_relations WHERE knowledge_id = %s AND source_entity = %s "
            "AND relation = %s AND target_entity = %s",
            (
                knowledge_id,
                source_entity,
                relation,
                target_entity,
            ),
        )

    def replace_knowledge_index_hints(
        self,
        *,
        project_id: int,
        knowledge_id: int,
        terms: list[dict[str, Any]],
        entities: list[dict[str, Any]],
    ) -> None:
        if terms:
            self._pool.execute(
                "DELETE FROM knowledge_terms WHERE knowledge_id = %s",
                (knowledge_id,),
            )
            placeholders = ", ".join(["(%s, %s, %s, %s, %s)"] * len(terms))
            args: list[Any] = []
            for term in terms:
                args.extend(
                    [
                        project_id,
                        knowledge_id,
                        term["term"],
                        term["field"],
                        term["weight"],
                    ]
                )
            self._pool.execute(
                "INSERT INTO knowledge_terms "
                "(project_id, knowledge_id, term, field, weight) "
                f"VALUES {placeholders}",
                tuple(args),
            )

        if entities:
            self._pool.execute(
                "DELETE FROM knowledge_entities WHERE knowledge_id = %s",
                (knowledge_id,),
            )
            placeholders = ", ".join(["(%s, %s, %s, %s)"] * len(entities))
            args: list[Any] = []
            for entity in entities:
                args.extend(
                    [
                        project_id,
                        knowledge_id,
                        entity["entity_name"],
                        entity["confidence"],
                    ]
                )
            self._pool.execute(
                "INSERT INTO knowledge_entities "
                "(project_id, knowledge_id, entity_name, confidence) "
                f"VALUES {placeholders}",
                tuple(args),
            )

    def replace_knowledge(
        self,
        *,
        knowledge_id: int,
        project_id: int,
        category: str | None = None,
        title: str | None = None,
        content: str | None = None,
        agent_id: int | None = None,
        source_session_id: int | None = None,
        source_turn_ids: list[int] | None = None,
        source_file: str | None = None,
        source_type: str | None = None,
        tags: list[str] | None = None,
        relevance_score: float | None = None,
        confidence: float | None = None,
        ttl_days: int | None = None,
        compute_embedding: bool = True,
        search_metadata_json: dict[str, Any] | None = None,
        content_hash: str | None = None,
        bump_revision: bool = False,
        refresh_derived_indexes: bool = True,
    ) -> None:
        previous = self.get_knowledge(knowledge_id)
        if not previous:
            raise ValueError(f"Knowledge entry {knowledge_id} not found")

        effective_category = category or str(previous.get("category") or "")
        effective_title = title or str(previous.get("title") or "")
        effective_content = content if content is not None else str(previous.get("content") or "")
        effective_source_file = (
            source_file if source_file is not None else previous.get("source_file")
        )
        effective_source_type = self._infer_source_type(
            source_type=source_type or str(previous.get("source_type") or "") or None,
            category=effective_category,
            source_file=effective_source_file,
        )
        effective_tags = tags
        if effective_tags is None:
            parsed_tags = previous.get("tags")
            effective_tags = parsed_tags if isinstance(parsed_tags, list) else None
        else:
            effective_tags = [str(tag).strip() for tag in effective_tags if str(tag).strip()] or None

        effective_agent_id = agent_id if agent_id is not None else previous.get("agent_id")
        effective_source_session_id = (
            source_session_id
            if source_session_id is not None
            else previous.get("source_session_id")
        )
        effective_source_turn_ids = (
            source_turn_ids if source_turn_ids is not None else previous.get("source_turn_ids")
        )
        effective_relevance = (
            relevance_score if relevance_score is not None else float(previous.get("relevance_score") or 1.0)
        )
        effective_confidence = (
            confidence if confidence is not None else float(previous.get("confidence") or 1.0)
        )
        effective_ttl_days = ttl_days if ttl_days is not None else previous.get("ttl_days")
        effective_expires_at = previous.get("expires_at")
        if ttl_days is not None:
            effective_expires_at = datetime.now() + timedelta(days=ttl_days)

        previous_metadata = self._ensure_dict(previous.get("search_metadata_json")) or {}
        effective_metadata = dict(previous_metadata)
        if search_metadata_json:
            effective_metadata.update(search_metadata_json)
        effective_metadata = self._build_default_search_metadata(
            source_type=effective_source_type,
            source_file=effective_source_file,
            tags=effective_tags,
            search_metadata_json=effective_metadata,
        )
        effective_content_hash = content_hash or self._compute_content_hash(effective_content)

        vec_sql = "NULL"
        embed_args: list[Any] = []
        if compute_embedding:
            vec_sql, embed_args = embedder.resolve_embed_sql(
                f"{effective_title}\n{effective_content}",
                self._tidb_embedding_model,
                backend=self._embedding_backend,
            )
        previous_version = int(previous.get("version") or 1)

        sql = (
            "UPDATE knowledge SET "
            "agent_id = %s, source_session_id = %s, source_turn_ids = %s, "
            "source_file = %s, source_type = %s, category = %s, title = %s, content = %s, "
            f"embedding = {vec_sql}, tags = %s, relevance_score = %s, confidence = %s, "
            "ttl_days = %s, expires_at = %s, content_hash = %s, version = %s, "
            "search_metadata_json = %s "
            "WHERE id = %s AND version = %s"
        )
        args: list[Any] = [
            effective_agent_id,
            effective_source_session_id,
            json.dumps(effective_source_turn_ids) if effective_source_turn_ids else None,
            effective_source_file,
            effective_source_type,
            effective_category,
            effective_title,
            effective_content,
        ]
        args.extend(embed_args)
        args.extend(
            [
                json.dumps(effective_tags) if effective_tags else None,
                effective_relevance,
                effective_confidence,
                effective_ttl_days,
                effective_expires_at,
                effective_content_hash,
                previous_version + 1,
                json.dumps(effective_metadata) if effective_metadata else None,
                knowledge_id,
                previous_version,
            ]
        )
        updated = self._pool.execute(sql, tuple(args))
        if updated != 1:
            raise RuntimeError(
                f"Knowledge entry {knowledge_id} was modified concurrently",
            )
        if refresh_derived_indexes:
            entities = self._replace_knowledge_graph(
                project_id=project_id,
                knowledge_id=knowledge_id,
                title=effective_title,
                content=effective_content,
                tags=effective_tags,
                source_file=effective_source_file,
            )
            self._replace_knowledge_terms(
                project_id=project_id,
                knowledge_id=knowledge_id,
                title=effective_title,
                content=effective_content,
                tags=effective_tags,
                source_file=effective_source_file,
                entities=entities,
            )
        if bump_revision:
            self._bump_project_memory_revision(project_id)

    def expire_knowledge(self, knowledge_id: int, *, project_id: int) -> None:
        self._supersede_knowledge(knowledge_id=knowledge_id, project_id=project_id)

    def store_knowledge(
        self, project_id: int, category: str, title: str, content: str,
        agent_id: int | None = None, source_session_id: int | None = None,
        source_turn_ids: list[int] | None = None, source_file: str | None = None,
        tags: list[str] | None = None, relevance_score: float = 1.0,
        confidence: float = 1.0, ttl_days: int | None = None,
        compute_embedding: bool = True, source_type: str | None = None,
        search_metadata_json: dict[str, Any] | None = None,
        merged_from: list[dict[str, Any]] | None = None,
        content_hash: str | None = None,
        version: int = 1,
        bump_revision: bool = False,
        refresh_derived_indexes: bool = True,
    ) -> int:
        effective_source_type = self._infer_source_type(
            source_type=source_type,
            category=category,
            source_file=source_file,
        )
        effective_content_hash = content_hash or self._compute_content_hash(content)
        effective_metadata = self._build_default_search_metadata(
            source_type=effective_source_type,
            source_file=source_file,
            tags=tags,
            search_metadata_json=search_metadata_json,
        )
        vec_sql = "NULL"
        embed_args: list[Any] = []
        if compute_embedding:
            vec_sql, embed_args = embedder.resolve_embed_sql(
                f"{title}\n{content}",
                self._tidb_embedding_model,
                backend=self._embedding_backend,
            )

        expires_at = None
        if ttl_days is not None:
            expires_at = datetime.now() + timedelta(days=ttl_days)

        sql = (
            "INSERT INTO knowledge (project_id, agent_id, source_session_id, "
            "source_turn_ids, source_file, source_type, category, title, content, embedding, "
            "tags, relevance_score, confidence, ttl_days, expires_at, content_hash, "
            "version, search_metadata_json, merged_from) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, {vec_sql}, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        args: list[Any] = [
            project_id, agent_id, source_session_id,
            json.dumps(source_turn_ids) if source_turn_ids else None,
            source_file, effective_source_type, category, title, content,
        ]
        args.extend(embed_args)
        args.extend([
            json.dumps(tags) if tags else None,
            relevance_score, confidence, ttl_days, expires_at,
            effective_content_hash,
            version,
            json.dumps(effective_metadata) if effective_metadata else None,
            json.dumps(merged_from) if merged_from else None,
        ])

        knowledge_id = self._pool.insert_returning_id(sql, tuple(args))
        if refresh_derived_indexes:
            entities = self._replace_knowledge_graph(
                project_id=project_id,
                knowledge_id=knowledge_id,
                title=title,
                content=content,
                tags=tags,
                source_file=source_file,
            )
            self._replace_knowledge_terms(
                project_id=project_id,
                knowledge_id=knowledge_id,
                title=title,
                content=content,
                tags=tags,
                source_file=source_file,
                entities=entities,
            )
        if bump_revision:
            self._bump_project_memory_revision(project_id)
        return knowledge_id

    def upsert_knowledge(
        self,
        project_id: int,
        category: str,
        title: str,
        content: str,
        *,
        source_type: str | None = None,
        agent_id: int | None = None,
        source_session_id: int | None = None,
        source_turn_ids: list[int] | None = None,
        source_file: str | None = None,
        tags: list[str] | None = None,
        relevance_score: float = 1.0,
        confidence: float = 1.0,
        ttl_days: int | None = None,
        compute_embedding: bool = True,
        search_metadata_json: dict[str, Any] | None = None,
        merge_threshold: float = 0.85,
        merge_runner: "AgentRunner | None" = None,
    ) -> tuple[int, str]:
        effective_source_type = self._infer_source_type(
            source_type=source_type,
            category=category,
            source_file=source_file,
        )
        effective_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()] or None
        effective_hash = self._compute_content_hash(content)
        effective_metadata = self._build_default_search_metadata(
            source_type=effective_source_type,
            source_file=source_file,
            tags=effective_tags,
            search_metadata_json=search_metadata_json,
        )

        if category in TRANSIENT_KNOWLEDGE_CATEGORIES:
            knowledge_id = self.store_knowledge(
                project_id=project_id,
                category=category,
                title=title,
                content=content,
                agent_id=agent_id,
                source_session_id=source_session_id,
                source_turn_ids=source_turn_ids,
                source_file=source_file,
                tags=effective_tags,
                relevance_score=relevance_score,
                confidence=confidence,
                ttl_days=ttl_days,
                compute_embedding=compute_embedding,
                source_type=effective_source_type,
                search_metadata_json=effective_metadata,
                content_hash=effective_hash,
                bump_revision=True,
            )
            return knowledge_id, "created"

        existing = self._find_exact_knowledge_in_scope(
            project_id=project_id,
            category=category,
            source_type=effective_source_type,
            source_file=source_file,
            content_hash=effective_hash,
        )
        if existing:
            return int(existing["id"]), "skipped"

        normalized_title = self._normalize_text(title)
        candidates = self._find_merge_candidates(
            project_id=project_id,
            category=category,
            source_type=effective_source_type,
            source_file=source_file,
            normalized_title=normalized_title,
        )

        best_candidate: dict | None = None
        best_score = -1.0
        for candidate in candidates:
            score = self._compute_merge_score(
                title=title,
                content=content,
                candidate=candidate,
                source_type=effective_source_type,
                source_file=source_file,
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        normalized_content = self._normalize_text(content)
        if best_candidate is not None:
            existing_content = best_candidate.get("content") or ""
            normalized_existing = self._normalize_text(existing_content)
            existing_tags = best_candidate.get("tags")
            if isinstance(existing_tags, str):
                existing_tags = self._parse_json_field(existing_tags)
            merged_tags = self._merge_tags(existing_tags, effective_tags)

            if normalized_existing in normalized_content:
                merged_from_entry = {
                    "merged_at": datetime.now().isoformat(timespec="seconds"),
                    "source_type": effective_source_type,
                    "source_session_id": source_session_id,
                    "source_turn_ids": source_turn_ids,
                    "source_file": source_file,
                }
                self._update_knowledge_after_merge(
                    knowledge_id=int(best_candidate["id"]),
                    project_id=project_id,
                    title=title,
                    content=content,
                    source_file=source_file,
                    tags=merged_tags,
                    relevance_score=max(relevance_score, float(best_candidate.get("relevance_score") or 0.0)),
                    confidence=max(confidence, float(best_candidate.get("confidence") or 0.0)),
                    search_metadata_json=effective_metadata,
                    previous_row=best_candidate,
                    compute_embedding=compute_embedding,
                    content_hash=effective_hash,
                    merged_from_entry=merged_from_entry,
                )
                return int(best_candidate["id"]), "merged"

            if normalized_content in normalized_existing:
                changed = merged_tags != existing_tags
                if changed:
                    self._update_knowledge_after_merge(
                        knowledge_id=int(best_candidate["id"]),
                        project_id=project_id,
                        title=str(best_candidate.get("title") or title),
                        content=existing_content,
                        source_file=str(best_candidate.get("source_file") or source_file or ""),
                        tags=merged_tags,
                        relevance_score=max(relevance_score, float(best_candidate.get("relevance_score") or 0.0)),
                        confidence=max(confidence, float(best_candidate.get("confidence") or 0.0)),
                        search_metadata_json=effective_metadata or self._ensure_dict(
                            best_candidate.get("search_metadata_json"),
                        ),
                        previous_row=best_candidate,
                        compute_embedding=compute_embedding,
                        content_hash=str(best_candidate.get("content_hash") or effective_hash),
                    )
                    return int(best_candidate["id"]), "merged"
                return int(best_candidate["id"]), "skipped"

            if best_score >= min(merge_threshold, LLM_MERGE_THRESHOLD) and merge_runner is not None:
                merged_content = self._merge_with_runner(
                    runner=merge_runner,
                    title=title,
                    existing_content=existing_content,
                    new_content=content,
                )
                if merged_content:
                    merged_hash = self._compute_content_hash(merged_content)
                    self._update_knowledge_after_merge(
                        knowledge_id=int(best_candidate["id"]),
                        project_id=project_id,
                        title=title,
                        content=merged_content,
                        source_file=source_file,
                        tags=merged_tags,
                        relevance_score=max(relevance_score, float(best_candidate.get("relevance_score") or 0.0)),
                        confidence=max(confidence, float(best_candidate.get("confidence") or 0.0)),
                        search_metadata_json=effective_metadata,
                        previous_row=best_candidate,
                        compute_embedding=compute_embedding,
                        content_hash=merged_hash,
                        merged_from_entry={
                            "merged_at": datetime.now().isoformat(timespec="seconds"),
                            "source_type": effective_source_type,
                            "source_session_id": source_session_id,
                            "source_turn_ids": source_turn_ids,
                            "source_file": source_file,
                        },
                    )
                    return int(best_candidate["id"]), "merged"

            if (
                best_score >= merge_threshold
                and effective_source_type == "document"
                and source_file
                and best_candidate.get("source_file") == source_file
                and self._normalize_text(best_candidate.get("title") or "") == normalized_title
                and confidence >= float(best_candidate.get("confidence") or 0.0)
            ):
                self._supersede_knowledge(
                    knowledge_id=int(best_candidate["id"]),
                    project_id=project_id,
                )
                knowledge_id = self.store_knowledge(
                    project_id=project_id,
                    category=category,
                    title=title,
                    content=content,
                    agent_id=agent_id,
                    source_session_id=source_session_id,
                    source_turn_ids=source_turn_ids,
                    source_file=source_file,
                    tags=effective_tags,
                    relevance_score=relevance_score,
                    confidence=confidence,
                    ttl_days=ttl_days,
                    compute_embedding=compute_embedding,
                    source_type=effective_source_type,
                    search_metadata_json=effective_metadata,
                    content_hash=effective_hash,
                    bump_revision=True,
                )
                return knowledge_id, "superseded"

            if best_score < merge_threshold:
                best_candidate = None

        knowledge_id = self.store_knowledge(
            project_id=project_id,
            category=category,
            title=title,
            content=content,
            agent_id=agent_id,
            source_session_id=source_session_id,
            source_turn_ids=source_turn_ids,
            source_file=source_file,
            tags=effective_tags,
            relevance_score=relevance_score,
            confidence=confidence,
            ttl_days=ttl_days,
            compute_embedding=compute_embedding,
            source_type=effective_source_type,
            search_metadata_json=effective_metadata,
            content_hash=effective_hash,
            bump_revision=True,
        )
        return knowledge_id, "created"

    def search_knowledge(
        self, project_id: int, query: str,
        agent_id: int | None = None, category: str | None = None,
        tags: list[str] | None = None, source_type: str | None = None, limit: int = 10,
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
        if source_type:
            conditions.append("k.source_type = %s")
            where_args.append(source_type)

        # Build score components
        score_parts = ["k.relevance_score"]

        if use_fulltext and query:
            query_terms = self._query_terms(query)
            if query_terms:
                placeholders = ", ".join(["%s"] * len(query_terms))
                score_parts.append(
                    "COALESCE(("
                    "SELECT SUM(kt.weight) FROM knowledge_terms kt "
                    "WHERE kt.project_id = k.project_id "
                    "AND kt.knowledge_id = k.id "
                    f"AND kt.term IN ({placeholders})"
                    "), 0)"
                )
                score_args.extend(query_terms)

        if use_vector and query:
            vec_sql, vec_args = embedder.resolve_embed_sql(
                query,
                self._tidb_embedding_model,
                backend=self._embedding_backend,
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

        try:
            return self._pool.fetch_all(sql, tuple(args))
        except pymysql.err.OperationalError as exc:
            if use_vector and self._is_missing_embedding_function(exc):
                return self.search_knowledge(
                    project_id=project_id,
                    query=query,
                    agent_id=agent_id,
                    category=category,
                    tags=tags,
                    source_type=source_type,
                    limit=limit,
                    use_vector=False,
                    use_fulltext=use_fulltext,
                )
            raise

    def search_knowledge_fulltext_only(
        self, project_id: int, query: str, limit: int = 10, source_type: str | None = None,
    ) -> list[dict]:
        """Keyword-only search (no embedding model needed)."""
        return self.search_knowledge(
            project_id=project_id, query=query, limit=limit,
            use_vector=False, use_fulltext=True, source_type=source_type,
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
        self._pool.execute(
            "DELETE kt FROM knowledge_terms kt "
            "JOIN knowledge k ON kt.knowledge_id = k.id "
            "WHERE k.project_id = %s AND k.category = %s",
            (project_id, category),
        )
        self._pool.execute(
            "DELETE ke FROM knowledge_entities ke "
            "JOIN knowledge k ON ke.knowledge_id = k.id "
            "WHERE k.project_id = %s AND k.category = %s",
            (project_id, category),
        )
        self._pool.execute(
            "DELETE kr FROM knowledge_relations kr "
            "JOIN knowledge k ON kr.knowledge_id = k.id "
            "WHERE k.project_id = %s AND k.category = %s",
            (project_id, category),
        )
        return self._pool.execute(
            "DELETE FROM knowledge WHERE project_id = %s AND category = %s",
            (project_id, category),
        )

    def delete_knowledge_by_source_file(self, project_id: int, source_file: str) -> int:
        self._pool.execute(
            "DELETE kt FROM knowledge_terms kt "
            "JOIN knowledge k ON kt.knowledge_id = k.id "
            "WHERE k.project_id = %s AND k.source_file = %s",
            (project_id, source_file),
        )
        self._pool.execute(
            "DELETE ke FROM knowledge_entities ke "
            "JOIN knowledge k ON ke.knowledge_id = k.id "
            "WHERE k.project_id = %s AND k.source_file = %s",
            (project_id, source_file),
        )
        self._pool.execute(
            "DELETE kr FROM knowledge_relations kr "
            "JOIN knowledge k ON kr.knowledge_id = k.id "
            "WHERE k.project_id = %s AND k.source_file = %s",
            (project_id, source_file),
        )
        return self._pool.execute(
            "DELETE FROM knowledge WHERE project_id = %s AND source_file = %s",
            (project_id, source_file),
        )

    def delete_knowledge(self, knowledge_id: int) -> int:
        """Delete a single knowledge entry by ID."""
        self._pool.execute(
            "DELETE FROM knowledge_terms WHERE knowledge_id = %s", (knowledge_id,),
        )
        self._pool.execute(
            "DELETE FROM knowledge_entities WHERE knowledge_id = %s", (knowledge_id,),
        )
        self._pool.execute(
            "DELETE FROM knowledge_relations WHERE knowledge_id = %s", (knowledge_id,),
        )
        return self._pool.execute(
            "DELETE FROM knowledge WHERE id = %s", (knowledge_id,),
        )

    def decay_relevance(self, decay_factor: float = 0.95) -> int:
        """Reduce relevance scores for aging knowledge entries."""
        return self._pool.execute(
            "UPDATE knowledge SET relevance_score = relevance_score * %s "
            "WHERE relevance_score > 0.1", (decay_factor,),
        )

    def expire_stale_knowledge(self) -> int:
        """Delete knowledge entries past their TTL."""
        self._pool.execute(
            "DELETE kt FROM knowledge_terms kt "
            "JOIN knowledge k ON kt.knowledge_id = k.id "
            "WHERE k.expires_at IS NOT NULL AND k.expires_at < NOW()",
        )
        self._pool.execute(
            "DELETE ke FROM knowledge_entities ke "
            "JOIN knowledge k ON ke.knowledge_id = k.id "
            "WHERE k.expires_at IS NOT NULL AND k.expires_at < NOW()",
        )
        self._pool.execute(
            "DELETE kr FROM knowledge_relations kr "
            "JOIN knowledge k ON kr.knowledge_id = k.id "
            "WHERE k.expires_at IS NOT NULL AND k.expires_at < NOW()",
        )
        return self._pool.execute(
            "DELETE FROM knowledge WHERE expires_at IS NOT NULL AND expires_at < NOW()",
        )

    def get_document_source(self, project_id: int, source_file: str) -> dict | None:
        source_file_hash = self._compute_raw_hash(source_file)
        return self._pool.fetch_one(
            "SELECT * FROM document_sources WHERE project_id = %s AND source_file_hash = %s",
            (project_id, source_file_hash),
        )

    def upsert_document_source(self, project_id: int, source_file: str, content_hash: str) -> None:
        source_file_hash = self._compute_raw_hash(source_file)
        existing = self.get_document_source(project_id, source_file)
        if existing:
            self._pool.execute(
                "UPDATE document_sources SET source_file = %s, content_hash = %s "
                "WHERE project_id = %s AND source_file_hash = %s",
                (source_file, content_hash, project_id, source_file_hash),
            )
            return
        self._pool.insert_returning_id(
            "INSERT INTO document_sources "
            "(project_id, source_file, source_file_hash, content_hash) "
            "VALUES (%s, %s, %s, %s)",
            (project_id, source_file, source_file_hash, content_hash),
        )

    def match_entities(self, project_id: int, query: str, limit: int = 5) -> list[str]:
        query_terms = self._query_terms(query)
        if not query_terms:
            return []
        placeholders = ", ".join(["%s"] * len(query_terms))
        sql = (
            "SELECT DISTINCT entity_name FROM knowledge_entities "
            "WHERE project_id = %s "
            f"AND LOWER(entity_name) IN ({placeholders}) "
            "ORDER BY entity_name LIMIT %s"
        )
        rows = self._pool.fetch_all(sql, tuple([project_id, *query_terms, limit]))
        return [str(row["entity_name"]) for row in rows]

    def get_related_entities(self, project_id: int, entities: list[str], limit: int = 8) -> list[dict]:
        if not entities:
            return []
        normalized = [entity.casefold() for entity in entities if entity]
        if not normalized:
            return []
        placeholders = ", ".join(["%s"] * len(normalized))
        sql = (
            "SELECT source_entity, relation, target_entity FROM knowledge_relations "
            "WHERE project_id = %s AND ("
            f"LOWER(source_entity) IN ({placeholders}) OR LOWER(target_entity) IN ({placeholders})"
            ") LIMIT %s"
        )
        rows = self._pool.fetch_all(
            sql,
            tuple([project_id, *normalized, *normalized, limit]),
        )
        related: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        normalized_set = set(normalized)
        for row in rows:
            source = str(row["source_entity"])
            target = str(row["target_entity"])
            relation = str(row["relation"])
            if source.casefold() in normalized_set:
                related_entity = target
            else:
                related_entity = source
            key = (source.casefold(), relation, target.casefold())
            if key in seen:
                continue
            seen.add(key)
            related.append({
                "source_entity": source,
                "related_entity": related_entity,
                "relation": relation,
            })
        return related

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

    # ──────────────────────── Delivery Slice States ─────────────────────

    def upsert_slice_state(
        self,
        work_item_id: int,
        slice_id: str,
        title: str,
        status: str,
        *,
        metadata_json: str | None = None,
    ) -> None:
        """Insert or update a single slice state row.

        On conflict, resets runtime columns (workspace_branch, workspace_path,
        stage_run_id, review_cycle_id) to NULL so stale values from prior runs
        don't linger after a re-persist.
        """
        self._pool.execute(
            "INSERT INTO delivery_slice_states "
            "(work_item_id, slice_id, title, status, metadata_json) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE title = VALUES(title), "
            "status = VALUES(status), metadata_json = VALUES(metadata_json), "
            "workspace_branch = NULL, workspace_path = NULL, "
            "stage_run_id = NULL, review_cycle_id = NULL",
            (work_item_id, slice_id, title, status, metadata_json),
        )

    def update_slice_state(
        self,
        work_item_id: int,
        slice_id: str,
        *,
        status: str | None = None,
        stage_run_id: int | None = _UNSET,
        review_cycle_id: int | None = _UNSET,
        workspace_branch: str | None = _UNSET,
        workspace_path: str | None = _UNSET,
        metadata_json: str | None = _UNSET,
    ) -> None:
        """Atomic single-row UPDATE with only the provided columns.

        Uses _UNSET sentinel: only columns explicitly passed are included
        in the SET clause. Pass None to null out a column.
        """
        sets: list[str] = []
        params: list[Any] = []

        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if stage_run_id is not _UNSET:
            sets.append("stage_run_id = %s")
            params.append(stage_run_id)
        if review_cycle_id is not _UNSET:
            sets.append("review_cycle_id = %s")
            params.append(review_cycle_id)
        if workspace_branch is not _UNSET:
            sets.append("workspace_branch = %s")
            params.append(workspace_branch)
        if workspace_path is not _UNSET:
            sets.append("workspace_path = %s")
            params.append(workspace_path)
        if metadata_json is not _UNSET:
            sets.append("metadata_json = %s")
            params.append(metadata_json)

        if not sets:
            return

        params.extend([work_item_id, slice_id])
        self._pool.execute(
            f"UPDATE delivery_slice_states SET {', '.join(sets)} "
            f"WHERE work_item_id = %s AND slice_id = %s",
            tuple(params),
        )

    def get_slice_states(self, work_item_id: int) -> list[dict]:
        """Load all slice state rows for a work item."""
        rows = self._pool.fetch_all(
            "SELECT * FROM delivery_slice_states "
            "WHERE work_item_id = %s ORDER BY id",
            (work_item_id,),
        )
        result = []
        for row in rows:
            d = dict(row)
            # Parse metadata_json string to dict
            meta = d.get("metadata_json")
            if isinstance(meta, str):
                try:
                    d["metadata_json"] = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    d["metadata_json"] = {}
            elif meta is None:
                d["metadata_json"] = {}
            result.append(d)
        return result

    def delete_slice_states(self, work_item_id: int) -> None:
        """Delete all slice state rows for a work item."""
        self._pool.execute(
            "DELETE FROM delivery_slice_states WHERE work_item_id = %s",
            (work_item_id,),
        )

    # ──────────────────────────── Stage Runs ────────────────────────────

    def create_stage_run(
        self,
        *,
        work_item_id: int,
        stage_name: str,
        stage_index: int = 0,
        iteration: int = 1,
        owner_agent_id: int | None = None,
        owner_role: str | None = None,
        status: str = "pending",
        summary: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> int:
        return self._pool.insert_returning_id(
            "INSERT INTO stage_runs "
            "(work_item_id, stage_name, stage_index, iteration, owner_agent_id, owner_role, "
            "status, summary, metadata_json) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                work_item_id,
                stage_name,
                stage_index,
                iteration,
                owner_agent_id,
                owner_role,
                status,
                summary,
                json.dumps(metadata_json) if metadata_json else None,
            ),
        )

    def get_stage_run(self, stage_run_id: int) -> StageRun | None:
        row = self._pool.fetch_one("SELECT * FROM stage_runs WHERE id = %s", (stage_run_id,))
        return StageRun(**row) if row else None

    def get_latest_stage_run(self, work_item_id: int, stage_name: str) -> StageRun | None:
        row = self._pool.fetch_one(
            "SELECT * FROM stage_runs WHERE work_item_id = %s AND stage_name = %s "
            "ORDER BY stage_index DESC, iteration DESC, id DESC LIMIT 1",
            (work_item_id, stage_name),
        )
        return StageRun(**row) if row else None

    def list_stage_runs(self, work_item_id: int) -> list[StageRun]:
        rows = self._pool.fetch_all(
            "SELECT * FROM stage_runs WHERE work_item_id = %s "
            "ORDER BY stage_index, iteration, id",
            (work_item_id,),
        )
        return [StageRun(**row) for row in rows]

    def update_stage_run(
        self,
        stage_run_id: int,
        *,
        status: str | None = None,
        summary: str | None = None,
        completed: bool = False,
        claimed_by_runtime_id: int | None = None,
        lease_expires_at: datetime | None = None,
        output_artifact_id: int | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> None:
        row = self._pool.fetch_one("SELECT metadata_json FROM stage_runs WHERE id = %s", (stage_run_id,))
        existing_metadata = self._parse_json_field(row.get("metadata_json")) if row else None
        merged_metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
        if metadata_json:
            merged_metadata.update(metadata_json)

        assignments: list[str] = []
        args: list[Any] = []
        if status is not None:
            assignments.append("status = %s")
            args.append(status)
        if summary is not None:
            assignments.append("summary = %s")
            args.append(summary)
        if claimed_by_runtime_id is not None:
            assignments.append("claimed_by_runtime_id = %s")
            args.append(claimed_by_runtime_id)
            assignments.append("claimed_at = NOW()")
        if lease_expires_at is not None:
            assignments.append("lease_expires_at = %s")
            args.append(lease_expires_at)
        if output_artifact_id is not None:
            assignments.append("output_artifact_id = %s")
            args.append(output_artifact_id)
        if metadata_json is not None:
            assignments.append("metadata_json = %s")
            args.append(json.dumps(merged_metadata) if merged_metadata else None)
        if completed:
            assignments.append("completed_at = NOW()")
        if not assignments:
            return
        args.append(stage_run_id)
        self._pool.execute(
            f"UPDATE stage_runs SET {', '.join(assignments)} WHERE id = %s",
            tuple(args),
        )

    def claim_stage_run(
        self,
        *,
        project_id: int,
        owner_role: str,
        runtime_registration_id: int,
        lease_seconds: int = 300,
    ) -> StageRun | None:
        now = datetime.now()
        row = self._pool.fetch_one(
            "SELECT sr.* FROM stage_runs sr "
            "JOIN work_items wi ON sr.work_item_id = wi.id "
            "WHERE wi.project_id = %s AND wi.status IN ('pending', 'in_progress', 'review') "
            "AND sr.owner_role = %s "
            "AND ("
            "sr.status = 'pending' "
            "OR (sr.status = 'claimed' AND (sr.lease_expires_at IS NULL OR sr.lease_expires_at < %s))"
            ") "
            "ORDER BY sr.stage_index, sr.iteration, sr.id LIMIT 1",
            (project_id, owner_role, now),
        )
        if not row:
            return None

        lease_expires_at = now + timedelta(seconds=max(lease_seconds, 30))
        updated = self._pool.execute(
            "UPDATE stage_runs SET status = 'claimed', claimed_by_runtime_id = %s, "
            "claimed_at = NOW(), lease_expires_at = %s "
            "WHERE id = %s AND (status = 'pending' OR (status = 'claimed' AND (lease_expires_at IS NULL OR lease_expires_at < %s)))",
            (runtime_registration_id, lease_expires_at, row["id"], now),
        )
        if updated != 1:
            return None
        return self.get_stage_run(int(row["id"]))

    def renew_stage_run_lease(
        self,
        stage_run_id: int,
        *,
        runtime_registration_id: int,
        lease_seconds: int = 300,
    ) -> bool:
        lease_expires_at = datetime.now() + timedelta(seconds=max(lease_seconds, 30))
        updated = self._pool.execute(
            "UPDATE stage_runs SET lease_expires_at = %s "
            "WHERE id = %s AND status = 'claimed' AND claimed_by_runtime_id = %s",
            (lease_expires_at, stage_run_id, runtime_registration_id),
        )
        return updated == 1

    def cancel_open_stage_runs(
        self,
        work_item_id: int,
        *,
        summary: str,
        status: str = "cancelled",
    ) -> None:
        excluded_statuses = ["completed", "blocked", "cancelled", "failed"]
        if status == "paused":
            excluded_statuses.append("paused")
        placeholders = ", ".join(f"'{value}'" for value in excluded_statuses)
        self._pool.execute(
            "UPDATE stage_runs SET status = %s, summary = %s, completed_at = NOW(), lease_expires_at = NULL "
            f"WHERE work_item_id = %s AND status NOT IN ({placeholders})",
            (status, summary[:4000], work_item_id),
        )

    # ──────────────────────────── Coordination Events ────────────────────────────

    def append_coordination_event(
        self,
        *,
        work_item_id: int,
        event_type: str,
        summary: str,
        stage_run_id: int | None = None,
        stage_name: str | None = None,
        title: str | None = None,
        from_agent_id: int | None = None,
        from_role: str | None = None,
        to_agent_id: int | None = None,
        to_role: str | None = None,
        payload_json: dict[str, Any] | None = None,
    ) -> CoordinationEvent:
        event_id = self._pool.insert_returning_id(
            "INSERT INTO coordination_events "
            "(work_item_id, stage_run_id, stage_name, event_type, title, summary, "
            "from_agent_id, from_role, to_agent_id, to_role, payload_json) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                work_item_id,
                stage_run_id,
                stage_name,
                event_type,
                title,
                summary,
                from_agent_id,
                from_role,
                to_agent_id,
                to_role,
                json.dumps(payload_json) if payload_json else None,
            ),
        )
        row = self._pool.fetch_one("SELECT * FROM coordination_events WHERE id = %s", (event_id,))
        if row:
            return CoordinationEvent(**row)
        return CoordinationEvent(
            id=event_id,
            work_item_id=work_item_id,
            stage_run_id=stage_run_id,
            stage_name=stage_name,
            event_type=event_type,
            title=title,
            summary=summary,
            from_agent_id=from_agent_id,
            from_role=from_role,
            to_agent_id=to_agent_id,
            to_role=to_role,
            payload_json=payload_json,
        )

    def list_coordination_events(
        self,
        work_item_id: int,
        *,
        stage_name: str | None = None,
        limit: int = 20,
    ) -> list[CoordinationEvent]:
        sql = "SELECT * FROM coordination_events WHERE work_item_id = %s"
        args: list[Any] = [work_item_id]
        if stage_name:
            sql += " AND stage_name = %s"
            args.append(stage_name)
        sql += " ORDER BY created_at DESC, id DESC LIMIT %s"
        args.append(limit)
        rows = self._pool.fetch_all(sql, tuple(args))
        return [CoordinationEvent(**row) for row in rows]

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

    def get_latest_artifact_by_type(self, work_item_id: int, artifact_type: str) -> dict | None:
        """Return the most recently modified artifact of a given type, or None.

        Orders by updated_at (not id or iteration) so that in-place upserts
        are correctly ranked above older rows with higher ids.
        """
        rows = self._pool.fetch_all(
            "SELECT * FROM artifacts WHERE work_item_id = %s AND artifact_type = %s "
            "ORDER BY updated_at DESC, id DESC LIMIT 1",
            (work_item_id, artifact_type),
        )
        return rows[0] if rows else None

    # ──────────────────────────── Review Cycles ────────────────────────────

    def create_review_cycle(
        self, work_item_id: int, iteration: int,
        proposer_agent_id: int, reviewer_agent_id: int,
        artifact_id: int,
        proposal_session_id: int | None = None,
        stage_name: str | None = None,
        reviewer_role: str | None = None,
        status: str = "pending",
        task_json: dict[str, Any] | None = None,
    ) -> int:
        """Create a review cycle (idempotent on duplicate artifact+reviewer).

        The UNIQUE constraint on (artifact_id, reviewer_agent_id) prevents the
        same reviewer from reviewing the same artifact twice. On retry, returns
        the existing cycle ID instead of crashing.
        """
        try:
            return self._pool.insert_returning_id(
                "INSERT INTO review_cycles (work_item_id, artifact_id, iteration, "
                "stage_name, proposer_agent_id, reviewer_agent_id, reviewer_role, "
                "proposal_session_id, status, task_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    work_item_id,
                    artifact_id,
                    iteration,
                    stage_name,
                    proposer_agent_id,
                    reviewer_agent_id,
                    reviewer_role,
                    proposal_session_id,
                    status,
                    json.dumps(task_json) if task_json else None,
                ),
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
        status: str | None = None,
        claimed_by_runtime_id: int | None = None,
        lease_expires_at: datetime | None = None,
        only_if_active: bool = False,
    ) -> bool:
        assignments = [
            "verdict = %s",
            "verdict_json = %s",
            "review_session_id = %s",
        ]
        args: list[Any] = [
            verdict,
            json.dumps(verdict_json) if verdict_json else None,
            review_session_id,
        ]
        if status is not None:
            assignments.append("status = %s")
            args.append(status)
        if claimed_by_runtime_id is not None:
            assignments.append("claimed_by_runtime_id = %s")
            args.append(claimed_by_runtime_id)
            assignments.append("claimed_at = NOW()")
        if lease_expires_at is not None:
            assignments.append("lease_expires_at = %s")
            args.append(lease_expires_at)
        if status in {"completed", "blocked", "cancelled"}:
            assignments.append("completed_at = NOW()")
        where_clauses = ["id = %s"]
        if only_if_active:
            where_clauses.append("status NOT IN ('completed', 'blocked', 'cancelled')")
        args.append(cycle_id)
        updated = self._pool.execute(
            f"UPDATE review_cycles SET {', '.join(assignments)} WHERE {' AND '.join(where_clauses)}",
            tuple(args),
        )
        return updated == 1

    def get_review_cycle(self, cycle_id: int) -> dict | None:
        row = self._pool.fetch_one(
            "SELECT * FROM review_cycles WHERE id = %s",
            (cycle_id,),
        )
        if row and row.get("task_json") is not None:
            row["task_json"] = self._parse_json_field(row["task_json"])
        if row and row.get("verdict_json") is not None:
            row["verdict_json"] = self._parse_json_field(row["verdict_json"])
        return row

    def claim_review_cycle(
        self,
        *,
        project_id: int,
        reviewer_role: str,
        runtime_registration_id: int,
        lease_seconds: int = 300,
    ) -> dict | None:
        now = datetime.now()
        row = self._pool.fetch_one(
            "SELECT rc.* FROM review_cycles rc "
            "JOIN work_items wi ON rc.work_item_id = wi.id "
            "WHERE wi.project_id = %s AND wi.status IN ('pending', 'in_progress', 'review') "
            "AND rc.reviewer_role = %s "
            "AND ("
            "rc.status = 'pending' "
            "OR (rc.status = 'claimed' AND (rc.lease_expires_at IS NULL OR rc.lease_expires_at < %s))"
            ") "
            "ORDER BY rc.created_at, rc.id LIMIT 1",
            (project_id, reviewer_role, now),
        )
        if not row:
            return None

        lease_expires_at = now + timedelta(seconds=max(lease_seconds, 30))
        updated = self._pool.execute(
            "UPDATE review_cycles SET status = 'claimed', claimed_by_runtime_id = %s, "
            "claimed_at = NOW(), lease_expires_at = %s "
            "WHERE id = %s AND (status = 'pending' OR (status = 'claimed' AND (lease_expires_at IS NULL OR lease_expires_at < %s)))",
            (runtime_registration_id, lease_expires_at, row["id"], now),
        )
        if updated != 1:
            return None
        return self.get_review_cycle(int(row["id"]))

    def renew_review_cycle_lease(
        self,
        cycle_id: int,
        *,
        runtime_registration_id: int,
        lease_seconds: int = 300,
    ) -> bool:
        lease_expires_at = datetime.now() + timedelta(seconds=max(lease_seconds, 30))
        updated = self._pool.execute(
            "UPDATE review_cycles SET lease_expires_at = %s "
            "WHERE id = %s AND status = 'claimed' AND claimed_by_runtime_id = %s",
            (lease_expires_at, cycle_id, runtime_registration_id),
        )
        return updated == 1

    def list_recent_review_history(
        self,
        *,
        work_item_id: int,
        stage_name: str,
        before_iteration: int,
        round_limit: int = 3,
    ) -> list[dict]:
        """Fallback review-history query for callers that do not keep an in-memory cache."""
        if not stage_name or before_iteration <= 1 or round_limit <= 0:
            return []
        rows = self._pool.fetch_all(
            "SELECT * FROM review_cycles "
            "WHERE work_item_id = %s AND stage_name = %s AND iteration < %s "
            "AND status IN ('completed', 'blocked', 'cancelled') "
            "AND verdict IN ('lgtm', 'changes_requested', 'failed') "
            "ORDER BY iteration DESC, reviewer_role ASC, id ASC",
            (
                work_item_id,
                stage_name,
                before_iteration,
            ),
        )
        limited_rows: list[dict] = []
        iterations: set[int] = set()
        for row in rows:
            iteration = int(row.get("iteration") or 0)
            if iteration <= 0:
                continue
            if iteration not in iterations and len(iterations) >= round_limit:
                continue
            iterations.add(iteration)
            if row.get("task_json") is not None:
                row["task_json"] = self._parse_json_field(row["task_json"])
            if row.get("verdict_json") is not None:
                row["verdict_json"] = self._parse_json_field(row["verdict_json"])
            limited_rows.append(row)
        return limited_rows

    def cancel_review_cycles_by_ids(
        self,
        cycle_ids: list[int],
        *,
        summary: str,
        status: str = "cancelled",
        verdict: str = "cancelled",
    ) -> int:
        if not cycle_ids:
            return 0
        excluded_statuses = ["completed", "blocked", "cancelled"]
        if status == "paused":
            excluded_statuses.append("paused")
        verdict_json = json.dumps(
            {
                "verdict": verdict,
                "issues": [],
                "summary": summary[:4000],
            }
        )
        placeholders = ", ".join(["%s"] * len(cycle_ids))
        excluded_placeholders = ", ".join(f"'{value}'" for value in excluded_statuses)
        updated = self._pool.execute(
            f"UPDATE review_cycles SET status = %s, verdict = %s, verdict_json = %s, "
            f"completed_at = NOW(), lease_expires_at = NULL "
            f"WHERE id IN ({placeholders}) AND status NOT IN ({excluded_placeholders})",
            (status, verdict, verdict_json, *cycle_ids),
        )
        return int(updated or 0)

    def cancel_open_review_cycles(
        self,
        work_item_id: int,
        *,
        summary: str,
        status: str = "cancelled",
        verdict: str = "cancelled",
    ) -> None:
        excluded_statuses = ["completed", "blocked", "cancelled"]
        if status == "paused":
            excluded_statuses.append("paused")
        verdict_json = json.dumps(
            {
                "verdict": verdict,
                "issues": [],
                "summary": summary[:4000],
            }
        )
        placeholders = ", ".join(f"'{value}'" for value in excluded_statuses)
        self._pool.execute(
            "UPDATE review_cycles SET status = %s, verdict = %s, verdict_json = %s, "
            "completed_at = NOW(), lease_expires_at = NULL "
            f"WHERE work_item_id = %s AND status NOT IN ({placeholders})",
            (status, verdict, verdict_json, work_item_id),
        )

    def get_review_cycles(self, work_item_id: int) -> list[dict]:
        rows = self._pool.fetch_all(
            "SELECT * FROM review_cycles WHERE work_item_id = %s ORDER BY created_at, id",
            (work_item_id,),
        )
        for row in rows:
            if row.get("task_json") is not None:
                row["task_json"] = self._parse_json_field(row["task_json"])
            if row.get("verdict_json") is not None:
                row["verdict_json"] = self._parse_json_field(row["verdict_json"])
        return rows

    def get_review_cycles_by_ids(self, cycle_ids: list[int]) -> list[dict]:
        if not cycle_ids:
            return []
        placeholders = ", ".join(["%s"] * len(cycle_ids))
        rows = self._pool.fetch_all(
            f"SELECT * FROM review_cycles WHERE id IN ({placeholders}) ORDER BY created_at, id",
            tuple(cycle_ids),
        )
        for row in rows:
            if row.get("task_json") is not None:
                row["task_json"] = self._parse_json_field(row["task_json"])
            if row.get("verdict_json") is not None:
                row["verdict_json"] = self._parse_json_field(row["verdict_json"])
        return rows

    def reactivate_review_cycle(
        self,
        cycle_id: int,
        *,
        iteration: int,
        stage_name: str | None,
        proposal_session_id: int | None,
        task_json: dict[str, Any] | None = None,
    ) -> bool:
        updated = self._pool.execute(
            "UPDATE review_cycles SET status = 'pending', verdict = 'pending', verdict_json = NULL, "
            "iteration = %s, stage_name = %s, proposal_session_id = %s, task_json = %s, "
            "review_session_id = NULL, claimed_by_runtime_id = NULL, claimed_at = NULL, "
            "lease_expires_at = NULL, completed_at = NULL "
            "WHERE id = %s AND status IN ('paused', 'cancelled')",
            (
                iteration,
                stage_name,
                proposal_session_id,
                json.dumps(task_json) if task_json else None,
                cycle_id,
            ),
        )
        return updated == 1
