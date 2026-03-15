"""Search planning and multi-branch retrieval for project knowledge."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from myswat.memory.store import MemoryStore

PROFILE_LIMIT_MULTIPLIER = {
    "quick": 1,
    "standard": 2,
    "precise": 3,
}

BRANCH_WEIGHTS_BY_MODE = {
    "auto": {"lexical": 1.05, "vector": 1.0, "graph": 0.35},
    "exact": {"lexical": 1.20, "vector": 0.90, "graph": 0.25},
    "concept": {"lexical": 0.75, "vector": 1.20, "graph": 0.30},
    "relation": {"lexical": 0.90, "vector": 0.90, "graph": 0.75},
}

ROLE_CATEGORY_PREFERENCES = {
    "architect": ["architecture", "decision", "pattern", "protocol", "invariant"],
    "developer": ["architecture", "bug_fix", "api_reference", "configuration", "failure_mode"],
    "qa_main": ["bug_fix", "review_feedback", "configuration", "api_reference", "failure_mode"],
    "qa_vice": ["bug_fix", "review_feedback", "configuration", "api_reference", "failure_mode"],
}

STAGE_CATEGORY_HINTS = (
    ("design", ["architecture", "decision", "pattern", "protocol", "invariant"]),
    ("plan", ["architecture", "decision", "pattern"]),
    ("review", ["bug_fix", "review_feedback", "failure_mode"]),
    ("test", ["bug_fix", "configuration", "failure_mode", "api_reference"]),
    ("implement", ["architecture", "bug_fix", "api_reference", "configuration"]),
)


@dataclass(slots=True)
class SearchPlan:
    project_id: int
    query: str
    keywords: list[str]
    entities: list[str]
    agent_id: int | None = None
    agent_role: str | None = None
    current_stage: str | None = None
    category: str | None = None
    source_type: str | None = None
    preferred_categories: list[str] | None = None
    limit: int = 10
    mode: str = "auto"
    profile: str = "standard"
    use_vector: bool = True
    use_fulltext: bool = True


class SearchPlanBuilder:
    """Builds a search plan from query shape plus workflow context."""

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        seen: set[str] = set()
        keywords: list[str] = []
        for token in re.findall(r"[A-Za-z0-9_:\-./]+", query):
            normalized = token.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            keywords.append(token)
        return keywords

    @staticmethod
    def _extract_entities(query: str) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_:\-./]+", query):
            if not (
                any(ch.isupper() for ch in token)
                or any(sep in token for sep in ("_", "-", "::", "/", "."))
            ):
                continue
            normalized = token.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            entities.append(token)
        return entities

    @staticmethod
    def _category_preferences(agent_role: str | None, current_stage: str | None, query: str) -> list[str]:
        prefs: list[str] = []
        seen: set[str] = set()

        def add_many(values: list[str]) -> None:
            for value in values:
                if value not in seen:
                    seen.add(value)
                    prefs.append(value)

        if agent_role:
            add_many(ROLE_CATEGORY_PREFERENCES.get(agent_role, []))

        stage_text = (current_stage or "").casefold()
        for hint, categories in STAGE_CATEGORY_HINTS:
            if hint in stage_text:
                add_many(categories)

        query_text = query.casefold()
        if "bug" in query_text or "panic" in query_text or "error" in query_text:
            add_many(["bug_fix", "failure_mode"])
        if "config" in query_text or "setting" in query_text:
            add_many(["configuration", "api_reference"])
        if "why " in query_text or "how " in query_text:
            add_many(["architecture", "protocol"])
        if "review" in query_text:
            add_many(["review_feedback", "bug_fix"])

        return prefs

    @staticmethod
    def build(
        *,
        project_id: int,
        query: str,
        agent_id: int | None = None,
        agent_role: str | None = None,
        current_stage: str | None = None,
        category: str | None = None,
        source_type: str | None = None,
        limit: int = 10,
        mode: str = "auto",
        profile: str = "standard",
    ) -> SearchPlan:
        normalized_query = query.strip()
        effective_mode = mode if mode in {"auto", "exact", "concept", "relation"} else "auto"
        effective_profile = profile if profile in {"quick", "standard", "precise"} else "standard"

        use_fulltext = bool(normalized_query)
        use_vector = bool(normalized_query)
        if effective_mode == "concept":
            use_fulltext = False
        elif effective_mode == "exact":
            use_fulltext = True
            use_vector = True

        effective_limit = max(limit, 1)
        if effective_profile == "quick":
            effective_limit = min(effective_limit, 8)
        elif effective_profile == "precise":
            effective_limit = max(effective_limit, 12)

        if not normalized_query:
            use_fulltext = False
            use_vector = False

        keywords = SearchPlanBuilder._extract_keywords(normalized_query)
        entities = SearchPlanBuilder._extract_entities(normalized_query)
        preferred_categories = SearchPlanBuilder._category_preferences(
            agent_role,
            current_stage,
            normalized_query,
        )

        return SearchPlan(
            project_id=project_id,
            query=normalized_query,
            keywords=keywords,
            entities=entities,
            agent_id=agent_id,
            agent_role=agent_role,
            current_stage=current_stage,
            category=category,
            source_type=source_type,
            preferred_categories=preferred_categories or None,
            limit=effective_limit,
            mode=effective_mode,
            profile=effective_profile,
            use_vector=use_vector,
            use_fulltext=use_fulltext,
        )


class KnowledgeSearchEngine:
    """Search orchestration layer over MemoryStore."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def _candidate_limit(self, plan: SearchPlan) -> int:
        return max(plan.limit * PROFILE_LIMIT_MULTIPLIER.get(plan.profile, 2), plan.limit)

    @staticmethod
    def _rrf_score(rank: int) -> float:
        return 1.0 / (60 + rank)

    @staticmethod
    def _row_key(row: dict, rank: int, branch_name: str) -> Any:
        row_id = row.get("id")
        if row_id is not None:
            return ("id", row_id)
        return (
            "synthetic",
            str(row.get("title") or ""),
            str(row.get("content") or ""),
            branch_name,
            rank,
        )

    def _metadata_boost(self, plan: SearchPlan, row: dict) -> float:
        boost = 0.0
        category = str(row.get("category") or "")
        if plan.preferred_categories and category in plan.preferred_categories:
            pref_index = plan.preferred_categories.index(category)
            boost += max(0.18 - pref_index * 0.02, 0.06)

        confidence = float(row.get("confidence") or 0.0)
        boost += min(max(confidence, 0.0), 1.0) * 0.05

        source_type = str(row.get("source_type") or "")
        if plan.mode == "exact" and source_type == "document":
            boost += 0.04
        if plan.mode == "relation" and category in {"architecture", "protocol", "invariant"}:
            boost += 0.05
        return boost

    def _graph_candidates(self, plan: SearchPlan) -> tuple[list[dict], dict[str, str]]:
        if plan.profile == "quick":
            return [], {}
        if plan.mode not in {"relation", "auto"} and plan.profile != "precise":
            return [], {}
        matched_entities = self._store.match_entities(plan.project_id, plan.query, limit=5)
        if not matched_entities and plan.entities:
            matched_entities = self._store.match_entities(
                plan.project_id,
                " ".join(plan.entities),
                limit=5,
            )
        if plan.mode == "auto" and not matched_entities:
            return [], {}
        related = self._store.get_related_entities(plan.project_id, matched_entities, limit=8)
        related_lookup = {
            str(item["related_entity"]).casefold(): str(item["relation"])
            for item in related
        }
        candidates: list[dict] = []
        for rel in related[:4]:
            query = str(rel["related_entity"])
            extra = self._store.search_knowledge(
                project_id=plan.project_id,
                query=query,
                agent_id=plan.agent_id,
                category=plan.category,
                source_type=plan.source_type,
                limit=3,
                use_vector=plan.use_vector,
                use_fulltext=True,
            )
            candidates.extend(extra)
        return candidates, related_lookup

    def _fuse(self, plan: SearchPlan, branches: dict[str, list[dict]]) -> list[dict]:
        weights = BRANCH_WEIGHTS_BY_MODE[plan.mode]
        fused: dict[Any, dict[str, Any]] = {}
        for branch_name, rows in branches.items():
            branch_weight = weights.get(branch_name, 0.0)
            for rank, row in enumerate(rows, start=1):
                row_id = self._row_key(row, rank, branch_name)
                entry = fused.setdefault(row_id, {
                    "row": dict(row),
                    "score": 0.0,
                    "branches": set(),
                })
                entry["branches"].add(branch_name)
                entry["score"] += branch_weight * self._rrf_score(rank)
                entry["score"] += self._metadata_boost(plan, row)

        ranked = sorted(
            fused.values(),
            key=lambda item: (item["score"], item["row"].get("confidence", 0.0)),
            reverse=True,
        )
        results: list[dict] = []
        for item in ranked[:plan.limit]:
            row = dict(item["row"])
            row["search_score"] = round(float(item["score"]), 6)
            row["_branches"] = sorted(item["branches"])
            results.append(row)
        return results

    def _search_with_context(self, plan: SearchPlan) -> tuple[list[dict], dict[str, str]]:
        if not plan.query:
            return self._store.search_knowledge(
                project_id=plan.project_id,
                query="",
                agent_id=plan.agent_id,
                category=plan.category,
                source_type=plan.source_type,
                limit=plan.limit,
                use_vector=False,
                use_fulltext=False,
            ), {}

        candidate_limit = self._candidate_limit(plan)
        lexical_results = self._store.search_knowledge(
            project_id=plan.project_id,
            query=plan.query,
            agent_id=plan.agent_id,
            category=plan.category,
            source_type=plan.source_type,
            limit=candidate_limit,
            use_vector=False,
            use_fulltext=plan.use_fulltext,
        ) if plan.use_fulltext else []
        vector_results = self._store.search_knowledge(
            project_id=plan.project_id,
            query=plan.query,
            agent_id=plan.agent_id,
            category=plan.category,
            source_type=plan.source_type,
            limit=candidate_limit,
            use_vector=plan.use_vector,
            use_fulltext=False,
        ) if plan.use_vector else []
        graph_results, _related_lookup = self._graph_candidates(plan)

        return self._fuse(
            plan,
            {
                "lexical": lexical_results,
                "vector": vector_results,
                "graph": graph_results,
            },
        ), _related_lookup

    def search(self, plan: SearchPlan) -> list[dict]:
        return self._search_with_context(plan)[0]

    def search_with_explanations(self, plan: SearchPlan) -> list[dict]:
        query_terms = set(self._store._query_terms(plan.query))
        results, related_lookup = self._search_with_context(plan)
        explained: list[dict] = []
        for row in results:
            item = dict(row)
            reasons: list[str] = []
            title = str(item.get("title") or "")
            source_file = str(item.get("source_file") or "")
            tags = item.get("tags")
            if isinstance(tags, str):
                parsed = self._store._parse_json_field(tags)
                tags = parsed if isinstance(parsed, list) else []
            if not isinstance(tags, list):
                tags = []

            lower_title = title.casefold()
            lower_source_file = source_file.casefold()
            lower_tags = [str(tag).casefold() for tag in tags]
            for term in sorted(query_terms):
                if term in lower_title:
                    reasons.append(f"title match: {term}")
                    continue
                if source_file and term in lower_source_file:
                    reasons.append(f"source_file match: {term}")
                    continue
                if any(term in tag for tag in lower_tags):
                    reasons.append(f"tag match: {term}")

            for term, relation in related_lookup.items():
                if term in lower_title:
                    reasons.append(f"graph expansion: {relation} {term}")

            branches = item.get("_branches") or []
            if "vector" in branches and not reasons and plan.query:
                reasons.append("semantic match")
            elif "vector" in branches and "semantic match" not in reasons:
                reasons.append("semantic match")
            if "lexical" in branches and not any("match" in reason for reason in reasons):
                reasons.append("lexical match")
            if plan.preferred_categories and item.get("category") in plan.preferred_categories:
                reasons.append(f"category bias: {item['category']}")

            if not reasons and plan.use_vector and plan.query:
                reasons.append("semantic match")
            item["why"] = reasons
            explained.append(item)
        return explained

    @staticmethod
    def render_for_context(results: list[dict], budget_tokens: int) -> str:
        if not results or budget_tokens <= 0:
            return ""

        grouped: dict[str, list[dict]] = {}
        order: list[str] = []
        for row in results:
            category = str(row.get("category") or "uncategorized")
            if category not in grouped:
                grouped[category] = []
                order.append(category)
            grouped[category].append(row)

        lines = ["## Relevant Knowledge\n"]
        tokens_used = len(lines[0]) // 4
        for category in order:
            header = f"### [{category}]\n"
            header_tokens = len(header) // 4
            if tokens_used + header_tokens > budget_tokens:
                break
            lines.append(header)
            tokens_used += header_tokens

            for entry in grouped[category]:
                title = str(entry.get("title") or "Untitled")
                content = " ".join(str(entry.get("content") or "").split())
                if len(content) > 220:
                    content = content[:220] + "... [truncated]"
                line = f"- **{title}**: {content}\n"
                line_tokens = len(line) // 4
                if tokens_used + line_tokens > budget_tokens:
                    break
                lines.append(line)
                tokens_used += line_tokens

        return "\n".join(lines).strip() if len(lines) > 1 else ""
