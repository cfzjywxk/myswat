"""Apply worker-produced learn actions to persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from myswat.memory.store import MemoryStore
from myswat.models.learn import (
    IndexHint,
    KnowledgeAction,
    KnowledgeLocator,
    LearnActionEnvelope,
    LearnRequest,
    RelationAction,
)


@dataclass
class ActionExecutionSummary:
    knowledge_created: int = 0
    knowledge_updated: int = 0
    knowledge_deleted: int = 0
    relations_added: int = 0
    relations_deleted: int = 0
    index_hints_applied: int = 0
    created_knowledge_ids: list[int] = field(default_factory=list)


class ActionExecutor:
    """Thin persistence executor for hidden worker action envelopes."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def execute(
        self,
        request: LearnRequest,
        envelope: LearnActionEnvelope,
    ) -> ActionExecutionSummary:
        summary = ActionExecutionSummary()
        resolved_ids: dict[tuple[str, str, str, str], int] = {}

        for action in envelope.knowledge_actions:
            knowledge_id = self._apply_knowledge_action(
                request=request,
                action=action,
                resolved_ids=resolved_ids,
                summary=summary,
            )
            if action.match is not None:
                resolved_ids[action.match.cache_key()] = knowledge_id
            target = action.target_locator()
            if target is not None:
                resolved_ids[target.cache_key()] = knowledge_id

        for action in envelope.relation_actions:
            self._apply_relation_action(
                request=request,
                action=action,
                resolved_ids=resolved_ids,
                summary=summary,
            )

        for hint in envelope.index_hints:
            self._apply_index_hint(
                request=request,
                hint=hint,
                resolved_ids=resolved_ids,
                summary=summary,
            )

        return summary

    def _apply_knowledge_action(
        self,
        *,
        request: LearnRequest,
        action: KnowledgeAction,
        resolved_ids: dict[tuple[str, str, str, str], int],
        summary: ActionExecutionSummary,
    ) -> int:
        metadata = self._build_provenance_metadata(request, action.metadata_json)

        if action.op == "create":
            knowledge_id = self._store.store_knowledge(
                project_id=request.project_id,
                category=action.category or "",
                title=action.title or "",
                content=action.content or "",
                source_session_id=request.source_session_id,
                source_file=action.source_file,
                tags=action.tags or None,
                relevance_score=action.relevance_score,
                confidence=action.confidence,
                ttl_days=action.ttl_days,
                source_type=action.source_type,
                search_metadata_json=metadata,
                bump_revision=True,
                refresh_derived_indexes=False,
            )
            summary.knowledge_created += 1
            summary.created_knowledge_ids.append(knowledge_id)
            return knowledge_id

        row = self._resolve_knowledge_row(
            project_id=request.project_id,
            knowledge_id=action.knowledge_id,
            locator=action.match,
            resolved_ids=resolved_ids,
        )
        if row is None:
            locator = action.match.model_dump(mode="json") if action.match else {"knowledge_id": action.knowledge_id}
            raise ValueError(f"Knowledge target not found for action: {locator}")

        knowledge_id = int(row["id"])
        if action.op == "delete":
            self._store.expire_knowledge(knowledge_id, project_id=request.project_id)
            summary.knowledge_deleted += 1
            return knowledge_id

        self._store.replace_knowledge(
            knowledge_id=knowledge_id,
            project_id=request.project_id,
            category=action.category,
            title=action.title,
            content=action.content,
            source_session_id=request.source_session_id,
            source_file=action.source_file,
            source_type=action.source_type,
            tags=action.tags or None,
            relevance_score=action.relevance_score,
            confidence=action.confidence,
            ttl_days=action.ttl_days,
            search_metadata_json=metadata,
            bump_revision=True,
            refresh_derived_indexes=False,
        )
        summary.knowledge_updated += 1
        return knowledge_id

    def _apply_relation_action(
        self,
        *,
        request: LearnRequest,
        action: RelationAction,
        resolved_ids: dict[tuple[str, str, str, str], int],
        summary: ActionExecutionSummary,
    ) -> None:
        row = self._resolve_knowledge_row(
            project_id=request.project_id,
            knowledge_id=action.knowledge_id,
            locator=action.knowledge_match,
            resolved_ids=resolved_ids,
        )
        if row is None:
            raise ValueError("Relation action target not found")
        knowledge_id = int(row["id"])

        if action.op == "delete":
            self._store.delete_knowledge_relation(
                knowledge_id=knowledge_id,
                source_entity=action.source_entity,
                relation=action.relation,
                target_entity=action.target_entity,
            )
            summary.relations_deleted += 1
            return

        self._store.add_knowledge_relation(
            project_id=request.project_id,
            knowledge_id=knowledge_id,
            source_entity=action.source_entity,
            relation=action.relation,
            target_entity=action.target_entity,
            confidence=action.confidence,
        )
        summary.relations_added += 1

    def _apply_index_hint(
        self,
        *,
        request: LearnRequest,
        hint: IndexHint,
        resolved_ids: dict[tuple[str, str, str, str], int],
        summary: ActionExecutionSummary,
    ) -> None:
        row = self._resolve_knowledge_row(
            project_id=request.project_id,
            knowledge_id=hint.knowledge_id,
            locator=hint.knowledge_match,
            resolved_ids=resolved_ids,
        )
        if row is None:
            raise ValueError("Index hint target not found")
        knowledge_id = int(row["id"])

        self._store.replace_knowledge_index_hints(
            project_id=request.project_id,
            knowledge_id=knowledge_id,
            terms=[
                {"term": term.term, "field": term.field, "weight": term.weight}
                for term in hint.terms
            ],
            entities=[
                {"entity_name": entity.entity_name, "confidence": entity.confidence}
                for entity in hint.entities
            ],
        )

        summary.index_hints_applied += 1

    def _resolve_knowledge_row(
        self,
        *,
        project_id: int,
        knowledge_id: int | None,
        locator: KnowledgeLocator | None,
        resolved_ids: dict[tuple[str, str, str, str], int],
    ) -> dict | None:
        if knowledge_id is not None:
            row = self._store.get_knowledge(knowledge_id)
            if row is None:
                return None
            if int(row.get("project_id") or 0) != project_id:
                return None
            return row

        if locator is None:
            return None

        cached_id = resolved_ids.get(locator.cache_key())
        if cached_id is not None:
            row = self._store.get_knowledge(cached_id)
            if row is not None and int(row.get("project_id") or 0) == project_id:
                return row

        return self._store.find_active_knowledge(
            project_id=project_id,
            category=locator.category,
            title=locator.title,
            source_type=locator.source_type,
            source_file=locator.source_file,
        )

    @staticmethod
    def _build_provenance_metadata(
        request: LearnRequest,
        metadata_json: dict[str, object],
    ) -> dict[str, object]:
        metadata = {
            "learn_source_kind": request.source_kind,
            "learn_trigger_kind": request.trigger_kind,
        }
        if request.id is not None:
            metadata["learn_request_id"] = request.id
        if request.source_session_id is not None:
            metadata["learn_source_session_id"] = request.source_session_id
        if request.source_work_item_id is not None:
            metadata["learn_source_work_item_id"] = request.source_work_item_id
        metadata.update(metadata_json)
        return json.loads(json.dumps(metadata))
