"""Tests for ActionExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from myswat.memory.action_executor import ActionExecutor
from myswat.models.learn import LearnActionEnvelope, LearnRequest


def test_execute_create_action_attaches_provenance() -> None:
    store = MagicMock()
    store.store_knowledge.return_value = 101
    executor = ActionExecutor(store)
    request = LearnRequest(
        id=9,
        project_id=1,
        source_kind="chat",
        trigger_kind="explicit_user_request",
        source_session_id=7,
    )
    envelope = LearnActionEnvelope.model_validate(
        {
            "knowledge_actions": [
                {
                    "op": "create",
                    "category": "architecture",
                    "title": "Build",
                    "content": "Use uv",
                    "metadata_json": {"worker_label": "memory"},
                }
            ],
            "relation_actions": [],
            "index_hints": [],
        }
    )

    summary = executor.execute(request, envelope)

    assert summary.knowledge_created == 1
    kwargs = store.store_knowledge.call_args.kwargs
    assert kwargs["search_metadata_json"]["learn_request_id"] == 9
    assert kwargs["search_metadata_json"]["learn_source_session_id"] == 7
    assert kwargs["search_metadata_json"]["worker_label"] == "memory"
    assert kwargs["refresh_derived_indexes"] is False


def test_execute_applies_relation_and_index_hints_for_created_knowledge() -> None:
    store = MagicMock()
    store.store_knowledge.return_value = 101
    store.get_knowledge.return_value = {"id": 101, "project_id": 1}
    executor = ActionExecutor(store)
    request = LearnRequest(project_id=1, source_kind="work", trigger_kind="workflow_summary")
    envelope = LearnActionEnvelope.model_validate(
        {
            "knowledge_actions": [
                {
                    "op": "create",
                    "category": "architecture",
                    "title": "Build",
                    "content": "Use uv",
                }
            ],
            "relation_actions": [
                {
                    "op": "add",
                    "knowledge_match": {"category": "architecture", "title": "Build"},
                    "source_entity": "CLI",
                    "relation": "uses",
                    "target_entity": "uv",
                }
            ],
            "index_hints": [
                {
                    "knowledge_match": {"category": "architecture", "title": "Build"},
                    "terms": [{"term": "uv", "field": "content", "weight": 1.5}],
                    "entities": [{"entity_name": "uv", "confidence": 0.95}],
                }
            ],
        }
    )

    summary = executor.execute(request, envelope)

    assert summary.knowledge_created == 1
    assert summary.relations_added == 1
    assert summary.index_hints_applied == 1
    store.add_knowledge_relation.assert_called_once_with(
        project_id=1,
        knowledge_id=101,
        source_entity="CLI",
        relation="uses",
        target_entity="uv",
        confidence=1.0,
    )
    store.replace_knowledge_index_hints.assert_called_once()


def test_execute_update_raises_when_target_missing() -> None:
    store = MagicMock()
    store.find_active_knowledge.return_value = None
    executor = ActionExecutor(store)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")
    envelope = LearnActionEnvelope.model_validate(
        {
            "knowledge_actions": [
                {
                    "op": "update",
                    "match": {"category": "architecture", "title": "Build"},
                    "content": "new content",
                }
            ],
            "relation_actions": [],
            "index_hints": [],
        }
    )

    with pytest.raises(ValueError, match="Knowledge target not found"):
        executor.execute(request, envelope)


def test_execute_rejects_cross_project_knowledge_id() -> None:
    store = MagicMock()
    store.get_knowledge.return_value = {"id": 17, "project_id": 2}
    executor = ActionExecutor(store)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")
    envelope = LearnActionEnvelope.model_validate(
        {
            "knowledge_actions": [
                {
                    "op": "update",
                    "knowledge_id": 17,
                    "content": "new content",
                }
            ],
            "relation_actions": [],
            "index_hints": [],
        }
    )

    with pytest.raises(ValueError, match="Knowledge target not found"):
        executor.execute(request, envelope)
