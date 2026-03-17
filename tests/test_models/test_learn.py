"""Tests for unified learn models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from myswat.models.learn import (
    IndexHint,
    IndexTerm,
    KnowledgeAction,
    KnowledgeLocator,
    LearnActionEnvelope,
    LearnRequest,
    RelationAction,
)


def test_knowledge_locator_cache_key_normalizes_case() -> None:
    locator = KnowledgeLocator(
        category="Architecture",
        title="Build Graph",
        source_type="Document",
        source_file="Docs/ARCH.md",
    )
    assert locator.cache_key() == (
        "architecture",
        "build graph",
        "document",
        "docs/arch.md",
    )


def test_learn_request_rejects_blank_trigger() -> None:
    with pytest.raises(ValidationError):
        LearnRequest(project_id=1, source_kind="chat", trigger_kind=" ")


def test_knowledge_action_create_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        KnowledgeAction(op="create", category="architecture", title="Build")


def test_knowledge_action_update_accepts_match_and_changed_content() -> None:
    action = KnowledgeAction(
        op="update",
        match=KnowledgeLocator(category="architecture", title="Build"),
        content="new content",
    )
    assert action.content == "new content"


def test_relation_action_requires_target_reference() -> None:
    with pytest.raises(ValidationError):
        RelationAction(op="add", source_entity="A", relation="depends_on", target_entity="B")


def test_index_hint_requires_terms_or_entities() -> None:
    with pytest.raises(ValidationError):
        IndexHint(knowledge_match=KnowledgeLocator(category="architecture", title="Build"))


def test_learn_action_envelope_coerces_single_index_hint_object() -> None:
    envelope = LearnActionEnvelope.model_validate(
        {
            "knowledge_actions": [],
            "relation_actions": [],
            "index_hints": {
                "knowledge_match": {"category": "architecture", "title": "Build"},
                "terms": [{"term": "build", "field": "title", "weight": 2.0}],
            },
        }
    )
    assert len(envelope.index_hints) == 1
    assert isinstance(envelope.index_hints[0].terms[0], IndexTerm)
