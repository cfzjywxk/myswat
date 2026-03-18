"""Tests for learn history and exact knowledge helper methods on MemoryStore."""

from __future__ import annotations

import json

import pytest

import myswat.memory.store as store_module
from myswat.memory.store import MemoryStore


def test_create_and_get_learn_request(mock_pool) -> None:
    store = MemoryStore(mock_pool)
    mock_pool.insert_returning_id.return_value = 41
    request_id = store.create_learn_request(
        project_id=7,
        source_kind="chat",
        trigger_kind="explicit_user_request",
        payload_json={"summary": "remember this"},
        source_session_id=3,
    )
    assert request_id == 41

    mock_pool.fetch_one.return_value = {
        "id": 41,
        "project_id": 7,
        "source_kind": "chat",
        "trigger_kind": "explicit_user_request",
        "source_session_id": 3,
        "source_work_item_id": None,
        "payload_json": json.dumps({"summary": "remember this"}),
        "status": "pending",
    }
    row = store.get_learn_request(41)
    assert row["payload_json"] == {"summary": "remember this"}


def test_create_complete_and_fail_learn_run(mock_pool) -> None:
    store = MemoryStore(mock_pool)
    mock_pool.insert_returning_id.return_value = 9
    run_id = store.create_learn_run(
        learn_request_id=4,
        worker_backend="codex",
        worker_model="gpt-5.4",
        input_context_json={"project": {"id": 1}},
    )
    assert run_id == 9

    store.complete_learn_run(9, output_envelope_json={"knowledge_actions": []})
    store.fail_learn_run(9, error_text="worker exploded")

    sql_statements = [call.args[0] for call in mock_pool.execute.call_args_list]
    assert any("UPDATE learn_runs SET status = 'completed'" in sql for sql in sql_statements)
    assert any("UPDATE learn_runs SET status = 'failed'" in sql for sql in sql_statements)


def test_find_active_knowledge_filters_exact_scope(mock_pool) -> None:
    store = MemoryStore(mock_pool)
    mock_pool.fetch_one.return_value = {
        "id": 5,
        "project_id": 1,
        "category": "architecture",
        "title": "Build",
        "content": "details",
    }
    row = store.find_active_knowledge(
        project_id=1,
        category="architecture",
        title="Build",
        source_type="document",
        source_file="docs/build.md",
    )
    assert row["id"] == 5
    sql, args = mock_pool.fetch_one.call_args.args
    assert "source_type = %s" in sql
    assert "source_file = %s" in sql
    assert args == (1, "architecture", "Build", "document", "docs/build.md")


def test_replace_knowledge_updates_row_and_rebuilds_indexes(mock_pool, monkeypatch) -> None:
    monkeypatch.setattr(
        store_module.embedder,
        "resolve_embed_sql",
        lambda text, tidb_model="", backend="auto": ("NULL", []),
    )
    store = MemoryStore(mock_pool)
    mock_pool.fetch_one.return_value = {
        "id": 5,
        "project_id": 1,
        "agent_id": None,
        "source_session_id": 2,
        "source_turn_ids": json.dumps([11]),
        "source_file": "docs/build.md",
        "source_type": "document",
        "category": "architecture",
        "title": "Build",
        "content": "old content",
        "tags": json.dumps(["old"]),
        "relevance_score": 0.4,
        "confidence": 0.5,
        "ttl_days": None,
        "expires_at": None,
        "version": 3,
        "search_metadata_json": json.dumps({"existing": True}),
    }
    graph_calls: list[tuple] = []
    term_calls: list[tuple] = []
    mock_pool.execute.return_value = 1
    monkeypatch.setattr(store, "_replace_knowledge_graph", lambda **kwargs: graph_calls.append(kwargs) or ["Entity"])
    monkeypatch.setattr(store, "_replace_knowledge_terms", lambda **kwargs: term_calls.append(kwargs))
    monkeypatch.setattr(store, "_bump_project_memory_revision", lambda project_id: graph_calls.append({"bump": project_id}))

    store.replace_knowledge(
        knowledge_id=5,
        project_id=1,
        title="Build",
        content="new content",
        tags=["fresh"],
        search_metadata_json={"learn_request_id": 99},
        bump_revision=True,
    )

    update_sql = mock_pool.execute.call_args.args[0]
    assert "UPDATE knowledge SET" in update_sql
    assert graph_calls[0]["knowledge_id"] == 5
    assert term_calls[0]["knowledge_id"] == 5
    assert {"bump": 1} in graph_calls


def test_replace_knowledge_raises_on_version_conflict(mock_pool, monkeypatch) -> None:
    monkeypatch.setattr(
        store_module.embedder,
        "resolve_embed_sql",
        lambda text, tidb_model="", backend="auto": ("NULL", []),
    )
    store = MemoryStore(mock_pool)
    mock_pool.fetch_one.return_value = {
        "id": 5,
        "project_id": 1,
        "agent_id": None,
        "source_session_id": 2,
        "source_turn_ids": json.dumps([11]),
        "source_file": None,
        "source_type": "session",
        "category": "architecture",
        "title": "Build",
        "content": "old content",
        "tags": None,
        "relevance_score": 0.4,
        "confidence": 0.5,
        "ttl_days": None,
        "expires_at": None,
        "version": 3,
        "search_metadata_json": None,
    }
    mock_pool.execute.return_value = 0

    with pytest.raises(RuntimeError, match="modified concurrently"):
        store.replace_knowledge(
            knowledge_id=5,
            project_id=1,
            content="new content",
            refresh_derived_indexes=False,
        )


def test_expire_knowledge_delegates_to_supersede(mock_pool, monkeypatch) -> None:
    store = MemoryStore(mock_pool)
    called = {}
    monkeypatch.setattr(store, "_supersede_knowledge", lambda **kwargs: called.update(kwargs))

    store.expire_knowledge(17, project_id=3)

    assert called == {"knowledge_id": 17, "project_id": 3}


def test_store_knowledge_sql_placeholders_match_args(mock_pool) -> None:
    store = MemoryStore(mock_pool)
    mock_pool.insert_returning_id.return_value = 12

    store.store_knowledge(
        project_id=1,
        category="architecture",
        title="Build",
        content="Use uv",
        compute_embedding=False,
        refresh_derived_indexes=False,
    )

    sql, args = mock_pool.insert_returning_id.call_args.args
    assert sql.count("%s") == len(args)


def test_replace_knowledge_index_hints_updates_only_requested_tables(mock_pool) -> None:
    store = MemoryStore(mock_pool)

    store.replace_knowledge_index_hints(
        project_id=1,
        knowledge_id=9,
        terms=[{"term": "raft", "field": "content", "weight": 1.5}],
        entities=[],
    )

    sql_statements = [call.args[0] for call in mock_pool.execute.call_args_list]
    assert any("DELETE FROM knowledge_terms" in sql for sql in sql_statements)
    assert any("INSERT INTO knowledge_terms" in sql for sql in sql_statements)
    assert not any("DELETE FROM knowledge_entities" in sql for sql in sql_statements)
