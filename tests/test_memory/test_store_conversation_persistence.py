"""Tests for Phase 2 conversation persistence store APIs."""

from __future__ import annotations

from datetime import datetime

from myswat.memory.store import MemoryStore


def test_mark_session_fully_compacted_sets_status_and_timestamp(mock_pool) -> None:
    store = MemoryStore(mock_pool)

    store.mark_session_fully_compacted(7)

    mock_pool.execute.assert_called_once_with(
        "UPDATE sessions SET status = 'compacted', compacted_at = NOW() WHERE id = %s",
        (7,),
    )


def test_get_recent_turns_by_project_groups_rows_by_role(mock_pool) -> None:
    mock_pool.fetch_all.return_value = [
        {
            "agent_role": "architect",
            "role": "user",
            "content": "specify the design",
            "created_at": "2026-03-13T10:00:00",
            "turn_id": 11,
            "session_id": 101,
        },
        {
            "agent_role": "architect",
            "role": "assistant",
            "content": "here is the design",
            "created_at": "2026-03-13T10:01:00",
            "turn_id": 12,
            "session_id": 101,
        },
        {
            "agent_role": "developer",
            "role": "user",
            "content": "implement it",
            "created_at": "2026-03-13T10:02:00",
            "turn_id": 21,
            "session_id": 202,
        },
    ]
    store = MemoryStore(mock_pool)

    result = store.get_recent_turns_by_project(1, per_role_limit=10, exclude_session_id=99)

    assert result == [
        {
            "agent_role": "architect",
            "turns": [
                {
                    "turn_id": 11,
                    "session_id": 101,
                    "role": "user",
                    "content": "specify the design",
                    "created_at": "2026-03-13T10:00:00",
                },
                {
                    "turn_id": 12,
                    "session_id": 101,
                    "role": "assistant",
                    "content": "here is the design",
                    "created_at": "2026-03-13T10:01:00",
                },
            ],
        },
        {
            "agent_role": "developer",
            "turns": [
                {
                    "turn_id": 21,
                    "session_id": 202,
                    "role": "user",
                    "content": "implement it",
                    "created_at": "2026-03-13T10:02:00",
                },
            ],
        },
    ]

    sql, args = mock_pool.fetch_all.call_args[0]
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY a.role ORDER BY st.created_at DESC, st.id DESC" in sql
    assert "AND s.id != %s" in sql
    assert "ORDER BY agent_role, created_at ASC, turn_id ASC" in sql
    assert args == (1, 99, 10)


def test_get_recent_turns_by_project_omits_exclude_predicate_when_none(mock_pool) -> None:
    store = MemoryStore(mock_pool)

    store.get_recent_turns_by_project(5, per_role_limit=3, exclude_session_id=None)

    sql, args = mock_pool.fetch_all.call_args[0]
    assert "AND s.id != %s" not in sql
    assert args == (5, 3)


def test_get_recent_turns_global_returns_chronological_rows(mock_pool) -> None:
    mock_pool.fetch_all.return_value = [
        {
            "turn_id": 31,
            "session_id": 3,
            "turn_index": 2,
            "role": "assistant",
            "content": "newest",
            "created_at": "2026-03-13T10:03:00",
            "agent_role": "developer",
        },
        {
            "turn_id": 30,
            "session_id": 3,
            "turn_index": 1,
            "role": "user",
            "content": "older",
            "created_at": "2026-03-13T10:02:00",
            "agent_role": "developer",
        },
    ]
    store = MemoryStore(mock_pool)

    result = store.get_recent_turns_global(8, limit=2)

    assert [row["content"] for row in result] == ["older", "newest"]
    sql, args = mock_pool.fetch_all.call_args[0]
    assert "a.role AS agent_role" in sql
    assert "ORDER BY st.created_at DESC, st.id DESC LIMIT %s" in sql
    assert args == (8, 2)


def test_get_recent_turns_global_applies_role_filter(mock_pool) -> None:
    store = MemoryStore(mock_pool)

    store.get_recent_turns_global(3, limit=5, role="developer")

    sql, args = mock_pool.fetch_all.call_args[0]
    assert "AND a.role = %s" in sql
    assert args == (3, "developer", 5)


def test_gc_compacted_turns_returns_zero_when_project_has_too_few_turns(mock_pool) -> None:
    mock_pool.fetch_one.return_value = None
    store = MemoryStore(mock_pool)

    result = store.gc_compacted_turns(11)

    assert result == {"turns_deleted": 0, "sessions_affected": 0}
    mock_pool.execute.assert_not_called()


def test_gc_compacted_turns_supports_dry_run(mock_pool) -> None:
    cutoff = {"created_at": datetime(2026, 3, 13, 10, 0, 0), "id": 55}
    counts = {"turns_deleted": 7, "sessions_affected": 3}
    mock_pool.fetch_one.side_effect = [cutoff, counts]
    store = MemoryStore(mock_pool)

    result = store.gc_compacted_turns(4, grace_days=9, keep_recent=50, dry_run=True)

    assert result == {"turns_deleted": 7, "sessions_affected": 3}
    assert mock_pool.fetch_one.call_count == 2
    cutoff_sql, cutoff_args = mock_pool.fetch_one.call_args_list[0][0]
    assert "ORDER BY st.created_at DESC, st.id DESC" in cutoff_sql
    assert cutoff_args == (4, 49)
    count_sql, count_args = mock_pool.fetch_one.call_args_list[1][0]
    assert "COUNT(DISTINCT s.id) AS sessions_affected" in count_sql
    assert "s.status = 'compacted'" in count_sql
    assert "s.compacted_at < NOW() - INTERVAL %s DAY" in count_sql
    assert count_args == (4, 9, cutoff["created_at"], cutoff["created_at"], 55)
    mock_pool.execute.assert_not_called()


def test_gc_compacted_turns_deletes_rows_after_counting(mock_pool) -> None:
    cutoff = {"created_at": datetime(2026, 3, 13, 10, 0, 0), "id": 77}
    counts = {"turns_deleted": 5, "sessions_affected": 2}
    mock_pool.fetch_one.side_effect = [cutoff, counts]
    mock_pool.execute.return_value = 5
    store = MemoryStore(mock_pool)

    result = store.gc_compacted_turns(9, grace_days=7, keep_recent=20, dry_run=False)

    assert result == {"turns_deleted": 5, "sessions_affected": 2}
    mock_pool.execute.assert_called_once()
    sql, args = mock_pool.execute.call_args[0]
    assert sql.startswith("DELETE st FROM session_turns st")
    assert "s.status = 'compacted'" in sql
    assert args == (9, 7, cutoff["created_at"], cutoff["created_at"], 77)


def test_gc_compacted_turns_skips_delete_when_nothing_matches(mock_pool) -> None:
    cutoff = {"created_at": datetime(2026, 3, 13, 10, 0, 0), "id": 88}
    counts = {"turns_deleted": 0, "sessions_affected": 0}
    mock_pool.fetch_one.side_effect = [cutoff, counts]
    store = MemoryStore(mock_pool)

    result = store.gc_compacted_turns(2, dry_run=False)

    assert result == {"turns_deleted": 0, "sessions_affected": 0}
    mock_pool.execute.assert_not_called()
