"""Comprehensive tests for KnowledgeCompactor and parse_compaction_output."""

import json
from unittest.mock import MagicMock

import pytest

from myswat.agents.base import AgentResponse
from myswat.memory.compactor import KnowledgeCompactor, parse_compaction_output
from myswat.models.session import SessionTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(turn_index, role="user", content="hello", session_id=1, turn_id=None):
    return SessionTurn(
        id=turn_id or (turn_index + 1),
        session_id=session_id,
        turn_index=turn_index,
        role=role,
        content=content,
        token_count_est=10,
    )


def _make_store():
    store = MagicMock()
    store.get_session.return_value = None
    store.get_session_turns.return_value = []
    store.count_uncompacted_turns.return_value = 0
    store.count_session_turns.return_value = 0
    store.get_compactable_sessions.return_value = []
    store.store_knowledge.return_value = 1
    return store


def _make_runner(output="[]", success=True):
    runner = MagicMock()
    runner.invoke.return_value = AgentResponse(
        content=output, exit_code=0 if success else 1,
    )
    return runner


# ===========================================================================
# 1. parse_compaction_output
# ===========================================================================


class TestParseCompactionOutput:
    def test_valid_json_array(self):
        raw = json.dumps([{"summary": "a"}, {"summary": "b"}])
        result, ok = parse_compaction_output(raw)
        assert ok is True
        assert len(result) == 2
        assert result[0]["summary"] == "a"

    def test_json_code_block(self):
        inner = json.dumps([{"key": "value"}])
        raw = f"Here:\n```json\n{inner}\n```\nDone."
        result, ok = parse_compaction_output(raw)
        assert ok is True
        assert len(result) == 1
        assert result[0]["key"] == "value"

    def test_plain_code_block(self):
        inner = json.dumps([{"k": 1}])
        raw = f"```\n{inner}\n```"
        result, ok = parse_compaction_output(raw)
        assert ok is True
        assert len(result) == 1

    def test_find_array_in_surrounding_text(self):
        raw = 'Preamble\n[{"item": "found"}]\nTrailing'
        result, ok = parse_compaction_output(raw)
        assert ok is True
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        assert parse_compaction_output("not json {{{") == ([], False)

    def test_non_list_returns_empty(self):
        assert parse_compaction_output(json.dumps({"not": "list"})) == ([], False)

    def test_empty_string(self):
        assert parse_compaction_output("") == ([], False)

    def test_empty_array(self):
        assert parse_compaction_output("[]") == ([], True)


# ===========================================================================
# 2. should_compact
# ===========================================================================


class TestShouldCompact:
    def test_no_session_returns_false(self):
        store = _make_store()
        store.get_session.return_value = None
        compactor = KnowledgeCompactor(store)
        assert compactor.should_compact(1) is False

    def test_compacted_status_returns_false(self):
        store = _make_store()
        store.get_session.return_value = {"status": "compacted", "token_count_est": 0}
        compactor = KnowledgeCompactor(store)
        assert compactor.should_compact(1) is False

    def test_turns_exceed_threshold(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "token_count_est": 0}
        store.count_uncompacted_turns.return_value = 12
        compactor = KnowledgeCompactor(store, threshold_turns=10)
        assert compactor.should_compact(1) is True

    def test_tokens_exceed_threshold_is_ignored(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "token_count_est": 6000}
        store.count_uncompacted_turns.return_value = 3
        compactor = KnowledgeCompactor(store, threshold_tokens=5000)
        assert compactor.should_compact(1) is False

    def test_neither_exceeded(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "token_count_est": 2000}
        store.count_uncompacted_turns.return_value = 5
        compactor = KnowledgeCompactor(store, threshold_turns=10, threshold_tokens=5000)
        assert compactor.should_compact(1) is False

    def test_exact_turn_threshold(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "token_count_est": 0}
        store.count_uncompacted_turns.return_value = 10
        compactor = KnowledgeCompactor(store, threshold_turns=10)
        assert compactor.should_compact(1) is True

    def test_exact_token_threshold(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "token_count_est": 5000}
        store.count_uncompacted_turns.return_value = 1
        compactor = KnowledgeCompactor(store, threshold_tokens=5000)
        assert compactor.should_compact(1) is False


# ===========================================================================
# 3. build_transcript
# ===========================================================================


class TestBuildTranscript:
    def test_formats_turns(self):
        store = _make_store()
        turns = [
            _make_turn(0, role="user", content="What is 2+2?"),
            _make_turn(1, role="assistant", content="4"),
        ]
        store.get_session_turns.return_value = turns
        compactor = KnowledgeCompactor(store)
        transcript = compactor.build_transcript(1)
        assert "[USER]: What is 2+2?" in transcript
        assert "[ASSISTANT]: 4" in transcript

    def test_truncation(self):
        store = _make_store()
        turns = [_make_turn(i, content="x" * 200) for i in range(100)]
        store.get_session_turns.return_value = turns
        compactor = KnowledgeCompactor(store)
        transcript = compactor.build_transcript(1, max_chars=500)
        assert "truncated" in transcript

    def test_empty_turns(self):
        store = _make_store()
        store.get_session_turns.return_value = []
        compactor = KnowledgeCompactor(store)
        transcript = compactor.build_transcript(1)
        assert transcript.strip() == ""

    def test_preserves_order(self):
        store = _make_store()
        turns = [
            _make_turn(0, role="user", content="FIRST"),
            _make_turn(1, role="assistant", content="SECOND"),
        ]
        store.get_session_turns.return_value = turns
        compactor = KnowledgeCompactor(store)
        transcript = compactor.build_transcript(1)
        assert transcript.find("FIRST") < transcript.find("SECOND")


# ===========================================================================
# 4. compact_session
# ===========================================================================


class TestCompactSession:
    def test_no_runner_returns_empty(self):
        store = _make_store()
        compactor = KnowledgeCompactor(store, runner=None)
        assert compactor.compact_session(1, 1) == []

    def test_no_session_returns_empty(self):
        store = _make_store()
        store.get_session.return_value = None
        runner = _make_runner()
        compactor = KnowledgeCompactor(store, runner=runner)
        assert compactor.compact_session(1, 1) == []

    def test_compacted_session_returns_empty(self):
        store = _make_store()
        store.get_session.return_value = {"status": "compacted", "compacted_through_turn_index": -1}
        runner = _make_runner()
        compactor = KnowledgeCompactor(store, runner=runner)
        assert compactor.compact_session(1, 1) == []

    def test_fewer_than_2_turns_returns_empty(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [_make_turn(0)]
        runner = _make_runner()
        compactor = KnowledgeCompactor(store, runner=runner)
        assert compactor.compact_session(1, 1) == []

    def test_successful_compaction(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
        ]
        items = [
            {"category": "progress", "title": "Greeting", "content": "A greeting exchange",
             "tags": [], "relevance_score": 0.8, "confidence": 0.9},
        ]
        runner = _make_runner(output=json.dumps(items))
        store.store_knowledge.return_value = 42

        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_session(1, 1)

        assert len(result) == 1
        assert result[0] == 42
        store.store_knowledge.assert_called_once()
        assert store.store_knowledge.call_args.kwargs["agent_id"] is None
        store.advance_compaction_watermark.assert_called_once_with(1, 1)
        store.mark_session_fully_compacted.assert_called_once_with(1)
        store.mark_session_compacted.assert_not_called()

    def test_mark_compacted_false_advances_watermark(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
        ]
        items = [{"category": "progress", "title": "T", "content": "C"}]
        runner = _make_runner(output=json.dumps(items))
        store.store_knowledge.return_value = 1

        compactor = KnowledgeCompactor(store, runner=runner)
        compactor.compact_session(1, 1, mark_compacted=False)

        store.mark_session_fully_compacted.assert_not_called()
        store.mark_session_compacted.assert_not_called()
        store.advance_compaction_watermark.assert_called_once_with(1, 1)

    def test_runner_failure(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
        ]
        runner = _make_runner(success=False)
        compactor = KnowledgeCompactor(store, runner=runner)
        assert compactor.compact_session(1, 1) == []
        store.advance_compaction_watermark.assert_not_called()
        store.mark_session_fully_compacted.assert_not_called()

    def test_empty_items_still_advances_watermark(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
        ]
        runner = _make_runner(output="[]")
        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_session(1, 1, mark_compacted=False)

        assert result == []
        store.advance_compaction_watermark.assert_called_once_with(1, 1)
        store.mark_session_fully_compacted.assert_not_called()

    def test_parse_failure_does_not_advance_watermark(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
        ]
        runner = _make_runner(output="not json")

        compactor = KnowledgeCompactor(store, runner=runner)
        assert compactor.compact_session(1, 1) == []

        store.advance_compaction_watermark.assert_not_called()
        store.mark_session_fully_compacted.assert_not_called()

    def test_empty_items_mark_compacted_true_marks_fully_compacted(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
        ]
        runner = _make_runner(output="[]")

        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_session(1, 1)

        assert result == []
        store.advance_compaction_watermark.assert_called_once_with(1, 1)
        store.mark_session_fully_compacted.assert_called_once_with(1)

    def test_truncation_only_advances_processed_prefix(self):
        store = _make_store()
        store.get_session.return_value = {"status": "completed", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
            _make_turn(2, role="user", content="x" * 100000),
        ]
        items = [{"category": "progress", "title": "Greeting", "content": "A greeting exchange"}]
        runner = _make_runner(output=json.dumps(items))
        store.store_knowledge.return_value = 99

        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_session(1, 1, mark_compacted=True)

        assert result == [99]
        store.advance_compaction_watermark.assert_called_once_with(1, 1)
        store.mark_session_fully_compacted.assert_not_called()
        assert store.store_knowledge.call_args.kwargs["source_turn_ids"] == [1, 2]

    def test_filters_turns_by_watermark(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": 2}
        all_turns = [
            _make_turn(0, role="user", content="Old 0"),
            _make_turn(1, role="assistant", content="Old 1"),
            _make_turn(2, role="user", content="Old 2"),
            _make_turn(3, role="assistant", content="New 3"),
            _make_turn(4, role="user", content="New 4"),
            _make_turn(5, role="assistant", content="New 5"),
        ]
        store.get_session_turns.return_value = all_turns

        items = [{"category": "progress", "title": "New", "content": "New stuff"}]
        runner = _make_runner(output=json.dumps(items))
        store.store_knowledge.return_value = 1

        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_session(1, 1)

        assert len(result) == 1
        # Verify the transcript passed to the runner excludes old turns
        invoke_call = runner.invoke.call_args[0][0]
        assert "Old 0" not in invoke_call
        assert "New 3" in invoke_call

    def test_zero_turns(self):
        store = _make_store()
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = []
        runner = _make_runner()
        compactor = KnowledgeCompactor(store, runner=runner)
        assert compactor.compact_session(1, 1) == []
        runner.invoke.assert_not_called()


# ===========================================================================
# 5. compact_all_pending
# ===========================================================================


class TestCompactAllPending:
    def test_skips_sessions_with_fewer_than_2_turns(self):
        store = _make_store()
        store.get_compactable_sessions.return_value = [{"id": 1, "agent_id": 1}]
        store.count_session_turns.return_value = 1

        runner = _make_runner()
        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_all_pending(1)

        assert result["skipped"] == 1
        assert result["compacted"] == 0
        store.mark_session_fully_compacted.assert_called_once_with(1)

    def test_compacts_sessions_with_enough_turns(self):
        store = _make_store()
        store.get_compactable_sessions.return_value = [{"id": 1, "agent_id": 1}]
        store.count_session_turns.return_value = 3
        store.get_session.return_value = {"status": "active", "compacted_through_turn_index": -1}
        store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Hello"),
            _make_turn(1, role="assistant", content="Hi"),
            _make_turn(2, role="user", content="Bye"),
        ]
        items = [{"category": "progress", "title": "T", "content": "C"}]
        runner = _make_runner(output=json.dumps(items))
        store.store_knowledge.return_value = 1

        compactor = KnowledgeCompactor(store, runner=runner)
        result = compactor.compact_all_pending(1)

        assert result["compacted"] == 1
        assert result["knowledge_created"] == 1

    def test_returns_summary_dict(self):
        store = _make_store()
        store.get_compactable_sessions.return_value = []
        compactor = KnowledgeCompactor(store, runner=_make_runner())
        result = compactor.compact_all_pending(1)

        assert isinstance(result, dict)
        assert "compacted" in result
        assert "knowledge_created" in result
        assert "skipped" in result

    def test_no_compactable_sessions(self):
        store = _make_store()
        store.get_compactable_sessions.return_value = []
        compactor = KnowledgeCompactor(store, runner=_make_runner())
        result = compactor.compact_all_pending(1)

        assert result == {"compacted": 0, "knowledge_created": 0, "skipped": 0}
