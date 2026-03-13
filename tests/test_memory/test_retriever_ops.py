"""Comprehensive tests for MemoryRetriever."""

from unittest.mock import MagicMock

import pytest

from myswat.memory.retriever import MemoryRetriever
from myswat.models.session import SessionTurn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_turn(turn_index=0, role="user", content="hello", session_id=1):
    return SessionTurn(
        id=turn_index + 1,
        session_id=session_id,
        turn_index=turn_index,
        role=role,
        content=content,
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_project.return_value = {"id": 1, "slug": "proj", "repo_path": "/tmp/proj"}
    store.list_knowledge.return_value = []
    store.search_knowledge.return_value = []
    store.get_session.return_value = None
    store.get_session_turns.return_value = []
    store.get_work_item.return_value = None
    store.get_work_item_state.return_value = {}
    store.list_work_items.return_value = []
    store.get_recent_artifacts_for_project.return_value = []
    store.get_recent_turns_by_project.return_value = []
    store.get_recent_history_for_agent.return_value = []
    return store


@pytest.fixture
def retriever(mock_store):
    return MemoryRetriever(mock_store)


# ===================================================================
# 1. _load_project_ops
# ===================================================================

class TestLoadProjectOps:

    def test_file_with_marker(self, retriever, tmp_path):
        md_file = tmp_path / "myswat.md"
        md_file.write_text(
            "# Header\nPreamble.\n"
            "## Project Operations Knowledge\nOps info.\n"
        )
        result = retriever._load_project_ops(1, str(tmp_path))
        assert result.startswith("## Project Operations Knowledge")
        assert "Ops info." in result

    def test_file_without_marker(self, retriever, tmp_path):
        md_file = tmp_path / "myswat.md"
        md_file.write_text("# Useful notes\nSome content.\n")
        result = retriever._load_project_ops(1, str(tmp_path))
        assert "Some content." in result

    def test_file_missing_falls_back_to_tidb(self, retriever, mock_store, tmp_path):
        mock_store.list_knowledge.return_value = [
            {"title": "Deploy Process", "content": "Run deploy.sh"},
        ]
        result = retriever._load_project_ops(1, str(tmp_path))
        assert "Deploy Process" in result
        assert "Run deploy.sh" in result

    def test_tidb_entries_formatted(self, retriever, mock_store, tmp_path):
        mock_store.list_knowledge.return_value = [
            {"title": "CI Pipeline", "content": "Use GitHub Actions"},
            {"title": "Hotfix Policy", "content": "Cherry-pick to release"},
        ]
        result = retriever._load_project_ops(1, str(tmp_path))
        assert "## Project Operations Knowledge" in result
        assert "### CI Pipeline" in result
        assert "### Hotfix Policy" in result

    def test_tidb_empty(self, retriever, mock_store, tmp_path):
        mock_store.list_knowledge.return_value = []
        result = retriever._load_project_ops(1, str(tmp_path))
        assert result == ""

    def test_repo_path_none(self, retriever, mock_store):
        mock_store.list_knowledge.return_value = []
        result = retriever._load_project_ops(1, None)
        mock_store.list_knowledge.assert_called_once()
        assert result == ""


# ===================================================================
# 2. build_context_for_agent
# ===================================================================

class TestBuildContextForAgent:

    def test_includes_project_ops(self, retriever, mock_store, tmp_path):
        md_file = tmp_path / "myswat.md"
        md_file.write_text("## Project Operations Knowledge\nOps info.\n")
        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, repo_path=str(tmp_path),
        )
        assert "Ops info." in result

    def test_includes_knowledge_results(self, retriever, mock_store):
        mock_store.search_knowledge.return_value = [
            {"title": "Auth Flow", "content": "OAuth2 based", "category": "knowledge"},
        ]
        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="implement auth",
        )
        assert "Auth Flow" in result

    def test_includes_myswat_command_guide(self, retriever, mock_store):
        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, repo_path="/tmp/proj",
        )
        assert "MySwat Project Access" in result
        assert "./myswat status -p proj" in result
        assert "./myswat history -p proj --turns 50" in result
        assert "/status" in result

    def test_includes_current_session_turns(self, retriever, mock_store):
        mock_store.get_session.return_value = {"compacted_through_turn_index": -1}
        mock_store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="Please fix the bug"),
        ]
        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, current_session_id=1,
        )
        assert "fix the bug" in result

    def test_includes_current_task_state(self, retriever, mock_store):
        mock_store.get_session.return_value = {
            "compacted_through_turn_index": -1,
            "work_item_id": 42,
        }
        mock_store.get_work_item.return_value = {
            "id": 42,
            "title": "Implement feature X",
            "status": "in_progress",
            "metadata_json": {},
        }
        mock_store.get_work_item_state.return_value = {
            "current_stage": "phase_2_under_review",
            "latest_summary": "Implemented the parser and added tests.",
            "next_todos": ["Address QA feedback", "Re-run tests"],
            "open_issues": ["Edge case for empty input"],
        }

        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, current_session_id=1,
        )

        assert "Current Task State" in result
        assert "Implement feature X" in result
        assert "Address QA feedback" in result
        assert "Edge case for empty input" in result

    def test_includes_work_items(self, retriever, mock_store):
        mock_store.list_work_items.return_value = [
            {"title": "Refactor DB layer", "status": "in_progress", "priority": 2},
        ]
        result = retriever.build_context_for_agent(project_id=1, agent_id=1)
        assert "Refactor DB layer" in result

    def test_includes_active_work_item_flow_summary(self, retriever, mock_store):
        mock_store.list_work_items.return_value = [
            {
                "title": "Refactor DB layer",
                "status": "in_progress",
                "priority": 2,
                "metadata_json": {
                    "task_state": {
                        "current_stage": "review_loop_reviewing",
                        "process_log": [
                            {
                                "from_role": "architect",
                                "to_role": "developer",
                                "title": "Architect delegation",
                                "summary": "Update the design doc",
                            }
                        ],
                    }
                },
            },
        ]
        result = retriever.build_context_for_agent(project_id=1, agent_id=1)
        assert "flow: architect -> developer" in result

    def test_includes_recent_project_turns(self, retriever, mock_store):
        mock_store.search_knowledge.return_value = [
            {"title": f"K{i}", "content": f"c{i}", "category": "k"}
            for i in range(2)
        ]
        mock_store.get_recent_turns_by_project.return_value = [
            {
                "agent_role": "developer",
                "turns": [{"role": "assistant", "content": "I fixed the tests"}],
            },
        ]
        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="continue",
        )
        assert "Recent Project Conversation" in result
        assert "I fixed the tests" in result
        mock_store.get_recent_turns_by_project.assert_called_once()
        mock_store.get_recent_history_for_agent.assert_not_called()

    def test_still_includes_recent_turns_when_knowledge_sufficient(self, retriever, mock_store):
        mock_store.search_knowledge.return_value = [
            {"title": f"K{i}", "content": f"c{i}", "category": "k"}
            for i in range(3)
        ]
        mock_store.get_recent_turns_by_project.return_value = [
            {
                "agent_role": "architect",
                "turns": [{"role": "user", "content": "keep this context"}],
            },
        ]
        result = retriever.build_context_for_agent(
            project_id=1, agent_id=1, task_description="do something",
        )
        assert "keep this context" in result
        mock_store.get_recent_turns_by_project.assert_called_once()
        mock_store.get_recent_history_for_agent.assert_not_called()

    def test_empty_context(self, retriever, mock_store):
        result = retriever.build_context_for_agent(project_id=1, agent_id=1)
        assert "MySwat Project Access" in result


# ===================================================================
# 3. _build_current_session_context
# ===================================================================

class TestBuildCurrentSessionContext:

    def test_ignores_watermark_and_includes_all_present_turns(self, retriever, mock_store):
        mock_store.get_session.return_value = {"compacted_through_turn_index": 1}
        mock_store.get_session_turns.return_value = [
            _make_turn(0, role="user", content="old message"),
            _make_turn(1, role="assistant", content="old reply"),
            _make_turn(2, role="user", content="new message"),
            _make_turn(3, role="assistant", content="new reply"),
        ]
        result = retriever._build_current_session_context(1, budget_tokens=5000)
        assert "new message" in result
        assert "old message" in result

    def test_truncates_long_content(self, retriever, mock_store):
        mock_store.get_session.return_value = {"compacted_through_turn_index": -1}
        mock_store.get_session_turns.return_value = [
            _make_turn(0, content="x" * 5000),
        ]
        result = retriever._build_current_session_context(1, budget_tokens=500)
        assert "truncated" in result

    def test_respects_budget(self, retriever, mock_store):
        mock_store.get_session.return_value = {"compacted_through_turn_index": -1}
        mock_store.get_session_turns.return_value = [
            _make_turn(i, content=f"Message {i} with text")
            for i in range(50)
        ]
        result = retriever._build_current_session_context(1, budget_tokens=200)
        # Can't include all 50 turns in 200 tokens
        assert len(result) < 200 * 8

    def test_empty_turns(self, retriever, mock_store):
        mock_store.get_session.return_value = {"compacted_through_turn_index": -1}
        mock_store.get_session_turns.return_value = []
        result = retriever._build_current_session_context(1, budget_tokens=5000)
        assert result == ""

    def test_no_session(self, retriever, mock_store):
        mock_store.get_session.return_value = None
        mock_store.get_session_turns.return_value = []
        result = retriever._build_current_session_context(1, budget_tokens=5000)
        assert result == ""


# ===================================================================
# 4. _build_cross_role_history
# ===================================================================

class TestBuildCrossRoleHistory:

    def test_formats_role_groups(self, retriever):
        history = [
            {
                "agent_role": "developer",
                "turns": [
                    {"role": "user", "content": "Add login endpoint"},
                    {"role": "assistant", "content": "Done, added /login"},
                ],
            },
        ]
        result = retriever._build_cross_role_history(history, budget_tokens=5000)
        assert "### [developer] Recent Turns" in result
        assert "Add login endpoint" in result
        assert "Done, added /login" in result

    def test_respects_budget_by_omitting_earlier_turns(self, retriever):
        history = [
            {
                "agent_role": f"role_{i}",
                "turns": [
                    {"role": "user", "content": f"Q{i} " + "x" * 500},
                    {"role": "assistant", "content": f"A{i} " + "y" * 500},
                ],
            }
            for i in range(20)
        ]
        result = retriever._build_cross_role_history(history, budget_tokens=100)
        # Should not include all 20 sessions' full content (would be ~20k chars)
        assert len(result) < 5000

    def test_empty_history(self, retriever):
        result = retriever._build_cross_role_history([], budget_tokens=5000)
        assert result == ""


# ===================================================================
# 5. search
# ===================================================================

class TestSearch:

    def test_passes_through(self, retriever, mock_store):
        expected = [{"title": "Result", "content": "body"}]
        mock_store.search_knowledge.return_value = expected
        result = retriever.search(
            project_id=1, query="deployment", agent_id=1, category="ops", limit=5,
        )
        mock_store.search_knowledge.assert_called_with(
            project_id=1, query="deployment", agent_id=1, category="ops", limit=5,
        )
        assert result == expected

    def test_default_params(self, retriever, mock_store):
        mock_store.search_knowledge.return_value = []
        result = retriever.search(project_id=1, query="test")
        assert result == []
