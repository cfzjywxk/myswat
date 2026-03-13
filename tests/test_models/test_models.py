"""Comprehensive tests for myswat models: Session, SessionTurn, WorkItem, Artifact, ReviewVerdict, ReviewCycle."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from myswat.models.session import Session, SessionTurn
from myswat.models.work_item import Artifact, ReviewCycle, ReviewVerdict, WorkItem


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TestSession:
    """Tests for the Session model."""

    def test_required_fields_only(self):
        """Creating a Session with only the required fields should succeed."""
        session = Session(agent_id=1, session_uuid="uuid-abc-123")
        assert session.agent_id == 1
        assert session.session_uuid == "uuid-abc-123"

    def test_missing_agent_id_raises(self):
        """Omitting agent_id should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Session(session_uuid="uuid-abc-123")

    def test_missing_session_uuid_raises(self):
        """Omitting session_uuid should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Session(agent_id=1)

    def test_default_values(self):
        """All optional fields should have their documented defaults."""
        session = Session(agent_id=1, session_uuid="uuid-1")
        assert session.id is None
        assert session.parent_session_id is None
        assert session.status == "active"
        assert session.purpose is None
        assert session.work_item_id is None
        assert session.token_count_est == 0
        assert session.compacted_through_turn_index == -1
        assert session.compacted_at is None
        assert session.created_at is None
        assert session.updated_at is None

    def test_all_fields_set(self):
        """Setting every field explicitly should work."""
        now = datetime.now()
        session = Session(
            id=42,
            agent_id=7,
            session_uuid="uuid-full",
            parent_session_id=10,
            status="completed",
            purpose="Run integration tests",
            work_item_id=5,
            token_count_est=1500,
            compacted_through_turn_index=3,
            compacted_at=now,
            created_at=now,
            updated_at=now,
        )
        assert session.id == 42
        assert session.agent_id == 7
        assert session.session_uuid == "uuid-full"
        assert session.parent_session_id == 10
        assert session.status == "completed"
        assert session.purpose == "Run integration tests"
        assert session.work_item_id == 5
        assert session.token_count_est == 1500
        assert session.compacted_through_turn_index == 3
        assert session.compacted_at == now
        assert session.created_at == now
        assert session.updated_at == now

    @pytest.mark.parametrize(
        "status",
        ["active", "completed", "paused", "error", "archived"],
    )
    def test_status_values_are_accepted(self, status: str):
        """Various status strings should be accepted (no enum constraint)."""
        session = Session(agent_id=1, session_uuid="uuid-s", status=status)
        assert session.status == status

    def test_token_count_est_zero_default(self):
        """token_count_est defaults to 0."""
        session = Session(agent_id=1, session_uuid="uuid-t")
        assert session.token_count_est == 0

    def test_compacted_through_turn_index_default(self):
        """compacted_through_turn_index defaults to -1."""
        session = Session(agent_id=1, session_uuid="uuid-c")
        assert session.compacted_through_turn_index == -1


# ---------------------------------------------------------------------------
# SessionTurn
# ---------------------------------------------------------------------------


class TestSessionTurn:
    """Tests for the SessionTurn model."""

    def test_required_fields_only(self):
        """Creating a SessionTurn with only required fields should succeed."""
        turn = SessionTurn(
            session_id=1,
            turn_index=0,
            role="user",
            content="Hello",
        )
        assert turn.session_id == 1
        assert turn.turn_index == 0
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_missing_session_id_raises(self):
        with pytest.raises(ValidationError):
            SessionTurn(turn_index=0, role="user", content="hi")

    def test_missing_turn_index_raises(self):
        with pytest.raises(ValidationError):
            SessionTurn(session_id=1, role="user", content="hi")

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            SessionTurn(session_id=1, turn_index=0, content="hi")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            SessionTurn(session_id=1, turn_index=0, role="user")

    def test_default_values(self):
        """Optional fields should have their documented defaults."""
        turn = SessionTurn(session_id=1, turn_index=0, role="user", content="x")
        assert turn.id is None
        assert turn.token_count_est == 0
        assert turn.metadata_json is None
        assert turn.created_at is None

    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_role_values(self, role: str):
        """All documented role values should be accepted."""
        turn = SessionTurn(session_id=1, turn_index=0, role=role, content="msg")
        assert turn.role == role

    # -- metadata_json validator -------------------------------------------

    def test_metadata_json_string_parsed_to_dict(self):
        """A JSON string should be parsed into a dict by the validator."""
        raw = json.dumps({"key": "value", "num": 42})
        turn = SessionTurn(
            session_id=1,
            turn_index=0,
            role="user",
            content="x",
            metadata_json=raw,
        )
        assert turn.metadata_json == {"key": "value", "num": 42}
        assert isinstance(turn.metadata_json, dict)

    def test_metadata_json_dict_passthrough(self):
        """A dict should pass through the validator unchanged."""
        data = {"already": "parsed"}
        turn = SessionTurn(
            session_id=1,
            turn_index=0,
            role="user",
            content="x",
            metadata_json=data,
        )
        assert turn.metadata_json == {"already": "parsed"}

    def test_metadata_json_none_passthrough(self):
        """None should pass through the validator and remain None."""
        turn = SessionTurn(
            session_id=1,
            turn_index=0,
            role="user",
            content="x",
            metadata_json=None,
        )
        assert turn.metadata_json is None

    def test_metadata_json_empty_string_parsed(self):
        """An empty JSON object string should parse to an empty dict."""
        turn = SessionTurn(
            session_id=1,
            turn_index=0,
            role="user",
            content="x",
            metadata_json="{}",
        )
        assert turn.metadata_json == {}

    def test_metadata_json_nested_string(self):
        """A nested JSON string should be correctly parsed."""
        nested = json.dumps({"outer": {"inner": [1, 2, 3]}})
        turn = SessionTurn(
            session_id=1,
            turn_index=0,
            role="user",
            content="x",
            metadata_json=nested,
        )
        assert turn.metadata_json == {"outer": {"inner": [1, 2, 3]}}

    def test_metadata_json_invalid_string_raises(self):
        """An invalid JSON string should raise a ValidationError."""
        with pytest.raises(ValidationError):
            SessionTurn(
                session_id=1,
                turn_index=0,
                role="user",
                content="x",
                metadata_json="not valid json{{{",
            )

    def test_all_fields_set(self):
        now = datetime.now()
        turn = SessionTurn(
            id=99,
            session_id=5,
            turn_index=3,
            role="assistant",
            content="Response text",
            token_count_est=250,
            metadata_json={"tool": "search"},
            created_at=now,
        )
        assert turn.id == 99
        assert turn.session_id == 5
        assert turn.turn_index == 3
        assert turn.role == "assistant"
        assert turn.content == "Response text"
        assert turn.token_count_est == 250
        assert turn.metadata_json == {"tool": "search"}
        assert turn.created_at == now


# ---------------------------------------------------------------------------
# WorkItem
# ---------------------------------------------------------------------------


class TestWorkItem:
    """Tests for the WorkItem model."""

    def test_required_fields_only(self):
        """Creating a WorkItem with only required fields should succeed."""
        item = WorkItem(project_id=1, title="Fix bug", item_type="task")
        assert item.project_id == 1
        assert item.title == "Fix bug"
        assert item.item_type == "task"

    def test_missing_project_id_raises(self):
        with pytest.raises(ValidationError):
            WorkItem(title="T", item_type="task")

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            WorkItem(project_id=1, item_type="task")

    def test_missing_item_type_raises(self):
        with pytest.raises(ValidationError):
            WorkItem(project_id=1, title="T")

    def test_default_values(self):
        """All optional fields should have their documented defaults."""
        item = WorkItem(project_id=1, title="T", item_type="task")
        assert item.id is None
        assert item.description is None
        assert item.status == "pending"
        assert item.assigned_agent_id is None
        assert item.parent_item_id is None
        assert item.priority == 3
        assert item.metadata_json is None
        assert item.created_at is None
        assert item.updated_at is None

    def test_all_fields_set(self):
        now = datetime.now()
        item = WorkItem(
            id=10,
            project_id=2,
            title="Implement feature",
            description="Detailed description here",
            item_type="feature",
            status="in_progress",
            assigned_agent_id=3,
            parent_item_id=5,
            priority=1,
            metadata_json={"tags": ["urgent"]},
            created_at=now,
            updated_at=now,
        )
        assert item.id == 10
        assert item.project_id == 2
        assert item.title == "Implement feature"
        assert item.description == "Detailed description here"
        assert item.item_type == "feature"
        assert item.status == "in_progress"
        assert item.assigned_agent_id == 3
        assert item.parent_item_id == 5
        assert item.priority == 1
        assert item.metadata_json == {"tags": ["urgent"]}
        assert item.created_at == now
        assert item.updated_at == now

    def test_metadata_json_string_parsed(self):
        item = WorkItem(
            project_id=1,
            title="Task",
            item_type="task",
            metadata_json='{"task_state": {"current_stage": "design"}}',
        )
        assert item.metadata_json == {"task_state": {"current_stage": "design"}}

    @pytest.mark.parametrize(
        "item_type",
        ["task", "feature", "bug", "epic", "story", "subtask"],
    )
    def test_item_type_values(self, item_type: str):
        """Various item_type strings should be accepted."""
        item = WorkItem(project_id=1, title="T", item_type=item_type)
        assert item.item_type == item_type

    @pytest.mark.parametrize(
        "status",
        ["pending", "in_progress", "completed", "blocked", "cancelled", "review"],
    )
    def test_status_values(self, status: str):
        """Various status strings should be accepted."""
        item = WorkItem(project_id=1, title="T", item_type="task", status=status)
        assert item.status == status

    def test_priority_default_is_3(self):
        item = WorkItem(project_id=1, title="T", item_type="task")
        assert item.priority == 3

    def test_priority_can_be_set(self):
        item = WorkItem(project_id=1, title="T", item_type="task", priority=1)
        assert item.priority == 1


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


class TestArtifact:
    """Tests for the Artifact model."""

    def test_required_fields_only(self):
        """Creating an Artifact with only required fields should succeed."""
        artifact = Artifact(
            work_item_id=1,
            agent_id=2,
            iteration=1,
            artifact_type="code",
            content="print('hello')",
        )
        assert artifact.work_item_id == 1
        assert artifact.agent_id == 2
        assert artifact.iteration == 1
        assert artifact.artifact_type == "code"
        assert artifact.content == "print('hello')"

    def test_missing_work_item_id_raises(self):
        with pytest.raises(ValidationError):
            Artifact(agent_id=2, iteration=1, artifact_type="code", content="x")

    def test_missing_agent_id_raises(self):
        with pytest.raises(ValidationError):
            Artifact(work_item_id=1, iteration=1, artifact_type="code", content="x")

    def test_missing_iteration_raises(self):
        with pytest.raises(ValidationError):
            Artifact(work_item_id=1, agent_id=2, artifact_type="code", content="x")

    def test_missing_artifact_type_raises(self):
        with pytest.raises(ValidationError):
            Artifact(work_item_id=1, agent_id=2, iteration=1, content="x")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            Artifact(work_item_id=1, agent_id=2, iteration=1, artifact_type="code")

    def test_default_values(self):
        """Optional fields should have their documented defaults."""
        artifact = Artifact(
            work_item_id=1,
            agent_id=2,
            iteration=1,
            artifact_type="code",
            content="x",
        )
        assert artifact.id is None
        assert artifact.title is None
        assert artifact.metadata_json is None
        assert artifact.created_at is None

    def test_all_fields_set(self):
        now = datetime.now()
        artifact = Artifact(
            id=50,
            work_item_id=3,
            agent_id=7,
            iteration=2,
            artifact_type="document",
            title="Design Doc",
            content="# Architecture\n...",
            metadata_json={"format": "markdown"},
            created_at=now,
        )
        assert artifact.id == 50
        assert artifact.work_item_id == 3
        assert artifact.agent_id == 7
        assert artifact.iteration == 2
        assert artifact.artifact_type == "document"
        assert artifact.title == "Design Doc"
        assert artifact.content == "# Architecture\n..."
        assert artifact.metadata_json == {"format": "markdown"}
        assert artifact.created_at == now

    @pytest.mark.parametrize(
        "artifact_type",
        ["code", "document", "test", "config", "diagram"],
    )
    def test_artifact_type_values(self, artifact_type: str):
        """Various artifact_type strings should be accepted."""
        artifact = Artifact(
            work_item_id=1,
            agent_id=1,
            iteration=1,
            artifact_type=artifact_type,
            content="x",
        )
        assert artifact.artifact_type == artifact_type


# ---------------------------------------------------------------------------
# ReviewVerdict
# ---------------------------------------------------------------------------


class TestReviewVerdict:
    """Tests for the ReviewVerdict model."""

    def test_required_fields_only(self):
        """Creating a ReviewVerdict with only verdict should succeed."""
        rv = ReviewVerdict(verdict="approved")
        assert rv.verdict == "approved"

    def test_missing_verdict_raises(self):
        with pytest.raises(ValidationError):
            ReviewVerdict()

    def test_default_issues_empty_list(self):
        """issues should default to an empty list."""
        rv = ReviewVerdict(verdict="approved")
        assert rv.issues == []
        assert isinstance(rv.issues, list)

    def test_default_summary_empty_string(self):
        """summary should default to an empty string."""
        rv = ReviewVerdict(verdict="approved")
        assert rv.summary == ""

    def test_all_fields_set(self):
        rv = ReviewVerdict(
            verdict="rejected",
            issues=["Missing tests", "No docstrings"],
            summary="Needs more work before merging.",
        )
        assert rv.verdict == "rejected"
        assert rv.issues == ["Missing tests", "No docstrings"]
        assert rv.summary == "Needs more work before merging."

    @pytest.mark.parametrize(
        "verdict",
        ["approved", "rejected", "pending", "needs_revision"],
    )
    def test_verdict_values(self, verdict: str):
        """Various verdict strings should be accepted."""
        rv = ReviewVerdict(verdict=verdict)
        assert rv.verdict == verdict

    def test_model_dump_all_defaults(self):
        """model_dump should include all fields with their defaults."""
        rv = ReviewVerdict(verdict="approved")
        dumped = rv.model_dump()
        assert dumped == {
            "verdict": "approved",
            "issues": [],
            "summary": "",
        }

    def test_model_dump_with_values(self):
        """model_dump should reflect explicitly set values."""
        rv = ReviewVerdict(
            verdict="rejected",
            issues=["Issue A", "Issue B"],
            summary="Two problems found.",
        )
        dumped = rv.model_dump()
        assert dumped == {
            "verdict": "rejected",
            "issues": ["Issue A", "Issue B"],
            "summary": "Two problems found.",
        }

    def test_model_dump_roundtrip(self):
        """A ReviewVerdict created from model_dump output should be equal."""
        original = ReviewVerdict(
            verdict="approved",
            issues=["minor nit"],
            summary="Looks good overall.",
        )
        reconstructed = ReviewVerdict(**original.model_dump())
        assert reconstructed.verdict == original.verdict
        assert reconstructed.issues == original.issues
        assert reconstructed.summary == original.summary

    def test_issues_list_is_independent_copy(self):
        """Default issues list should not be shared across instances."""
        rv1 = ReviewVerdict(verdict="a")
        rv2 = ReviewVerdict(verdict="b")
        rv1.issues.append("added")
        assert rv2.issues == []


# ---------------------------------------------------------------------------
# ReviewCycle
# ---------------------------------------------------------------------------


class TestReviewCycle:
    """Tests for the ReviewCycle model."""

    def test_required_fields_only(self):
        """Creating a ReviewCycle with only required fields should succeed."""
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
        )
        assert rc.work_item_id == 1
        assert rc.proposer_agent_id == 2
        assert rc.reviewer_agent_id == 3

    def test_missing_work_item_id_raises(self):
        with pytest.raises(ValidationError):
            ReviewCycle(proposer_agent_id=2, reviewer_agent_id=3)

    def test_missing_proposer_agent_id_raises(self):
        with pytest.raises(ValidationError):
            ReviewCycle(work_item_id=1, reviewer_agent_id=3)

    def test_missing_reviewer_agent_id_raises(self):
        with pytest.raises(ValidationError):
            ReviewCycle(work_item_id=1, proposer_agent_id=2)

    def test_default_values(self):
        """Optional fields should have their documented defaults."""
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
        )
        assert rc.id is None
        assert rc.artifact_id is None
        assert rc.iteration == 1
        assert rc.proposal_session_id is None
        assert rc.review_session_id is None
        assert rc.verdict == "pending"
        assert rc.verdict_json is None
        assert rc.created_at is None
        assert rc.updated_at is None

    def test_all_fields_set(self):
        now = datetime.now()
        verdict_obj = ReviewVerdict(
            verdict="approved",
            issues=[],
            summary="All good.",
        )
        rc = ReviewCycle(
            id=100,
            work_item_id=5,
            artifact_id=20,
            iteration=3,
            proposer_agent_id=10,
            reviewer_agent_id=11,
            proposal_session_id=50,
            review_session_id=51,
            verdict="approved",
            verdict_json=verdict_obj,
            created_at=now,
            updated_at=now,
        )
        assert rc.id == 100
        assert rc.work_item_id == 5
        assert rc.artifact_id == 20
        assert rc.iteration == 3
        assert rc.proposer_agent_id == 10
        assert rc.reviewer_agent_id == 11
        assert rc.proposal_session_id == 50
        assert rc.review_session_id == 51
        assert rc.verdict == "approved"
        assert rc.verdict_json == verdict_obj
        assert rc.created_at == now
        assert rc.updated_at == now

    def test_verdict_json_accepts_review_verdict(self):
        """verdict_json should accept a ReviewVerdict instance."""
        rv = ReviewVerdict(verdict="rejected", issues=["bad"], summary="No good")
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
            verdict_json=rv,
        )
        assert rc.verdict_json is not None
        assert rc.verdict_json.verdict == "rejected"
        assert rc.verdict_json.issues == ["bad"]
        assert rc.verdict_json.summary == "No good"

    def test_verdict_json_accepts_none(self):
        """verdict_json should accept None."""
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
            verdict_json=None,
        )
        assert rc.verdict_json is None

    def test_verdict_json_accepts_dict(self):
        """verdict_json should accept a dict that matches ReviewVerdict shape."""
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
            verdict_json={"verdict": "approved", "issues": [], "summary": "ok"},
        )
        assert rc.verdict_json is not None
        assert isinstance(rc.verdict_json, ReviewVerdict)
        assert rc.verdict_json.verdict == "approved"

    @pytest.mark.parametrize(
        "verdict",
        ["pending", "approved", "rejected", "needs_revision"],
    )
    def test_verdict_values(self, verdict: str):
        """Various verdict strings should be accepted on the cycle itself."""
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
            verdict=verdict,
        )
        assert rc.verdict == verdict

    def test_iteration_default_is_1(self):
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
        )
        assert rc.iteration == 1

    def test_iteration_can_be_overridden(self):
        rc = ReviewCycle(
            work_item_id=1,
            proposer_agent_id=2,
            reviewer_agent_id=3,
            iteration=5,
        )
        assert rc.iteration == 5
