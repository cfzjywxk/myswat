"""Workflow review-cycle persistence tests."""

from __future__ import annotations

from types import SimpleNamespace

from myswat.workflow.engine import WorkflowEngine


class _FakeResponse:
    def __init__(self, content: str, success: bool = True) -> None:
        self.content = content
        self.success = success
        self.exit_code = 0


class _FakeSessionManager:
    def __init__(self, agent_id: int, agent_role: str, responses: list[str], session_id: int) -> None:
        self.agent_id = agent_id
        self.agent_role = agent_role
        self._responses = list(responses)
        self.session = SimpleNamespace(id=session_id)

    def send(self, prompt: str, task_description: str | None = None) -> _FakeResponse:
        return _FakeResponse(self._responses.pop(0))


class _FakeStore:
    def __init__(self) -> None:
        self._next_artifact_id = 100
        self._next_cycle_id = 200
        self.artifacts: list[dict] = []
        self.cycles: list[dict] = []
        self.verdict_updates: list[dict] = []

    def create_artifact(
        self,
        work_item_id: int,
        agent_id: int,
        iteration: int,
        artifact_type: str,
        title: str,
        content: str,
        metadata_json: dict | None = None,
    ) -> int:
        artifact_id = self._next_artifact_id
        self._next_artifact_id += 1
        self.artifacts.append({
            "id": artifact_id,
            "work_item_id": work_item_id,
            "agent_id": agent_id,
            "iteration": iteration,
            "artifact_type": artifact_type,
            "title": title,
            "content": content,
            "metadata_json": metadata_json,
        })
        return artifact_id

    def create_review_cycle(
        self,
        work_item_id: int,
        iteration: int,
        proposer_agent_id: int,
        reviewer_agent_id: int,
        artifact_id: int,
        proposal_session_id: int | None = None,
    ) -> int:
        key = (artifact_id, reviewer_agent_id)
        if any((cycle["artifact_id"], cycle["reviewer_agent_id"]) == key for cycle in self.cycles):
            raise AssertionError(f"duplicate review cycle for {key}")

        cycle_id = self._next_cycle_id
        self._next_cycle_id += 1
        self.cycles.append({
            "id": cycle_id,
            "work_item_id": work_item_id,
            "iteration": iteration,
            "proposer_agent_id": proposer_agent_id,
            "reviewer_agent_id": reviewer_agent_id,
            "artifact_id": artifact_id,
            "proposal_session_id": proposal_session_id,
        })
        return cycle_id

    def update_review_verdict(
        self,
        cycle_id: int,
        verdict: str,
        verdict_json: dict | None = None,
        review_session_id: int | None = None,
    ) -> None:
        self.verdict_updates.append({
            "cycle_id": cycle_id,
            "verdict": verdict,
            "verdict_json": verdict_json,
            "review_session_id": review_session_id,
        })


def test_review_cycles_use_distinct_artifacts_across_stages() -> None:
    store = _FakeStore()
    dev = _FakeSessionManager(agent_id=1, agent_role="developer", responses=[], session_id=11)
    qa = _FakeSessionManager(
        agent_id=2,
        agent_role="qa_main",
        responses=[
            '{"verdict": "lgtm", "issues": [], "summary": "design ok"}',
            '{"verdict": "lgtm", "issues": [], "summary": "plan ok"}',
        ],
        session_id=22,
    )

    engine = WorkflowEngine(
        store=store,
        dev_sm=dev,
        qa_sms=[qa],
        project_id=7,
        work_item_id=42,
        ask_user=lambda _prompt: "y",
    )

    design, design_iters = engine._run_review_loop("design artifact", "design", context="req")
    plan, plan_iters = engine._run_review_loop("plan artifact", "plan", context="req")

    assert design == "design artifact"
    assert plan == "plan artifact"
    assert design_iters == 1
    assert plan_iters == 1
    assert len(store.artifacts) == 2
    assert [artifact["artifact_type"] for artifact in store.artifacts] == ["design_doc", "proposal"]
    assert len(store.cycles) == 2
    assert store.cycles[0]["artifact_id"] != store.cycles[1]["artifact_id"]
    assert all(cycle["proposal_session_id"] == 11 for cycle in store.cycles)
    assert [update["review_session_id"] for update in store.verdict_updates] == [22, 22]


def test_review_cycles_track_new_artifact_each_iteration() -> None:
    store = _FakeStore()
    dev = _FakeSessionManager(
        agent_id=1,
        agent_role="developer",
        responses=["revised design artifact"],
        session_id=11,
    )
    qa = _FakeSessionManager(
        agent_id=2,
        agent_role="qa_main",
        responses=[
            '{"verdict": "changes_requested", "issues": ["tighten design"], "summary": "needs work"}',
            '{"verdict": "lgtm", "issues": [], "summary": "looks good"}',
        ],
        session_id=22,
    )

    engine = WorkflowEngine(
        store=store,
        dev_sm=dev,
        qa_sms=[qa],
        project_id=7,
        work_item_id=42,
        ask_user=lambda _prompt: "y",
    )

    artifact, iterations = engine._run_review_loop("initial design artifact", "design", context="req")

    assert artifact == "revised design artifact"
    assert iterations == 2
    assert len(store.artifacts) == 2
    assert [cycle["artifact_id"] for cycle in store.cycles] == [100, 101]
    assert [update["verdict"] for update in store.verdict_updates] == ["changes_requested", "lgtm"]
