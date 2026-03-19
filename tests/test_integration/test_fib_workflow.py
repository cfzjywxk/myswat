"""End-to-end workflow test for the MCP-oriented fib demo flow."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from itertools import count
from types import SimpleNamespace

from myswat.cli.worker_cmd import run_worker
from myswat.server.contracts import AssignmentEnvelope, ReviewVerdictEnvelope, StageRunCompletion
from myswat.server.mcp_stdio import MySwatMCPDispatcher
from myswat.workflow.kernel import WorkflowKernel
from myswat.workflow.modes import WorkMode
from myswat.workflow.runtime import WorkflowRuntime


def _participant(agent_id: int, role: str) -> WorkflowRuntime:
    return WorkflowRuntime(
        agent_row={
            "id": agent_id,
            "role": role,
            "display_name": role,
        }
    )


class _MemoryStore:
    def __init__(self) -> None:
        self.artifacts: list[dict] = []
        self.stage_runs: dict[int, dict] = {}
        self.project = {"id": 1, "slug": "fib-demo", "name": "fib-demo", "repo_path": "/tmp/fib-demo"}

    def get_latest_artifact_by_type(self, work_item_id: int, artifact_type: str) -> dict | None:
        matches = [
            row for row in self.artifacts
            if row["work_item_id"] == work_item_id and row["artifact_type"] == artifact_type
        ]
        return matches[-1] if matches else None

    def list_artifacts(self, work_item_id: int) -> list[dict]:
        return [row for row in self.artifacts if row["work_item_id"] == work_item_id]

    def get_latest_stage_run(self, work_item_id: int, stage_name: str):
        matches = [
            row for row in self.stage_runs.values()
            if row["work_item_id"] == work_item_id and row["stage_name"] == stage_name
        ]
        if not matches:
            return None
        latest = matches[-1]
        return SimpleNamespace(iteration=latest["iteration"])

    def get_project(self, project_id: int) -> dict | None:
        if project_id == int(self.project["id"]):
            return dict(self.project)
        return None


@dataclass
class _StageRecord:
    work_item_id: int
    stage_name: str
    owner_agent_id: int | None
    owner_role: str | None
    iteration: int
    status: str
    summary: str
    prompt: str
    focus: str
    artifact_type: str | None
    artifact_title: str | None
    artifact_id: int | None = None
    artifact_content: str = ""


@dataclass
class _ReviewRecord:
    work_item_id: int
    cycle_id: int
    artifact_id: int
    reviewer_role: str
    reviewer_agent_id: int
    stage_name: str
    iteration: int
    prompt: str
    focus: str
    status: str = "pending"
    verdict: str = "pending"
    issues: list[str] | None = None
    summary: str = ""


class _WorkflowCoordinationService:
    def __init__(self, store: _MemoryStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._runtime_ids = count(1)
        self._stage_ids = count(100)
        self._artifact_ids = count(1000)
        self._cycle_ids = count(2000)
        self.stage_runs: dict[int, _StageRecord] = {}
        self.review_cycles: dict[int, _ReviewRecord] = {}
        self.runtime_status: dict[int, str] = {}
        self.decisions: list[dict] = []

    def resolve_project(self, request):
        return SimpleNamespace(
            model_dump=lambda: {
                "project_id": int(self.store.project["id"]),
                "project_slug": str(self.store.project["slug"]),
                "name": str(self.store.project["name"]),
                "repo_path": self.store.project["repo_path"],
            }
        )

    def register_runtime(self, request):
        runtime_id = next(self._runtime_ids)
        self.runtime_status[runtime_id] = "online"
        return SimpleNamespace(model_dump=lambda: {"runtime_registration_id": runtime_id})

    def heartbeat_runtime(self, request) -> None:
        self.runtime_status[int(request.runtime_registration_id)] = "online"

    def renew_stage_run_lease(self, request) -> None:
        return

    def renew_review_cycle_lease(self, request) -> None:
        return

    def update_runtime_status(self, request) -> dict:
        self.runtime_status[int(request.runtime_registration_id)] = str(request.status)
        return {
            "runtime_registration_id": int(request.runtime_registration_id),
            "status": str(request.status),
        }

    def report_status(self, request) -> dict:
        return {"ok": True, "summary": request.summary}

    def get_work_item_snapshot(self, request) -> dict:
        return {
            "project": dict(self.store.project),
            "work_item": {"id": int(request.work_item_id), "project_id": int(self.store.project["id"])},
            "task_state": {},
            "recent_artifacts": list(self.store.artifacts[-5:]),
            "recent_events": [],
            "knowledge": [],
            "system_context": "",
        }

    def search_knowledge(self, request) -> list[dict]:
        return []

    def get_recent_artifacts(self, request) -> list[dict]:
        return list(self.store.artifacts[-int(request.limit):])

    def submit_artifact(self, request):
        artifact_id = next(self._artifact_ids)
        artifact = {
            "id": artifact_id,
            "work_item_id": int(request.work_item_id),
            "agent_id": int(request.agent_id),
            "iteration": int(request.iteration),
            "artifact_type": str(request.artifact_type),
            "title": request.title,
            "content": str(request.content),
            "metadata_json": request.metadata_json or {},
        }
        with self._cond:
            self.store.artifacts.append(artifact)
            self._cond.notify_all()
        return {"artifact_id": artifact_id}

    def start_stage_run(self, request):
        stage_run_id = next(self._stage_ids)
        record = _StageRecord(
            work_item_id=int(request.work_item_id),
            stage_name=str(request.stage_name),
            owner_agent_id=request.owner_agent_id,
            owner_role=request.owner_role,
            iteration=int(request.iteration),
            status=str(request.status),
            summary=str(request.summary or ""),
            prompt=str(request.task_prompt or ""),
            focus=str(request.task_focus or ""),
            artifact_type=request.artifact_type,
            artifact_title=request.artifact_title,
        )
        with self._cond:
            self.stage_runs[stage_run_id] = record
            self.store.stage_runs[stage_run_id] = {
                "work_item_id": record.work_item_id,
                "stage_name": record.stage_name,
                "iteration": record.iteration,
            }
            self._cond.notify_all()
        return SimpleNamespace(stage_run_id=stage_run_id)

    def wait_for_stage_run_completion(self, request):
        stage_run_id = int(request.stage_run_id)
        with self._cond:
            while self.stage_runs[stage_run_id].status not in {"completed", "blocked", "cancelled"}:
                self._cond.wait(timeout=request.poll_interval_seconds)
            record = self.stage_runs[stage_run_id]
        return StageRunCompletion(
            stage_run_id=stage_run_id,
            work_item_id=record.work_item_id,
            stage_name=record.stage_name,
            status=record.status,
            summary=record.summary,
            artifact_id=record.artifact_id,
            artifact_content=record.artifact_content,
        )

    def claim_next_assignment(self, request):
        runtime_id = int(request.runtime_registration_id)
        role = str(request.agent_role)
        with self._cond:
            for stage_run_id in sorted(self.stage_runs):
                record = self.stage_runs[stage_run_id]
                if record.owner_role != role or record.status != "pending":
                    continue
                record.status = "claimed"
                return AssignmentEnvelope(
                    assignment_kind="stage",
                    runtime_registration_id=runtime_id,
                    project_id=int(request.project_id),
                    work_item_id=record.work_item_id,
                    stage_run_id=stage_run_id,
                    stage_name=record.stage_name,
                    agent_id=record.owner_agent_id,
                    agent_role=record.owner_role,
                    iteration=record.iteration,
                    prompt=record.prompt,
                    focus=record.focus,
                    system_context=f"kind=stage;stage={record.stage_name}",
                    artifact_type=record.artifact_type,
                    artifact_title=record.artifact_title,
                )

            for cycle_id in sorted(self.review_cycles):
                record = self.review_cycles[cycle_id]
                if record.reviewer_role != role or record.status != "pending":
                    continue
                record.status = "claimed"
                artifact = next(item for item in self.store.artifacts if item["id"] == record.artifact_id)
                return AssignmentEnvelope(
                    assignment_kind="review",
                    runtime_registration_id=runtime_id,
                    project_id=int(request.project_id),
                    work_item_id=record.work_item_id,
                    review_cycle_id=record.cycle_id,
                    stage_name=record.stage_name,
                    agent_id=record.reviewer_agent_id,
                    agent_role=record.reviewer_role,
                    iteration=record.iteration,
                    prompt=record.prompt,
                    focus=record.focus,
                    system_context=f"kind=review;stage={record.stage_name}",
                    artifact_id=record.artifact_id,
                    metadata_json={"artifact_content": artifact["content"]},
                )

        return AssignmentEnvelope(
            assignment_kind="none",
            runtime_registration_id=runtime_id,
            project_id=int(request.project_id),
        )

    def complete_stage_task(self, request):
        artifact_id = next(self._artifact_ids)
        artifact = {
            "id": artifact_id,
            "work_item_id": int(request.work_item_id),
            "agent_id": int(request.agent_id),
            "iteration": int(request.iteration),
            "artifact_type": str(request.artifact_type),
            "title": request.title,
            "content": str(request.content),
            "metadata_json": request.metadata_json or {},
        }
        with self._cond:
            self.store.artifacts.append(artifact)
            record = self.stage_runs[int(request.stage_run_id)]
            record.status = "completed"
            record.summary = str(request.summary or "")
            record.artifact_id = artifact_id
            record.artifact_content = str(request.content)
            self._cond.notify_all()
        return StageRunCompletion(
            stage_run_id=int(request.stage_run_id),
            work_item_id=int(request.work_item_id),
            stage_name=str(request.stage_name),
            status="completed",
            summary=str(request.summary or ""),
            artifact_id=artifact_id,
            artifact_content=str(request.content),
        )

    def fail_stage_task(self, request):
        with self._cond:
            record = self.stage_runs[int(request.stage_run_id)]
            record.status = "blocked"
            record.summary = str(request.summary)
            self._cond.notify_all()
        return StageRunCompletion(
            stage_run_id=int(request.stage_run_id),
            work_item_id=int(request.work_item_id),
            stage_name=str(request.stage_name),
            status="blocked",
            summary=str(request.summary),
        )

    def request_review(self, request):
        cycle_id = next(self._cycle_ids)
        with self._cond:
            self.review_cycles[cycle_id] = _ReviewRecord(
                work_item_id=int(request.work_item_id),
                cycle_id=cycle_id,
                artifact_id=int(request.artifact_id),
                reviewer_role=str(request.reviewer_role),
                reviewer_agent_id=int(request.reviewer_agent_id),
                stage_name=str(request.stage),
                iteration=int(request.iteration),
                prompt=str(request.task_prompt or ""),
                focus=str(request.task_focus or ""),
            )
            self._cond.notify_all()
        return SimpleNamespace(cycle_id=cycle_id)

    def wait_for_review_verdicts(self, request):
        cycle_ids = [int(cycle_id) for cycle_id in request.cycle_ids]
        with self._cond:
            while True:
                pending = [
                    self.review_cycles[cycle_id]
                    for cycle_id in cycle_ids
                    if self.review_cycles[cycle_id].status != "completed"
                ]
                if not pending:
                    break
                self._cond.wait(timeout=request.poll_interval_seconds)
            records = [self.review_cycles[cycle_id] for cycle_id in cycle_ids]
        return [
            ReviewVerdictEnvelope(
                cycle_id=record.cycle_id,
                reviewer_role=record.reviewer_role,
                verdict=record.verdict,  # type: ignore[arg-type]
                issues=list(record.issues or []),
                summary=record.summary,
            )
            for record in records
        ]

    def publish_review_verdict(self, request):
        with self._cond:
            record = self.review_cycles[int(request.cycle_id)]
            record.status = "completed"
            record.verdict = str(request.verdict)
            record.issues = list(request.issues)
            record.summary = str(request.summary)
            self._cond.notify_all()
        return {"ok": True}

    def persist_decision(self, request):
        payload = {
            "project_id": int(request.project_id),
            "title": str(request.title),
            "content": str(request.content),
            "category": str(request.category),
        }
        self.decisions.append(payload)
        return {"knowledge_id": len(self.decisions), "action": "created"}


class _WorkflowRunner:
    def __init__(self, role: str) -> None:
        self._role = role

    def reset_session(self) -> None:
        return

    def invoke(self, prompt: str, system_context: str | None = None):
        text = system_context or ""
        kind_match = re.search(r"\bkind=([a-z_]+)", text)
        stage_match = re.search(r"\bstage=([a-z0-9_]+)", text)
        kind = kind_match.group(1) if kind_match else None
        stage = stage_match.group(1) if stage_match else None

        if kind == "review":
            return SimpleNamespace(
                success=True,
                content='{"verdict":"lgtm","issues":[],"summary":"Looks good."}',
            )
        if stage == "design":
            return SimpleNamespace(
                success=True,
                content="# Technical Design\nImplement a Rust fibonacci sequence generator with tests.\n",
            )
        if stage == "plan":
            return SimpleNamespace(
                success=True,
                content="Phase 1: Implement the fibonacci generator and unit tests.\n",
            )
        if stage == "phase_1":
            return SimpleNamespace(
                success=True,
                content="Implemented the Rust fibonacci generator and verified the unit tests.\n",
            )
        if stage == "test_plan":
            return SimpleNamespace(
                success=True,
                content="1. Run cargo test\n2. Verify fibonacci(10) == 55\n",
            )
        if stage == "ga_test":
            return SimpleNamespace(
                success=True,
                content=json.dumps(
                    {
                        "status": "pass",
                        "summary": "All fibonacci checks passed.",
                        "tests_failed": 0,
                        "bugs": [],
                    }
                ),
            )
        if stage == "report":
            return SimpleNamespace(
                success=True,
                content="Final report: implemented and validated the Rust fibonacci generator.\n",
            )
        raise AssertionError(f"Unhandled assignment for role={self._role}, stage={stage}, kind={kind}")


class _DirectMCPClient:
    def __init__(self, service) -> None:
        self._dispatcher = MySwatMCPDispatcher(service)

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._dispatcher.call_tool(name, arguments)
        wire_body = json.dumps({"result": result}, ensure_ascii=False, default=str)
        parsed = json.loads(wire_body)
        structured = parsed["result"].get("structuredContent")
        return {} if structured is None else structured


def test_full_fib_workflow_completes_via_mcp_workers():
    store = _MemoryStore()
    service = _WorkflowCoordinationService(store)
    dispatcher_client = _DirectMCPClient(service)

    settings = SimpleNamespace(server=SimpleNamespace(request_timeout_seconds=5))
    project_row = {"id": 1, "slug": "fib-demo", "repo_path": "/tmp/fib-demo"}
    workers: list[threading.Thread] = []
    worker_results: dict[str, dict] = {}

    def _start_worker(role: str, agent_id: int) -> None:
        agent_row = {
            "id": agent_id,
            "role": role,
            "display_name": role,
            "cli_backend": "fake",
            "model_name": "fake-model",
            "cli_path": "fake",
        }

        def _run() -> None:
            worker_results[role] = run_worker(
                project_slug="fib-demo",
                role=role,
                server_url="http://unused",
                project_row=project_row,
                agent_row=agent_row,
                runner=_WorkflowRunner(role),
                settings=settings,
                mcp_client=dispatcher_client,
                idle_exit_seconds=2.0,
            )

        thread = threading.Thread(target=_run, daemon=True)
        workers.append(thread)
        thread.start()

    _start_worker("architect", 10)
    _start_worker("developer", 20)
    _start_worker("qa_main", 30)

    kernel = WorkflowKernel(
        store=store,
        service=service,
        dev=_participant(20, "developer"),
        qas=[_participant(30, "qa_main")],
        arch=_participant(10, "architect"),
        project_id=1,
        work_item_id=1,
        mode=WorkMode.full,
        auto_approve=True,
    )

    result_holder: dict[str, object] = {}

    def _run_kernel() -> None:
        result_holder["result"] = kernel.run("implement a fibonacci sequence generator in rust")

    kernel_thread = threading.Thread(target=_run_kernel, daemon=True)
    kernel_thread.start()
    kernel_thread.join(timeout=10)
    assert kernel_thread.is_alive() is False

    for thread in workers:
        thread.join(timeout=5)
        assert thread.is_alive() is False

    result = result_holder["result"]
    assert result.success is True
    assert "fibonacci" in result.design.lower()
    assert "Phase 1" in result.plan
    assert len(result.phases) == 1
    assert result.phases[0].committed is True
    assert result.ga_test is not None
    assert result.ga_test.passed is True
    assert "fibonacci generator" in result.final_report.lower()

    artifact_types = [row["artifact_type"] for row in store.artifacts]
    assert artifact_types == [
        "design_doc",
        "implementation_plan",
        "phase_result",
        "test_plan",
        "test_report",
        "final_report",
    ]
    assert worker_results["architect"]["stage_assignments"] == 1
    assert worker_results["developer"]["stage_assignments"] == 3
    assert worker_results["qa_main"]["review_assignments"] == 3
    assert service.runtime_status
    assert all(status == "offline" for status in service.runtime_status.values())
