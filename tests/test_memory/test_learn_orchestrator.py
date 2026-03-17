"""Tests for LearnOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from myswat.memory.action_executor import ActionExecutionSummary
from myswat.memory.learn_orchestrator import AsyncLearnJob, LearnOrchestrator
from myswat.models.learn import LearnActionEnvelope, LearnRequest


def test_execute_persists_request_run_and_completion() -> None:
    store = MagicMock()
    store.create_learn_request.return_value = 12
    store.create_learn_run.return_value = 34
    store.get_project.return_value = {"id": 1, "slug": "demo"}
    store.get_project_memory_revision.return_value = 5
    worker = MagicMock()
    worker.backend = "codex"
    worker.model = "gpt-5.4"
    worker.run.return_value = LearnActionEnvelope()
    executor = MagicMock()
    executor.execute.return_value = ActionExecutionSummary(knowledge_created=1)

    orchestrator = LearnOrchestrator(store=store, worker=worker, executor=executor)
    request = LearnRequest(
        project_id=1,
        source_kind="chat",
        trigger_kind="explicit_user_request",
        payload_json={"summary": "remember this"},
    )

    result = orchestrator.execute(request)

    assert result.request_id == 12
    assert result.run_id == 34
    assert result.summary.knowledge_created == 1
    assert request.id == 12
    assert store.update_learn_request_status.call_args_list == [
        call(12, "started"),
        call(12, "completed"),
    ]
    store.complete_learn_run.assert_called_once()


def test_execute_marks_failure_when_worker_raises() -> None:
    store = MagicMock()
    store.create_learn_request.return_value = 12
    store.create_learn_run.return_value = 34
    store.get_project.return_value = {"id": 1}
    store.get_project_memory_revision.return_value = 0
    worker = MagicMock()
    worker.backend = "codex"
    worker.model = "gpt-5.4"
    worker.run.side_effect = RuntimeError("bad worker")
    executor = MagicMock()

    orchestrator = LearnOrchestrator(store=store, worker=worker, executor=executor)
    request = LearnRequest(project_id=1, source_kind="work", trigger_kind="workflow_summary")

    with pytest.raises(RuntimeError, match="bad worker"):
        orchestrator.execute(request)

    store.fail_learn_run.assert_called_once_with(34, error_text="bad worker")
    assert store.update_learn_request_status.call_args_list[-1] == call(12, "failed")


def test_execute_async_returns_joinable_job() -> None:
    store = MagicMock()
    store.create_learn_request.return_value = 12
    store.create_learn_run.return_value = 34
    store.get_project.return_value = {"id": 1}
    store.get_project_memory_revision.return_value = 0
    worker = MagicMock()
    worker.backend = "codex"
    worker.model = "gpt-5.4"
    worker.run.return_value = LearnActionEnvelope()
    executor = MagicMock()
    executor.execute.return_value = ActionExecutionSummary()

    orchestrator = LearnOrchestrator(store=store, worker=worker, executor=executor)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")

    job = orchestrator.execute_async(request)

    assert isinstance(job, AsyncLearnJob)
    result = job.join(timeout=5)
    assert result is not None
    assert result.request_id == 12
