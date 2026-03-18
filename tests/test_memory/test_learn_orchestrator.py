"""Tests for LearnOrchestrator."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, call

import pytest

from myswat.agents.base import AgentResponse
from myswat.large_payloads import extract_markdown_path, read_markdown_file
from myswat.memory.action_executor import ActionExecutionSummary
from myswat.memory.learn_orchestrator import AsyncLearnJob, LearnOrchestrator
from myswat.memory.memory_worker import MemoryWorker
from myswat.models.learn import LearnActionEnvelope, LearnRequest
from myswat.models.session import SessionTurn


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


def test_execute_session_request_handles_datetime_context() -> None:
    store = MagicMock()
    store.create_learn_request.return_value = 12
    store.create_learn_run.return_value = 34
    store.get_project.return_value = {
        "id": 1,
        "slug": "demo",
        "created_at": datetime(2026, 3, 18, 22, 0, 0),
    }
    store.get_project_memory_revision.return_value = 5
    store.get_session.return_value = {
        "id": 19,
        "purpose": "chat",
        "created_at": datetime(2026, 3, 18, 22, 1, 0),
    }
    store.count_session_turns.return_value = 3
    store.get_session_turns.return_value = [
        SessionTurn(
            session_id=19,
            turn_index=0,
            role="user",
            content="hello",
            created_at=datetime(2026, 3, 18, 22, 2, 0),
        ),
    ]
    worker_runner = MagicMock()
    worker_runner.invoke.return_value = AgentResponse(
        content='{"knowledge_actions":[],"relation_actions":[],"index_hints":[]}',
        exit_code=0,
    )
    worker = MemoryWorker(runner=worker_runner)
    executor = MagicMock()
    executor.execute.return_value = ActionExecutionSummary()

    orchestrator = LearnOrchestrator(store=store, worker=worker, executor=executor)
    request = LearnRequest(
        project_id=1,
        source_kind="session",
        trigger_kind="session_termination",
        source_session_id=19,
        payload_json={"agent_role": "architect"},
    )

    result = orchestrator.execute(request)

    assert result.request_id == 12
    sent_prompt = worker_runner.invoke.call_args.args[0]
    prompt_text = read_markdown_file(extract_markdown_path(sent_prompt)) or sent_prompt
    assert "2026-03-18T22:01:00" in prompt_text
