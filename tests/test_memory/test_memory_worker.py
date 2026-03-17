"""Tests for the hidden memory worker wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from myswat.agents.base import AgentResponse
from myswat.config.settings import MySwatSettings
from myswat.memory.memory_worker import MemoryWorker
from myswat.models.learn import LearnRequest


def test_build_prompt_includes_request_and_context() -> None:
    runner = MagicMock()
    worker = MemoryWorker(settings=MySwatSettings(), runner=runner)
    request = LearnRequest(
        project_id=1,
        source_kind="chat",
        trigger_kind="explicit_user_request",
        payload_json={"summary": "remember this"},
    )

    prompt = worker.build_prompt(request=request, context={"project": {"slug": "demo"}})

    assert '"trigger_kind": "explicit_user_request"' in prompt
    assert '"slug": "demo"' in prompt


def test_run_parses_valid_json_envelope() -> None:
    runner = MagicMock()
    runner.invoke.return_value = AgentResponse(
        content="""```json
        {"knowledge_actions":[{"op":"create","category":"architecture","title":"Build","content":"Use uv"}],"relation_actions":[],"index_hints":[]}
        ```""",
        exit_code=0,
    )
    worker = MemoryWorker(settings=MySwatSettings(), runner=runner)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")

    envelope = worker.run(request=request, context={"project": {"id": 1}})

    assert envelope.knowledge_actions[0].title == "Build"
    runner.invoke.assert_called_once()


def test_run_raises_when_runner_fails() -> None:
    runner = MagicMock()
    runner.invoke.return_value = AgentResponse(content="boom", exit_code=1)
    worker = MemoryWorker(settings=MySwatSettings(), runner=runner)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")

    with pytest.raises(RuntimeError, match="Memory worker failed"):
        worker.run(request=request, context={})


def test_run_rejects_non_object_json() -> None:
    runner = MagicMock()
    runner.invoke.return_value = AgentResponse(content='["not-an-envelope"]', exit_code=0)
    worker = MemoryWorker(settings=MySwatSettings(), runner=runner)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")

    with pytest.raises(ValueError, match="JSON object envelope"):
        worker.run(request=request, context={})
