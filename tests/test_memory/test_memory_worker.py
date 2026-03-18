"""Tests for the hidden memory worker wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from myswat.agents.base import AgentResponse
from myswat.config.settings import MySwatSettings
from myswat.large_payloads import extract_markdown_path, write_temp_markdown
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


def test_run_externalizes_large_prompt_and_includes_file_prompt() -> None:
    runner = MagicMock()
    runner.invoke.return_value = AgentResponse(
        content='{"knowledge_actions":[],"relation_actions":[],"index_hints":[]}',
        exit_code=0,
    )
    worker = MemoryWorker(settings=MySwatSettings(), runner=runner)
    request = LearnRequest(
        project_id=1,
        source_kind="chat",
        trigger_kind="explicit_user_request",
        payload_json={"summary": "A" * 1600},
    )

    worker.run(request=request, context={"notes": "B" * 1600})

    sent_prompt = runner.invoke.call_args.args[0]
    prompt_path = extract_markdown_path(sent_prompt)
    assert prompt_path is not None
    assert Path(prompt_path).exists()
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    assert '"summary": "' in prompt_text
    assert "B" * 200 in prompt_text

    sent_system_context = runner.invoke.call_args.kwargs["system_context"]
    assert "Large Payload Handling" in sent_system_context
    assert "Return exactly one JSON object" in sent_system_context


def test_run_reads_externalized_json_response() -> None:
    runner = MagicMock()
    response_path = write_temp_markdown(
        '{"knowledge_actions":[],"relation_actions":[],"index_hints":[]}',
        label="memory-worker-response",
        heading=None,
    )
    runner.invoke.return_value = AgentResponse(
        content=f"The detailed response is in `{response_path}`.",
        exit_code=0,
    )
    worker = MemoryWorker(settings=MySwatSettings(), runner=runner)
    request = LearnRequest(project_id=1, source_kind="chat", trigger_kind="explicit_user_request")

    envelope = worker.run(request=request, context={})

    assert envelope.knowledge_actions == []
