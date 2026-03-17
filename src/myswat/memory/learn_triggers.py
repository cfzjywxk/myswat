"""Best-effort learn trigger helpers for chat and workflow entrypoints."""

from __future__ import annotations

import re
from typing import Any

from myswat.config.settings import MySwatSettings
from myswat.memory.learn_orchestrator import (
    AsyncLearnJob,
    LearnExecutionResult,
    LearnOrchestrator,
)
from myswat.memory.store import MemoryStore
from myswat.models.learn import LearnRequest

_EXPLICIT_LEARN_PATTERNS = (
    re.compile(r"^(please\s+)?(learn|remember|memorize)\b", re.IGNORECASE),
    re.compile(r"\b(learn|remember|memorize)\s+(this|that|it|these|the following)\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+this\s+in\s+mind\b", re.IGNORECASE),
    re.compile(r"\bstore\s+this\b", re.IGNORECASE),
)


def is_explicit_learn_request(text: str) -> bool:
    normalized = " ".join(str(text).split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _EXPLICIT_LEARN_PATTERNS)


def submit_chat_learn_request(
    *,
    store: MemoryStore,
    settings: MySwatSettings,
    project_id: int,
    user_message: str,
    assistant_response: str,
    workdir: str | None = None,
    source_session_id: int | None = None,
    asynchronous: bool | None = None,
) -> LearnExecutionResult | AsyncLearnJob | None:
    if not is_explicit_learn_request(user_message):
        return None

    orchestrator = LearnOrchestrator(store=store, settings=settings, workdir=workdir)
    request = LearnRequest(
        project_id=project_id,
        source_kind="chat",
        trigger_kind="explicit_user_request",
        source_session_id=source_session_id,
        payload_json={
            "user_message": user_message,
            "assistant_response": assistant_response,
        },
    )
    return orchestrator.submit(request, asynchronous=asynchronous)


def submit_workflow_summary_learn_request(
    *,
    store: MemoryStore,
    settings: MySwatSettings,
    project_id: int,
    source_work_item_id: int,
    requirement: str,
    final_status: str,
    final_summary: str,
    mode: str,
    workdir: str | None = None,
    source_session_id: int | None = None,
    payload_json: dict[str, Any] | None = None,
    asynchronous: bool | None = None,
) -> LearnExecutionResult | AsyncLearnJob:
    orchestrator = LearnOrchestrator(store=store, settings=settings, workdir=workdir)
    request_payload = {
        "requirement": requirement,
        "final_status": final_status,
        "final_summary": final_summary,
        "work_mode": mode,
    }
    if payload_json:
        request_payload.update(payload_json)
    request = LearnRequest(
        project_id=project_id,
        source_kind="work",
        trigger_kind="workflow_summary",
        source_session_id=source_session_id,
        source_work_item_id=source_work_item_id,
        payload_json=request_payload,
    )
    return orchestrator.submit(request, asynchronous=asynchronous)


def submit_session_summary_learn_request(
    *,
    store: MemoryStore,
    settings: MySwatSettings,
    project_id: int,
    source_session_id: int,
    agent_role: str,
    purpose: str | None = None,
    source_work_item_id: int | None = None,
    workdir: str | None = None,
    payload_json: dict[str, Any] | None = None,
    asynchronous: bool | None = None,
) -> LearnExecutionResult | AsyncLearnJob:
    orchestrator = LearnOrchestrator(store=store, settings=settings, workdir=workdir)
    request_payload = {
        "agent_role": agent_role,
        "purpose": purpose,
    }
    if payload_json:
        request_payload.update(payload_json)
    request = LearnRequest(
        project_id=project_id,
        source_kind="session",
        trigger_kind="session_termination",
        source_session_id=source_session_id,
        source_work_item_id=source_work_item_id,
        payload_json=request_payload,
    )
    return orchestrator.submit(request, asynchronous=asynchronous)
