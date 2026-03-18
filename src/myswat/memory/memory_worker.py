"""Hidden worker wrapper for the unified learn pipeline."""

from __future__ import annotations

import json
from typing import Any

from myswat.agents.base import AgentRunner
from myswat.agents.factory import make_memory_worker_runner
from myswat.config.settings import MySwatSettings
from myswat.large_payloads import (
    AGENT_FILE_PROMPT,
    maybe_externalize_prompt,
    maybe_externalize_system_context,
    resolve_externalized_text,
)
from myswat.models.learn import LearnActionEnvelope, LearnRequest

MEMORY_WORKER_SYSTEM_PROMPT = """You are MySwat's hidden memory worker.

Return exactly one JSON object with this shape:
{
  "knowledge_actions": [...],
  "relation_actions": [...],
  "index_hints": [...]
}

Rules:
- Do not include markdown fences or commentary.
- Only emit actions you are confident should be persisted.
- Use exact create/update/delete knowledge actions.
- Use relation_actions and index_hints only when you have specific structured data.
"""

FILE_AWARE_MEMORY_WORKER_SYSTEM_PROMPT = "\n\n---\n\n".join(
    [AGENT_FILE_PROMPT, MEMORY_WORKER_SYSTEM_PROMPT],
)


def _extract_json_block(text: str) -> dict | list | None:
    """Extract a JSON object or array from text that may contain markdown."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        start = text.find(start_ch)
        end = text.rfind(end_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class MemoryWorker:
    """Dispatches learn requests to the hidden LLM worker and validates output."""

    def __init__(
        self,
        *,
        settings: MySwatSettings | None = None,
        runner: AgentRunner | None = None,
        workdir: str | None = None,
    ) -> None:
        self._settings = settings or MySwatSettings()
        self._backend = self._settings.memory_worker.backend
        self._model = self._settings.memory_worker.model
        self._runner = runner or make_memory_worker_runner(self._settings, workdir=workdir)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    def build_prompt(
        self,
        *,
        request: LearnRequest,
        context: dict[str, Any],
    ) -> str:
        payload = {
            "request": request.model_dump(mode="json"),
            "context": context,
        }
        return (
            "Analyze the learn request and return a strict action envelope.\n\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )

    def run(
        self,
        *,
        request: LearnRequest,
        context: dict[str, Any],
    ) -> LearnActionEnvelope:
        prompt = self.build_prompt(request=request, context=context)
        sent_prompt, _ = maybe_externalize_prompt(
            prompt,
            label="memory-worker-request",
        )
        sent_system_context, _ = maybe_externalize_system_context(
            FILE_AWARE_MEMORY_WORKER_SYSTEM_PROMPT,
            label="memory-worker-context",
        )
        response = self._runner.invoke(
            sent_prompt,
            system_context=sent_system_context,
        )
        if not response.success:
            raise RuntimeError(
                f"Memory worker failed (backend={self._backend}, model={self._model}, "
                f"exit={response.exit_code})"
            )

        data = _extract_json_block(resolve_externalized_text(response.content))
        if not isinstance(data, dict):
            raise ValueError("Memory worker did not return a JSON object envelope")
        return LearnActionEnvelope.model_validate(data)
