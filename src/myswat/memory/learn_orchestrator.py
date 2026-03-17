"""Unified learn orchestration."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from myswat.config.settings import MySwatSettings
from myswat.memory.action_executor import ActionExecutionSummary, ActionExecutor
from myswat.memory.memory_worker import MemoryWorker
from myswat.memory.store import MemoryStore
from myswat.models.learn import LearnRequest


@dataclass
class LearnExecutionResult:
    request_id: int
    run_id: int
    summary: ActionExecutionSummary


class AsyncLearnJob:
    """Background learn execution handle."""

    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread
        self.result: LearnExecutionResult | None = None
        self.error: Exception | None = None

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    def join(self, timeout: float | None = None) -> LearnExecutionResult | None:
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            return None
        if self.error is not None:
            raise self.error
        return self.result


class LearnOrchestrator:
    """Persist learn triggers, run the hidden worker, and apply the envelope."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        settings: MySwatSettings | None = None,
        worker: MemoryWorker | None = None,
        executor: ActionExecutor | None = None,
        workdir: str | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or MySwatSettings()
        self._worker = worker or MemoryWorker(settings=self._settings, workdir=workdir)
        self._executor = executor or ActionExecutor(store)

    def build_context(self, request: LearnRequest) -> dict[str, Any]:
        project = self._store.get_project(request.project_id) or {}
        context: dict[str, Any] = {
            "project": project,
            "memory_revision": self._store.get_project_memory_revision(request.project_id),
            "payload": request.payload_json,
        }

        if request.source_session_id is not None:
            turn_limit = 20
            if request.trigger_kind in {"session_termination", "workflow_summary"}:
                turn_limit = 200
            turns = self._store.get_session_turns(request.source_session_id, limit=turn_limit)
            session = self._store.get_session(request.source_session_id)
            context["source_session"] = {
                "id": request.source_session_id,
                "session": session,
                "turn_count": self._store.count_session_turns(request.source_session_id),
                "turns": [turn.model_dump(mode="json") for turn in turns],
            }

        if request.source_work_item_id is not None:
            work_item = self._store.get_work_item(request.source_work_item_id)
            if work_item is not None:
                context["source_work_item"] = work_item

        return context

    def execute(self, request: LearnRequest) -> LearnExecutionResult:
        request_id = request.id
        run_id: int | None = None

        try:
            if request_id is None:
                request_id = self._store.create_learn_request(
                    project_id=request.project_id,
                    source_kind=request.source_kind,
                    trigger_kind=request.trigger_kind,
                    payload_json=request.payload_json,
                    source_session_id=request.source_session_id,
                    source_work_item_id=request.source_work_item_id,
                    status="pending",
                )
                request.id = request_id

            self._store.update_learn_request_status(request_id, "started")
            request.status = "started"

            context = self.build_context(request)
            run_id = self._store.create_learn_run(
                learn_request_id=request_id,
                worker_backend=self._worker.backend,
                worker_model=self._worker.model,
                input_context_json=context,
                status="started",
            )

            envelope = self._worker.run(request=request, context=context)
            summary = self._executor.execute(request, envelope)

            self._store.complete_learn_run(
                run_id,
                output_envelope_json=envelope.model_dump(mode="json"),
            )
            self._store.update_learn_request_status(request_id, "completed")
            request.status = "completed"
            return LearnExecutionResult(request_id=request_id, run_id=run_id, summary=summary)

        except Exception as exc:
            if run_id is not None:
                self._store.fail_learn_run(run_id, error_text=str(exc))
            if request_id is not None:
                self._store.update_learn_request_status(request_id, "failed")
                request.status = "failed"
            raise

    def execute_async(self, request: LearnRequest) -> AsyncLearnJob:
        job: AsyncLearnJob

        def _target() -> None:
            try:
                job.result = self.execute(request)
            except Exception as exc:  # pragma: no cover - exercised via join()
                job.error = exc

        thread = threading.Thread(target=_target, daemon=True)
        job = AsyncLearnJob(thread)
        thread.start()
        return job

    def submit(
        self,
        request: LearnRequest,
        *,
        asynchronous: bool | None = None,
    ) -> LearnExecutionResult | AsyncLearnJob:
        use_async = (
            self._settings.memory_worker.async_enabled
            if asynchronous is None
            else asynchronous
        )
        if use_async:
            return self.execute_async(request)
        return self.execute(request)
