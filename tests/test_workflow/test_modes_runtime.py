"""Coverage-focused tests for workflow mode and runtime helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from myswat.workflow.modes import WorkMode, normalize_delegation_mode, resolve_cli_work_mode
from myswat.workflow.review_loop import _parse_verdict
from myswat.workflow.runtime import WorkflowRuntime


def test_normalize_delegation_mode_and_cli_mode_resolution():
    assert normalize_delegation_mode(" FULL ") == "full"
    assert normalize_delegation_mode("testplan") == "testplan"
    assert normalize_delegation_mode("custom-mode") == "custom-mode"

    assert resolve_cli_work_mode(design=False, develop=False, test=False) == WorkMode.full
    assert resolve_cli_work_mode(design=True, develop=False, test=False) == WorkMode.design
    assert resolve_cli_work_mode(design=False, develop=True, test=False) == WorkMode.develop
    assert resolve_cli_work_mode(design=False, develop=False, test=True) == WorkMode.test

    with pytest.raises(ValueError, match="multiple work modes selected"):
        resolve_cli_work_mode(design=True, develop=True, test=False)


def test_workflow_runtime_exposes_agent_fields_and_display_fallback():
    runtime = WorkflowRuntime(
        agent_row={
            "id": "7",
            "role": "developer",
            "cli_backend": "codex",
            "model_name": "gpt-5.4",
        }
    )

    assert runtime.agent_id == 7
    assert runtime.agent_role == "developer"
    assert runtime.display_name == "developer"
    assert runtime.cli_backend == "codex"
    assert runtime.model_name == "gpt-5.4"
    assert runtime.agent_row == {
        "id": "7",
        "role": "developer",
        "cli_backend": "codex",
        "model_name": "gpt-5.4",
    }


def test_parse_verdict_resolves_externalized_plain_text_lgtm():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write("LGTM after reading the externalized file.")
        path = handle.name
    try:
        verdict = _parse_verdict(f"See `{path}`")
        assert verdict.verdict == "lgtm"
        assert "LGTM" in verdict.summary
    finally:
        Path(path).unlink(missing_ok=True)


def test_parse_verdict_resolves_externalized_json_payload():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write('{"verdict":"lgtm","issues":[],"summary":"Structured from file."}')
        path = handle.name
    try:
        verdict = _parse_verdict(f"The detailed response is in `{path}`.")
        assert verdict.verdict == "lgtm"
        assert verdict.summary == "Structured from file."
    finally:
        Path(path).unlink(missing_ok=True)


def test_parse_verdict_recovers_externalized_markdown_review_as_changes_requested():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write(
            "# QA Review\n\n## Issue Detail\n\n### Missing explicit rollback trigger\n\n"
            "The design does not explain where the executor invokes statement rollback.\n"
        )
        path = handle.name
    try:
        verdict = _parse_verdict(f"The detailed response is in `{path}`.")
        assert verdict.verdict == "changes_requested"
        assert verdict.summary == "Missing explicit rollback trigger"
        assert verdict.issues[0] == "Missing explicit rollback trigger"
    finally:
        Path(path).unlink(missing_ok=True)
