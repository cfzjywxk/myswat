from __future__ import annotations

import pytest

from myswat.workflow.prd_support import derive_requirement_title, resolve_prd_requirement


def test_resolve_prd_requirement_without_reference_returns_original():
    store = object()

    result = resolve_prd_requirement(
        store=store,
        project_id=1,
        requirement="ship the billing revamp",
    )

    assert result.submitted_requirement == "ship the billing revamp"
    assert result.effective_requirement == "ship the billing revamp"
    assert result.uses_prd_artifact is False


def test_resolve_prd_requirement_uses_artifact_and_appends_extra_instructions():
    class _Store:
        def get_artifact(self, artifact_id: int):
            assert artifact_id == 12
            return {
                "id": 12,
                "work_item_id": 7,
                "artifact_type": "prd_doc",
                "title": "PRD: Billing Revamp",
                "content": "# PRD: Billing Revamp\n\n## Problem Statement\n\nLegacy billing is brittle.",
            }

        def get_work_item(self, work_item_id: int):
            assert work_item_id == 7
            return {"id": 7, "project_id": 1}

    result = resolve_prd_requirement(
        store=_Store(),
        project_id=1,
        requirement="PRD_ARTIFACT: 12\nImplement this end-to-end.",
    )

    assert result.source_artifact_id == 12
    assert result.source_work_item_id == 7
    assert result.additional_instructions == "Implement this end-to-end."
    assert "## Additional Run Instructions" in result.effective_requirement
    assert "Legacy billing is brittle." in result.effective_requirement


def test_resolve_prd_requirement_rejects_wrong_artifact_type():
    class _Store:
        def get_artifact(self, artifact_id: int):
            return {
                "id": artifact_id,
                "work_item_id": 7,
                "artifact_type": "design_doc",
                "title": "Design",
                "content": "design",
            }

        def get_work_item(self, work_item_id: int):
            return {"id": work_item_id, "project_id": 1}

    with pytest.raises(ValueError, match="expected 'prd_doc'"):
        resolve_prd_requirement(
            store=_Store(),
            project_id=1,
            requirement="PRD artifact #12",
        )


def test_derive_requirement_title_prefers_additional_instructions_then_source_title():
    resolution = resolve_prd_requirement(
        store=type(
            "_Store",
            (),
            {
                "get_artifact": lambda self, artifact_id: {
                    "id": artifact_id,
                    "work_item_id": 7,
                    "artifact_type": "prd_doc",
                    "title": "PRD: Billing Revamp",
                    "content": "# PRD: Billing Revamp",
                },
                "get_work_item": lambda self, work_item_id: {"id": work_item_id, "project_id": 1},
            },
        )(),
        project_id=1,
        requirement="PRD_ARTIFACT: 5\nShip the first slice.",
    )

    assert derive_requirement_title(
        submitted_requirement="PRD_ARTIFACT: 5\nShip the first slice.",
        resolution=resolution,
    ) == "Ship the first slice."
