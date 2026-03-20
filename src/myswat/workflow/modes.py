"""Workflow mode definitions and public delegation routing metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorkMode(StrEnum):
    """Workflow engine modes.

    `full`, `design`, `develop`, and `test` are public entry points.
    `architect_design` and `testplan_design` are internal engine-only modes
    used by chat-led delegation flows.
    """

    full = "full"
    design = "design"
    develop = "develop"
    test = "test"
    architect_design = "architect_design"
    testplan_design = "testplan_design"


PUBLIC_WORK_MODES: tuple[WorkMode, ...] = (
    WorkMode.full,
    WorkMode.design,
    WorkMode.develop,
    WorkMode.test,
)

INTERNAL_WORK_MODES: tuple[WorkMode, ...] = (
    WorkMode.architect_design,
    WorkMode.testplan_design,
)

DEFAULT_DELEGATION_MODE = "develop"

ARCHITECT_DELEGATION_MODES: tuple[str, ...] = (
    WorkMode.full.value,
    WorkMode.design.value,
    DEFAULT_DELEGATION_MODE,
)

QA_DELEGATION_MODES: tuple[str, ...] = ("testplan",)


@dataclass(frozen=True)
class DelegationModeSpec:
    delegation_mode: str
    engine_mode: WorkMode
    chat_handler: str
    allowed_roles: frozenset[str]
    banner: str
    detail: str | None = None
    save_session_before_run: bool = False
    reset_role_session: bool = False


DELEGATION_MODE_SPECS: dict[str, DelegationModeSpec] = {
    DEFAULT_DELEGATION_MODE: DelegationModeSpec(
        delegation_mode=DEFAULT_DELEGATION_MODE,
        engine_mode=WorkMode.develop,
        chat_handler="workflow",
        allowed_roles=frozenset({"architect"}),
        banner="Architect delegated implementation workflow",
        detail="Starting implementation workflow automatically.",
        save_session_before_run=True,
        reset_role_session=True,
    ),
    WorkMode.design.value: DelegationModeSpec(
        delegation_mode=WorkMode.design.value,
        engine_mode=WorkMode.architect_design,
        chat_handler="design_review",
        allowed_roles=frozenset({"architect"}),
        banner="Architect started a design review workflow",
    ),
    WorkMode.full.value: DelegationModeSpec(
        delegation_mode=WorkMode.full.value,
        engine_mode=WorkMode.full,
        chat_handler="full_workflow",
        allowed_roles=frozenset({"architect"}),
        banner="Architect delegated full workflow",
        detail="Starting architect-led full workflow (design -> plan -> develop -> test -> report).",
        reset_role_session=True,
    ),
    QA_DELEGATION_MODES[0]: DelegationModeSpec(
        delegation_mode=QA_DELEGATION_MODES[0],
        engine_mode=WorkMode.testplan_design,
        chat_handler="testplan_review",
        allowed_roles=frozenset({"qa_main", "qa_vice"}),
        banner="QA started a test-plan review workflow",
    ),
}

_DELEGATION_MODE_ALIASES: dict[str, str] = {
    "": DEFAULT_DELEGATION_MODE,
    DEFAULT_DELEGATION_MODE: DEFAULT_DELEGATION_MODE,
    WorkMode.design.value: WorkMode.design.value,
    WorkMode.full.value: WorkMode.full.value,
    QA_DELEGATION_MODES[0]: QA_DELEGATION_MODES[0],
}


def normalize_delegation_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or "").strip().lower()
    return _DELEGATION_MODE_ALIASES.get(mode, mode)


def resolve_cli_work_mode(*, design: bool, develop: bool, test: bool) -> WorkMode:
    selected: list[WorkMode] = []
    if design:
        selected.append(WorkMode.design)
    if develop:
        selected.append(WorkMode.develop)
    if test:
        selected.append(WorkMode.test)
    if len(selected) > 1:
        raise ValueError("multiple work modes selected")
    return selected[0] if selected else WorkMode.full
