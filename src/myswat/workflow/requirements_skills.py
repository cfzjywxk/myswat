"""Skill-pack helpers for requirements-heavy workflow prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap


REQUIREMENTS_SKILL_NAMES: tuple[str, ...] = (
    "improve-codebase-architecture",
    "write-a-prd",
    "tdd",
    "prd-to-plan",
    "prd-to-issues",
)


def _default_requirements_skills_root() -> Path | None:
    repo_sibling = Path(__file__).resolve().parents[4] / "skills"
    if repo_sibling.exists():
        return repo_sibling
    return None


def _extract_tagged_block(text: str, tag: str) -> str:
    pattern = re.compile(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_second_level_headings(text: str) -> tuple[str, ...]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            headings.append(stripped[3:].strip())
    return tuple(headings)


def _normalize_block(text: str) -> str:
    return textwrap.dedent(text).strip()


@dataclass(frozen=True)
class RequirementsSkillPack:
    """Resolved requirement-oriented workflow guidance derived from skill files."""

    root: Path | None
    available_skills: frozenset[str]
    prd_sections: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.root is not None and bool(self.available_skills)

    def design_guidance(self) -> str:
        if not self.enabled:
            return ""
        prd_sections = ", ".join(self.prd_sections) if self.prd_sections else (
            "Problem Statement, Solution, User Stories, Implementation Decisions, "
            "Testing Decisions, Out of Scope, Further Notes"
        )
        return _normalize_block(
            f"""
            ## Integrated Requirement Skills
            Use the configured requirement skill pack ({", ".join(sorted(self.available_skills))}) to
            shape this artifact as a requirement-to-design handoff, not just architecture notes.

            Before the core design sections, include a compact `## PRD Snapshot` using this order:
            {prd_sections}.

            Inside the design, explicitly cover:
            - Module partitioning with one clear responsibility per module or bounded context.
            - Why the boundaries improve cohesion and reduce coupling, including dependency direction.
            - One to three deep modules or interfaces that hide most internal complexity behind a small public surface.
            - Boundary invariants, public inputs/outputs, and which behaviors tests should validate through those interfaces.
            - A TDD-ready validation strategy: identify the first tracer-bullet behavior and how red-green-refactor would proceed.
            - A final `## Issue-Ready Delivery Slices` section with thin vertical slices, dependencies, and what can run in parallel.
            """
        )

    def design_review_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Also verify that:
            - The PRD snapshot is complete, scoped, and consistent with the design.
            - Module boundaries maximize cohesion and minimize unnecessary cross-module coupling.
            - Deep-module interfaces and boundary invariants are explicit.
            - The testing strategy exercises public interfaces and supports tracer-bullet TDD.
            - Delivery slices are vertical, issue-ready, and parallel-friendly rather than horizontal layer splits.
            """
        )

    def plan_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Before listing phases, add a `## Delivery Slices` section that breaks the work into issue-ready
            vertical slices. For each slice include:
            - Title
            - Type: AFK or HITL
            - Blocked by
            - Covers: user stories or acceptance criteria
            - Parallelization notes
            - Done when: observable behavior works through a public interface and tests pass

            Prefer thin vertical slices that cut through interface, implementation, and tests together.
            Do NOT create horizontal slices such as "schema first", "API later", or "tests later".

            After the slices, keep the sequential `Phase N:` section count as low as possible. A phase may
            group one or more slices, but tests must stay inside the same slice or phase as the behavior they verify.
            """
        )

    def plan_review_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Reject plans that:
            - Decompose work by layer instead of vertical slice.
            - Separate testing into a later phase or dedicated cleanup issue.
            - Hide dependencies needed for safe parallel work.
            - Fail to expose which slices can run independently.
            """
        )

    def phase_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Execute this phase in TDD mode whenever the repo allows it:
            1. Pick the next user-visible behavior from the current slice.
            2. Write or extend one failing boundary test through a public interface.
            3. Implement only enough code to pass.
            4. Refactor behind the same interface.

            Avoid tests that lock onto private helpers or internal call sequences. Do not defer tests into
            future work inside this phase.
            """
        )

    def code_review_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Also inspect whether:
            - The changed code preserves the planned module boundaries.
            - Tests validate public behavior instead of implementation details.
            - The phase actually completes a vertical slice rather than a partial horizontal layer.
            """
        )

    def test_plan_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Organize this plan around behaviors, acceptance criteria, and public boundaries instead of internal methods.
            Include:
            - An acceptance-criteria coverage map
            - Boundary or contract tests for each major module
            - Regression checks per delivery slice
            - The merge gate: which checks must pass before parallel slices combine
            Keep the plan compatible with tracer-bullet TDD and public-interface validation.
            """
        )

    def test_plan_review_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            """
            ## Integrated Requirement Skills
            Also verify that the plan maps back to acceptance criteria, exercises public interfaces,
            and defines merge gates for independently developed slices.
            """
        )


def append_skill_guidance(base_prompt: str, guidance: str) -> str:
    if not guidance.strip():
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{guidance.strip()}\n"


def load_requirements_skill_pack(skills_root: str | Path | None) -> RequirementsSkillPack:
    resolved_root: Path | None = None
    if skills_root:
        candidate = Path(skills_root).expanduser().resolve()
        if candidate.exists():
            resolved_root = candidate
    else:
        resolved_root = _default_requirements_skills_root()

    if resolved_root is None:
        return RequirementsSkillPack(
            root=None,
            available_skills=frozenset(),
            prd_sections=tuple(),
        )

    available: set[str] = set()
    prd_sections: tuple[str, ...] = tuple()

    for skill_name in REQUIREMENTS_SKILL_NAMES:
        skill_path = resolved_root / skill_name / "SKILL.md"
        if not skill_path.exists():
            continue
        available.add(skill_name)
        if skill_name == "write-a-prd":
            text = skill_path.read_text(encoding="utf-8")
            template = _extract_tagged_block(text, "prd-template")
            prd_sections = _extract_second_level_headings(template)

    return RequirementsSkillPack(
        root=resolved_root,
        available_skills=frozenset(available),
        prd_sections=prd_sections,
    )
