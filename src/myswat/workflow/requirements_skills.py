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
    "grill-me",
    "ubiquitous-language",
    "design-an-interface",
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


# Canonical PRD section order matching the interactive PRD prompt.
# Used by design_guidance() for the PRD Snapshot section regardless of what
# the external skill template lists.
_PRD_SNAPSHOT_SECTIONS: tuple[str, ...] = (
    "Problem Statement",
    "Solution",
    "User Stories",
    "Ubiquitous Language",
    "Module Sketch",
    "Implementation Decisions",
    "Testing Decisions",
    "Out of Scope",
    "Open Questions",
)


@dataclass(frozen=True)
class RequirementsSkillPack:
    """Resolved requirement-oriented workflow guidance derived from skill files."""

    root: Path | None
    available_skills: frozenset[str]
    prd_sections: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.root is not None and bool(self.available_skills)

    def prd_guidance(self) -> str:
        if not self.enabled:
            return ""
        return _normalize_block(
            f"""
            ## Integrated Requirement Skills
            This PRD workflow is backed by the requirement skill pack
            ({", ".join(sorted(self.available_skills))}).

            Additional guidance beyond the base instructions above:
            - Group ubiquitous-language glossary terms by subdomain or lifecycle stage.
              Show relationships between terms with cardinality where useful.
            - When interviewing, if a question can be answered by exploring the codebase,
              explore instead of asking. Provide your recommended answer alongside every question.
            - When sketching modules, explicitly identify dependency categories: in-process,
              local-substitutable, remote-owned, or true-external.
            - For modules that need tests, prefer boundary tests through public interfaces
              over mocking internals.
            """
        )

    def design_guidance(self) -> str:
        if not self.enabled:
            return ""
        # Always use the canonical PRD section order defined by the interactive
        # PRD prompt, regardless of what the external skill template lists.
        prd_sections = ", ".join(_PRD_SNAPSHOT_SECTIONS)
        return _normalize_block(
            f"""
            ## Integrated Requirement Skills
            Use the configured requirement skill pack ({", ".join(sorted(self.available_skills))}) to
            shape this artifact as a requirement-to-design handoff, not just architecture notes.

            Before the core design sections, include a compact `## PRD Snapshot` using this order:
            {prd_sections}.

            ### Module and Interface Design
            - Partition into modules with one clear responsibility each.
            - Favor deep modules: a small public interface hiding significant internal complexity.
              Avoid shallow modules where the interface is nearly as complex as the implementation.
            - For each new module or changed boundary, consider at least two radically different
              interface shapes (e.g. minimize method count vs. maximize flexibility vs. optimize
              for the common case). Record the chosen interface AND the alternatives considered
              with a brief rationale for the choice.
            - State dependency direction between modules. Prefer dependencies that point inward
              (callers depend on stable interfaces, not the reverse).
            - For each module, specify: public surface, what complexity it hides, boundary invariants.

            ### Testing Strategy
            - Identify which behaviors tests should validate through public interfaces.
            - Identify the first tracer-bullet behavior: the thinnest end-to-end path that proves
              the architecture works. Describe how red-green-refactor would proceed for that path.
            - Tests should verify observable behavior, not implementation details.

            ### Delivery Slices
            - Include a final `## Issue-Ready Delivery Slices` section.
            - Each slice cuts vertically through interface, implementation, and tests together.
            - Note dependencies between slices and which can run in parallel.
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
            - Modules are deep, not shallow: the public interface should be significantly simpler
              than the implementation it hides. Flag any module whose interface is nearly as complex
              as its internals.
            - At least two interface alternatives were considered for new modules or changed
              boundaries. The design records why the chosen shape wins on simplicity, depth,
              or ease of correct use.
            - Dependencies point inward toward stable interfaces, not outward toward callers.
            - Boundary invariants and public inputs/outputs are explicit for each module.
            - The testing strategy exercises public interfaces and supports tracer-bullet TDD.
              Tests should target observable behavior, not implementation details.
            - Delivery slices are vertical (interface + implementation + tests together),
              issue-ready, and parallel-friendly — not horizontal layer splits.
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
            Execute this phase using vertical-slice TDD:

            ### Tracer Bullet First
            Start with ONE test for the thinnest end-to-end behavior in this slice.
            RED: write the test (it fails). GREEN: implement minimal code to pass.
            This proves the path works before adding detail.

            ### Incremental Loop
            For each remaining behavior in this slice:
            1. RED — write one failing test through a public interface.
            2. GREEN — implement only enough code to make it pass. Do not anticipate future tests.
            3. Move to the next behavior.

            ### Refactor After Green
            After tests pass, refactor: extract duplication, deepen modules, apply SOLID where natural.
            Never refactor while RED — get GREEN first. Run tests after each refactor step.

            ### What to Avoid
            - Never write all tests first then all implementation (horizontal slicing).
            - Avoid tests that mock internal collaborators or assert on call counts/order.
            - Avoid tests that lock onto private helpers or internal call sequences.
            - Do not defer tests into future work inside this phase.
            - Each test should describe WHAT (behavior), not HOW (implementation).
            - Each test should survive an internal refactor without breaking.
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
            - Modules remain deep: public interfaces are simpler than the implementation they hide.
              Flag any new module whose interface is nearly as complex as its internals.
            - Tests validate observable behavior through public interfaces, not implementation details.
              Red flags: mocking internal collaborators, testing private methods, asserting on call
              counts or ordering, tests that break on refactor without behavior change.
            - The phase completes a vertical slice (interface + implementation + tests together),
              not a partial horizontal layer (e.g. schema-only, API-only, or tests-deferred).
            - Tests were written alongside code (TDD discipline), not bolted on after the fact.
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
