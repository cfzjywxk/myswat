from __future__ import annotations

from myswat.workflow.requirements_skills import append_skill_guidance, load_requirements_skill_pack


def test_load_requirements_skill_pack_extracts_prd_sections(tmp_path):
    skills_root = tmp_path / "skills"
    write_prd_dir = skills_root / "write-a-prd"
    write_prd_dir.mkdir(parents=True)
    (skills_root / "tdd").mkdir(parents=True)
    (write_prd_dir / "SKILL.md").write_text(
        """---
name: write-a-prd
---
<prd-template>
## Problem Statement
## Solution
## User Stories
## Testing Decisions
</prd-template>
""",
        encoding="utf-8",
    )
    ((skills_root / "tdd") / "SKILL.md").write_text("---\nname: tdd\n---\n", encoding="utf-8")

    pack = load_requirements_skill_pack(skills_root)

    assert pack.enabled is True
    assert "write-a-prd" in pack.available_skills
    assert "tdd" in pack.available_skills
    assert pack.prd_sections == (
        "Problem Statement",
        "Solution",
        "User Stories",
        "Testing Decisions",
    )


def test_load_requirements_skill_pack_is_disabled_when_root_missing(tmp_path):
    pack = load_requirements_skill_pack(tmp_path / "missing-skills")

    assert pack.enabled is False
    assert pack.available_skills == frozenset()
    assert pack.prd_sections == ()


def test_append_skill_guidance_preserves_base_prompt_when_guidance_empty():
    assert append_skill_guidance("base prompt", "") == "base prompt"


def test_append_skill_guidance_appends_extra_section():
    result = append_skill_guidance("base prompt", "extra guidance")

    assert result.startswith("base prompt")
    assert result.endswith("extra guidance\n")


def _make_skill_pack(tmp_path, skill_names=("write-a-prd", "tdd")):
    """Helper to create a minimal skill pack on disk."""
    skills_root = tmp_path / "skills"
    for name in skill_names:
        skill_dir = skills_root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        content = "---\nname: {}\n---\n".format(name)
        if name == "write-a-prd":
            content = "---\nname: write-a-prd\n---\n<prd-template>\n## Problem Statement\n## Solution\n</prd-template>\n"
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return load_requirements_skill_pack(skills_root)


def test_new_skill_names_are_detected(tmp_path):
    pack = _make_skill_pack(
        tmp_path,
        ("write-a-prd", "tdd", "grill-me", "ubiquitous-language", "design-an-interface"),
    )

    assert pack.enabled is True
    assert "grill-me" in pack.available_skills
    assert "ubiquitous-language" in pack.available_skills
    assert "design-an-interface" in pack.available_skills


def test_design_guidance_is_dedented(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.design_guidance()

    assert guidance.startswith("## Integrated Requirement Skills")
    assert "\n            ## Integrated Requirement Skills" not in guidance


def test_design_guidance_includes_interface_alternatives(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.design_guidance()

    assert "two radically different" in guidance.lower() or "interface shapes" in guidance.lower()
    assert "deep module" in guidance.lower()
    assert "tracer-bullet" in guidance.lower()


def test_design_review_guidance_includes_deep_module_criteria(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.design_review_guidance()

    assert "shallow" in guidance.lower()
    assert "alternatives" in guidance.lower()


def test_plan_guidance_encourages_parallel_independent_slices(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.plan_guidance()

    assert "independent slices run in parallel" in guidance.lower()
    assert "compile, test, or run" in guidance.lower()
    assert "concrete data, api, or schema dependency" in guidance.lower()
    assert "shared contract slice" in guidance.lower()


def test_plan_review_guidance_rejects_artificial_linear_dependencies(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.plan_review_guidance()

    assert "compile, test, or run" in guidance.lower()
    assert "concrete data, api, or schema dependency" in guidance.lower()
    assert "shared contract slice" in guidance.lower()
    assert "fail to compile without slice a's code" not in guidance.lower()


def test_phase_guidance_includes_vertical_slice_tdd(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.phase_guidance()

    assert "tracer bullet" in guidance.lower()
    assert "red" in guidance.lower()
    assert "green" in guidance.lower()
    assert "horizontal" in guidance.lower()


def test_code_review_guidance_includes_test_quality_criteria(tmp_path):
    pack = _make_skill_pack(tmp_path)
    guidance = pack.code_review_guidance()

    assert "mocking internal" in guidance.lower()
    assert "vertical slice" in guidance.lower()


def test_prd_guidance_is_empty_when_disabled(tmp_path):
    pack = load_requirements_skill_pack(tmp_path / "missing-skills")

    assert pack.prd_guidance() == ""


def test_prd_guidance_includes_skill_specific_extras(tmp_path):
    pack = _make_skill_pack(tmp_path, ("write-a-prd", "tdd", "grill-me"))
    guidance = pack.prd_guidance()

    assert guidance.startswith("## Integrated Requirement Skills")
    # Should reference skill pack names
    assert "grill-me" in guidance
    # Should add extras beyond the base prompt (dependency categories, boundary tests)
    assert "dependency categories" in guidance.lower()
    assert "boundary tests" in guidance.lower()
    # Should NOT duplicate the base prompt's questioning/domain/module sections
    assert "## Questioning Discipline" not in guidance
    assert "## Domain Language" not in guidance
    assert "## Module Sketching" not in guidance


def test_design_guidance_uses_canonical_prd_sections(tmp_path):
    """PRD Snapshot always uses the canonical section order, not the skill file template."""
    pack = _make_skill_pack(tmp_path, ("write-a-prd", "tdd"))
    guidance = pack.design_guidance()

    # All canonical sections present in the correct order
    assert "Ubiquitous Language" in guidance
    assert "Module Sketch" in guidance
    assert "Open Questions" in guidance
    assert "Problem Statement" in guidance
    assert "Solution" in guidance
    # Ubiquitous Language should appear before Implementation Decisions
    assert guidance.index("Ubiquitous Language") < guidance.index("Implementation Decisions")
    # Further Notes (from the skill file) should NOT appear
    assert "Further Notes" not in guidance
