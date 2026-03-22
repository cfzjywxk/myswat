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


def test_design_guidance_is_dedented(tmp_path):
    skills_root = tmp_path / "skills"
    write_prd_dir = skills_root / "write-a-prd"
    write_prd_dir.mkdir(parents=True)
    (skills_root / "tdd").mkdir(parents=True)
    (write_prd_dir / "SKILL.md").write_text("---\nname: write-a-prd\n---\n", encoding="utf-8")
    ((skills_root / "tdd") / "SKILL.md").write_text("---\nname: tdd\n---\n", encoding="utf-8")

    pack = load_requirements_skill_pack(skills_root)
    guidance = pack.design_guidance()

    assert guidance.startswith("## Integrated Requirement Skills")
    assert "\n            ## Integrated Requirement Skills" not in guidance
