"""Helpers for PRD artifacts and requirement resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re


_PRD_LINE_RE = re.compile(r"^\s*PRD_ARTIFACT\s*:\s*#?(\d+)\s*$", re.IGNORECASE)
_PRD_INLINE_RE = re.compile(r"\bPRD\s+artifact\s*#?(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PRDRequirementResolution:
    submitted_requirement: str
    effective_requirement: str
    additional_instructions: str = ""
    source_artifact_id: int | None = None
    source_work_item_id: int | None = None
    source_title: str = ""

    @property
    def uses_prd_artifact(self) -> bool:
        return self.source_artifact_id is not None


def _strip_prd_reference(requirement: str) -> tuple[int | None, str]:
    artifact_id: int | None = None
    kept_lines: list[str] = []

    for line in requirement.splitlines():
        match = _PRD_LINE_RE.match(line)
        if match and artifact_id is None:
            artifact_id = int(match.group(1))
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    if artifact_id is not None:
        return artifact_id, cleaned

    inline_match = _PRD_INLINE_RE.search(requirement)
    if inline_match is None:
        return None, requirement.strip()

    artifact_id = int(inline_match.group(1))
    cleaned = (requirement[:inline_match.start()] + requirement[inline_match.end():]).strip()
    return artifact_id, cleaned


def resolve_prd_requirement(*, store, project_id: int, requirement: str) -> PRDRequirementResolution:
    submitted_requirement = str(requirement or "").strip()
    artifact_id, additional_instructions = _strip_prd_reference(submitted_requirement)
    if artifact_id is None:
        return PRDRequirementResolution(
            submitted_requirement=submitted_requirement,
            effective_requirement=submitted_requirement,
        )

    artifact = store.get_artifact(artifact_id)
    if not artifact:
        raise ValueError(f"PRD artifact #{artifact_id} was not found.")

    artifact_type = str(artifact.get("artifact_type") or "")
    if artifact_type != "prd_doc":
        raise ValueError(
            f"Artifact #{artifact_id} is type '{artifact_type or 'unknown'}', expected 'prd_doc'."
        )

    source_work_item_id = int(artifact.get("work_item_id") or 0)
    if source_work_item_id <= 0:
        raise ValueError(f"PRD artifact #{artifact_id} is missing its source work item.")

    source_work_item = store.get_work_item(source_work_item_id)
    if not source_work_item:
        raise ValueError(
            f"PRD artifact #{artifact_id} references missing work item #{source_work_item_id}."
        )

    source_project_id = int(source_work_item.get("project_id") or 0)
    if source_project_id != int(project_id):
        raise ValueError(
            f"PRD artifact #{artifact_id} belongs to project #{source_project_id}, not project #{project_id}."
        )

    prd_content = str(artifact.get("content") or "").strip()
    if not prd_content:
        raise ValueError(f"PRD artifact #{artifact_id} has no content.")

    effective_requirement = prd_content
    if additional_instructions:
        effective_requirement += (
            "\n\n## Additional Run Instructions\n\n"
            f"{additional_instructions.strip()}\n"
        )

    return PRDRequirementResolution(
        submitted_requirement=submitted_requirement,
        effective_requirement=effective_requirement,
        additional_instructions=additional_instructions.strip(),
        source_artifact_id=artifact_id,
        source_work_item_id=source_work_item_id,
        source_title=str(artifact.get("title") or ""),
    )


def derive_requirement_title(
    *,
    submitted_requirement: str,
    resolution: PRDRequirementResolution,
) -> str:
    cleaned = resolution.additional_instructions.strip()
    if cleaned:
        return cleaned[:200]
    if resolution.source_artifact_id is not None:
        title = resolution.source_title.strip()
        if title:
            return f"Use {title}"[:200]
        return f"Use PRD artifact #{resolution.source_artifact_id}"[:200]
    return submitted_requirement[:200]
