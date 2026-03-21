"""Shared helpers for parsing reviewer output."""

from __future__ import annotations

import json
import re

from myswat.large_payloads import resolve_externalized_text, resolve_externalized_value
from myswat.models.work_item import ReviewVerdict

UNSTRUCTURED_REVIEW_SUMMARY_LIMIT = 280
UNSTRUCTURED_REVIEW_ISSUE_LIMIT = 500
UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES = 6


def strip_review_code_fences(value: str) -> str:
    stripped = value.strip()
    if "```json" in stripped:
        return stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("{"):
                return candidate
    return stripped


def _resolved_review_text(raw: str) -> str:
    return resolve_externalized_text(str(raw or "")).strip()


def _collapse_review_whitespace(value: str) -> str:
    return " ".join(str(value or "").split())


def _strip_markdown_decoration(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return _collapse_review_whitespace(text)


def _is_generic_review_heading(value: str) -> bool:
    normalized = _collapse_review_whitespace(value).strip(" :.-").lower()
    return normalized in {
        "",
        "issue detail",
        "issues",
        "review",
        "review comments",
        "review summary",
        "summary",
        "details",
        "risks",
        "recommendation",
        "recommendations",
        "next steps",
        "fix",
    }


def _normalize_review_line(value: str) -> str:
    line = str(value or "").strip()
    if not line:
        return ""
    if line.startswith(">"):
        line = re.sub(r"^(>\s*)+", "", line).strip()
    if not line:
        return ""
    label_match = re.match(r"^\*\*([^*]+):\*\*(.*)$", line)
    if label_match:
        label = _strip_markdown_decoration(label_match.group(1)).strip(" -:")
        remainder = _strip_markdown_decoration(label_match.group(2)).strip(" -:")
        if not remainder:
            return ""
        return f"{label}: {remainder}" if label else remainder
    return line


def _paragraph_candidates(text: str) -> list[str]:
    paragraphs: list[str] = []
    chunks = re.split(r"\n\s*\n", str(text or ""))
    for chunk in chunks:
        lines = []
        in_code_block = False
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if line.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or not line:
                continue
            if re.match(r"^#{1,6}\s+", line):
                continue
            normalized = _normalize_review_line(line)
            if normalized:
                lines.append(normalized)
        candidate = _strip_markdown_decoration(" ".join(lines)).strip(" -:")
        if candidate:
            paragraphs.append(candidate)
    return paragraphs


def looks_like_structured_review_payload(raw: str) -> bool:
    text = _resolved_review_text(raw)
    if not text:
        return False
    stripped = strip_review_code_fences(text).lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def parse_structured_review_verdict(raw: str) -> ReviewVerdict | None:
    candidates = [strip_review_code_fences(str(raw or ""))]
    resolved_text = _resolved_review_text(raw)
    if resolved_text and resolved_text != candidates[0]:
        candidates.append(strip_review_code_fences(resolved_text))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = resolve_externalized_value(json.loads(candidate))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        if not isinstance(payload, dict):
            continue
        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in {"lgtm", "changes_requested"}:
            continue
        raw_issues = payload.get("issues") or []
        if not isinstance(raw_issues, list):
            raw_issues = [raw_issues]
        return ReviewVerdict(
            verdict=verdict,
            issues=[str(item) for item in raw_issues if str(item).strip()],
            summary=str(payload.get("summary") or ""),
        )
    return None


def parse_plain_text_lgtm_verdict(raw: str, *, summary_limit: int = 200) -> ReviewVerdict | None:
    resolved_text = _resolved_review_text(raw)
    if not resolved_text:
        return None
    lowered = resolved_text.lower()
    if "lgtm" in lowered and "changes_requested" not in lowered:
        return ReviewVerdict(
            verdict="lgtm",
            issues=[],
            summary=resolved_text[:summary_limit],
        )
    return None


def parse_unstructured_changes_requested_verdict(
    raw: str,
    *,
    summary_limit: int = UNSTRUCTURED_REVIEW_SUMMARY_LIMIT,
    issue_limit: int = UNSTRUCTURED_REVIEW_ISSUE_LIMIT,
    max_paragraph_issues: int = UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES,
) -> ReviewVerdict | None:
    resolved_text = _resolved_review_text(raw)
    if not resolved_text:
        return None

    heading_issues: list[str] = []
    for line in resolved_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("### "):
            continue
        heading = _strip_markdown_decoration(stripped[4:]).strip(" -:")
        if heading and not _is_generic_review_heading(heading):
            heading_issues.append(heading)

    paragraph_issues = _paragraph_candidates(resolved_text)
    issues: list[str] = []
    for candidate in heading_issues:
        cleaned = candidate[:issue_limit].strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(existing.lower() == lowered for existing in issues):
            continue
        issues.append(cleaned)
    paragraph_issue_count = 0
    for candidate in paragraph_issues:
        if paragraph_issue_count >= max_paragraph_issues:
            break
        cleaned = candidate[:issue_limit].strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(existing.lower() == lowered for existing in issues):
            continue
        issues.append(cleaned)
        paragraph_issue_count += 1

    if not issues:
        fallback = _strip_markdown_decoration(resolved_text)[:issue_limit].strip()
        if not fallback:
            return None
        issues = [fallback]

    summary_source = heading_issues[0] if heading_issues else issues[0]
    summary = summary_source[:summary_limit].strip()
    if not summary:
        summary = "Reviewer provided unstructured feedback; treating as changes_requested."

    return ReviewVerdict(
        verdict="changes_requested",
        issues=issues,
        summary=summary,
    )
