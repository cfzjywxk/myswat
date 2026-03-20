"""Shared helpers for parsing reviewer output."""

from __future__ import annotations

import json

from myswat.large_payloads import resolve_externalized_text, resolve_externalized_value
from myswat.models.work_item import ReviewVerdict


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


def parse_structured_review_verdict(raw: str) -> ReviewVerdict | None:
    candidates = [strip_review_code_fences(str(raw or ""))]
    resolved_text = resolve_externalized_text(str(raw or "")).strip()
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
    resolved_text = resolve_externalized_text(str(raw or "")).strip()
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
