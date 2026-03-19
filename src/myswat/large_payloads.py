"""Helpers for spilling large prompts and responses to temporary markdown files."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

MAX_INLINE_CHARS = 1000

AGENT_FILE_PROMPT = """## Large Payload Handling
- Read any referenced `/tmp/*.md` files before answering.
- When a request, response, or supporting detail would exceed 1000 characters, write the full body to a temporary markdown file under `/tmp` and refer to that file instead of inlining everything.
- If an output must stay in JSON, keep the same JSON schema and replace oversized string fields with a short `See /tmp/...md` reference.
"""


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    text = text.strip("-")
    return text or "payload"


def _preview(text: str, limit: int = 220) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) > limit:
        return collapsed[:limit].rstrip() + "..."
    return collapsed


def write_temp_markdown(text: str, *, label: str, heading: str | None = None) -> str:
    prefix = f"myswat-{_slug(label)[:32]}-"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=tempfile.gettempdir(),
        prefix=prefix,
        suffix=".md",
    ) as handle:
        title = heading or label.replace("_", " ").strip().title()
        if title:
            handle.write(f"# {title}\n\n")
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        return handle.name


def maybe_externalize_text(
    text: str | None,
    *,
    label: str,
    intro: str,
    followup: str = "",
    threshold: int = MAX_INLINE_CHARS,
    preview_chars: int = 220,
    heading: str | None = None,
) -> tuple[str, str | None]:
    content = str(text or "")
    if len(content) <= threshold:
        return content, None

    path = write_temp_markdown(content, label=label, heading=heading)
    parts = [f"{intro} `{path}`."]
    if followup:
        parts.append(followup)
    preview = _preview(content, limit=preview_chars)
    if preview:
        parts.append(f"Preview: {preview}")
    return "\n\n".join(parts), path


def maybe_externalize_prompt(text: str | None, *, label: str) -> tuple[str, str | None]:
    return maybe_externalize_text(
        text,
        label=label,
        intro="The full request is in",
        followup="Read that markdown file completely and follow it exactly.",
        heading=f"{label.replace('_', ' ').title()} Request",
    )


def maybe_externalize_system_context(text: str | None, *, label: str) -> tuple[str, str | None]:
    return maybe_externalize_text(
        text,
        label=label,
        intro="The full system context is in",
        followup="Read it before responding and treat it as part of the system instructions.",
        heading=f"{label.replace('_', ' ').title()} System Context",
    )


def maybe_externalize_response(text: str | None, *, label: str) -> tuple[str, str | None]:
    return maybe_externalize_text(
        text,
        label=label,
        intro="The detailed response is in",
        heading=f"{label.replace('_', ' ').title()} Response",
    )


def maybe_externalize_summary(text: str | None, *, label: str) -> str:
    summary, _ = maybe_externalize_text(
        text,
        label=label,
        intro="Detailed workflow data is in",
        preview_chars=180,
        heading=f"{label.replace('_', ' ').title()} Details",
    )
    return summary


def maybe_externalize_list(items: list[str] | None, *, label: str) -> list[str] | None:
    if items is None:
        return None
    return [
        maybe_externalize_summary(item, label=f"{label}-{idx + 1}")
        for idx, item in enumerate(items)
    ]


def extract_markdown_path(text: str) -> str | None:
    match = re.search(r"`([^`]+\.md)`", text)
    if match:
        return match.group(1)
    match = re.search(r"(/tmp/[^\s]+\.md)", text)
    if match:
        return match.group(1)
    return None


def read_markdown_file(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def resolve_externalized_text(text: str | None) -> str:
    content = str(text or "")
    path = extract_markdown_path(content)
    if not path:
        return content
    try:
        resolved = read_markdown_file(path)
    except OSError:
        return content
    return resolved or content


def resolve_externalized_value(value: Any):
    if isinstance(value, str):
        return resolve_externalized_text(value)
    if isinstance(value, list):
        return [resolve_externalized_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: resolve_externalized_value(item)
            for key, item in value.items()
        }
    return value
