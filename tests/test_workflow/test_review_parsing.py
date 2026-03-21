"""Focused tests for review_parsing helpers."""

from __future__ import annotations

from myswat.workflow.review_parsing import (
    UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES,
    parse_unstructured_changes_requested_verdict,
)


def test_parse_unstructured_changes_requested_preserves_bold_label_content():
    verdict = parse_unstructured_changes_requested_verdict(
        """
        # QA Review

        **Problem:** The function crashes on NULL input.

        **Fix:** Add a nil guard before dereferencing the pointer.
        """
    )

    assert verdict is not None
    assert verdict.verdict == "changes_requested"
    assert "Problem: The function crashes on NULL input." in verdict.issues
    assert "Fix: Add a nil guard before dereferencing the pointer." in verdict.issues


def test_parse_unstructured_changes_requested_preserves_blockquote_content():
    verdict = parse_unstructured_changes_requested_verdict(
        """
        # QA Review

        > The retry logic doesn't handle timeout errors and reports success.
        """
    )

    assert verdict is not None
    assert verdict.verdict == "changes_requested"
    assert verdict.summary == "The retry logic doesn't handle timeout errors and reports success."
    assert verdict.issues == ["The retry logic doesn't handle timeout errors and reports success."]


def test_parse_unstructured_changes_requested_keeps_more_than_three_paragraph_issues():
    review = "\n\n".join(
        f"**Issue {index}:** Problem {index}."
        for index in range(1, UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES + 1)
    )
    verdict = parse_unstructured_changes_requested_verdict(review)

    assert verdict is not None
    assert verdict.verdict == "changes_requested"
    assert len(verdict.issues) == UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES
    assert verdict.issues[-1] == (
        f"Issue {UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES}: "
        f"Problem {UNSTRUCTURED_REVIEW_MAX_PARAGRAPH_ISSUES}."
    )
