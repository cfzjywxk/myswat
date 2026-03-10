"""Tests for pure helper functions in CLI modules."""

import re
import pytest
from unittest.mock import MagicMock, patch

from myswat.cli.chat_cmd import _extract_delegation
from myswat.cli.init_cmd import _slugify
from myswat.cli.main import _infer_stage_labels
from myswat.cli.progress import _fmt_duration, _build_live_display


# ── _fmt_duration ──


class TestFmtDuration:
    def test_seconds_only(self):
        assert _fmt_duration(5) == "5s"

    def test_zero_seconds(self):
        assert _fmt_duration(0) == "0s"

    def test_59_seconds(self):
        assert _fmt_duration(59) == "59s"

    def test_one_minute(self):
        assert _fmt_duration(60) == "1m00s"

    def test_minutes_and_seconds(self):
        assert _fmt_duration(95) == "1m35s"

    def test_59_minutes(self):
        assert _fmt_duration(3599) == "59m59s"

    def test_one_hour(self):
        assert _fmt_duration(3600) == "1h00m00s"

    def test_hours_minutes_seconds(self):
        assert _fmt_duration(3725) == "1h02m05s"

    def test_float_truncated(self):
        assert _fmt_duration(5.9) == "5s"


# ── _extract_delegation ──


class TestExtractDelegation:
    def test_extracts_task_line(self):
        text = "Here is my plan:\n```delegate\nTASK: implement the auth module\n```\nDone."
        result = _extract_delegation(text)
        assert result == "implement the auth module"

    def test_task_case_insensitive(self):
        text = "```delegate\ntask: do something\n```"
        result = _extract_delegation(text)
        assert result == "do something"

    def test_no_delegate_block(self):
        text = "Just a normal response with no delegation."
        result = _extract_delegation(text)
        assert result is None

    def test_delegate_block_without_task_line(self):
        text = "```delegate\nsome instructions to follow\n```"
        result = _extract_delegation(text)
        assert result == "some instructions to follow"

    def test_empty_delegate_block(self):
        text = "```delegate\n```"
        result = _extract_delegation(text)
        assert result is None

    def test_whitespace_only_delegate_block(self):
        text = "```delegate\n   \n```"
        result = _extract_delegation(text)
        assert result is None

    def test_multiple_lines_in_block_returns_task(self):
        text = "```delegate\nCONTEXT: auth system\nTASK: add login endpoint\nPRIORITY: high\n```"
        result = _extract_delegation(text)
        assert result == "add login endpoint"

    def test_delegate_block_fallback_whole_content(self):
        text = "```delegate\nno task prefix here\njust raw instructions\n```"
        result = _extract_delegation(text)
        assert "no task prefix here" in result


# ── _build_live_display ──


class TestBuildLiveDisplay:
    def test_returns_text_object(self):
        result = _build_live_display(0, 5.0, [])
        from rich.text import Text
        assert isinstance(result, Text)

    def test_shows_elapsed_time(self):
        result = _build_live_display(0, 65.0, [])
        plain = result.plain
        assert "1m05s" in plain

    def test_shows_live_lines(self):
        result = _build_live_display(0, 1.0, ["line1", "line2"])
        plain = result.plain
        assert "line1" in plain
        assert "line2" in plain

    def test_truncates_long_output(self):
        lines = [f"line {i}" for i in range(20)]
        result = _build_live_display(0, 1.0, lines)
        plain = result.plain
        assert "earlier lines" in plain

    def test_empty_lines(self):
        result = _build_live_display(0, 0.0, [])
        plain = result.plain
        assert "Waiting for AI agent" in plain


# ── _slugify ──


class TestSlugify:
    def test_basic_name(self):
        assert _slugify("My Project") == "my-project"

    def test_special_characters(self):
        assert _slugify("My Project!@#") == "my-project"

    def test_leading_trailing_spaces(self):
        assert _slugify("  hello world  ") == "hello-world"

    def test_multiple_special_chars(self):
        assert _slugify("foo---bar") == "foo-bar"

    def test_numbers_preserved(self):
        assert _slugify("project 42") == "project-42"

    def test_already_slug(self):
        assert _slugify("my-project") == "my-project"

    def test_uppercase(self):
        assert _slugify("MY_PROJECT") == "my-project"

    def test_empty_string(self):
        assert _slugify("") == ""

    def test_only_special_chars(self):
        assert _slugify("!@#$%") == ""


# ── _infer_stage_labels ──


class TestInferStageLabels:
    def test_design_review(self):
        rounds = [{"proposer_role": "developer", "reviewer_role": "qa_main"}]
        labels = _infer_stage_labels(rounds)
        assert labels == ["Design Review"]

    def test_plan_review(self):
        rounds = [
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
        ]
        labels = _infer_stage_labels(rounds)
        assert labels[0] == "Design Review"
        assert labels[1] == "Plan Review"

    def test_code_review_phases(self):
        rounds = [
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
        ]
        labels = _infer_stage_labels(rounds)
        assert labels[2] == "Code Review (phase 1)"
        assert labels[3] == "Code Review (phase 2)"

    def test_test_plan_review(self):
        rounds = [{"proposer_role": "qa_main", "reviewer_role": "developer"}]
        labels = _infer_stage_labels(rounds)
        assert labels == ["Test Plan Review"]

    def test_unknown_direction(self):
        rounds = [{"proposer_role": "architect", "reviewer_role": "qa_main"}]
        labels = _infer_stage_labels(rounds)
        assert "architect" in labels[0]
        assert "qa_main" in labels[0]

    def test_empty_rounds(self):
        assert _infer_stage_labels([]) == []

    def test_mixed_stages(self):
        rounds = [
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "developer", "reviewer_role": "qa_main"},
            {"proposer_role": "qa_main", "reviewer_role": "developer"},
        ]
        labels = _infer_stage_labels(rounds)
        assert labels[0] == "Design Review"
        assert labels[1] == "Plan Review"
        assert labels[2] == "Code Review (phase 1)"
        assert labels[3] == "Test Plan Review"

    def test_qa_vice_as_reviewer(self):
        rounds = [{"proposer_role": "developer", "reviewer_role": "qa_vice"}]
        labels = _infer_stage_labels(rounds)
        assert labels == ["Design Review"]
