"""Tests for pure helper functions in CLI modules."""

import re
import pytest
from unittest.mock import MagicMock, patch

from myswat.cli.chat_cmd import (
    _detect_direct_delegation_request,
    _extract_delegation,
    _strip_wrapping_quotes,
)
from myswat.cli.init_cmd import _slugify
from myswat.cli.main import _infer_stage_labels
from myswat.cli.progress import _fmt_duration, _build_live_display
from myswat.workflow.engine import WorkMode
from myswat.workflow.modes import (
    DEFAULT_DELEGATION_MODE,
    DELEGATION_MODE_SPECS,
    INTERNAL_WORK_MODES,
    PUBLIC_WORK_MODES,
    normalize_delegation_mode,
)


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
        assert result == ("implement the auth module", "develop")

    def test_task_case_insensitive(self):
        text = "```delegate\ntask: do something\n```"
        result = _extract_delegation(text)
        assert result == ("do something", "develop")

    def test_explicit_mode_is_parsed(self):
        text = "```delegate\nMODE: design\nTASK: formalize the cache design\n```"
        result = _extract_delegation(text)
        assert result == ("formalize the cache design", "design")

    def test_mode_case_insensitive(self):
        text = "```delegate\nMoDe: TESTPLAN\nTASK: finalize release checks\n```"
        result = _extract_delegation(text)
        assert result == ("finalize release checks", "testplan")

    def test_mode_blank_defaults_to_code(self):
        text = "```delegate\nMODE:   \nTASK: finalize release checks\n```"
        result = _extract_delegation(text)
        assert result == ("finalize release checks", "develop")

    def test_no_delegate_block(self):
        text = "Just a normal response with no delegation."
        result = _extract_delegation(text)
        assert result is None

    def test_delegate_block_without_task_line(self):
        text = "```delegate\nsome instructions to follow\n```"
        result = _extract_delegation(text)
        assert result == ("some instructions to follow", "develop")

    def test_delegate_block_without_task_ignores_mode_metadata(self):
        text = "```delegate\nMODE: design\nsome instructions to follow\n```"
        result = _extract_delegation(text)
        assert result == ("some instructions to follow", "design")

    def test_delegate_block_with_only_mode_returns_none(self):
        text = "```delegate\nMODE: design\n```"
        result = _extract_delegation(text)
        assert result is None

    def test_delegate_block_with_blank_task_ignores_task_metadata(self):
        text = "```delegate\nMODE: design\nTASK:   \nsome instructions to follow\n```"
        result = _extract_delegation(text)
        assert result == ("some instructions to follow", "design")

    def test_delegate_block_with_only_blank_task_returns_none(self):
        text = "```delegate\nTASK:   \n```"
        result = _extract_delegation(text)
        assert result is None

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
        assert result == ("add login endpoint", "develop")

    def test_mode_full_is_parsed(self):
        text = "```delegate\nMODE: full\nTASK: design and implement the auth module\n```"
        result = _extract_delegation(text)
        assert result == ("design and implement the auth module", "full")

    def test_mode_full_case_insensitive(self):
        text = "```delegate\nMODE: Full\nTASK: build out the feature\n```"
        result = _extract_delegation(text)
        assert result == ("build out the feature", "full")

    def test_delegate_block_fallback_whole_content(self):
        text = "```delegate\nno task prefix here\njust raw instructions\n```"
        result = _extract_delegation(text)
        assert result is not None
        assert "no task prefix here" in result[0]
        assert result[1] == "develop"


class TestDirectChatDelegation:
    def test_strip_wrapping_quotes_removes_outer_pair_only(self):
        assert _strip_wrapping_quotes('"hello world"') == "hello world"
        assert _strip_wrapping_quotes("'hello world'") == "hello world"
        assert _strip_wrapping_quotes('"hello') == '"hello'

    def test_detect_direct_architect_full_workflow_request(self):
        result = _detect_direct_delegation_request(
            "architect",
            '"Design and implement the auth module with your team"',
        )
        assert result == ("Design and implement the auth module with your team", "full")

    def test_detect_direct_request_requires_team_language(self):
        assert _detect_direct_delegation_request(
            "architect",
            "Design and implement the auth module",
        ) is None


class TestWorkflowModes:
    def test_normalize_delegation_aliases(self):
        assert normalize_delegation_mode(None) == DEFAULT_DELEGATION_MODE
        assert normalize_delegation_mode("") == DEFAULT_DELEGATION_MODE
        assert normalize_delegation_mode("develop") == DEFAULT_DELEGATION_MODE

    def test_delegation_specs_capture_public_to_engine_mapping(self):
        assert DELEGATION_MODE_SPECS["full"].engine_mode == WorkMode.full
        assert DELEGATION_MODE_SPECS["design"].engine_mode == WorkMode.architect_design
        assert DELEGATION_MODE_SPECS["develop"].engine_mode == WorkMode.develop
        assert DELEGATION_MODE_SPECS["testplan"].engine_mode == WorkMode.testplan_design

    def test_internal_modes_are_not_user_facing(self):
        assert WorkMode.architect_design in INTERNAL_WORK_MODES
        assert WorkMode.testplan_design in INTERNAL_WORK_MODES
        assert WorkMode.architect_design not in PUBLIC_WORK_MODES
        assert WorkMode.testplan_design not in PUBLIC_WORK_MODES


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

    def test_test_plan_review_with_architect_reviewer(self):
        rounds = [{"proposer_role": "qa_main", "reviewer_role": "architect"}]
        labels = _infer_stage_labels(rounds)
        assert labels == ["Test Plan Review"]

    def test_architect_to_qa_is_architect_design_review(self):
        rounds = [{"proposer_role": "architect", "reviewer_role": "qa_main"}]
        labels = _infer_stage_labels(rounds)
        assert labels == ["Architect Design Review"]

    def test_architect_to_developer_is_architect_design_review(self):
        rounds = [{"proposer_role": "architect", "reviewer_role": "developer"}]
        labels = _infer_stage_labels(rounds)
        assert labels == ["Architect Design Review"]

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
