"""Comprehensive tests for WorkflowEngine and helper functions."""

import json
import pytest
from unittest.mock import MagicMock, patch

from myswat.workflow.engine import WorkMode, WorkflowEngine, _extract_json_block


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def mock_dev_sm():
    return MagicMock()


@pytest.fixture
def mock_qa_sms():
    return [MagicMock(), MagicMock()]


@pytest.fixture
def engine(mock_store, mock_dev_sm, mock_qa_sms):
    """Return a WorkflowEngine with default settings."""
    return WorkflowEngine(
        store=mock_store,
        dev_sm=mock_dev_sm,
        qa_sms=mock_qa_sms,
        project_id="proj-1",
        work_item_id="wi-1",
    )


@pytest.fixture
def engine_custom_ask(mock_store, mock_dev_sm, mock_qa_sms):
    """Return a WorkflowEngine with a custom ask_user callable."""
    custom_ask = MagicMock(return_value="user response")
    return WorkflowEngine(
        store=mock_store,
        dev_sm=mock_dev_sm,
        qa_sms=mock_qa_sms,
        project_id="proj-2",
        work_item_id="wi-2",
        max_review_iterations=3,
        ask_user=custom_ask,
    )


# ===================================================================
# 1. _extract_json_block
# ===================================================================

class TestExtractJsonBlock:
    """Tests for the module-level _extract_json_block helper."""

    # -- ```json fenced block ------------------------------------------

    def test_json_fenced_block_object(self):
        text = 'Some preamble\n```json\n{"key": "value"}\n```\ntrailing'
        result = _extract_json_block(text)
        assert result == {"key": "value"}

    def test_json_fenced_block_array(self):
        text = 'Header\n```json\n[1, 2, 3]\n```\nfooter'
        result = _extract_json_block(text)
        assert result == [1, 2, 3]

    def test_json_fenced_block_multiline(self):
        payload = {"name": "test", "items": [1, 2], "nested": {"a": True}}
        text = f"Intro\n```json\n{json.dumps(payload, indent=2)}\n```\nDone"
        result = _extract_json_block(text)
        assert result == payload

    # -- ``` generic fenced block --------------------------------------

    def test_generic_fenced_block_object(self):
        text = 'Prefix\n```\n{"status": "ok"}\n```\nSuffix'
        result = _extract_json_block(text)
        assert result == {"status": "ok"}

    def test_generic_fenced_block_array(self):
        text = '```\n["a", "b"]\n```'
        result = _extract_json_block(text)
        assert result == ["a", "b"]

    # -- Direct JSON (no fences) ---------------------------------------

    def test_direct_json_object(self):
        text = '{"direct": true}'
        result = _extract_json_block(text)
        assert result == {"direct": True}

    def test_direct_json_array(self):
        text = '[{"id": 1}, {"id": 2}]'
        result = _extract_json_block(text)
        assert result == [{"id": 1}, {"id": 2}]

    # -- Nested / complex JSON ----------------------------------------

    def test_nested_complex_json(self):
        payload = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"deep": True}],
                },
                "siblings": ["x", "y"],
            },
            "flag": False,
        }
        text = f"```json\n{json.dumps(payload)}\n```"
        result = _extract_json_block(text)
        assert result == payload

    def test_complex_array_of_objects(self):
        payload = [
            {"name": "bug1", "severity": "high"},
            {"name": "bug2", "severity": "low", "details": {"line": 42}},
        ]
        text = json.dumps(payload)
        result = _extract_json_block(text)
        assert result == payload

    # -- No valid JSON -------------------------------------------------

    def test_no_valid_json_returns_none(self):
        text = "This is just plain text with no JSON at all."
        result = _extract_json_block(text)
        assert result is None

    def test_malformed_json_returns_none(self):
        text = '{"broken": True, missing_quote: 1}'
        result = _extract_json_block(text)
        assert result is None

    def test_empty_string_returns_none(self):
        result = _extract_json_block("")
        assert result is None

    def test_only_braces_invalid_json(self):
        text = "{ not json }"
        result = _extract_json_block(text)
        assert result is None

    # -- Edge cases ----------------------------------------------------

    def test_json_block_with_extra_whitespace(self):
        text = '```json\n  \n  {"spaced": true}  \n  \n```'
        result = _extract_json_block(text)
        assert result == {"spaced": True}

    def test_multiple_json_blocks_returns_first(self):
        text = (
            '```json\n{"first": 1}\n```\n'
            'Some text\n'
            '```json\n{"second": 2}\n```'
        )
        result = _extract_json_block(text)
        # Should extract from the first block encountered
        assert result is not None
        assert "first" in result or "second" in result


# ===================================================================
# 2. _parse_phases
# ===================================================================

class TestParsePhases:
    """Tests for WorkflowEngine._parse_phases."""

    def test_phase_n_format(self, engine):
        plan = (
            "Phase 1: Setup environment\n"
            "Phase 2: Implement core logic\n"
            "Phase 3: Add error handling\n"
        )
        result = engine._parse_phases(plan)
        assert result == [
            "Setup environment",
            "Implement core logic",
            "Add error handling",
        ]

    def test_step_n_format(self, engine):
        plan = (
            "Step 1: Initialize database\n"
            "Step 2: Create models\n"
            "Step 3: Write migrations\n"
        )
        result = engine._parse_phases(plan)
        assert result == [
            "Initialize database",
            "Create models",
            "Write migrations",
        ]

    def test_markdown_header_phase_format(self, engine):
        plan = (
            "# Project Plan\n"
            "## Phase 1: Architecture design\n"
            "Details about architecture...\n"
            "## Phase 2: Implementation\n"
            "Details about implementation...\n"
            "## Phase 3: Testing\n"
            "Details about testing...\n"
        )
        result = engine._parse_phases(plan)
        assert result == [
            "Architecture design",
            "Implementation",
            "Testing",
        ]

    def test_numbered_list_format(self, engine):
        plan = (
            "1. Design the API\n"
            "2. Implement endpoints\n"
            "3. Write integration tests\n"
        )
        result = engine._parse_phases(plan)
        assert result == [
            "Design the API",
            "Implement endpoints",
            "Write integration tests",
        ]

    def test_no_phases_returns_full_implementation(self, engine):
        plan = "Just do everything at once, no structure here."
        result = engine._parse_phases(plan)
        assert result == ["Full implementation"]

    def test_empty_plan_returns_full_implementation(self, engine):
        result = engine._parse_phases("")
        assert result == ["Full implementation"]

    def test_mixed_formats_picks_up_phases(self, engine):
        plan = (
            "Here is the plan:\n"
            "Phase 1: First thing\n"
            "Some description text\n"
            "Phase 2: Second thing\n"
        )
        result = engine._parse_phases(plan)
        assert "First thing" in result
        assert "Second thing" in result

    def test_single_phase(self, engine):
        plan = "Phase 1: Only one phase"
        result = engine._parse_phases(plan)
        assert result == ["Only one phase"]

    def test_single_numbered_item(self, engine):
        plan = "1. Single item plan"
        result = engine._parse_phases(plan)
        assert result == ["Single item plan"]


# ===================================================================
# 3. _parse_test_results
# ===================================================================

class TestParseTestResults:
    """Tests for WorkflowEngine._parse_test_results."""

    def test_json_status_pass_returns_empty(self, engine):
        output = json.dumps({"status": "pass", "details": "all good"})
        result = engine._parse_test_results(output)
        assert result == []

    def test_json_status_pass_in_fenced_block(self, engine):
        output = f'Results:\n```json\n{json.dumps({"status": "pass"})}\n```'
        result = engine._parse_test_results(output)
        assert result == []

    def test_json_with_bugs_list_returned(self, engine):
        bugs = [
            {"description": "NullPointerException in handler", "severity": "high"},
            {"description": "Off-by-one error in loop", "severity": "medium"},
        ]
        output = json.dumps({"bugs": bugs})
        result = engine._parse_test_results(output)
        assert result == bugs

    def test_json_with_empty_bugs_list(self, engine):
        output = json.dumps({"bugs": []})
        result = engine._parse_test_results(output)
        assert result == []

    def test_keyword_all_tests_pass_returns_empty(self, engine):
        output = "Ran 42 tests. all tests pass. No issues found."
        result = engine._parse_test_results(output)
        assert result == []

    def test_keyword_all_tests_pass_case_insensitive(self, engine):
        output = "ALL TESTS PASS - everything looks good"
        result = engine._parse_test_results(output)
        assert result == []

    def test_keyword_fail_returns_generic_bug(self, engine):
        output = "test_something FAIL: expected 5 got 3"
        result = engine._parse_test_results(output)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_keyword_bug_returns_generic_bug(self, engine):
        output = "Found a bug in the authentication module"
        result = engine._parse_test_results(output)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_empty_output_returns_empty(self, engine):
        result = engine._parse_test_results("")
        assert result == []

    def test_no_match_returns_empty(self, engine):
        output = "This output mentions nothing about testing outcomes."
        result = engine._parse_test_results(output)
        assert result == []

    def test_json_bugs_in_fenced_block(self, engine):
        bugs = [{"description": "Memory leak", "severity": "critical"}]
        output = f'```json\n{json.dumps({"bugs": bugs})}\n```'
        result = engine._parse_test_results(output)
        assert result == bugs


# ===================================================================
# 4. _parse_bug_estimation
# ===================================================================

class TestParseBugEstimation:
    """Tests for WorkflowEngine._parse_bug_estimation."""

    def test_json_simple_fix(self, engine):
        output = json.dumps({"assessment": "simple_fix"})
        result = engine._parse_bug_estimation(output)
        assert result == "simple_fix"

    def test_json_arch_change(self, engine):
        output = json.dumps({"assessment": "arch_change"})
        result = engine._parse_bug_estimation(output)
        assert result == "arch_change"

    def test_json_assessment_in_fenced_block(self, engine):
        output = f'```json\n{json.dumps({"assessment": "simple_fix"})}\n```'
        result = engine._parse_bug_estimation(output)
        assert result == "simple_fix"

    def test_keyword_arch_change(self, engine):
        output = "This bug requires an arch_change to fix properly."
        result = engine._parse_bug_estimation(output)
        assert result == "arch_change"

    def test_keyword_architecture_change(self, engine):
        output = "The fix requires an architecture change in the data layer."
        result = engine._parse_bug_estimation(output)
        assert result == "arch_change"

    def test_keyword_redesign(self, engine):
        output = "We need a full redesign of the module to address this."
        result = engine._parse_bug_estimation(output)
        assert result == "arch_change"

    def test_default_returns_simple_fix(self, engine):
        output = "This seems straightforward to resolve."
        result = engine._parse_bug_estimation(output)
        assert result == "simple_fix"

    def test_empty_output_returns_simple_fix(self, engine):
        result = engine._parse_bug_estimation("")
        assert result == "simple_fix"

    def test_json_arch_change_in_fenced_block(self, engine):
        output = f'```json\n{json.dumps({"assessment": "arch_change"})}\n```'
        result = engine._parse_bug_estimation(output)
        assert result == "arch_change"


# ===================================================================
# 5. _review_artifact_type
# ===================================================================

class TestReviewArtifactType:
    """Tests for WorkflowEngine._review_artifact_type."""

    def test_design_maps_to_design_doc(self, engine):
        assert engine._review_artifact_type("design") == "design_doc"

    def test_test_plan_maps_to_test_plan(self, engine):
        assert engine._review_artifact_type("test_plan") == "test_plan"

    def test_code_maps_to_diff(self, engine):
        assert engine._review_artifact_type("code") == "diff"

    def test_unknown_type_maps_to_proposal(self, engine):
        assert engine._review_artifact_type("unknown") == "proposal"

    def test_empty_string_maps_to_proposal(self, engine):
        assert engine._review_artifact_type("") == "proposal"

    def test_random_string_maps_to_proposal(self, engine):
        assert engine._review_artifact_type("something_else") == "proposal"


# ===================================================================
# 6. WorkflowEngine __init__
# ===================================================================

class TestWorkflowEngineInit:
    """Tests for WorkflowEngine constructor defaults and overrides."""

    def test_default_initialization(self, mock_store, mock_dev_sm, mock_qa_sms):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-default",
        )
        assert engine._store is mock_store
        assert engine._dev is mock_dev_sm
        assert engine._qas is mock_qa_sms
        assert engine._project_id == "proj-default"
        assert engine._work_item_id is None
        assert engine._max_review == 5
        assert engine._mode == WorkMode.full
        assert engine._ask is not None
        assert callable(engine._ask)

    def test_custom_work_item_id(self, mock_store, mock_dev_sm, mock_qa_sms):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-1",
            work_item_id="wi-42",
        )
        assert engine._work_item_id == "wi-42"

    def test_custom_max_review_iterations(self, mock_store, mock_dev_sm, mock_qa_sms):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-1",
            max_review_iterations=10,
        )
        assert engine._max_review == 10

    def test_custom_ask_user(self, mock_store, mock_dev_sm, mock_qa_sms):
        custom_ask = MagicMock(return_value="yes")
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-1",
            ask_user=custom_ask,
        )
        assert engine._ask is custom_ask
        # Verify the custom callable works as expected
        assert engine._ask("prompt") == "yes"
        custom_ask.assert_called_once_with("prompt")

    def test_all_parameters_set(self, mock_store, mock_dev_sm, mock_qa_sms):
        custom_ask = lambda prompt: "answer"
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-full",
            work_item_id="wi-full",
            max_review_iterations=2,
            mode=WorkMode.test,
            ask_user=custom_ask,
        )
        assert engine._store is mock_store
        assert engine._dev is mock_dev_sm
        assert engine._qas is mock_qa_sms
        assert engine._project_id == "proj-full"
        assert engine._work_item_id == "wi-full"
        assert engine._max_review == 2
        assert engine._mode == WorkMode.test
        assert engine._ask is custom_ask

    def test_empty_qa_sms_list(self, mock_store, mock_dev_sm):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=[],
            project_id="proj-1",
        )
        assert engine._qas == []


class TestWorkflowModeDispatch:
    """Tests for the phase-1 mode dispatch scaffold."""

    def test_run_dispatches_to_full_mode(self, engine):
        expected = MagicMock()

        with patch.object(engine, "_run_full", return_value=expected) as mock_run_full:
            result = engine.run("build feature")

        assert result is expected
        mock_run_full.assert_called_once()
        requirement, dispatch_result = mock_run_full.call_args.args
        assert requirement == "build feature"
        assert dispatch_result.requirement == "build feature"
        engine._store.update_work_item_state.assert_called_once()
        engine._store.append_work_item_process_event.assert_called_once()

    def test_run_dispatches_to_design_mode(self, mock_store, mock_dev_sm, mock_qa_sms):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-design",
            work_item_id="wi-design",
            mode=WorkMode.design,
        )
        expected = MagicMock()

        with patch.object(engine, "_run_design_mode", return_value=expected) as mock_run_design_mode:
            result = engine.run("build feature")

        assert result is expected
        mock_run_design_mode.assert_called_once()
        requirement, dispatch_result = mock_run_design_mode.call_args.args
        assert requirement == "build feature"
        assert dispatch_result.requirement == "build feature"
        engine._store.update_work_item_state.assert_called_once()
        engine._store.append_work_item_process_event.assert_called_once()

    def test_run_dispatches_to_development_mode(self, mock_store, mock_dev_sm, mock_qa_sms):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-development",
            work_item_id="wi-development",
            mode=WorkMode.development,
        )
        expected = MagicMock()

        with patch.object(engine, "_run_development_mode", return_value=expected) as mock_run_development_mode:
            result = engine.run("build feature")

        assert result is expected
        mock_run_development_mode.assert_called_once()
        requirement, dispatch_result = mock_run_development_mode.call_args.args
        assert requirement == "build feature"
        assert dispatch_result.requirement == "build feature"
        engine._store.update_work_item_state.assert_called_once()
        engine._store.append_work_item_process_event.assert_called_once()

    def test_run_dispatches_to_test_mode(self, mock_store, mock_dev_sm, mock_qa_sms):
        engine = WorkflowEngine(
            store=mock_store,
            dev_sm=mock_dev_sm,
            qa_sms=mock_qa_sms,
            project_id="proj-test",
            work_item_id="wi-test",
            mode=WorkMode.test,
        )
        expected = MagicMock()

        with patch.object(engine, "_run_test_mode", return_value=expected) as mock_run_test_mode:
            result = engine.run("build feature")

        assert result is expected
        mock_run_test_mode.assert_called_once()
        requirement, dispatch_result = mock_run_test_mode.call_args.args
        assert requirement == "build feature"
        assert dispatch_result.requirement == "build feature"
        engine._store.update_work_item_state.assert_called_once()
        engine._store.append_work_item_process_event.assert_called_once()


# ===================================================================
# 7. _build_review_prompt and _build_address_prompt (smoke tests)
# ===================================================================

class TestBuildPrompts:
    """Basic smoke tests for prompt-building methods."""

    def test_build_review_prompt_returns_string(self, engine):
        result = engine._build_review_prompt(
            artifact_type="design",
            context="Build a REST API",
            artifact="Here is the design document...",
            iteration=1,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_review_prompt_includes_artifact(self, engine):
        artifact = "The proposed architecture uses microservices."
        result = engine._build_review_prompt(
            artifact_type="code",
            context="Refactor the monolith",
            artifact=artifact,
            iteration=2,
        )
        assert artifact in result

    def test_build_review_prompt_includes_iteration(self, engine):
        result = engine._build_review_prompt(
            artifact_type="test_plan",
            context="Test coverage",
            artifact="Test plan v3",
            iteration=3,
        )
        assert "3" in result

    def test_build_address_prompt_returns_string(self, engine):
        result = engine._build_address_prompt(
            artifact_type="code",
            artifact="def foo(): pass",
            feedback="Add error handling",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_address_prompt_includes_feedback(self, engine):
        feedback = "Please add input validation for the email field."
        result = engine._build_address_prompt(
            artifact_type="design",
            artifact="Design doc v1",
            feedback=feedback,
        )
        assert feedback in result

    def test_build_address_prompt_includes_artifact(self, engine):
        artifact = "class UserService:\n    pass"
        result = engine._build_address_prompt(
            artifact_type="code",
            artifact=artifact,
            feedback="Implement the methods",
        )
        assert artifact in result
