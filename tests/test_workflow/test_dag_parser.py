"""Tests for enriched delivery slice parsing."""

import pytest

from myswat.workflow.dag import DeliverySlice, SliceDAGError, generate_slice_id
from myswat.workflow.kernel import WorkflowKernel


# We can't easily instantiate WorkflowKernel with all its deps,
# but _parse_dag_delivery_slices only uses self._extract_markdown_section.
# So we test _extract_markdown_section + the parser logic together
# by creating a minimal stub.

class _StubKernel:
    """Minimal stub that provides the methods the parser needs."""
    _extract_markdown_section = staticmethod(WorkflowKernel._extract_markdown_section)

    def _parse_dag_delivery_slices(self, plan, work_item_id):
        return WorkflowKernel._parse_dag_delivery_slices(self, plan, work_item_id)


SAMPLE_PLAN = """\
# Implementation Plan

## Delivery Slices

### Slice 1: SQL Parser Extensions [AFK]
Parse new SQL syntax for window functions and CTEs.

Done when:
- Window function parsing produces correct AST nodes
- CTE parsing handles recursive and non-recursive forms
- Error messages include line/column info

### Slice 2: Planner Tree Support
Extend the query planner to handle window functions.

Blocked by: SQL Parser Extensions

Acceptance criteria:
- Planner generates WindowAgg nodes for window functions
- Cost estimation accounts for window partitioning

Parallel notes: Can run after parser is branch_complete

### Slice 3: SQL Executor [HITL]
Implement execution of window function plans.

Blocked by: Planner Tree Support, SQL Parser Extensions

Done when:
- Window functions execute correctly on partitioned data
- Performance meets baseline benchmarks

## Next Steps
"""


class TestEnrichedParser:
    def test_parse_basic_slices(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        assert len(slices) == 3
        assert slices[0].title == "SQL Parser Extensions"
        assert slices[1].title == "Planner Tree Support"
        assert slices[2].title == "SQL Executor"

    def test_slice_ids_are_content_hashes(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        expected_id = generate_slice_id("SQL Parser Extensions", 42)
        assert slices[0].id == expected_id
        assert len(slices[0].id) == 12

    def test_execution_modes(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        assert slices[0].execution_mode == "AFK"
        assert slices[1].execution_mode == "AFK"  # default
        assert slices[2].execution_mode == "HITL"

    def test_blocked_by_resolved(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        # Slice 1 has no deps
        assert slices[0].blocked_by == []

        # Slice 2 blocked by "SQL Parser Extensions"
        parser_id = generate_slice_id("SQL Parser Extensions", 42)
        assert slices[1].blocked_by == [parser_id]

        # Slice 3 blocked by both
        planner_id = generate_slice_id("Planner Tree Support", 42)
        assert set(slices[2].blocked_by) == {planner_id, parser_id}

    def test_acceptance_criteria_extracted(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        # Slice 1 uses "Done when:"
        assert len(slices[0].acceptance_criteria) == 3
        assert "Window function parsing" in slices[0].acceptance_criteria[0]

        # Slice 2 uses "Acceptance criteria:"
        assert len(slices[1].acceptance_criteria) == 2

    def test_description_captured(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        assert "Parse new SQL syntax" in slices[0].description

    def test_plan_positions(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        assert slices[0].plan_position == 0
        assert slices[1].plan_position == 1
        assert slices[2].plan_position == 2

    def test_parallelization_notes(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        assert "branch_complete" in slices[1].parallelization_notes

    def test_empty_plan(self):
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices("no delivery slices here", work_item_id=42)
        assert slices == []

    def test_duplicate_titles_raises(self):
        dup_plan = """\
## Delivery Slices

### Slice 1: Parser
First parser slice.

### Slice 2: Parser
Duplicate title.
"""
        kernel = _StubKernel()
        with pytest.raises(SliceDAGError, match="Duplicate"):
            kernel._parse_dag_delivery_slices(dup_plan, work_item_id=1)

    def test_blocked_by_slice_number(self):
        """Test 'Blocked by: Slice 1' syntax."""
        plan = """\
## Delivery Slices

### Slice 1: Parser
Parse things.

### Slice 2: Executor
Execute things.

Blocked by: Slice 1
"""
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(plan, work_item_id=1)

        parser_id = generate_slice_id("Parser", 1)
        assert slices[1].blocked_by == [parser_id]

    def test_id_stability_across_parses(self):
        """Same plan, same work_item_id → same IDs."""
        kernel = _StubKernel()
        slices1 = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)
        slices2 = kernel._parse_dag_delivery_slices(SAMPLE_PLAN, work_item_id=42)

        for s1, s2 in zip(slices1, slices2):
            assert s1.id == s2.id

    def test_unresolved_blocker_raises(self):
        """Unresolved Blocked by: reference must raise, not silently drop."""
        plan = """\
## Delivery Slices

### Slice 1: Parser
Parse things.

### Slice 2: Executor
Execute things.

Blocked by: NonExistent Slice
"""
        kernel = _StubKernel()
        with pytest.raises(SliceDAGError, match="Unresolved blocked_by"):
            kernel._parse_dag_delivery_slices(plan, work_item_id=1)

    def test_ambiguous_substring_blocker_raises(self):
        """Substring match that hits multiple titles must raise, not pick first."""
        plan = """\
## Delivery Slices

### Slice 1: SQL Parser Core
Core parser logic.

### Slice 2: SQL Parser Extensions
Extended parser logic.

### Slice 3: Executor
Execute things.

Blocked by: SQL Parser
"""
        kernel = _StubKernel()
        with pytest.raises(SliceDAGError, match="Ambiguous blocked_by"):
            kernel._parse_dag_delivery_slices(plan, work_item_id=1)

    def test_unique_substring_blocker_resolves(self):
        """Substring match that hits exactly one title resolves correctly."""
        plan = """\
## Delivery Slices

### Slice 1: SQL Parser Core
Core parser logic.

### Slice 2: Query Optimizer
Optimization logic.

### Slice 3: Executor
Execute things.

Blocked by: Optimizer
"""
        kernel = _StubKernel()
        slices = kernel._parse_dag_delivery_slices(plan, work_item_id=1)

        opt_id = generate_slice_id("Query Optimizer", 1)
        assert slices[2].blocked_by == [opt_id]
