"""Tests for SliceDAG — delivery slice dependency graph with state machine."""

import json
import pytest

from myswat.workflow.dag import (
    DeliverySlice,
    SliceDAG,
    SliceDAGError,
    SliceStatus,
    WorkspaceRef,
    generate_slice_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice(
    title: str,
    *,
    sid: str | None = None,
    blocked_by: list[str] | None = None,
    execution_mode: str = "AFK",
    position: int = 0,
    status: SliceStatus = SliceStatus.pending,
) -> DeliverySlice:
    """Shortcut to build a test slice."""
    return DeliverySlice(
        id=sid or generate_slice_id(title, 1),
        title=title,
        blocked_by=blocked_by or [],
        execution_mode=execution_mode,
        plan_position=position,
        status=status,
    )


def _dag(*slices: DeliverySlice) -> SliceDAG:
    """Build a DAG from slices, bypassing persistence."""
    return SliceDAG.from_slices(list(slices))


# ---------------------------------------------------------------------------
# Construction & Validation
# ---------------------------------------------------------------------------

class TestSliceDAGConstruction:
    def test_linear_chain(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[b.id], position=2)
        dag = _dag(a, b, c)

        assert len(dag.slices) == 3
        assert dag.slices[a.id].status == SliceStatus.ready
        assert dag.slices[b.id].status == SliceStatus.pending
        assert dag.slices[c.id].status == SliceStatus.pending

    def test_fan_out(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[a.id], position=2)
        dag = _dag(a, b, c)

        assert dag.slices[a.id].status == SliceStatus.ready
        assert dag.slices[b.id].status == SliceStatus.pending
        assert dag.slices[c.id].status == SliceStatus.pending

    def test_diamond(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[a.id], position=2)
        d = _slice("D", blocked_by=[b.id, c.id], position=3)
        dag = _dag(a, b, c, d)

        assert dag.slices[a.id].status == SliceStatus.ready
        assert dag.slices[d.id].status == SliceStatus.pending

    def test_independent_roots(self):
        a = _slice("A", position=0)
        b = _slice("B", position=1)
        dag = _dag(a, b)

        assert dag.slices[a.id].status == SliceStatus.ready
        assert dag.slices[b.id].status == SliceStatus.ready

    def test_cycle_detection(self):
        a_id = generate_slice_id("A", 1)
        b_id = generate_slice_id("B", 1)
        a = _slice("A", sid=a_id, blocked_by=[b_id], position=0)
        b = _slice("B", sid=b_id, blocked_by=[a_id], position=1)

        with pytest.raises(SliceDAGError, match="Cycle"):
            _dag(a, b)

    def test_duplicate_titles(self):
        a = _slice("Same", sid="aaa", position=0)
        b = _slice("Same", sid="bbb", position=1)

        with pytest.raises(SliceDAGError, match="Duplicate"):
            _dag(a, b)

    def test_missing_blocked_by_ref(self):
        a = _slice("A", blocked_by=["nonexistent"], position=0)

        with pytest.raises(SliceDAGError, match="unknown"):
            _dag(a)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestSliceDAGQueries:
    def test_ready_slices(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)

        ready = dag.ready_slices()
        assert [s.title for s in ready] == ["A"]

    def test_dispatchable_slices_ordering(self):
        """needs_revision before ready, plan position within groups."""
        a = _slice("A", position=0)
        b = _slice("B", position=1)
        dag = _dag(a, b)
        # B is needs_revision
        dag.slices[b.id].status = SliceStatus.needs_revision

        result = dag.dispatchable_slices()
        assert [s.title for s in result] == ["B", "A"]

    def test_all_terminal_false(self):
        a = _slice("A", position=0)
        dag = _dag(a)
        assert not dag.all_terminal()

    def test_all_terminal_true(self):
        a = _slice("A", position=0)
        b = _slice("B", position=1)
        dag = _dag(a, b)
        dag.slices[a.id].status = SliceStatus.done
        dag.slices[b.id].status = SliceStatus.failed
        assert dag.all_terminal()

    def test_all_terminal_with_branch_complete(self):
        a = _slice("A", position=0)
        dag = _dag(a)
        dag.slices[a.id].status = SliceStatus.branch_complete
        assert dag.all_terminal()

    def test_topological_order_simple(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[b.id], position=2)
        dag = _dag(a, b, c)

        order = dag.topological_order()
        assert [s.title for s in order] == ["A", "B", "C"]

    def test_topological_order_fan_out(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[a.id], position=2)
        dag = _dag(a, b, c)

        order = dag.topological_order()
        assert order[0].title == "A"
        # B and C both come after A, order by plan_position
        assert [s.title for s in order[1:]] == ["B", "C"]

    def test_slices_in_status(self):
        a = _slice("A", position=0)
        b = _slice("B", position=1)
        dag = _dag(a, b)
        dag.slices[b.id].status = SliceStatus.done

        done = dag.slices_in_status(SliceStatus.done)
        assert [s.title for s in done] == ["B"]

    def test_can_make_progress_with_pending(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)
        dag.slices[a.id].status = SliceStatus.dev_in_progress
        assert dag.can_make_progress()

    def test_can_make_progress_all_deferred(self):
        a = _slice("A", position=0)
        dag = _dag(a)
        dag.slices[a.id].status = SliceStatus.hitl_deferred
        assert not dag.can_make_progress()

    def test_can_make_progress_all_terminal(self):
        a = _slice("A", position=0)
        dag = _dag(a)
        dag.slices[a.id].status = SliceStatus.done
        assert not dag.can_make_progress()


# ---------------------------------------------------------------------------
# State Transitions
# ---------------------------------------------------------------------------

class TestSliceDAGTransitions:
    def test_mark_done_unblocks(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)

        unblocked = dag.mark_done(a.id)
        assert len(unblocked) == 1
        assert unblocked[0].title == "B"
        assert dag.slices[b.id].status == SliceStatus.ready

    def test_mark_branch_complete_unblocks(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)

        unblocked = dag.mark_branch_complete(a.id)
        assert len(unblocked) == 1
        assert unblocked[0].title == "B"
        assert dag.slices[b.id].status == SliceStatus.ready

    def test_mark_done_diamond_unblock(self):
        """D needs both B and C. Only unblocked when both are done/branch_complete."""
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[a.id], position=2)
        d = _slice("D", blocked_by=[b.id, c.id], position=3)
        dag = _dag(a, b, c, d)

        dag.mark_done(a.id)  # B and C unblocked
        assert dag.slices[b.id].status == SliceStatus.ready
        assert dag.slices[c.id].status == SliceStatus.ready
        assert dag.slices[d.id].status == SliceStatus.pending

        dag.mark_done(b.id)  # D still blocked (C not done)
        assert dag.slices[d.id].status == SliceStatus.pending

        dag.mark_done(c.id)  # D now unblocked
        assert dag.slices[d.id].status == SliceStatus.ready

    def test_mark_failed_cascades(self):
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[b.id], position=2)
        dag = _dag(a, b, c)

        cascaded = dag.mark_failed(a.id)
        assert dag.slices[a.id].status == SliceStatus.failed
        assert dag.slices[b.id].status == SliceStatus.failed
        assert dag.slices[c.id].status == SliceStatus.failed
        assert len(cascaded) == 2

    def test_mark_needs_revision(self):
        a = _slice("A", position=0)
        dag = _dag(a)

        feedback = {"issues": ["fix spacing"], "summary": "needs work"}
        dag.mark_needs_revision(a.id, feedback=feedback)
        assert dag.slices[a.id].status == SliceStatus.needs_revision
        assert dag.slices[a.id].metadata_json["revision_feedback"] == feedback

    def test_mark_ready_for_serial_clears_feedback(self):
        a = _slice("A", position=0)
        dag = _dag(a)

        dag.slices[a.id].metadata_json["revision_feedback"] = {"test": True}
        dag.mark_ready_for_serial(a.id)
        assert dag.slices[a.id].status == SliceStatus.ready
        assert "revision_feedback" not in dag.slices[a.id].metadata_json

    def test_mark_enqueuing_dev(self):
        a = _slice("A", position=0)
        dag = _dag(a)
        ws = WorkspaceRef(branch="myswat/slice/1/abc", path="/tmp/ws")

        dag.mark_enqueuing_dev(a.id, workspace=ws)
        assert dag.slices[a.id].status == SliceStatus.enqueuing_dev
        assert dag.slices[a.id].workspace == ws

    def test_mark_dev_in_progress(self):
        a = _slice("A", position=0)
        dag = _dag(a)

        dag.mark_dev_in_progress(a.id, stage_run_id=42)
        assert dag.slices[a.id].status == SliceStatus.dev_in_progress
        assert dag.slices[a.id].stage_run_id == 42

    def test_mark_review(self):
        a = _slice("A", position=0)
        dag = _dag(a)

        dag.mark_review(a.id, review_cycle_id=99)
        assert dag.slices[a.id].status == SliceStatus.review
        assert dag.slices[a.id].review_cycle_id == 99

    def test_hitl_deferred_and_reactivation(self):
        a = _slice("A", execution_mode="HITL", position=0)
        dag = _dag(a)

        dag.mark_hitl_deferred(a.id)
        assert dag.slices[a.id].status == SliceStatus.hitl_deferred

        reactivated = dag.reactivate_hitl_deferred()
        assert len(reactivated) == 1
        assert dag.slices[a.id].status == SliceStatus.ready

    def test_clear_workspace(self):
        a = _slice("A", position=0)
        dag = _dag(a)
        ws = WorkspaceRef(branch="test", path="/tmp")
        dag.slices[a.id].workspace = ws

        dag.clear_workspace(a.id)
        assert dag.slices[a.id].workspace is None


# ---------------------------------------------------------------------------
# Slice ID
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Persist → Resume Round-Trip
# ---------------------------------------------------------------------------

class _MockStore:
    """In-memory store that mimics upsert/update/get for delivery_slice_states."""

    def __init__(self):
        self._rows: dict[tuple[int, str], dict] = {}  # (work_item_id, slice_id) -> row

    def upsert_slice_state(self, work_item_id, slice_id, title, status, *, metadata_json=None):
        key = (work_item_id, slice_id)
        self._rows[key] = {
            "work_item_id": work_item_id,
            "slice_id": slice_id,
            "title": title,
            "status": status,
            "workspace_branch": None,
            "workspace_path": None,
            "stage_run_id": None,
            "review_cycle_id": None,
            "metadata_json": metadata_json,
        }

    def update_slice_state(self, work_item_id, slice_id, *, status=None, **kwargs):
        key = (work_item_id, slice_id)
        if key not in self._rows:
            return
        if status is not None:
            self._rows[key]["status"] = status
        for k, v in kwargs.items():
            if k in self._rows[key]:
                self._rows[key][k] = v

    def get_slice_states(self, work_item_id):
        results = []
        for (wid, _), row in self._rows.items():
            if wid == work_item_id:
                # Parse metadata_json string to dict (same as real store)
                d = dict(row)
                meta = d.get("metadata_json")
                if isinstance(meta, str):
                    import json
                    try:
                        d["metadata_json"] = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        d["metadata_json"] = {}
                elif meta is None:
                    d["metadata_json"] = {}
                results.append(d)
        return results


class TestPersistResumeRoundTrip:
    def test_blocked_by_survives_persist_and_resume(self):
        """Critical regression test: blocked_by must survive persist → transition → from_store."""
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)

        store = _MockStore()
        dag.persist_initial(store, work_item_id=1)

        # Simulate a transition that rewrites metadata_json
        dag.mark_needs_revision(a.id, feedback={"issues": ["fix bug"]})

        # Resume from store
        dag2 = SliceDAG.from_store(store, work_item_id=1)

        # Critical check: B's blocked_by must still reference A
        assert dag2.slices[b.id].blocked_by == [a.id]
        assert dag2.slices[a.id].status == SliceStatus.needs_revision

    def test_blocked_by_survives_mark_ready_for_serial(self):
        """mark_ready_for_serial also rewrites metadata — structural fields must persist."""
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)

        store = _MockStore()
        dag.persist_initial(store, work_item_id=1)

        # Add revision feedback, then clear it via mark_ready_for_serial
        dag.slices[a.id].metadata_json["revision_feedback"] = {"test": True}
        dag.mark_ready_for_serial(a.id)

        # Resume
        dag2 = SliceDAG.from_store(store, work_item_id=1)

        # B's blocked_by should still reference A
        assert dag2.slices[b.id].blocked_by == [a.id]
        # revision_feedback should be cleared
        assert "revision_feedback" not in dag2.slices[a.id].metadata_json

    def test_acceptance_criteria_survives_roundtrip(self):
        a = DeliverySlice(
            id=generate_slice_id("A", 1),
            title="A",
            acceptance_criteria=["Tests pass", "No regressions"],
            plan_position=0,
        )
        dag = _dag(a)

        store = _MockStore()
        dag.persist_initial(store, work_item_id=1)

        dag2 = SliceDAG.from_store(store, work_item_id=1)
        assert dag2.slices[a.id].acceptance_criteria == ["Tests pass", "No regressions"]

    def test_full_lifecycle_roundtrip(self):
        """Simulate: persist → mark_done(A) → resume → B should be ready."""
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        dag = _dag(a, b)

        store = _MockStore()
        dag.persist_initial(store, work_item_id=1)

        dag.mark_done(a.id)
        assert dag.slices[b.id].status == SliceStatus.ready

        # Resume — B should still be ready
        dag2 = SliceDAG.from_store(store, work_item_id=1)
        assert dag2.slices[a.id].status == SliceStatus.done
        assert dag2.slices[b.id].status == SliceStatus.ready

    def test_dispatchable_after_resume_with_needs_revision(self):
        """After mark_needs_revision → resume, the slice should be dispatchable."""
        a = _slice("A", position=0)
        b = _slice("B", position=1)
        dag = _dag(a, b)

        store = _MockStore()
        dag.persist_initial(store, work_item_id=1)

        dag.mark_needs_revision(a.id, feedback={"issues": ["fix it"]})

        dag2 = SliceDAG.from_store(store, work_item_id=1)
        dispatchable = dag2.dispatchable_slices()
        titles = [s.title for s in dispatchable]
        assert "A" in titles
        assert dag2.slices[a.id].metadata_json.get("revision_feedback") == {"issues": ["fix it"]}


    def test_partial_persist_raises_on_missing_dep(self):
        """from_store with partial rows (missing dep) must raise, not silently drop."""
        a = _slice("A", position=0)
        b = _slice("B", blocked_by=[a.id], position=1)
        c = _slice("C", blocked_by=[b.id], position=2)
        dag = _dag(a, b, c)

        store = _MockStore()
        dag.persist_initial(store, work_item_id=1)

        # Simulate partial persistence: delete B's row
        key_b = (1, b.id)
        del store._rows[key_b]

        # Resume should raise because C references B which is missing
        with pytest.raises(SliceDAGError, match="missing from persisted rows"):
            SliceDAG.from_store(store, work_item_id=1)


class TestSliceID:
    def test_deterministic(self):
        id1 = generate_slice_id("SQL Parser", 42)
        id2 = generate_slice_id("SQL Parser", 42)
        assert id1 == id2

    def test_different_titles(self):
        id1 = generate_slice_id("SQL Parser", 42)
        id2 = generate_slice_id("SQL Executor", 42)
        assert id1 != id2

    def test_different_work_items(self):
        id1 = generate_slice_id("SQL Parser", 42)
        id2 = generate_slice_id("SQL Parser", 43)
        assert id1 != id2

    def test_length(self):
        sid = generate_slice_id("test", 1)
        assert len(sid) == 12
