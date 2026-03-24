"""SliceDAG — delivery slice dependency graph with state machine."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from myswat.memory.store import MemoryStore


class SliceStatus(StrEnum):
    pending = "pending"
    ready = "ready"
    needs_revision = "needs_revision"
    enqueuing_dev = "enqueuing_dev"
    dev_in_progress = "dev_in_progress"
    enqueuing_review = "enqueuing_review"
    review = "review"
    branch_complete = "branch_complete"
    done = "done"
    failed = "failed"
    hitl_deferred = "hitl_deferred"


_TERMINAL = frozenset({SliceStatus.done, SliceStatus.failed, SliceStatus.branch_complete})
_DEP_SATISFIED = frozenset({SliceStatus.done, SliceStatus.branch_complete})
_DISPATCHABLE = frozenset({SliceStatus.ready, SliceStatus.needs_revision})


@dataclass(frozen=True)
class WorkspaceRef:
    """Immutable reference to a slice's working directory."""

    branch: str  # e.g. "myswat/slice/42/a1b2c3d4e5f6"
    path: str  # filesystem path to worktree root


@dataclass
class DeliverySlice:
    """Mutable delivery slice with full lifecycle state."""

    id: str
    title: str
    description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    execution_mode: str = "AFK"  # AFK | HITL
    blocked_by: list[str] = field(default_factory=list)  # list of slice IDs
    user_stories: list[str] = field(default_factory=list)
    parallelization_notes: str = ""
    status: SliceStatus = SliceStatus.pending
    workspace: WorkspaceRef | None = None
    stage_run_id: int | None = None
    review_cycle_id: int | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)
    plan_position: int = 0  # order in original plan, for stable tie-breaking


def generate_slice_id(title: str, work_item_id: int) -> str:
    """Content-hash slice ID: sha256(title + work_item_id)[:12]."""
    raw = f"{title}:{work_item_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class SliceDAGError(Exception):
    """Raised for DAG construction errors (cycles, missing refs, etc.)."""


@dataclass
class SliceDAG:
    """Delivery slice dependency graph with state machine transitions.

    All mark_* methods update both in-memory state and persist to the
    delivery_slice_states table via the store.
    """

    slices: dict[str, DeliverySlice]  # id -> slice
    adjacency: dict[str, set[str]]  # id -> set of dependents (forward edges)
    reverse: dict[str, set[str]]  # id -> set of dependencies (backward edges)
    _store: MemoryStore | None = field(default=None, repr=False)
    _work_item_id: int | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_slices(cls, slices: list[DeliverySlice]) -> SliceDAG:
        """Build a DAG from freshly parsed slices (first run only).

        Validates: no cycles, all blocked_by references resolve, unique titles.
        """
        # Validate unique titles
        titles = [s.title for s in slices]
        seen_titles: set[str] = set()
        for t in titles:
            if t in seen_titles:
                raise SliceDAGError(f"Duplicate slice title: {t!r}")
            seen_titles.add(t)

        slice_map: dict[str, DeliverySlice] = {}
        for s in slices:
            slice_map[s.id] = s

        # Build adjacency and reverse edges
        adjacency: dict[str, set[str]] = defaultdict(set)
        reverse: dict[str, set[str]] = defaultdict(set)
        for s in slices:
            if s.id not in adjacency:
                adjacency[s.id] = set()
            if s.id not in reverse:
                reverse[s.id] = set()
            for dep_id in s.blocked_by:
                if dep_id not in slice_map:
                    raise SliceDAGError(
                        f"Slice {s.title!r} blocked_by unknown slice ID {dep_id!r}"
                    )
                adjacency[dep_id].add(s.id)
                reverse[s.id].add(dep_id)

        dag = cls(
            slices=slice_map,
            adjacency=dict(adjacency),
            reverse=dict(reverse),
        )

        # Validate no cycles
        dag._detect_cycles()

        # Set initial statuses: slices with no deps start as ready
        for s in slices:
            if not s.blocked_by:
                s.status = SliceStatus.ready

        return dag

    @classmethod
    def from_store(
        cls, store: MemoryStore, work_item_id: int
    ) -> SliceDAG:
        """Reconstruct DAG from persisted delivery_slice_states rows (resume path).

        Populates each DeliverySlice with: status, workspace, stage_run_id,
        review_cycle_id, metadata_json.
        """
        rows = store.get_slice_states(work_item_id)
        if not rows:
            raise SliceDAGError(
                f"No persisted slice states for work_item_id={work_item_id}"
            )

        slices: list[DeliverySlice] = []
        for idx, row in enumerate(rows):
            meta = row.get("metadata_json")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            elif meta is None:
                meta = {}

            workspace = None
            if row.get("workspace_branch") and row.get("workspace_path"):
                workspace = WorkspaceRef(
                    branch=row["workspace_branch"],
                    path=row["workspace_path"],
                )

            blocked_by = meta.get("blocked_by", [])
            if isinstance(blocked_by, str):
                blocked_by = [b.strip() for b in blocked_by.split(",") if b.strip()]

            s = DeliverySlice(
                id=row["slice_id"],
                title=row["title"],
                description=meta.get("description", ""),
                acceptance_criteria=meta.get("acceptance_criteria", []),
                execution_mode=meta.get("execution_mode", "AFK"),
                blocked_by=blocked_by,
                user_stories=meta.get("user_stories", []),
                parallelization_notes=meta.get("parallelization_notes", ""),
                status=SliceStatus(row["status"]),
                workspace=workspace,
                stage_run_id=row.get("stage_run_id"),
                review_cycle_id=row.get("review_cycle_id"),
                metadata_json=meta,
                plan_position=idx,
            )
            slices.append(s)

        # Build edges manually (skip from_slices validation — already validated on first run)
        slice_map: dict[str, DeliverySlice] = {s.id: s for s in slices}
        adjacency: dict[str, set[str]] = defaultdict(set)
        reverse: dict[str, set[str]] = defaultdict(set)
        for s in slices:
            if s.id not in adjacency:
                adjacency[s.id] = set()
            if s.id not in reverse:
                reverse[s.id] = set()
            for dep_id in s.blocked_by:
                if dep_id not in slice_map:
                    raise SliceDAGError(
                        f"Slice {s.title!r} references dependency {dep_id!r} "
                        f"which is missing from persisted rows — "
                        f"partial persist_initial? Re-persist required."
                    )
                adjacency[dep_id].add(s.id)
                reverse[s.id].add(dep_id)

        dag = cls(
            slices=slice_map,
            adjacency=dict(adjacency),
            reverse=dict(reverse),
            _store=store,
            _work_item_id=work_item_id,
        )
        return dag

    def persist_initial(self, store: MemoryStore, work_item_id: int) -> None:
        """Persist all slices to delivery_slice_states on first run."""
        self._store = store
        self._work_item_id = work_item_id
        for s in self._ordered_slices():
            meta = self._build_metadata(s)
            store.upsert_slice_state(
                work_item_id=work_item_id,
                slice_id=s.id,
                title=s.title,
                status=s.status.value,
                metadata_json=json.dumps(meta),
            )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def ready_slices(self) -> list[DeliverySlice]:
        """Return slices whose deps are all done/branch_complete and own status is ready."""
        result = []
        for s in self._ordered_slices():
            if s.status != SliceStatus.ready:
                continue
            if self._deps_satisfied(s):
                result.append(s)
        return result

    def dispatchable_slices(self) -> list[DeliverySlice]:
        """Return slices in any dispatchable state whose deps are satisfied.

        Ordered: needs_revision first (fastest re-dispatch), then ready.
        Within each group, ordered by plan position.
        """
        revision = []
        ready = []
        for s in self._ordered_slices():
            if s.status not in _DISPATCHABLE:
                continue
            if not self._deps_satisfied(s):
                continue
            if s.status == SliceStatus.needs_revision:
                revision.append(s)
            else:
                ready.append(s)
        return revision + ready

    def all_terminal(self) -> bool:
        """True when every slice is done, failed, or branch_complete."""
        return all(s.status in _TERMINAL for s in self.slices.values())

    def topological_order(self) -> list[DeliverySlice]:
        """Return all slices in dependency-respecting order (Kahn's algorithm).

        Ties broken by plan position.
        """
        in_degree: dict[str, int] = {sid: 0 for sid in self.slices}
        for sid, deps in self.reverse.items():
            in_degree[sid] = len(deps)

        queue: list[DeliverySlice] = sorted(
            [s for s in self.slices.values() if in_degree[s.id] == 0],
            key=lambda s: s.plan_position,
        )
        result: list[DeliverySlice] = []
        q = deque(queue)

        while q:
            s = q.popleft()
            result.append(s)
            dependents = sorted(
                [self.slices[d] for d in self.adjacency.get(s.id, set())],
                key=lambda s: s.plan_position,
            )
            for dep in dependents:
                in_degree[dep.id] -= 1
                if in_degree[dep.id] == 0:
                    q.append(dep)

        if len(result) != len(self.slices):
            raise SliceDAGError("Cycle detected in DAG during topological sort")

        return result

    def slices_in_status(self, status: SliceStatus) -> list[DeliverySlice]:
        """Return all slices with the given status, ordered by plan position."""
        return sorted(
            [s for s in self.slices.values() if s.status == status],
            key=lambda s: s.plan_position,
        )

    def can_make_progress(self) -> bool:
        """True if there exist non-terminal slices that could eventually become ready.

        hitl_deferred counts as 'no progress' since user already declined.
        base-ref conflicts are handled by the caller before this is checked.
        """
        _NO_PROGRESS = frozenset({SliceStatus.hitl_deferred})
        for s in self.slices.values():
            if s.status in _TERMINAL:
                continue
            if s.status in _NO_PROGRESS:
                continue
            # This slice is non-terminal and not deferred — progress is possible
            # (it's either in-flight, pending with potential deps, or dispatchable)
            return True
        return False

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_ready(self, slice_id: str) -> None:
        """Reset to ready. Does NOT clear metadata_json."""
        s = self.slices[slice_id]
        s.status = SliceStatus.ready
        self._persist_status(slice_id, SliceStatus.ready)

    def mark_ready_for_serial(self, slice_id: str) -> None:
        """Reset to ready AND clear stale metadata (revision_feedback)."""
        s = self.slices[slice_id]
        s.status = SliceStatus.ready
        s.metadata_json.pop("revision_feedback", None)
        self._persist_status(
            slice_id, SliceStatus.ready,
            metadata_json=json.dumps(self._build_metadata(s)),
        )

    def mark_needs_revision(self, slice_id: str, *, feedback: dict) -> None:
        """Review returned changes_requested. Persist feedback to metadata_json."""
        s = self.slices[slice_id]
        s.status = SliceStatus.needs_revision
        s.metadata_json["revision_feedback"] = feedback
        self._persist_status(
            slice_id, SliceStatus.needs_revision,
            metadata_json=json.dumps(self._build_metadata(s)),
        )

    def mark_enqueuing_dev(self, slice_id: str, *, workspace: WorkspaceRef) -> None:
        """Crash-safe phase 1 for dev: workspace set, about to queue dev assignment."""
        s = self.slices[slice_id]
        s.status = SliceStatus.enqueuing_dev
        s.workspace = workspace
        self._persist_status(
            slice_id, SliceStatus.enqueuing_dev,
            workspace_branch=workspace.branch,
            workspace_path=workspace.path,
        )

    def mark_enqueuing_review(self, slice_id: str) -> None:
        """Crash-safe phase 1 for review: dev done, about to queue review."""
        s = self.slices[slice_id]
        s.status = SliceStatus.enqueuing_review
        self._persist_status(slice_id, SliceStatus.enqueuing_review)

    def mark_dev_in_progress(self, slice_id: str, *, stage_run_id: int) -> None:
        """Crash-safe phase 2 for dev: assignment queued, ID recorded."""
        s = self.slices[slice_id]
        s.status = SliceStatus.dev_in_progress
        s.stage_run_id = stage_run_id
        self._persist_status(
            slice_id, SliceStatus.dev_in_progress,
            stage_run_id=stage_run_id,
        )

    def mark_review(self, slice_id: str, *, review_cycle_id: int) -> None:
        """Crash-safe phase 2 for review: review queued, ID recorded."""
        s = self.slices[slice_id]
        s.status = SliceStatus.review
        s.review_cycle_id = review_cycle_id
        self._persist_status(
            slice_id, SliceStatus.review,
            review_cycle_id=review_cycle_id,
        )

    def mark_branch_complete(self, slice_id: str) -> list[DeliverySlice]:
        """QA approved (LGTM). Branch ready for human merge.

        Returns newly-unblocked slices (pending -> ready).
        """
        s = self.slices[slice_id]
        s.status = SliceStatus.branch_complete
        self._persist_status(slice_id, SliceStatus.branch_complete)
        return self._unblock_dependents(slice_id)

    def mark_done(self, slice_id: str) -> list[DeliverySlice]:
        """Mark merged (human confirmed via myswat merge-done).

        Returns newly-unblocked slices (pending -> ready).
        """
        s = self.slices[slice_id]
        s.status = SliceStatus.done
        self._persist_status(slice_id, SliceStatus.done)
        return self._unblock_dependents(slice_id)

    def mark_failed(self, slice_id: str) -> list[DeliverySlice]:
        """Mark failed, cascade failure to all transitive dependents."""
        s = self.slices[slice_id]
        s.status = SliceStatus.failed
        self._persist_status(slice_id, SliceStatus.failed)
        cascaded = self._cascade_failure(slice_id)
        return cascaded

    def mark_hitl_deferred(self, slice_id: str) -> None:
        """User declined HITL slice."""
        s = self.slices[slice_id]
        s.status = SliceStatus.hitl_deferred
        self._persist_status(slice_id, SliceStatus.hitl_deferred)

    def reactivate_hitl_deferred(self) -> list[DeliverySlice]:
        """Reset all hitl_deferred slices back to ready."""
        reactivated = []
        for s in self.slices_in_status(SliceStatus.hitl_deferred):
            s.status = SliceStatus.ready
            self._persist_status(s.id, SliceStatus.ready)
            reactivated.append(s)
        return reactivated

    def clear_workspace(self, slice_id: str) -> None:
        """Null out workspace fields in memory and DB."""
        s = self.slices[slice_id]
        s.workspace = None
        if self._store and self._work_item_id is not None:
            self._store.update_slice_state(
                self._work_item_id, slice_id,
                workspace_branch=None,
                workspace_path=None,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ordered_slices(self) -> list[DeliverySlice]:
        """Return slices ordered by plan position."""
        return sorted(self.slices.values(), key=lambda s: s.plan_position)

    def _deps_satisfied(self, s: DeliverySlice) -> bool:
        """Check if all dependencies are done or branch_complete."""
        return all(
            self.slices[dep_id].status in _DEP_SATISFIED
            for dep_id in s.blocked_by
            if dep_id in self.slices
        )

    def _unblock_dependents(self, slice_id: str) -> list[DeliverySlice]:
        """Check dependents — if all their deps are satisfied, transition pending->ready."""
        unblocked = []
        for dep_id in self.adjacency.get(slice_id, set()):
            dep = self.slices[dep_id]
            if dep.status != SliceStatus.pending:
                continue
            if self._deps_satisfied(dep):
                dep.status = SliceStatus.ready
                self._persist_status(dep.id, SliceStatus.ready)
                unblocked.append(dep)
        return unblocked

    def _cascade_failure(self, slice_id: str) -> list[DeliverySlice]:
        """Cascade failure to all transitive dependents."""
        cascaded = []
        queue = deque(self.adjacency.get(slice_id, set()))
        visited: set[str] = set()
        while queue:
            dep_id = queue.popleft()
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep = self.slices[dep_id]
            if dep.status in {SliceStatus.done, SliceStatus.failed}:
                continue
            dep.status = SliceStatus.failed
            self._persist_status(dep_id, SliceStatus.failed)
            cascaded.append(dep)
            queue.extend(self.adjacency.get(dep_id, set()))
        return cascaded

    def _detect_cycles(self) -> None:
        """Detect cycles using topological sort (Kahn's algorithm)."""
        in_degree: dict[str, int] = {sid: len(deps) for sid, deps in self.reverse.items()}
        for sid in self.slices:
            if sid not in in_degree:
                in_degree[sid] = 0
        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        count = 0
        while queue:
            sid = queue.popleft()
            count += 1
            for dep_id in self.adjacency.get(sid, set()):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)
        if count != len(self.slices):
            raise SliceDAGError("Cycle detected in delivery slice dependencies")

    def _persist_status(self, slice_id: str, status: SliceStatus, **kwargs: Any) -> None:
        """Persist slice state to the store if available."""
        if self._store is None or self._work_item_id is None:
            return
        self._store.update_slice_state(
            self._work_item_id, slice_id,
            status=status.value,
            **kwargs,
        )

    @staticmethod
    def _build_metadata(s: DeliverySlice) -> dict[str, Any]:
        """Build the metadata_json dict for persistence."""
        meta = dict(s.metadata_json)
        # Always store structural fields in metadata
        meta["description"] = s.description
        meta["acceptance_criteria"] = s.acceptance_criteria
        meta["execution_mode"] = s.execution_mode
        meta["blocked_by"] = s.blocked_by
        meta["user_stories"] = s.user_stories
        meta["parallelization_notes"] = s.parallelization_notes
        return meta
