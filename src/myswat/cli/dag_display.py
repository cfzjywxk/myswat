"""Status display for delivery-slice dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from myswat.workflow.dag import SliceDAG

_STATUS_ICONS: dict[str, str] = {
    "done": "\u2713",              # ✓
    "branch_complete": "\u2605",   # ★
    "dev_in_progress": "\u25c9",   # ◉
    "review": "\u25c9",            # ◉
    "enqueuing_dev": "\u25c9",     # ◉
    "enqueuing_review": "\u25c9",  # ◉
    "needs_revision": "\u25c9",    # ◉
    "pending": "\u25cb",           # ○
    "failed": "\u2717",            # ✗
    "hitl_deferred": "\u25cc",     # ◌
    "ready": "\u00b7",             # ·
}

_STATUS_COLORS: dict[str, str] = {
    "done": "green",
    "branch_complete": "bright_yellow",
    "dev_in_progress": "yellow",
    "review": "cyan",
    "enqueuing_dev": "yellow",
    "enqueuing_review": "cyan",
    "needs_revision": "yellow",
    "pending": "dim",
    "failed": "red",
    "hitl_deferred": "dim yellow",
    "ready": "white",
}

_STATUS_LABELS: dict[str, str] = {
    "done": "done",
    "branch_complete": "branch_complete",
    "dev_in_progress": "dev",
    "review": "review",
    "enqueuing_dev": "dev",
    "enqueuing_review": "review",
    "needs_revision": "revision",
    "pending": "blocked",
    "failed": "failed",
    "hitl_deferred": "deferred",
    "ready": "ready",
}


def _active_style(state_label: str) -> tuple[str, str, str]:
    if state_label == "review":
        return "\u25c9", "cyan", "active review"
    return "\u25c9", "yellow", "active dev"


def _multiline_slice_text(items: list[dict], *, empty: str = "none") -> Text:
    text = Text()
    if not items:
        text.append(empty, style="dim")
        return text

    for index, item in enumerate(items):
        if index:
            text.append("\n")
        title = str(item.get("title") or item.get("slice_id") or "-")
        text.append(f"- {title}", style="bold")
        state_label = str(item.get("state_label") or "").strip()
        if state_label:
            text.append(f" | state {state_label}", style="dim")
        owner_label = str(item.get("owner_label") or "").strip()
        if owner_label:
            text.append(f" | owner {owner_label}", style="dim")
        stage_name = str(item.get("stage_name") or "").strip()
        if stage_name:
            text.append(f" | {stage_name}", style="dim")
        if item.get("inferred"):
            text.append(" | inferred", style="dim italic")
    return text


def _build_summary_grid(
    *,
    execution_model: str,
    active_slices: list[dict],
    parallel_ready_slices: list[dict],
    ready_to_merge: list[str],
) -> Table:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(style="bold", width=16, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row("Execution", Text(execution_model, style="yellow" if execution_model == "serial" else "white"))
    if active_slices:
        grid.add_row("Active slices", _multiline_slice_text(active_slices))
    if parallel_ready_slices:
        grid.add_row("Parallel-ready", _multiline_slice_text(parallel_ready_slices))
    if ready_to_merge:
        merge_text = Text()
        for index, name in enumerate(ready_to_merge):
            if index:
                merge_text.append("\n")
            merge_text.append(f"- {name}", style="bright_yellow")
        grid.add_row("Ready to merge", merge_text)
    return grid


def _build_slice_label(
    dag: SliceDAG,
    slice_id: str,
    *,
    extra_deps: list[str] | None = None,
    active_by_slice_id: dict[str, dict] | None = None,
    parallel_ready_ids: set[str] | None = None,
) -> Text:
    """Build a Rich Text label for a single slice node."""
    s = dag.slices[slice_id]
    active = (active_by_slice_id or {}).get(slice_id)
    if active is not None:
        icon, color, label = _active_style(str(active.get("state_label") or "dev"))
    else:
        status = s.status.value
        icon = _STATUS_ICONS.get(status, "?")
        color = _STATUS_COLORS.get(status, "white")
        label = _STATUS_LABELS.get(status, status)

    text = Text()
    text.append(f"{icon} ", style=color)
    text.append(s.title, style=f"bold {color}")

    padding = max(1, 50 - len(s.title))
    text.append(" " * padding)
    text.append(label, style=color)

    if active is not None:
        owner_label = str(active.get("owner_label") or "").strip()
        if owner_label:
            text.append(f" | owner {owner_label}", style=f"bold {color}")
        stage_name = str(active.get("stage_name") or "").strip()
        if stage_name:
            text.append(f" | {stage_name}", style="dim")
        if active.get("inferred"):
            text.append(" | inferred", style="dim italic")
    elif parallel_ready_ids and slice_id in parallel_ready_ids and s.status.value in {"ready", "needs_revision"}:
        text.append(" | parallel-ready", style="bright_cyan")

    if s.workspace is not None:
        text.append(f" \u2190 {s.workspace.branch}", style="dim")

    if extra_deps:
        dep_titles = [dag.slices[d].title for d in extra_deps if d in dag.slices]
        if dep_titles:
            text.append(f" (also needs: {', '.join(dep_titles)})", style="dim italic")

    return text


def render_dag_status(
    dag: SliceDAG,
    work_item_id: int,
    *,
    execution_model: str = "serial",
    active_slices: list[dict] | None = None,
    active_by_slice_id: dict[str, dict] | None = None,
    parallel_ready_slices: list[dict] | None = None,
) -> Panel:
    """Render the delivery-slice dependency graph with honest runtime context."""
    active_slices = active_slices or []
    active_by_slice_id = active_by_slice_id or {}
    parallel_ready_slices = parallel_ready_slices or []
    parallel_ready_ids = {str(item.get("slice_id") or "") for item in parallel_ready_slices}

    total = len(dag.slices)
    done_count = sum(1 for s in dag.slices.values() if s.status.value == "done")

    header = f"Delivery Slices (work item #{work_item_id})"
    progress = f"{done_count} of {total} complete"

    tree = Tree(
        Text.assemble(
            (header, "bold"),
            ("  " * max(1, 40 - len(header)), ""),
            (progress, "dim"),
        ),
        guide_style="dim",
    )

    # Root slices have no blockers and anchor the dependency tree.
    roots = [s for s in dag._ordered_slices() if not s.blocked_by]
    placed: set[str] = set()

    def _add_subtree(parent_tree: Tree, slice_id: str) -> None:
        if slice_id in placed:
            return
        placed.add(slice_id)

        s = dag.slices[slice_id]
        parent_dep = None
        for dep_id in s.blocked_by:
            if dep_id in placed or dep_id == slice_id:
                parent_dep = dep_id
                break
        extra_deps = [d for d in s.blocked_by if d != parent_dep] if parent_dep else []

        label = _build_slice_label(
            dag,
            slice_id,
            extra_deps=extra_deps,
            active_by_slice_id=active_by_slice_id,
            parallel_ready_ids=parallel_ready_ids,
        )
        node = parent_tree.add(label)

        dependents = sorted(
            [dag.slices[d] for d in dag.adjacency.get(slice_id, set())],
            key=lambda x: x.plan_position,
        )
        for dep in dependents:
            _add_subtree(node, dep.id)

    # Build the visible tree from each root, then add any defensive leftovers.
    for root in roots:
        label = _build_slice_label(
            dag,
            root.id,
            active_by_slice_id=active_by_slice_id,
            parallel_ready_ids=parallel_ready_ids,
        )
        node = tree.add(label)
        placed.add(root.id)
        dependents = sorted(
            [dag.slices[d] for d in dag.adjacency.get(root.id, set())],
            key=lambda x: x.plan_position,
        )
        for dep in dependents:
            _add_subtree(node, dep.id)

    for s in dag._ordered_slices():
        if s.id not in placed:
            label = _build_slice_label(
                dag,
                s.id,
                active_by_slice_id=active_by_slice_id,
                parallel_ready_ids=parallel_ready_ids,
            )
            tree.add(label)
            placed.add(s.id)

    ready_to_merge = []
    for s in dag._ordered_slices():
        if s.status.value != "branch_complete":
            continue
        branch_info = f" ({s.workspace.branch})" if s.workspace else ""
        ready_to_merge.append(f"{s.title}{branch_info}")

    # The summary grid makes the current serial-vs-parallel reality explicit.
    summary = _build_summary_grid(
        execution_model=execution_model,
        active_slices=active_slices,
        parallel_ready_slices=parallel_ready_slices,
        ready_to_merge=ready_to_merge,
    )
    panel_title = (
        "[bold]Serial Slice Dependencies[/bold]"
        if execution_model == "serial"
        else "[bold]Slice Dependencies[/bold]"
    )

    return Panel(
        Group(summary, Text(""), tree),
        title=panel_title,
        border_style="blue",
        padding=(1, 2),
    )
