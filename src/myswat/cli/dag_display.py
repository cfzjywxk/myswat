"""DAG status display for delivery slices using Rich Tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
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


def _build_slice_label(
    dag: SliceDAG,
    slice_id: str,
    *,
    extra_deps: list[str] | None = None,
) -> Text:
    """Build a Rich Text label for a single slice node."""
    s = dag.slices[slice_id]
    status = s.status.value
    icon = _STATUS_ICONS.get(status, "?")
    color = _STATUS_COLORS.get(status, "white")
    label = _STATUS_LABELS.get(status, status)

    text = Text()
    text.append(f"{icon} ", style=color)
    text.append(s.title, style=f"bold {color}")

    # Right-align status label
    padding = max(1, 50 - len(s.title))
    text.append(" " * padding)
    text.append(label, style=color)

    # Show branch name for slices with active workspaces
    if s.workspace is not None:
        text.append(f" \u2190 {s.workspace.branch}", style="dim")

    # Cross-reference note for multi-parent slices
    if extra_deps:
        dep_titles = [dag.slices[d].title for d in extra_deps if d in dag.slices]
        if dep_titles:
            text.append(f" (also needs: {', '.join(dep_titles)})", style="dim italic")

    return text


def render_dag_status(dag: SliceDAG, work_item_id: int) -> Panel:
    """Render the slice DAG as an ASCII tree with status icons and branch names.

    Slices with no dependencies are tree roots.
    Slices with dependencies are children of their first blocker,
    with cross-references for additional blockers.
    """
    total = len(dag.slices)
    terminal_count = sum(
        1 for s in dag.slices.values()
        if s.status.value in ("done", "branch_complete", "failed")
    )
    done_count = sum(
        1 for s in dag.slices.values()
        if s.status.value == "done"
    )

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

    # Identify root slices (no dependencies)
    roots = [
        s for s in dag._ordered_slices()
        if not s.blocked_by
    ]

    # Track which slices we've placed in the tree to handle diamonds
    placed: set[str] = set()

    def _add_subtree(parent_tree: Tree, slice_id: str) -> None:
        """Recursively add a slice and its dependents to the tree."""
        if slice_id in placed:
            return
        placed.add(slice_id)

        s = dag.slices[slice_id]
        # Find extra deps (deps beyond the one that placed us)
        extra = [d for d in s.blocked_by if d != slice_id and d in dag.slices]
        # If this slice has multiple deps and was placed under one,
        # show cross-references for the others
        parent_dep = None
        for dep_id in s.blocked_by:
            if dep_id in placed or dep_id == slice_id:
                parent_dep = dep_id
                break
        extra_deps = [d for d in s.blocked_by if d != parent_dep] if parent_dep else []

        label = _build_slice_label(dag, slice_id, extra_deps=extra_deps)
        node = parent_tree.add(label)

        # Add dependents (slices that depend on this one)
        dependents = sorted(
            [dag.slices[d] for d in dag.adjacency.get(slice_id, set())],
            key=lambda x: x.plan_position,
        )
        for dep in dependents:
            _add_subtree(node, dep.id)

    # Build tree from roots
    for root in roots:
        label = _build_slice_label(dag, root.id)
        node = tree.add(label)
        placed.add(root.id)

        # Add dependents
        dependents = sorted(
            [dag.slices[d] for d in dag.adjacency.get(root.id, set())],
            key=lambda x: x.plan_position,
        )
        for dep in dependents:
            _add_subtree(node, dep.id)

    # Handle any slices not yet placed (shouldn't happen with valid DAG but defensive)
    for s in dag._ordered_slices():
        if s.id not in placed:
            label = _build_slice_label(dag, s.id)
            tree.add(label)
            placed.add(s.id)

    # "Ready to merge" footer
    branch_complete = [
        s for s in dag._ordered_slices()
        if s.status.value == "branch_complete"
    ]

    footer_parts: list[Text] = []
    if branch_complete:
        footer = Text("\nReady to merge: ", style="bold bright_yellow")
        names = []
        for s in branch_complete:
            branch_info = f" ({s.workspace.branch})" if s.workspace else ""
            names.append(f"{s.title}{branch_info}")
        footer.append(", ".join(names), style="bright_yellow")
        footer_parts.append(footer)

    # Build final content
    content = tree
    subtitle = None
    if footer_parts:
        subtitle = footer_parts[0].plain

    return Panel(
        content,
        title="[bold]Slice DAG[/bold]",
        subtitle=subtitle,
        border_style="blue",
        padding=(1, 2),
    )
