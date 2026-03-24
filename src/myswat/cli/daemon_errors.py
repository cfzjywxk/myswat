"""Shared user-facing rendering for local daemon transport failures."""

from __future__ import annotations

from rich.console import Console


def print_daemon_error(exc: Exception, *, console: Console | None = None) -> None:
    console = console or Console()
    message = str(exc)
    console.print(f"[red]{message}[/red]")
    lowered = message.lower()
    if "unavailable at" in lowered or "connection refused" in lowered:
        console.print("[dim]Start the daemon first: myswat server[/dim]")
        return
    if "timed out" in lowered:
        console.print(
            "[dim]The daemon is running, but the request is still in progress or blocked.[/dim]"
        )
