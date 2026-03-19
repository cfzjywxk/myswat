"""CLI entrypoint for the persistent MySwat daemon."""

from __future__ import annotations

from rich.console import Console

from myswat.server.daemon import MySwatDaemon

console = Console()


def run_daemon_server() -> int:
    daemon = MySwatDaemon()
    console.print(f"[green]MySwat daemon listening on {daemon.base_url}[/green]")
    daemon.serve_forever()
    return 0
