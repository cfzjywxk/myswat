"""CLI entrypoint for the persistent MySwat daemon."""

from __future__ import annotations

import logging
from datetime import datetime

from rich.console import Console

from myswat.server.daemon import MySwatDaemon

console = Console()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().setLevel(logging.INFO)


def run_daemon_server() -> int:
    _configure_logging()
    daemon = MySwatDaemon()
    console.print(
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim] "
        f"[green]MySwat daemon listening on {daemon.base_url}[/green]"
    )
    daemon.serve_forever()
    return 0
