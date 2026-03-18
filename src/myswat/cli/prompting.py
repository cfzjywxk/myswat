"""Shared prompt_toolkit helpers for interactive CLI input."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import FileHistory


def _normalize_config_path(config_path: str | Path | None) -> Path:
    if isinstance(config_path, Path):
        return config_path.expanduser()
    if isinstance(config_path, str) and config_path:
        return Path(config_path).expanduser()
    return Path(tempfile.gettempdir()) / "myswat-config.toml"


def _history_file(config_path: str | Path | None, history_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", history_name).strip("-") or "default"
    history_dir = _normalize_config_path(config_path).parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / f"{safe_name}.history"


def create_prompt_session(
    *,
    config_path: str | Path | None,
    history_name: str,
    prompt_session_factory: Callable[..., Any],
    key_bindings_factory: Callable[[], Any],
):
    """Build a PromptSession with shell-like Emacs bindings and file history."""
    bindings = key_bindings_factory()

    @bindings.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return prompt_session_factory(
        multiline=False,
        key_bindings=bindings,
        prompt_continuation="... ",
        editing_mode=EditingMode.EMACS,
        history=FileHistory(str(_history_file(config_path, history_name))),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )


def make_prompt_callback(prompt_session: Any | None = None) -> Callable[[str], str]:
    """Build a reusable ask_user callback backed by prompt_toolkit when available."""

    def ask_user(prompt_text: str) -> str:
        if prompt_session is not None:
            try:
                return prompt_session.prompt(prompt_text, multiline=False).strip()
            except (EOFError, KeyboardInterrupt):
                return "n"
        try:
            return input(f"\n{prompt_text}").strip()
        except (EOFError, KeyboardInterrupt):
            return "n"

    return ask_user
