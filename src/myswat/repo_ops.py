"""Helpers for repo-aware workflow automation."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GitProbeResult:
    available: bool
    is_git_repo: bool
    clean: bool
    message: str = ""


@dataclass(frozen=True)
class GitCommitResult:
    ok: bool
    committed: bool
    message: str = ""


@dataclass(frozen=True)
class GitPushResult:
    ok: bool
    pushed: bool
    message: str = ""


def _normalize_repo_path(repo_path: str | Path) -> Path:
    return Path(repo_path).expanduser().resolve()


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None


def _git_message(result: subprocess.CompletedProcess[str] | None, fallback: str) -> str:
    if result is None:
        return fallback
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return stdout or stderr or fallback


def _relative_repo_path(repo_path: Path, file_path: str | Path) -> str:
    candidate = Path(file_path).expanduser().resolve()
    try:
        return str(candidate.relative_to(repo_path))
    except ValueError:
        return os.path.relpath(candidate, repo_path)


def probe_git_repository(repo_path: str | Path | None) -> GitProbeResult:
    if not repo_path:
        return GitProbeResult(
            available=True,
            is_git_repo=False,
            clean=False,
            message="Repository path is not configured.",
        )

    repo = _normalize_repo_path(repo_path)
    if not repo.exists() or not repo.is_dir():
        return GitProbeResult(
            available=True,
            is_git_repo=False,
            clean=False,
            message=f"Repository path does not exist: {repo}",
        )

    probe = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    if probe is None:
        return GitProbeResult(
            available=False,
            is_git_repo=False,
            clean=False,
            message="git CLI is not available.",
        )
    if probe.returncode != 0:
        return GitProbeResult(
            available=True,
            is_git_repo=False,
            clean=False,
            message=_git_message(probe, f"{repo} is not a git repository."),
        )

    status = _run_git(repo, "status", "--porcelain")
    if status is None:
        return GitProbeResult(
            available=False,
            is_git_repo=True,
            clean=False,
            message="git CLI is not available.",
        )
    if status.returncode != 0:
        return GitProbeResult(
            available=True,
            is_git_repo=True,
            clean=False,
            message=_git_message(status, "Unable to inspect git status."),
        )
    return GitProbeResult(
        available=True,
        is_git_repo=True,
        clean=not bool((status.stdout or "").strip()),
    )


def ensure_git_repository(repo_path: str | Path | None) -> GitProbeResult:
    status = probe_git_repository(repo_path)
    if not repo_path or not status.available or status.is_git_repo:
        return status

    repo = _normalize_repo_path(repo_path)
    init_result = _run_git(repo, "init")
    if init_result is None:
        return GitProbeResult(
            available=False,
            is_git_repo=False,
            clean=False,
            message="git CLI is not available.",
        )
    if init_result.returncode != 0:
        return GitProbeResult(
            available=True,
            is_git_repo=False,
            clean=False,
            message=_git_message(init_result, f"Failed to initialize git repository at {repo}."),
        )

    refreshed = probe_git_repository(repo)
    if refreshed.available and refreshed.is_git_repo:
        return GitProbeResult(
            available=True,
            is_git_repo=True,
            clean=refreshed.clean,
            message=f"Initialized a git repository at {repo}.",
        )
    return refreshed


def render_design_plan_markdown(
    *,
    requirement: str,
    design: str,
    plan: str,
) -> str:
    return (
        "# MySwat Design Plan\n\n"
        "This file is maintained by MySwat and tracks the latest approved design and implementation plan.\n\n"
        "## Requirement\n\n"
        f"{requirement.strip()}\n\n"
        "## Technical Design\n\n"
        f"{design.strip()}\n\n"
        "## Implementation Plan\n\n"
        f"{plan.strip()}\n"
    )


def write_design_plan_doc(
    repo_path: str | Path,
    *,
    requirement: str,
    design: str,
    plan: str,
    filename: str = "myswat-design-plan.md",
) -> Path:
    repo = _normalize_repo_path(repo_path)
    doc_path = repo / filename
    doc_path.write_text(
        render_design_plan_markdown(
            requirement=requirement,
            design=design,
            plan=plan,
        ),
        encoding="utf-8",
    )
    return doc_path


def commit_repo_changes(
    repo_path: str | Path | None,
    *,
    message: str,
    paths: Iterable[str | Path] | None = None,
    trailers: Iterable[str] | None = None,
) -> GitCommitResult:
    status = probe_git_repository(repo_path)
    if not status.available:
        return GitCommitResult(ok=True, committed=False, message=status.message or "git CLI is not available.")
    if not status.is_git_repo:
        return GitCommitResult(ok=True, committed=False, message=status.message or "Repository is not under git.")
    if not repo_path:
        return GitCommitResult(ok=True, committed=False, message="Repository path is not configured.")

    repo = _normalize_repo_path(repo_path)
    path_args = [_relative_repo_path(repo, path) for path in paths or []]
    if path_args:
        add_result = _run_git(repo, "add", "--", *path_args)
    else:
        add_result = _run_git(repo, "add", "-A")
    if add_result is None:
        return GitCommitResult(ok=True, committed=False, message="git CLI is not available.")
    if add_result.returncode != 0:
        return GitCommitResult(ok=False, committed=False, message=_git_message(add_result, "git add failed."))

    if path_args:
        diff_result = _run_git(repo, "diff", "--cached", "--quiet", "--", *path_args)
    else:
        diff_result = _run_git(repo, "diff", "--cached", "--quiet")
    if diff_result is None:
        return GitCommitResult(ok=True, committed=False, message="git CLI is not available.")
    if diff_result.returncode == 0:
        return GitCommitResult(ok=True, committed=False, message="No changes to commit.")
    if diff_result.returncode != 1:
        return GitCommitResult(ok=False, committed=False, message=_git_message(diff_result, "Unable to inspect staged changes."))

    commit_args = ["commit", "-m", message]
    trailer_lines = [str(trailer).strip() for trailer in (trailers or []) if str(trailer).strip()]
    if trailer_lines:
        commit_args.extend(["-m", "\n".join(trailer_lines)])

    commit_result = _run_git(repo, *commit_args)
    if commit_result is None:
        return GitCommitResult(ok=True, committed=False, message="git CLI is not available.")
    if commit_result.returncode != 0:
        return GitCommitResult(ok=False, committed=False, message=_git_message(commit_result, "git commit failed."))
    return GitCommitResult(
        ok=True,
        committed=True,
        message=_git_message(commit_result, f"Committed local changes: {message}"),
    )


def push_repo_changes(
    repo_path: str | Path | None,
    *,
    remote: str | None = None,
    branch: str | None = None,
) -> GitPushResult:
    status = probe_git_repository(repo_path)
    if not status.available:
        return GitPushResult(ok=True, pushed=False, message=status.message or "git CLI is not available.")
    if not status.is_git_repo:
        return GitPushResult(ok=True, pushed=False, message=status.message or "Repository is not under git.")

    repo = _normalize_repo_path(repo_path)
    push_args = ["push"]
    if remote:
        push_args.append(remote)
    if branch:
        push_args.append(branch)

    push_result = _run_git(repo, *push_args)
    if push_result is None:
        return GitPushResult(ok=True, pushed=False, message="git CLI is not available.")
    if push_result.returncode != 0:
        return GitPushResult(ok=False, pushed=False, message=_git_message(push_result, "git push failed."))
    return GitPushResult(
        ok=True,
        pushed=True,
        message=_git_message(push_result, "Pushed local workflow commits."),
    )
