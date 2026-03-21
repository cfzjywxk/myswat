"""Helpers for repo-aware workflow automation."""

from __future__ import annotations

from datetime import datetime
import subprocess
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class GitChangedPathsResult:
    ok: bool
    paths: set[str] = field(default_factory=set)
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


def _git_path(repo_path: Path, pathspec: str) -> Path | None:
    result = _run_git(repo_path, "rev-parse", "--git-path", pathspec)
    if result is None or result.returncode != 0:
        return None

    raw_path = (result.stdout or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_path / candidate
    return candidate


def _ensure_git_exclude_pattern(repo_path: Path, pattern: str) -> None:
    exclude_path = _git_path(repo_path, "info/exclude")
    if exclude_path is None:
        return

    normalized_pattern = pattern.strip()
    if not normalized_pattern:
        return

    try:
        existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
        if normalized_pattern in {line.strip() for line in existing.splitlines()}:
            return

        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        if existing and not existing.endswith("\n"):
            existing += "\n"
        exclude_path.write_text(f"{existing}{normalized_pattern}\n", encoding="utf-8")
    except OSError:
        return


def _relative_repo_path(repo_path: Path, file_path: str | Path) -> str | None:
    candidate = Path(file_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_path / candidate
    candidate = candidate.resolve()
    try:
        return str(candidate.relative_to(repo_path))
    except ValueError:
        return None


def _timestamped_filename(stem: str, *, generated_at: datetime | None = None, suffix: str = ".md") -> str:
    timestamp = (generated_at or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    return f"{stem}-{timestamp}{suffix}"


def _parse_porcelain_paths(output: str) -> set[str]:
    parts = output.split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(parts):
        entry = parts[index]
        if not entry:
            index += 1
            continue
        if len(entry) < 4:
            index += 1
            continue

        status = entry[:2]
        primary_path = entry[3:]
        if primary_path:
            paths.add(primary_path)

        if "R" in status or "C" in status:
            if index + 1 < len(parts):
                secondary_path = parts[index + 1]
                if secondary_path:
                    paths.add(secondary_path)
                index += 1
        index += 1
    return paths


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


def list_changed_repo_paths(repo_path: str | Path | None) -> GitChangedPathsResult:
    status = probe_git_repository(repo_path)
    if not status.available:
        return GitChangedPathsResult(ok=False, message=status.message or "git CLI is not available.")
    if not status.is_git_repo:
        return GitChangedPathsResult(ok=False, message=status.message or "Repository is not under git.")
    if not repo_path:
        return GitChangedPathsResult(ok=False, message="Repository path is not configured.")

    repo = _normalize_repo_path(repo_path)
    result = _run_git(repo, "status", "--porcelain", "-z")
    if result is None:
        return GitChangedPathsResult(ok=False, message="git CLI is not available.")
    if result.returncode != 0:
        return GitChangedPathsResult(ok=False, message=_git_message(result, "Unable to inspect git status."))
    return GitChangedPathsResult(ok=True, paths=_parse_porcelain_paths(result.stdout or ""))


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
    filename: str | None = None,
    generated_at: datetime | None = None,
) -> Path:
    repo = _normalize_repo_path(repo_path)
    doc_path = repo / (filename or _timestamped_filename("myswat-design-plan", generated_at=generated_at))
    doc_path.write_text(
        render_design_plan_markdown(
            requirement=requirement,
            design=design,
            plan=plan,
        ),
        encoding="utf-8",
    )
    return doc_path


def write_workflow_report_doc(
    repo_path: str | Path,
    *,
    report: str,
    work_mode: str,
    filename: str | None = None,
    generated_at: datetime | None = None,
) -> Path:
    repo = _normalize_repo_path(repo_path)
    stem = f"myswat-{(work_mode or 'workflow').strip().lower()}-workflow-report"
    report_dir = repo / ".myswat" / "workflow-reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    _ensure_git_exclude_pattern(repo, ".myswat/")
    doc_path = report_dir / (filename or _timestamped_filename(stem, generated_at=generated_at))
    content = report.strip()
    if not content.endswith("\n"):
        content += "\n"
    doc_path.write_text(content, encoding="utf-8")
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
    if paths is not None:
        path_args: list[str] = []
        seen_paths: set[str] = set()
        for path in paths:
            relative = _relative_repo_path(repo, path)
            if not relative or relative in seen_paths:
                continue
            seen_paths.add(relative)
            path_args.append(relative)
        if not path_args:
            return GitCommitResult(ok=True, committed=False, message="No selected paths to commit.")
        add_result = _run_git(repo, "add", "-A", "--", *path_args)
    else:
        path_args = []
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
    if not remote:
        remote_result = _run_git(repo, "remote")
        if remote_result is None:
            return GitPushResult(ok=True, pushed=False, message="git CLI is not available.")
        if remote_result.returncode != 0:
            return GitPushResult(ok=False, pushed=False, message=_git_message(remote_result, "Unable to inspect git remotes."))
        if not any(line.strip() for line in (remote_result.stdout or "").splitlines()):
            return GitPushResult(ok=True, pushed=False, message="No configured push destination; skipping push.")

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
