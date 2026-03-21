"""Tests for repo-aware git helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess
from unittest.mock import patch

from myswat.repo_ops import (
    GitCommitResult,
    GitChangedPathsResult,
    GitProbeResult,
    commit_repo_changes,
    list_changed_repo_paths,
    write_design_plan_doc,
    write_workflow_report_doc,
)


def _completed_process(*args: str, returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=list(args), returncode=returncode, stdout=stdout, stderr=stderr)


def test_commit_repo_changes_appends_trailers_to_git_commit_message(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with patch(
        "myswat.repo_ops.probe_git_repository",
        return_value=GitProbeResult(available=True, is_git_repo=True, clean=False, message=""),
    ):
        with patch(
            "myswat.repo_ops._run_git",
            side_effect=[
                _completed_process("git", "add", returncode=0),
                _completed_process("git", "diff", returncode=1),
                _completed_process("git", "commit", returncode=0, stdout="[main abc123] phase 1: Ship it"),
            ],
        ) as mock_git:
            result = commit_repo_changes(
                repo_path,
                message="phase 1: Ship it",
                trailers=["Co-Authored-By: MySwat Dev (GPT-5.4) <noreply@myswat.invalid>"],
            )

    assert result.ok is True
    assert result.committed is True
    assert mock_git.call_args_list[2].args == (
        repo_path.resolve(),
        "commit",
        "-m",
        "phase 1: Ship it",
        "-m",
        "Co-Authored-By: MySwat Dev (GPT-5.4) <noreply@myswat.invalid>",
    )


def test_commit_repo_changes_uses_path_scoped_git_add_for_selected_paths(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with patch(
        "myswat.repo_ops.probe_git_repository",
        return_value=GitProbeResult(available=True, is_git_repo=True, clean=False, message=""),
    ):
        with patch(
            "myswat.repo_ops._run_git",
            side_effect=[
                _completed_process("git", "add", returncode=0),
                _completed_process("git", "diff", returncode=1),
                _completed_process("git", "commit", returncode=0, stdout="[main abc123] scoped commit"),
            ],
        ) as mock_git:
            result = commit_repo_changes(
                repo_path,
                message="phase 1: Ship it",
                paths=[repo_path / "src/lib.rs"],
            )

    assert result.ok is True
    assert result.committed is True
    assert mock_git.call_args_list[0].args == (
        repo_path.resolve(),
        "add",
        "-A",
        "--",
        "src/lib.rs",
    )


def test_commit_repo_changes_skips_paths_outside_the_repo(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    outside_path = tmp_path / "outside.txt"
    outside_path.write_text("outside", encoding="utf-8")

    with patch(
        "myswat.repo_ops.probe_git_repository",
        return_value=GitProbeResult(available=True, is_git_repo=True, clean=False, message=""),
    ):
        with patch("myswat.repo_ops._run_git") as mock_git:
            result = commit_repo_changes(
                repo_path,
                message="phase 1: Ship it",
                paths=[outside_path],
            )

    assert result == GitCommitResult(ok=True, committed=False, message="No selected paths to commit.")
    mock_git.assert_not_called()


def test_list_changed_repo_paths_parses_modified_untracked_and_renamed_files(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with patch(
        "myswat.repo_ops.probe_git_repository",
        return_value=GitProbeResult(available=True, is_git_repo=True, clean=False, message=""),
    ):
        with patch(
            "myswat.repo_ops._run_git",
            return_value=_completed_process(
                "git",
                "status",
                returncode=0,
                stdout=" M src/lib.rs\0?? notes.txt\0R  old.py\0new.py\0",
            ),
        ):
            result = list_changed_repo_paths(repo_path)

    assert result == GitChangedPathsResult(
        ok=True,
        paths={"src/lib.rs", "notes.txt", "old.py", "new.py"},
        message="",
    )


def test_repo_outputs_use_timestamped_root_level_filenames(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    generated_at = datetime(2026, 3, 21, 21, 48, 32, tzinfo=timezone.utc)

    design_path = write_design_plan_doc(
        repo_path,
        requirement="ship it",
        design="design",
        plan="plan",
        generated_at=generated_at,
    )
    report_path = write_workflow_report_doc(
        repo_path,
        report="# Workflow Report",
        work_mode="develop",
        generated_at=generated_at,
    )

    assert design_path == repo_path / "myswat-design-plan-20260321-214832.md"
    assert report_path == repo_path / "myswat-develop-workflow-report-20260321-214832.md"
