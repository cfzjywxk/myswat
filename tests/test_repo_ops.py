"""Tests for repo-aware git helpers."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from myswat.repo_ops import GitProbeResult, commit_repo_changes


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
