"""Extended tests for myswat.cli.learn_cmd — covering run_learn."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.agents.base import AgentResponse
from myswat.cli.learn_cmd import run_learn


class TestRunLearn:
    def _setup_mocks(self, tmp_path):
        """Create common mock setup."""
        settings = MagicMock()
        settings.compaction.compaction_backend = "codex"
        settings.compaction.threshold_turns = 200
        settings.compaction.threshold_tokens = 800000

        proj = {
            "id": 1, "slug": "proj", "name": "Proj",
            "repo_path": str(tmp_path),
        }
        agent_row = {
            "id": 1, "role": "architect", "display_name": "Architect",
            "cli_backend": "codex", "model_name": "gpt-5",
            "cli_path": "codex", "cli_extra_args": None,
        }
        return settings, proj, agent_row

    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_project_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                                mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_learn("missing")

    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_repo_path_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                                  mock_settings_cls):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": "/nonexistent/path",
        }
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_learn("proj", workdir="/nonexistent/path")

    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_no_indicator_files(self, mock_store_cls, mock_mig, mock_pool_cls,
                                 mock_settings_cls, tmp_path):
        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": str(tmp_path),
        }
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_learn("proj", workdir=str(tmp_path))

    @patch("myswat.cli.learn_cmd._store_learned_knowledge")
    @patch("myswat.cli.learn_cmd._write_myswat_md")
    @patch("myswat.cli.learn_cmd.make_runner_from_row")
    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_architect_not_found(self, mock_store_cls, mock_mig, mock_pool_cls,
                                  mock_settings_cls, mock_make_runner,
                                  mock_write_md, mock_store_learned,
                                  tmp_path):
        (tmp_path / "Makefile").write_text("all: build")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": str(tmp_path),
        }
        mock_store.get_agent.return_value = None
        mock_store_cls.return_value = mock_store

        with pytest.raises(ClickExit):
            run_learn("proj", workdir=str(tmp_path))

    @patch("myswat.cli.learn_cmd._store_learned_knowledge")
    @patch("myswat.cli.learn_cmd._write_myswat_md")
    @patch("myswat.cli.learn_cmd.make_runner_from_row")
    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_agent_failure(self, mock_store_cls, mock_mig, mock_pool_cls,
                            mock_settings_cls, mock_make_runner,
                            mock_write_md, mock_store_learned, tmp_path):
        (tmp_path / "Makefile").write_text("all: build")

        settings, proj, agent_row = self._setup_mocks(tmp_path)
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store_cls.return_value = mock_store

        runner = MagicMock()
        runner.invoke.return_value = AgentResponse(content="error", exit_code=1)
        mock_make_runner.return_value = runner

        with pytest.raises(ClickExit):
            run_learn("proj", workdir=str(tmp_path))

    @patch("myswat.cli.learn_cmd._store_learned_knowledge")
    @patch("myswat.cli.learn_cmd._write_myswat_md")
    @patch("myswat.cli.learn_cmd.make_runner_from_row")
    @patch("myswat.workflow.engine._extract_json_block")
    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_unparseable_output(self, mock_store_cls, mock_mig, mock_pool_cls,
                                 mock_settings_cls, mock_extract,
                                 mock_make_runner, mock_write_md,
                                 mock_store_learned, tmp_path):
        (tmp_path / "Makefile").write_text("all: build")

        settings, proj, agent_row = self._setup_mocks(tmp_path)
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store_cls.return_value = mock_store

        runner = MagicMock()
        runner.invoke.return_value = AgentResponse(content="no json", exit_code=0)
        mock_make_runner.return_value = runner
        mock_extract.return_value = "not a dict"

        with pytest.raises(ClickExit):
            run_learn("proj", workdir=str(tmp_path))

    @patch("myswat.cli.learn_cmd._store_learned_knowledge")
    @patch("myswat.cli.learn_cmd._write_myswat_md")
    @patch("myswat.cli.learn_cmd.make_runner_from_row")
    @patch("myswat.workflow.engine._extract_json_block")
    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_validation_failure(self, mock_store_cls, mock_mig, mock_pool_cls,
                                 mock_settings_cls, mock_extract,
                                 mock_make_runner, mock_write_md,
                                 mock_store_learned, tmp_path):
        (tmp_path / "Makefile").write_text("all: build")

        settings, proj, agent_row = self._setup_mocks(tmp_path)
        mock_settings_cls.return_value = settings

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store_cls.return_value = mock_store

        runner = MagicMock()
        runner.invoke.return_value = AgentResponse(content="{}", exit_code=0)
        mock_make_runner.return_value = runner
        # Returns a dict missing required keys
        mock_extract.return_value = {}

        with pytest.raises(ClickExit):
            run_learn("proj", workdir=str(tmp_path))

    @patch("myswat.cli.learn_cmd._store_learned_knowledge")
    @patch("myswat.cli.learn_cmd._write_myswat_md")
    @patch("myswat.cli.learn_cmd.make_runner_from_row")
    @patch("myswat.workflow.engine._extract_json_block")
    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_success(self, mock_store_cls, mock_mig, mock_pool_cls,
                      mock_settings_cls, mock_extract, mock_make_runner,
                      mock_write_md, mock_store_learned, tmp_path):
        (tmp_path / "Makefile").write_text("all: build")
        (tmp_path / "CLAUDE.md").write_text("Be helpful.")

        settings, proj, agent_row = self._setup_mocks(tmp_path)
        mock_settings_cls.return_value = settings
        mock_mig.return_value = ["v001"]

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store_cls.return_value = mock_store

        runner = MagicMock()
        runner.invoke.return_value = AgentResponse(content="json", exit_code=0)
        mock_make_runner.return_value = runner

        valid_data = {
            "build": {"commands": ["make build"]},
            "test": {"tiers": [{"name": "unit", "cmd": "pytest"}], "gate_command": "make test"},
            "structure": {"entry_points": ["src/main.py"]},
            "language": "Rust",
            "project_type": "library",
            "conventions": {"rules": ["use snake_case"]},
            "security": {"requirements": ["no hardcoded secrets"]},
            "invariants": ["never break build"],
        }
        mock_extract.return_value = valid_data
        mock_store_learned.return_value = 8
        mock_write_md.return_value = tmp_path / "myswat.md"

        run_learn("proj", workdir=str(tmp_path))
        mock_store_learned.assert_called_once()
        mock_write_md.assert_called_once()

    @patch("myswat.cli.learn_cmd._store_learned_knowledge")
    @patch("myswat.cli.learn_cmd._write_myswat_md")
    @patch("myswat.cli.learn_cmd.make_runner_from_row")
    @patch("myswat.workflow.engine._extract_json_block")
    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_success_minimal_data(self, mock_store_cls, mock_mig, mock_pool_cls,
                                   mock_settings_cls, mock_extract,
                                   mock_make_runner, mock_write_md,
                                   mock_store_learned, tmp_path):
        (tmp_path / "Makefile").write_text("all: build")

        settings, proj, agent_row = self._setup_mocks(tmp_path)
        mock_settings_cls.return_value = settings
        mock_mig.return_value = []

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = proj
        mock_store.get_agent.return_value = agent_row
        mock_store_cls.return_value = mock_store

        runner = MagicMock()
        runner.invoke.return_value = AgentResponse(content="json", exit_code=0)
        mock_make_runner.return_value = runner

        valid_data = {
            "build": {"commands": ["make build"]},
            "test": {"tiers": [{"name": "unit"}]},
            "structure": {"entry_points": ["src/main.py"]},
        }
        mock_extract.return_value = valid_data
        mock_store_learned.return_value = 4
        mock_write_md.return_value = tmp_path / "myswat.md"

        run_learn("proj", workdir=str(tmp_path))

    @patch("myswat.cli.learn_cmd.MySwatSettings")
    @patch("myswat.cli.learn_cmd.TiDBPool")
    @patch("myswat.cli.learn_cmd.run_migrations")
    @patch("myswat.cli.learn_cmd.MemoryStore")
    def test_uses_proj_repo_path_as_fallback(self, mock_store_cls, mock_mig,
                                              mock_pool_cls, mock_settings_cls,
                                              tmp_path):
        (tmp_path / "Makefile").write_text("all: build")

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {
            "id": 1, "repo_path": str(tmp_path),
        }
        mock_store.get_agent.return_value = None
        mock_store_cls.return_value = mock_store

        # Will fail at architect not found, but tests the repo_path resolution
        with pytest.raises(ClickExit):
            run_learn("proj")  # no workdir arg
