"""Tests for myswat.cli.init_cmd."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.cli.init_cmd import _slugify, _seed_default_agents, run_init


# ---------------------------------------------------------------------------
# _seed_default_agents
# ---------------------------------------------------------------------------
class TestSeedDefaultAgents:
    def test_creates_4_agents(self):
        store = MagicMock()
        store.get_agent.return_value = None
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "kimi"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)
        assert store.create_agent.call_count == 4

    def test_skips_existing_agents(self):
        store = MagicMock()
        store.get_agent.return_value = {"id": 1}  # all exist
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "kimi"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)
        store.create_agent.assert_not_called()

    def test_partial_existing(self):
        store = MagicMock()

        def get_agent_side(pid, role):
            if role == "architect":
                return {"id": 1}
            return None

        store.get_agent.side_effect = get_agent_side
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "kimi"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)
        assert store.create_agent.call_count == 3

    def test_uses_configured_backends(self):
        store = MagicMock()
        store.get_agent.return_value = None
        settings = MagicMock()
        settings.agents.architect_model = "claude-sonnet-4-6"
        settings.agents.developer_model = "claude-sonnet-4-6"
        settings.agents.qa_main_model = "kimi"
        settings.agents.qa_vice_model = "gpt-5"
        settings.agents.architect_backend = "claude"
        settings.agents.developer_backend = "claude"
        settings.agents.qa_main_backend = "kimi"
        settings.agents.qa_vice_backend = "codex"
        settings.agents.codex_path = "codex"
        settings.agents.kimi_path = "kimi"
        settings.agents.claude_path = "claude"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.kimi_default_flags = ["--print"]
        settings.agents.claude_default_flags = ["--print", "--output-format", "stream-json"]

        _seed_default_agents(store, settings, 1)

        create_calls = store.create_agent.call_args_list
        assert create_calls[0].kwargs["cli_backend"] == "claude"
        assert create_calls[0].kwargs["cli_path"] == "claude"
        assert create_calls[1].kwargs["cli_backend"] == "claude"
        assert create_calls[2].kwargs["cli_backend"] == "kimi"
        assert create_calls[3].kwargs["cli_backend"] == "codex"


# ---------------------------------------------------------------------------
# run_init
# ---------------------------------------------------------------------------
class TestRunInit:
    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_health_check_failure(self, mock_store_cls, mock_mig,
                                   mock_pool_cls, mock_settings_cls,
                                   mock_seed):
        pool = MagicMock()
        pool.health_check.return_value = False
        mock_pool_cls.return_value = pool

        with pytest.raises(ClickExit):
            run_init("My Project", None, None)

    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_new_project(self, mock_store_cls, mock_mig,
                          mock_pool_cls, mock_settings_cls, mock_seed):
        pool = MagicMock()
        pool.health_check.return_value = True
        mock_pool_cls.return_value = pool
        mock_mig.return_value = ["v001"]

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store.create_project.return_value = 1
        mock_store_cls.return_value = mock_store

        run_init("My Project", "/tmp/repo", "Test desc")
        mock_store.create_project.assert_called_once_with(
            slug="my-project", name="My Project",
            description="Test desc", repo_path="/tmp/repo",
        )
        mock_seed.assert_called_once()

    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_existing_project(self, mock_store_cls, mock_mig,
                               mock_pool_cls, mock_settings_cls, mock_seed):
        pool = MagicMock()
        pool.health_check.return_value = True
        mock_pool_cls.return_value = pool
        mock_mig.return_value = []

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = {"id": 42}
        mock_store_cls.return_value = mock_store

        run_init("My Project", None, None)
        mock_store.create_project.assert_not_called()
        mock_seed.assert_called_once()

    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_migrations_applied(self, mock_store_cls, mock_mig,
                                 mock_pool_cls, mock_settings_cls, mock_seed):
        pool = MagicMock()
        pool.health_check.return_value = True
        mock_pool_cls.return_value = pool
        mock_mig.return_value = ["v001", "v002"]

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store.create_project.return_value = 1
        mock_store_cls.return_value = mock_store

        run_init("Test", None, None)
        mock_mig.assert_called_once()

    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_no_migrations_needed(self, mock_store_cls, mock_mig,
                                   mock_pool_cls, mock_settings_cls,
                                   mock_seed):
        pool = MagicMock()
        pool.health_check.return_value = True
        mock_pool_cls.return_value = pool
        mock_mig.return_value = []

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store.create_project.return_value = 1
        mock_store_cls.return_value = mock_store

        run_init("Test", None, None)
