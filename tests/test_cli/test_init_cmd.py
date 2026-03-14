"""Tests for myswat.cli.init_cmd."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
import typer
from click.exceptions import Exit as ClickExit

from myswat.cli.init_cmd import (
    ARCHITECT_SYSTEM_PROMPT,
    DEVELOPER_SYSTEM_PROMPT,
    QA_MAIN_SYSTEM_PROMPT,
    QA_VICE_SYSTEM_PROMPT,
    TEAM_WORKFLOWS_KNOWLEDGE,
    _ensure_flag_value,
    _is_cli_available,
    _seed_default_agents,
    _seed_team_workflows,
    _slugify,
    run_init,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestInitHelpers:
    def test_ensure_flag_value_adds_missing_flag(self):
        assert _ensure_flag_value(["--print"], "--effort", "high") == [
            "--print", "--effort", "high",
        ]

    def test_ensure_flag_value_preserves_existing_flag(self):
        assert _ensure_flag_value(["--print", "--effort", "medium"], "--effort", "high") == [
            "--print", "--effort", "medium",
        ]

    def test_ensure_flag_value_preserves_existing_equals_style_flag(self):
        assert _ensure_flag_value(["--print", "--effort=medium"], "--effort", "high") == [
            "--print", "--effort=medium",
        ]

    def test_is_cli_available_empty_string_false(self):
        assert _is_cli_available("") is False

    def test_is_cli_available_nonexistent_absolute_path_false(self, tmp_path):
        missing = tmp_path / "missing-claude"
        assert _is_cli_available(str(missing)) is False

    @patch("myswat.cli.init_cmd.shutil.which", return_value="/usr/bin/claude")
    def test_is_cli_available_bare_name_found_in_path_true(self, mock_which):
        assert _is_cli_available("claude") is True
        mock_which.assert_called_once_with("claude")

    def test_is_cli_available_executable_path_true(self, tmp_path):
        target = tmp_path / "claude"
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
        assert _is_cli_available(str(target)) is True


# ---------------------------------------------------------------------------
# _seed_default_agents
# ---------------------------------------------------------------------------
class TestSeedDefaultAgents:
    @patch("myswat.cli.init_cmd._is_cli_available", return_value=True)
    def test_creates_4_agents(self, mock_available):
        store = MagicMock()
        store.get_agent.return_value = None
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "claude-opus-4-6"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.claude_path = "claude"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)
        assert store.create_agent.call_count == 4

    @patch("myswat.cli.init_cmd._is_cli_available", return_value=True)
    def test_seeded_agents_include_system_prompts(self, mock_available):
        store = MagicMock()
        store.get_agent.return_value = None
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "claude-opus-4-6"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.claude_path = "claude"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)

        prompts = [call.kwargs.get("system_prompt") for call in store.create_agent.call_args_list]
        assert ARCHITECT_SYSTEM_PROMPT in prompts
        assert DEVELOPER_SYSTEM_PROMPT in prompts
        assert QA_MAIN_SYSTEM_PROMPT in prompts
        assert QA_VICE_SYSTEM_PROMPT in prompts

    @patch("myswat.cli.init_cmd._is_cli_available", return_value=True)
    def test_skips_existing_agents(self, mock_available):
        store = MagicMock()
        store.get_agent.return_value = {"id": 1}
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "claude-opus-4-6"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.claude_path = "claude"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)
        store.create_agent.assert_not_called()

    @patch("myswat.cli.init_cmd._is_cli_available", return_value=True)
    def test_partial_existing(self, mock_available):
        store = MagicMock()

        def get_agent_side(pid, role):
            if role == "architect":
                return {"id": 1}
            return None

        store.get_agent.side_effect = get_agent_side
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "claude-opus-4-6"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.claude_path = "claude"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)
        assert store.create_agent.call_count == 3

    @patch("myswat.cli.init_cmd._is_cli_available", return_value=True)
    def test_uses_configured_backends(self, mock_available):
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
        assert "delegate" in create_calls[0].kwargs["system_prompt"]
        assert create_calls[1].kwargs["cli_backend"] == "claude"
        assert create_calls[1].kwargs["system_prompt"] == DEVELOPER_SYSTEM_PROMPT
        assert create_calls[2].kwargs["cli_backend"] == "kimi"
        assert "--effort" not in create_calls[2].kwargs["cli_extra_args"]
        assert "testplan" in create_calls[2].kwargs["system_prompt"]
        assert create_calls[3].kwargs["cli_backend"] == "codex"
        assert "testplan" in create_calls[3].kwargs["system_prompt"]

    @patch("myswat.cli.init_cmd._is_cli_available", return_value=False)
    def test_default_qamain_claude_unavailable_aborts_before_creating_agents(self, mock_available):
        store = MagicMock()
        store.get_agent.return_value = None
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "claude-opus-4-6"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.claude_path = "claude"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.kimi_default_flags = ["--print"]

        with pytest.raises(ClickExit):
            _seed_default_agents(store, settings, 1)

        store.create_agent.assert_not_called()

    @patch("myswat.cli.init_cmd._is_cli_available", return_value=True)
    def test_default_qamain_claude_gets_high_effort(self, mock_available):
        store = MagicMock()
        store.get_agent.return_value = None
        settings = MagicMock()
        settings.agents.architect_model = "gpt-5"
        settings.agents.developer_model = "gpt-5"
        settings.agents.qa_main_model = "claude-opus-4-6"
        settings.agents.qa_vice_model = "kimi"
        settings.agents.codex_path = "codex"
        settings.agents.claude_path = "claude"
        settings.agents.kimi_path = "kimi"
        settings.agents.codex_default_flags = ["--json"]
        settings.agents.claude_default_flags = ["--print"]
        settings.agents.kimi_default_flags = ["--print"]

        _seed_default_agents(store, settings, 1)

        qa_main_call = store.create_agent.call_args_list[2]
        assert qa_main_call.kwargs["cli_backend"] == "claude"
        assert qa_main_call.kwargs["model_name"] == "claude-opus-4-6"
        assert qa_main_call.kwargs["cli_extra_args"] == ["--print", "--effort", "high"]


# ---------------------------------------------------------------------------
# _seed_team_workflows
# ---------------------------------------------------------------------------
class TestSeedTeamWorkflows:
    def test_stores_knowledge_when_none_exists(self):
        store = MagicMock()
        store.list_knowledge.return_value = []

        _seed_team_workflows(store, 1)

        store.store_knowledge.assert_called_once()
        kwargs = store.store_knowledge.call_args.kwargs
        assert kwargs["category"] == "project_ops"
        assert kwargs["title"] == "Team Workflows"
        assert kwargs["content"] == TEAM_WORKFLOWS_KNOWLEDGE
        assert "workflow" in kwargs["tags"]

    def test_skips_when_content_unchanged(self):
        store = MagicMock()
        store.list_knowledge.return_value = [
            {"id": 99, "title": "Team Workflows", "content": TEAM_WORKFLOWS_KNOWLEDGE},
        ]

        _seed_team_workflows(store, 1)

        store.store_knowledge.assert_not_called()

    def test_replaces_when_content_changed(self):
        store = MagicMock()
        store.list_knowledge.return_value = [
            {"id": 99, "title": "Team Workflows", "content": "old content"},
        ]

        _seed_team_workflows(store, 1)

        store.delete_knowledge.assert_called_once_with(99)  # DELETE old
        store.store_knowledge.assert_called_once()  # INSERT new


# ---------------------------------------------------------------------------
# TEAM_WORKFLOWS_KNOWLEDGE content
# ---------------------------------------------------------------------------
class TestTeamWorkflowsKnowledge:
    """Verify the knowledge content covers all supported delegation modes."""

    def test_documents_all_modes(self):
        for mode in ("full", "design", "code", "testplan"):
            assert f"MODE: {mode}" in TEAM_WORKFLOWS_KNOWLEDGE

    def test_documents_delegation_format(self):
        assert "```delegate" in TEAM_WORKFLOWS_KNOWLEDGE

    def test_documents_role_permissions(self):
        assert "architect" in TEAM_WORKFLOWS_KNOWLEDGE
        assert "qa_main" in TEAM_WORKFLOWS_KNOWLEDGE or "QA roles" in TEAM_WORKFLOWS_KNOWLEDGE

    def test_includes_decision_guide(self):
        assert "Decision Guide" in TEAM_WORKFLOWS_KNOWLEDGE

    def test_includes_usage_examples(self):
        # Each mode section should have "Use when:" with example phrases
        assert TEAM_WORKFLOWS_KNOWLEDGE.count("Use when:") >= 4


# ---------------------------------------------------------------------------
# run_init
# ---------------------------------------------------------------------------
class TestRunInit:
    @patch("myswat.cli.init_cmd._seed_team_workflows")
    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_health_check_failure(self, mock_store_cls, mock_mig,
                                   mock_pool_cls, mock_settings_cls,
                                   mock_seed, mock_seed_wf):
        pool = MagicMock()
        pool.health_check.return_value = False
        mock_pool_cls.return_value = pool

        with pytest.raises(ClickExit):
            run_init("My Project", None, None)

    @patch("myswat.cli.init_cmd._seed_team_workflows")
    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_new_project(self, mock_store_cls, mock_mig,
                          mock_pool_cls, mock_settings_cls, mock_seed,
                          mock_seed_wf):
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
        mock_seed_wf.assert_called_once()

    @patch("myswat.cli.init_cmd._seed_team_workflows")
    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_existing_project(self, mock_store_cls, mock_mig,
                               mock_pool_cls, mock_settings_cls, mock_seed,
                               mock_seed_wf):
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
        mock_seed_wf.assert_called_once()

    @patch("myswat.cli.init_cmd._seed_team_workflows")
    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_migrations_applied(self, mock_store_cls, mock_mig,
                                 mock_pool_cls, mock_settings_cls, mock_seed,
                                 mock_seed_wf):
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

    @patch("myswat.cli.init_cmd._seed_team_workflows")
    @patch("myswat.cli.init_cmd._seed_default_agents")
    @patch("myswat.cli.init_cmd.MySwatSettings")
    @patch("myswat.cli.init_cmd.TiDBPool")
    @patch("myswat.cli.init_cmd.run_migrations")
    @patch("myswat.cli.init_cmd.MemoryStore")
    def test_no_migrations_needed(self, mock_store_cls, mock_mig,
                                   mock_pool_cls, mock_settings_cls,
                                   mock_seed, mock_seed_wf):
        pool = MagicMock()
        pool.health_check.return_value = True
        mock_pool_cls.return_value = pool
        mock_mig.return_value = []

        mock_store = MagicMock()
        mock_store.get_project_by_slug.return_value = None
        mock_store.create_project.return_value = 1
        mock_store_cls.return_value = mock_store

        run_init("Test", None, None)
