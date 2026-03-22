"""Comprehensive tests for myswat.config.settings classes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from myswat.config.settings import (
    AgentSettings,
    CompactionSettings,
    EmbeddingSettings,
    MemoryWorkerSettings,
    MySwatSettings,
    TiDBSettings,
    WorkflowSettings,
    get_workflow_review_limit,
)


# ---------------------------------------------------------------------------
# TiDBSettings
# ---------------------------------------------------------------------------

class TestTiDBSettings:
    """Tests for TiDBSettings defaults and env-var overrides."""

    def test_defaults(self):
        settings = TiDBSettings()
        assert settings.host == ""
        assert settings.port == 4000
        assert settings.user == ""
        assert settings.password == ""
        assert settings.ssl_ca == "/etc/ssl/certs/ca-certificates.crt"
        assert settings.database == "myswat"

    def test_env_override_host(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_HOST", "db.example.com")
        settings = TiDBSettings()
        assert settings.host == "db.example.com"

    def test_env_override_port(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_PORT", "3306")
        settings = TiDBSettings()
        assert settings.port == 3306

    def test_env_override_user(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_USER", "admin")
        settings = TiDBSettings()
        assert settings.user == "admin"

    def test_env_override_password(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_PASSWORD", "s3cret")
        settings = TiDBSettings()
        assert settings.password == "s3cret"

    def test_env_override_ssl_ca(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_SSL_CA", "/custom/ca.pem")
        settings = TiDBSettings()
        assert settings.ssl_ca == "/custom/ca.pem"

    def test_env_override_database(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_DATABASE", "other_db")
        settings = TiDBSettings()
        assert settings.database == "other_db"

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_TIDB_HOST", "10.0.0.1")
        monkeypatch.setenv("MYSWAT_TIDB_PORT", "5000")
        monkeypatch.setenv("MYSWAT_TIDB_USER", "root")
        monkeypatch.setenv("MYSWAT_TIDB_PASSWORD", "pw")
        monkeypatch.setenv("MYSWAT_TIDB_DATABASE", "test_db")
        settings = TiDBSettings()
        assert settings.host == "10.0.0.1"
        assert settings.port == 5000
        assert settings.user == "root"
        assert settings.password == "pw"
        assert settings.database == "test_db"


# ---------------------------------------------------------------------------
# AgentSettings
# ---------------------------------------------------------------------------

class TestAgentSettings:
    """Tests for AgentSettings defaults, list fields, and env-var overrides."""

    def test_defaults(self):
        settings = AgentSettings()
        assert settings.codex_path == "codex"
        assert settings.kimi_path == "kimi"
        assert settings.claude_path == "claude"
        assert settings.claude_required_ip == ""
        assert settings.claude_ip_check_timeout_seconds == 10
        assert settings.architect_backend == "codex"
        assert settings.developer_backend == "codex"
        assert settings.qa_main_backend == "claude"
        assert settings.qa_vice_enabled is False
        assert settings.qa_vice_backend == "kimi"
        assert settings.developer_model == "gpt-5.4"
        assert settings.architect_model == "gpt-5.4"
        assert settings.qa_main_model == "claude-opus-4-6"
        assert settings.qa_vice_model == "kimi-code/kimi-for-coding"

    def test_default_codex_flags(self):
        settings = AgentSettings()
        assert settings.codex_default_flags == ["--full-auto", "--json"]

    def test_default_kimi_flags(self):
        settings = AgentSettings()
        expected = [
            "--print",
            "--output-format",
            "text",
            "--yolo",
            "--final-message-only",
        ]
        assert settings.kimi_default_flags == expected

    def test_default_claude_flags(self):
        settings = AgentSettings()
        expected = [
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        assert settings.claude_default_flags == expected

    def test_list_fields_are_lists(self):
        settings = AgentSettings()
        assert isinstance(settings.codex_default_flags, list)
        assert isinstance(settings.kimi_default_flags, list)
        assert isinstance(settings.claude_default_flags, list)

    def test_env_override_developer_model(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_DEVELOPER_MODEL", "gpt-6")
        settings = AgentSettings()
        assert settings.developer_model == "gpt-6"

    def test_env_override_architect_model(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_ARCHITECT_MODEL", "claude-opus-5")
        settings = AgentSettings()
        assert settings.architect_model == "claude-opus-5"

    def test_env_override_codex_path(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_CODEX_PATH", "/usr/local/bin/codex")
        settings = AgentSettings()
        assert settings.codex_path == "/usr/local/bin/codex"

    def test_env_override_kimi_path(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_KIMI_PATH", "/opt/kimi")
        settings = AgentSettings()
        assert settings.kimi_path == "/opt/kimi"

    def test_env_override_claude_path(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_CLAUDE_PATH", "/opt/claude")
        settings = AgentSettings()
        assert settings.claude_path == "/opt/claude"

    def test_env_override_claude_required_ip(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_CLAUDE_REQUIRED_IP", "203.0.113.10")
        settings = AgentSettings()
        assert settings.claude_required_ip == "203.0.113.10"

    def test_env_override_claude_ip_check_timeout_seconds(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_CLAUDE_IP_CHECK_TIMEOUT_SECONDS", "7")
        settings = AgentSettings()
        assert settings.claude_ip_check_timeout_seconds == 7

    def test_env_override_role_backend(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_DEVELOPER_BACKEND", "claude")
        settings = AgentSettings()
        assert settings.developer_backend == "claude"

    def test_env_override_qa_main_model(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_QA_MAIN_MODEL", "custom/qa-model")
        settings = AgentSettings()
        assert settings.qa_main_model == "custom/qa-model"

    def test_env_override_qa_vice_model(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_QA_VICE_MODEL", "custom/vice-model")
        settings = AgentSettings()
        assert settings.qa_vice_model == "custom/vice-model"

    def test_env_override_qa_vice_enabled(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_AGENTS_QA_VICE_ENABLED", "true")
        settings = AgentSettings()
        assert settings.qa_vice_enabled is True


# ---------------------------------------------------------------------------
# WorkflowSettings
# ---------------------------------------------------------------------------

class TestWorkflowSettings:
    """Tests for WorkflowSettings defaults and env-var overrides."""

    def test_defaults(self):
        settings = WorkflowSettings()
        assert settings.design_plan_review_limit == 10
        assert settings.dev_plan_review_limit == 10
        assert settings.dev_code_review_limit == 10
        assert settings.ga_plan_review_limit == 2
        assert settings.ga_test_review_limit == 2
        assert settings.requirements_skills_root == ""

    def test_env_override_split_review_limit(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_WORKFLOW_DEV_CODE_REVIEW_LIMIT", "4")
        settings = WorkflowSettings()
        assert settings.dev_code_review_limit == 4
        assert settings.design_plan_review_limit == 10

    def test_env_override_ga_review_defaults(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_WORKFLOW_GA_PLAN_REVIEW_LIMIT", "5")
        monkeypatch.setenv("MYSWAT_WORKFLOW_GA_TEST_REVIEW_LIMIT", "6")
        settings = WorkflowSettings()
        assert settings.ga_plan_review_limit == 5
        assert settings.ga_test_review_limit == 6

    def test_get_workflow_review_limit_uses_split_limit_only(self):
        settings = WorkflowSettings(dev_plan_review_limit=6)
        assert get_workflow_review_limit(settings, "dev_plan_review_limit") == 6
        legacy = MagicMock()
        legacy.dev_code_review_limit = 4
        assert get_workflow_review_limit(legacy, "dev_code_review_limit") == 4
        assert get_workflow_review_limit(object(), "ga_plan_review_limit") == 2
        assert get_workflow_review_limit(object(), "dev_plan_review_limit", default=9) == 9

    def test_env_override_requirements_skills_root(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_WORKFLOW_REQUIREMENTS_SKILLS_ROOT", "/tmp/skills")
        settings = WorkflowSettings()
        assert settings.requirements_skills_root == "/tmp/skills"


# ---------------------------------------------------------------------------
# CompactionSettings
# ---------------------------------------------------------------------------

class TestCompactionSettings:
    """Tests for CompactionSettings defaults and env-var overrides."""

    def test_defaults(self):
        settings = CompactionSettings()
        assert settings.threshold_turns == 50
        assert settings.compaction_backend == "codex"

    def test_threshold_turns_default_is_50(self):
        settings = CompactionSettings()
        assert settings.threshold_turns == 50

    def test_env_override_threshold_turns(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_COMPACTION_THRESHOLD_TURNS", "500")
        settings = CompactionSettings()
        assert settings.threshold_turns == 500

    def test_env_override_compaction_backend(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_COMPACTION_COMPACTION_BACKEND", "kimi")
        settings = CompactionSettings()
        assert settings.compaction_backend == "kimi"


# ---------------------------------------------------------------------------
# EmbeddingSettings
# ---------------------------------------------------------------------------

class TestEmbeddingSettings:
    """Tests for EmbeddingSettings defaults and env-var overrides."""

    def test_defaults(self):
        settings = EmbeddingSettings()
        assert settings.backend == "auto"
        assert settings.tidb_model == "built-in"

    def test_env_override_backend(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_EMBEDDING_BACKEND", "tidb")
        settings = EmbeddingSettings()
        assert settings.backend == "tidb"

    def test_env_override_tidb_model(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_EMBEDDING_TIDB_MODEL", "text-embedding-3-small")
        settings = EmbeddingSettings()
        assert settings.tidb_model == "text-embedding-3-small"

    def test_env_disable_tidb_fallback(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_EMBEDDING_TIDB_MODEL", "")
        settings = EmbeddingSettings()
        assert settings.tidb_model == ""


# ---------------------------------------------------------------------------
# MemoryWorkerSettings
# ---------------------------------------------------------------------------

class TestMemoryWorkerSettings:
    """Tests for MemoryWorkerSettings defaults and env-var overrides."""

    def test_defaults(self):
        settings = MemoryWorkerSettings()
        assert settings.backend == "codex"
        assert settings.model == "gpt-5.4"
        assert settings.role_name == "_memory_worker"
        assert settings.async_enabled is True
        assert settings.trigger_mode == "events_only"

    def test_env_override_backend(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_MEMORY_WORKER_BACKEND", "kimi")
        settings = MemoryWorkerSettings()
        assert settings.backend == "kimi"

    def test_env_override_async_enabled(self, monkeypatch):
        monkeypatch.setenv("MYSWAT_MEMORY_WORKER_ASYNC_ENABLED", "false")
        settings = MemoryWorkerSettings()
        assert settings.async_enabled is False


# ---------------------------------------------------------------------------
# MySwatSettings
# ---------------------------------------------------------------------------

class TestMySwatSettings:
    """Tests for the top-level MySwatSettings aggregation."""

    def test_defaults_no_config_file(self, monkeypatch, tmp_path):
        """With no TOML file present the settings should still load with defaults."""
        nonexistent = tmp_path / "does_not_exist.toml"
        monkeypatch.setenv("MYSWAT_CONFIG_PATH", str(nonexistent))
        settings = MySwatSettings(config_path=nonexistent)

        # Sub-settings are populated with their own defaults
        assert isinstance(settings.tidb, TiDBSettings)
        assert isinstance(settings.agents, AgentSettings)
        assert isinstance(settings.workflow, WorkflowSettings)
        assert isinstance(settings.compaction, CompactionSettings)
        assert isinstance(settings.memory_worker, MemoryWorkerSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)

        # Spot-check a few nested defaults
        assert settings.tidb.port == 4000
        assert settings.agents.developer_model == "gpt-5.4"
        assert settings.workflow.design_plan_review_limit == 10
        assert settings.workflow.ga_plan_review_limit == 2
        assert settings.compaction.threshold_turns == 50

    def test_missing_config_file_no_error(self, tmp_path):
        """A missing TOML config file must not raise an error."""
        missing = tmp_path / "nonexistent" / "config.toml"
        settings = MySwatSettings(config_path=missing)
        # Should still have valid defaults
        assert settings.tidb.database == "myswat"

    def test_toml_loading(self, tmp_path):
        """Values from a TOML file should be loaded into settings."""
        toml_content = """\
[tidb]
host = "toml-host.example.com"
port = 3307
user = "toml_user"
password = "toml_password"
database = "toml_db"

[agents]
developer_model = "toml-model-v1"
architect_model = "toml-model-v2"

[workflow]
design_plan_review_limit = 8
dev_plan_review_limit = 7
dev_code_review_limit = 6
ga_plan_review_limit = 5
ga_test_review_limit = 4

[compaction]
threshold_turns = 300
compaction_backend = "kimi"

[memory_worker]
backend = "claude"
model = "claude-opus-4-6"
async_enabled = false
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        settings = MySwatSettings(config_path=config_file)

        assert settings.tidb.host == "toml-host.example.com"
        assert settings.tidb.port == 3307
        assert settings.tidb.user == "toml_user"
        assert settings.tidb.password == "toml_password"
        assert settings.tidb.database == "toml_db"
        assert settings.agents.developer_model == "toml-model-v1"
        assert settings.agents.architect_model == "toml-model-v2"
        assert settings.workflow.design_plan_review_limit == 8
        assert settings.workflow.dev_plan_review_limit == 7
        assert settings.workflow.dev_code_review_limit == 6
        assert settings.workflow.ga_plan_review_limit == 5
        assert settings.workflow.ga_test_review_limit == 4
        assert settings.compaction.threshold_turns == 300
        assert settings.compaction.compaction_backend == "kimi"
        assert settings.memory_worker.backend == "claude"
        assert settings.memory_worker.model == "claude-opus-4-6"
        assert settings.memory_worker.async_enabled is False

    def test_toml_values_present_when_loaded(self, tmp_path):
        """TOML values are loaded for nested settings sections."""
        toml_content = """\
[tidb]
host = "toml-host.example.com"
port = 3307
database = "toml_db"

[agents]
developer_model = "toml-model"

[workflow]
dev_code_review_limit = 8

[compaction]
threshold_turns = 300
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        settings = MySwatSettings(config_path=config_file)

        # TOML values loaded into nested settings
        assert settings.tidb.host == "toml-host.example.com"
        assert settings.tidb.port == 3307
        assert settings.tidb.database == "toml_db"
        assert settings.agents.developer_model == "toml-model"
        assert settings.workflow.dev_code_review_limit == 8
        assert settings.compaction.threshold_turns == 300

    def test_legacy_threshold_tokens_in_toml_is_ignored(self, tmp_path):
        """Old configs with threshold_tokens should still load cleanly."""
        toml_content = """\
[compaction]
threshold_turns = 300
threshold_tokens = 500000
compaction_backend = "kimi"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        settings = MySwatSettings(config_path=config_file)

        assert settings.compaction.threshold_turns == 300
        assert settings.compaction.compaction_backend == "kimi"

    def test_partial_toml(self, tmp_path):
        """A TOML file with only some sections should merge with defaults."""
        toml_content = """\
[tidb]
host = "partial-host"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        settings = MySwatSettings(config_path=config_file)

        # Overridden from TOML
        assert settings.tidb.host == "partial-host"
        # Defaults preserved
        assert settings.tidb.port == 4000
        assert settings.agents.developer_model == "gpt-5.4"
        assert settings.workflow.design_plan_review_limit == 10
        assert settings.workflow.ga_test_review_limit == 2
        assert settings.compaction.threshold_turns == 50

    def test_config_path_attribute(self, tmp_path):
        """config_path should be stored as a Path object."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        settings = MySwatSettings(config_path=config_file)
        assert isinstance(settings.config_path, Path)
        assert settings.config_path == config_file

    def test_empty_toml_file(self, tmp_path):
        """An empty TOML file should be handled gracefully, using all defaults."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        settings = MySwatSettings(config_path=config_file)

        assert settings.tidb.host == ""
        assert settings.tidb.port == 4000
        assert settings.agents.codex_path == "codex"
        assert settings.workflow.design_plan_review_limit == 10
        assert settings.workflow.ga_plan_review_limit == 2
        assert settings.compaction.threshold_turns == 50
