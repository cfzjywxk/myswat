"""Comprehensive tests for myswat models: Agent and Project."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from myswat.models.agent import Agent
from myswat.models.project import Project


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TestAgent:
    """Tests for the Agent model."""

    def test_required_fields_only(self):
        """Creating an Agent with only required fields should succeed."""
        agent = Agent(
            project_id=1,
            role="developer",
            display_name="Dev Agent",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
        )
        assert agent.project_id == 1
        assert agent.role == "developer"
        assert agent.display_name == "Dev Agent"
        assert agent.cli_backend == "codex"
        assert agent.model_name == "gpt-4"
        assert agent.cli_path == "/usr/bin/codex"

    def test_default_values(self):
        """All optional fields should default to None."""
        agent = Agent(
            project_id=1,
            role="architect",
            display_name="Arch",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
        )
        assert agent.id is None
        assert agent.cli_extra_args is None
        assert agent.system_prompt is None
        assert agent.created_at is None

    def test_all_fields_set(self):
        """Setting every field explicitly should work."""
        now = datetime.now()
        agent = Agent(
            id=42,
            project_id=7,
            role="qa_main",
            display_name="QA Main Agent",
            cli_backend="kimi",
            model_name="kimi-v1",
            cli_path="/opt/kimi/cli",
            cli_extra_args=["--verbose", "--timeout=60"],
            system_prompt="You are a QA agent.",
            created_at=now,
        )
        assert agent.id == 42
        assert agent.project_id == 7
        assert agent.role == "qa_main"
        assert agent.display_name == "QA Main Agent"
        assert agent.cli_backend == "kimi"
        assert agent.model_name == "kimi-v1"
        assert agent.cli_path == "/opt/kimi/cli"
        assert agent.cli_extra_args == ["--verbose", "--timeout=60"]
        assert agent.system_prompt == "You are a QA agent."
        assert agent.created_at == now

    def test_missing_project_id_raises(self):
        """Omitting project_id should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Agent(
                role="developer",
                display_name="Dev",
                cli_backend="codex",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
            )

    def test_missing_role_raises(self):
        """Omitting role should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                display_name="Dev",
                cli_backend="codex",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
            )

    def test_missing_display_name_raises(self):
        """Omitting display_name should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                role="developer",
                cli_backend="codex",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
            )

    def test_missing_cli_backend_raises(self):
        """Omitting cli_backend should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                role="developer",
                display_name="Dev",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
            )

    def test_missing_model_name_raises(self):
        """Omitting model_name should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                role="developer",
                display_name="Dev",
                cli_backend="codex",
                cli_path="/usr/bin/codex",
            )

    def test_missing_cli_path_raises(self):
        """Omitting cli_path should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                role="developer",
                display_name="Dev",
                cli_backend="codex",
                model_name="gpt-4",
            )

    def test_invalid_project_id_type_raises(self):
        """Passing a non-coercible type for project_id should raise."""
        with pytest.raises(ValidationError):
            Agent(
                project_id="not-an-int",
                role="developer",
                display_name="Dev",
                cli_backend="codex",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
            )

    def test_invalid_cli_extra_args_type_raises(self):
        """Passing a non-list type for cli_extra_args should raise."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                role="developer",
                display_name="Dev",
                cli_backend="codex",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
                cli_extra_args="not-a-list",
            )

    def test_invalid_created_at_type_raises(self):
        """Passing a non-datetime-parseable value for created_at should raise."""
        with pytest.raises(ValidationError):
            Agent(
                project_id=1,
                role="developer",
                display_name="Dev",
                cli_backend="codex",
                model_name="gpt-4",
                cli_path="/usr/bin/codex",
                created_at="not-a-datetime",
            )

    @pytest.mark.parametrize(
        "role",
        ["architect", "developer", "qa_main", "qa_vice"],
    )
    def test_role_values_are_accepted(self, role: str):
        """All documented role strings should be accepted."""
        agent = Agent(
            project_id=1,
            role=role,
            display_name="Agent",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
        )
        assert agent.role == role

    @pytest.mark.parametrize("backend", ["codex", "kimi", "claude"])
    def test_cli_backend_values_are_accepted(self, backend: str):
        """All documented cli_backend values should be accepted."""
        agent = Agent(
            project_id=1,
            role="developer",
            display_name="Agent",
            cli_backend=backend,
            model_name="model",
            cli_path="/usr/bin/cli",
        )
        assert agent.cli_backend == backend

    def test_cli_extra_args_empty_list(self):
        """An empty list for cli_extra_args should be accepted."""
        agent = Agent(
            project_id=1,
            role="developer",
            display_name="Dev",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
            cli_extra_args=[],
        )
        assert agent.cli_extra_args == []

    def test_cli_extra_args_list_parsing(self):
        """cli_extra_args should accept a list of strings."""
        args = ["--flag", "-o", "value"]
        agent = Agent(
            project_id=1,
            role="developer",
            display_name="Dev",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
            cli_extra_args=args,
        )
        assert agent.cli_extra_args == ["--flag", "-o", "value"]
        assert len(agent.cli_extra_args) == 3

    def test_created_at_datetime_string_parsing(self):
        """A valid ISO datetime string should be parsed into a datetime."""
        agent = Agent(
            project_id=1,
            role="developer",
            display_name="Dev",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
            created_at="2025-01-15T10:30:00",
        )
        assert isinstance(agent.created_at, datetime)
        assert agent.created_at.year == 2025
        assert agent.created_at.month == 1
        assert agent.created_at.day == 15

    def test_model_dump_required_fields_only(self):
        """model_dump should include all fields with defaults for optionals."""
        agent = Agent(
            project_id=1,
            role="developer",
            display_name="Dev",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
        )
        dumped = agent.model_dump()
        assert dumped == {
            "id": None,
            "project_id": 1,
            "role": "developer",
            "display_name": "Dev",
            "cli_backend": "codex",
            "model_name": "gpt-4",
            "cli_path": "/usr/bin/codex",
            "cli_extra_args": None,
            "system_prompt": None,
            "created_at": None,
        }

    def test_model_dump_all_fields(self):
        """model_dump should reflect all explicitly set values."""
        now = datetime(2025, 6, 15, 12, 0, 0)
        agent = Agent(
            id=10,
            project_id=2,
            role="qa_vice",
            display_name="QA Vice",
            cli_backend="kimi",
            model_name="kimi-v2",
            cli_path="/opt/kimi",
            cli_extra_args=["--debug"],
            system_prompt="Test all the things.",
            created_at=now,
        )
        dumped = agent.model_dump()
        assert dumped["id"] == 10
        assert dumped["project_id"] == 2
        assert dumped["role"] == "qa_vice"
        assert dumped["display_name"] == "QA Vice"
        assert dumped["cli_backend"] == "kimi"
        assert dumped["model_name"] == "kimi-v2"
        assert dumped["cli_path"] == "/opt/kimi"
        assert dumped["cli_extra_args"] == ["--debug"]
        assert dumped["system_prompt"] == "Test all the things."
        assert dumped["created_at"] == now

    def test_model_dump_roundtrip(self):
        """An Agent created from model_dump output should have equal fields."""
        now = datetime.now()
        original = Agent(
            id=5,
            project_id=3,
            role="architect",
            display_name="Architect",
            cli_backend="codex",
            model_name="gpt-4",
            cli_path="/usr/bin/codex",
            cli_extra_args=["--fast"],
            system_prompt="Design systems.",
            created_at=now,
        )
        reconstructed = Agent(**original.model_dump())
        assert reconstructed.id == original.id
        assert reconstructed.project_id == original.project_id
        assert reconstructed.role == original.role
        assert reconstructed.display_name == original.display_name
        assert reconstructed.cli_backend == original.cli_backend
        assert reconstructed.model_name == original.model_name
        assert reconstructed.cli_path == original.cli_path
        assert reconstructed.cli_extra_args == original.cli_extra_args
        assert reconstructed.system_prompt == original.system_prompt
        assert reconstructed.created_at == original.created_at


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class TestProject:
    """Tests for the Project model."""

    def test_required_fields_only(self):
        """Creating a Project with only required fields should succeed."""
        project = Project(slug="my-project", name="My Project")
        assert project.slug == "my-project"
        assert project.name == "My Project"

    def test_default_values(self):
        """All optional fields should default to None."""
        project = Project(slug="proj", name="Proj")
        assert project.id is None
        assert project.description is None
        assert project.repo_path is None
        assert project.config_json is None
        assert project.created_at is None
        assert project.updated_at is None

    def test_all_fields_set(self):
        """Setting every field explicitly should work."""
        now = datetime.now()
        project = Project(
            id=10,
            slug="full-project",
            name="Full Project",
            description="A fully configured project.",
            repo_path="/home/user/repos/full-project",
            config_json={"ci": True, "coverage_threshold": 80},
            created_at=now,
            updated_at=now,
        )
        assert project.id == 10
        assert project.slug == "full-project"
        assert project.name == "Full Project"
        assert project.description == "A fully configured project."
        assert project.repo_path == "/home/user/repos/full-project"
        assert project.config_json == {"ci": True, "coverage_threshold": 80}
        assert project.created_at == now
        assert project.updated_at == now

    def test_missing_slug_raises(self):
        """Omitting slug should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Project(name="My Project")

    def test_missing_name_raises(self):
        """Omitting name should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Project(slug="my-project")

    def test_invalid_id_type_raises(self):
        """Passing a non-coercible type for id should raise."""
        with pytest.raises(ValidationError):
            Project(id="not-an-int", slug="proj", name="Proj")

    def test_invalid_config_json_type_raises(self):
        """Passing a non-dict type for config_json should raise."""
        with pytest.raises(ValidationError):
            Project(slug="proj", name="Proj", config_json="not-a-dict")

    def test_invalid_created_at_type_raises(self):
        """Passing a non-datetime-parseable value for created_at should raise."""
        with pytest.raises(ValidationError):
            Project(slug="proj", name="Proj", created_at="not-a-datetime")

    def test_invalid_updated_at_type_raises(self):
        """Passing a non-datetime-parseable value for updated_at should raise."""
        with pytest.raises(ValidationError):
            Project(slug="proj", name="Proj", updated_at="not-a-datetime")

    def test_config_json_empty_dict(self):
        """An empty dict for config_json should be accepted."""
        project = Project(slug="proj", name="Proj", config_json={})
        assert project.config_json == {}

    def test_config_json_nested_dict(self):
        """config_json should accept nested dictionaries."""
        config = {
            "deploy": {
                "env": "production",
                "replicas": 3,
                "features": ["auth", "logging"],
            },
            "version": 2,
        }
        project = Project(slug="proj", name="Proj", config_json=config)
        assert project.config_json == config
        assert project.config_json["deploy"]["replicas"] == 3
        assert project.config_json["deploy"]["features"] == ["auth", "logging"]

    def test_config_json_mixed_value_types(self):
        """config_json should accept dicts with mixed value types (Any)."""
        config = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "null_val": None,
            "list_val": [1, "two", 3.0],
        }
        project = Project(slug="proj", name="Proj", config_json=config)
        assert project.config_json == config

    def test_created_at_datetime_string_parsing(self):
        """A valid ISO datetime string should be parsed into a datetime."""
        project = Project(
            slug="proj",
            name="Proj",
            created_at="2025-03-20T14:45:00",
        )
        assert isinstance(project.created_at, datetime)
        assert project.created_at.year == 2025
        assert project.created_at.month == 3
        assert project.created_at.day == 20

    def test_updated_at_datetime_string_parsing(self):
        """A valid ISO datetime string should be parsed for updated_at."""
        project = Project(
            slug="proj",
            name="Proj",
            updated_at="2025-06-01T08:00:00",
        )
        assert isinstance(project.updated_at, datetime)
        assert project.updated_at.year == 2025

    def test_model_dump_required_fields_only(self):
        """model_dump should include all fields with defaults for optionals."""
        project = Project(slug="test-proj", name="Test Project")
        dumped = project.model_dump()
        assert dumped == {
            "id": None,
            "slug": "test-proj",
            "name": "Test Project",
            "description": None,
            "repo_path": None,
            "config_json": None,
            "created_at": None,
            "updated_at": None,
        }

    def test_model_dump_all_fields(self):
        """model_dump should reflect all explicitly set values."""
        now = datetime(2025, 7, 1, 9, 0, 0)
        project = Project(
            id=5,
            slug="dumped",
            name="Dumped Project",
            description="For testing dumps.",
            repo_path="/tmp/repo",
            config_json={"key": "value"},
            created_at=now,
            updated_at=now,
        )
        dumped = project.model_dump()
        assert dumped["id"] == 5
        assert dumped["slug"] == "dumped"
        assert dumped["name"] == "Dumped Project"
        assert dumped["description"] == "For testing dumps."
        assert dumped["repo_path"] == "/tmp/repo"
        assert dumped["config_json"] == {"key": "value"}
        assert dumped["created_at"] == now
        assert dumped["updated_at"] == now

    def test_model_dump_roundtrip(self):
        """A Project created from model_dump output should have equal fields."""
        now = datetime.now()
        original = Project(
            id=8,
            slug="roundtrip",
            name="Roundtrip Project",
            description="Testing roundtrip.",
            repo_path="/home/user/repo",
            config_json={"nested": {"a": 1}},
            created_at=now,
            updated_at=now,
        )
        reconstructed = Project(**original.model_dump())
        assert reconstructed.id == original.id
        assert reconstructed.slug == original.slug
        assert reconstructed.name == original.name
        assert reconstructed.description == original.description
        assert reconstructed.repo_path == original.repo_path
        assert reconstructed.config_json == original.config_json
        assert reconstructed.created_at == original.created_at
        assert reconstructed.updated_at == original.updated_at
