"""Shared test fixtures for MySwat."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from myswat.agents.base import AgentResponse


@pytest.fixture
def mock_pool():
    """A mocked TiDBPool — all queries return empty/0 by default."""
    pool = MagicMock()
    pool.fetch_one.return_value = None
    pool.fetch_all.return_value = []
    pool.execute.return_value = 0
    pool.insert_returning_id.return_value = 1
    return pool


@pytest.fixture
def mock_store(mock_pool):
    """A MemoryStore backed by a mocked pool."""
    from myswat.memory.store import MemoryStore

    return MemoryStore(mock_pool)


@pytest.fixture
def mock_runner():
    """A mocked AgentRunner that returns success."""
    runner = MagicMock()
    runner.workdir = "/tmp/test-repo"
    runner.is_session_started = False
    runner.cli_session_id = None
    runner.invoke.return_value = AgentResponse(
        content="test response", exit_code=0,
    )
    return runner


@pytest.fixture
def fake_agent_row():
    """A realistic agent DB row."""
    return {
        "id": 1,
        "project_id": 1,
        "role": "developer",
        "display_name": "Dev (GPT-5.4)",
        "cli_backend": "codex",
        "model_name": "gpt-5.4",
        "cli_path": "codex",
        "cli_extra_args": None,
        "system_prompt": "You are a developer.",
    }


@pytest.fixture
def fake_project():
    """A realistic project DB row."""
    return {
        "id": 1,
        "slug": "test-project",
        "name": "Test Project",
        "description": "A test project",
        "repo_path": "/tmp/test-repo",
    }


@pytest.fixture
def success_response():
    return AgentResponse(content="done", exit_code=0)


@pytest.fixture
def failure_response():
    return AgentResponse(content="error occurred", exit_code=1)


def make_fake_session_manager(agent_id=1, agent_role="developer",
                               responses=None, session_id=1):
    """Create a fake SessionManager for workflow tests."""
    sm = MagicMock()
    sm.agent_id = agent_id
    sm.agent_role = agent_role
    sm.session = SimpleNamespace(id=session_id)

    if responses is not None:
        side_effects = []
        for r in responses:
            if isinstance(r, str):
                side_effects.append(AgentResponse(content=r, exit_code=0))
            else:
                side_effects.append(r)
        sm.send.side_effect = side_effects
    else:
        sm.send.return_value = AgentResponse(content="ok", exit_code=0)

    return sm
