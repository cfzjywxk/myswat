"""Additional coverage-focused tests for myswat.cli.chat_cmd."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit

from myswat.agents.base import AgentResponse
from myswat.workflow.modes import WorkMode
from myswat.cli.chat_cmd import (
    _extract_delegation,
    _run_full_workflow,
    _run_full_workflow_interactive,
    _run_design_review_interactive,
    _run_inline_review,
    _run_inline_review_interactive,
    _run_testplan_review,
    _run_testplan_review_interactive,
    _run_workflow,
    _run_workflow_interactive,
    _show_status,
    _show_task_details,
)


def _agent_row(role: str = "developer", backend: str = "codex") -> dict:
    return {
        "id": 1,
        "role": role,
        "display_name": f"Agent-{role}",
        "cli_backend": backend,
        "model_name": "gpt-5",
        "cli_path": backend,
        "cli_extra_args": None,
    }


def _proj() -> dict:
    return {"id": 1, "slug": "proj", "name": "Proj", "repo_path": "/tmp"}


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.workflow.max_review_iterations = 3
    settings.embedding.tidb_model = "built-in"
    settings.compaction.threshold_turns = 200
    return settings


def test_show_status_handles_non_dict_task_state_and_process_log():
    store = MagicMock()
    store.list_work_items.return_value = [
        {
            "id": 1,
            "status": "in_progress",
            "title": "Implement auth",
            "metadata_json": {"task_state": "bad"},
        },
        {
            "id": 2,
            "status": "review",
            "title": "Add tests",
            "metadata_json": {
                "task_state": {
                    "current_stage": "phase_2",
                    "process_log": [
                        {"type": "task_request", "summary": "start"},
                        "skip-me",
                    ],
                }
            },
        },
    ]

    with patch("myswat.cli.chat_cmd.console.print") as mock_print:
        _show_status(store, MagicMock(), _proj())

    rendered = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "Work Item #2 Flow" in rendered


def test_show_task_details_no_items_prints_empty_state():
    store = MagicMock()
    store.list_work_items.return_value = []

    with patch("myswat.cli.chat_cmd.console.print") as mock_print:
        _show_task_details(store, _proj())

    assert any("No work items yet." in str(call) for call in mock_print.call_args_list)


def test_show_task_details_renders_state_artifacts_and_missing_item():
    store = MagicMock()
    store.list_work_items.return_value = [
        {"id": 8, "status": "in_progress", "title": "Implement auth"},
    ]
    store.get_work_item.side_effect = [
        None,
        {"id": 8, "status": "in_progress", "item_type": "code_change", "title": "Implement auth"},
    ]
    store.get_work_item_state.return_value = {
        "current_stage": "phase_2",
        "latest_summary": "Added auth flow",
        "next_todos": ["Add tests"],
        "open_issues": ["Missing retry path"],
        "process_log": [
            {"type": "review_request", "summary": "please review"},
            "skip-me",
        ],
    }
    store.list_artifacts.return_value = [
        {"iteration": 1, "artifact_type": "proposal", "title": "Plan v1"},
    ]

    with patch("myswat.cli.chat_cmd.console.print") as mock_print:
        _show_task_details(store, _proj(), 7)
        _show_task_details(store, _proj())

    rendered = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "Work item 7 not found." in rendered
    assert "Stage:" in rendered
    assert "Next TODOs:" in rendered
    assert "Open Issues:" in rendered
    assert "Process Log:" in rendered


def test_extract_delegation_skips_blank_lines_in_fallback_block():
    text = "```delegate\nplain task line\n\nmore detail\n```"

    assert _extract_delegation(text) == ("plain task line\nmore detail", "develop")


def test_extract_delegation_fallback_ignores_mode_and_empty_task_lines():
    text = "```delegate\nMODE: testplan\nTASK:\nactual task line\n```"

    assert _extract_delegation(text) == ("actual task line", "testplan")


def test_show_task_details_selects_first_item_and_normalizes_non_dict_state():
    store = MagicMock()
    store.list_work_items.return_value = [
        {"id": 9, "status": "completed", "title": "Done already"},
    ]
    store.get_work_item.return_value = {
        "id": 9,
        "status": "completed",
        "item_type": "code_change",
        "title": "Done already",
    }
    store.get_work_item_state.return_value = "bad-state"
    store.list_artifacts.return_value = []

    _show_task_details(store, _proj())

    store.get_work_item.assert_called_once_with(9)
    store.get_work_item_state.assert_called_once_with(9)


@patch("myswat.cli.chat_cmd._run_workflow")
def test_run_inline_review_forwards_remote_workflow_args(mock_run_workflow):
    created = []

    _run_inline_review(
        MagicMock(),
        _proj(),
        "/tmp",
        _settings(),
        "do the work",
        should_cancel=lambda: True,
        on_work_item_created=created.append,
        register_cancel_target=MagicMock(),
    )

    kwargs = mock_run_workflow.call_args.kwargs
    assert kwargs["mode"] == WorkMode.develop
    assert kwargs["requirement"] == "do the work"
    assert kwargs["should_cancel"]()
    kwargs["on_work_item_created"](42)
    assert created == [42]


@patch("myswat.cli.chat_cmd._run_with_task_monitor")
@patch("myswat.cli.chat_cmd._run_workflow")
def test_inline_review_interactive_executes_worker_callback(mock_run_workflow, mock_monitor):
    def _run_monitor(**kwargs):
        kwargs["worker_fn"]()
        assert kwargs["work_item_ref"]["id"] == 42
        assert kwargs["cancel_targets"] == []

    def _run_workflow_side_effect(**kwargs):
        kwargs["on_work_item_created"](42)
        assert kwargs["mode"] == WorkMode.develop
        assert kwargs["should_cancel"]() is False

    mock_monitor.side_effect = _run_monitor
    mock_run_workflow.side_effect = _run_workflow_side_effect

    _run_inline_review_interactive(
        store=MagicMock(),
        proj=_proj(),
        workdir="/tmp",
        settings=_settings(),
        task="ship it",
    )


@patch("myswat.cli.chat_cmd._run_workflow")
def test_run_testplan_review_forwards_callbacks(mock_run_workflow):
    proposer_sm = MagicMock()
    created = []

    _run_testplan_review(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=proposer_sm,
        workdir="/tmp",
        settings=_settings(),
        task="finalize qa plan",
        should_cancel=lambda: False,
        on_work_item_created=created.append,
        register_cancel_target=MagicMock(),
        auto_approve=False,
    )

    kwargs = mock_run_workflow.call_args.kwargs
    assert kwargs["mode"] == WorkMode.testplan_design
    assert kwargs["proposer_sm"] is proposer_sm
    assert kwargs["requirement"] == "finalize qa plan"
    assert kwargs["should_cancel"]() is False
    kwargs["on_work_item_created"](55)
    assert created == [55]


@patch("myswat.cli.chat_cmd.console.print")
def test_run_testplan_review_requires_qa_session(mock_print):
    _run_testplan_review(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=None,
        workdir="/tmp",
        settings=_settings(),
        task="finalize qa plan",
    )

    assert any("Missing QA session." in str(call) for call in mock_print.call_args_list)


@patch("myswat.cli.chat_cmd._run_with_task_monitor")
@patch("myswat.cli.chat_cmd._run_workflow")
def test_testplan_review_interactive_executes_worker_callback(mock_run_workflow, mock_monitor):
    proposer_sm = MagicMock()

    def _run_monitor(**kwargs):
        kwargs["worker_fn"]()
        assert kwargs["work_item_ref"]["id"] == 55
        assert kwargs["cancel_targets"] == []

    def _run_workflow_side_effect(**kwargs):
        kwargs["on_work_item_created"](55)
        assert kwargs["mode"] == WorkMode.testplan_design
        assert kwargs["proposer_sm"] is proposer_sm

    mock_monitor.side_effect = _run_monitor
    mock_run_workflow.side_effect = _run_workflow_side_effect

    _run_testplan_review_interactive(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=proposer_sm,
        workdir="/tmp",
        settings=_settings(),
        task="plan tests",
        prompt_session=MagicMock(),
    )


@patch("myswat.cli.chat_cmd.console.print")
@patch("myswat.cli.chat_cmd._run_remote_workflow")
def test_run_workflow_requires_architect_for_architect_design_mode(mock_run_remote, mock_print):
    store = MagicMock()
    dev_agent = _agent_row("developer")
    qa_agent = _agent_row("qa_main", "kimi")
    store.get_agent.side_effect = lambda _pid, role: {
        "developer": dev_agent,
        "qa_main": qa_agent,
        "architect": None,
    }.get(role)

    _run_workflow(
        store=store,
        proj=_proj(),
        workdir="/tmp",
        settings=_settings(),
        requirement="design auth",
        mode=WorkMode.architect_design,
    )

    mock_run_remote.assert_not_called()
    assert any("Missing architect agent." in str(call) for call in mock_print.call_args_list)


@patch("myswat.cli.chat_cmd._run_remote_workflow")
def test_run_workflow_forwards_created_callback_and_mode(mock_run_remote):
    store = MagicMock()
    arch_agent = _agent_row("architect")
    dev_agent = _agent_row("developer")
    qa_main = _agent_row("qa_main", "kimi")
    qa_vice = _agent_row("qa_vice", "kimi")
    store.get_agent.side_effect = lambda _pid, role: {
        "architect": arch_agent,
        "developer": dev_agent,
        "qa_main": qa_main,
        "qa_vice": qa_vice,
    }.get(role)
    created = []

    _run_workflow(
        store=store,
        proj=_proj(),
        workdir="/tmp",
        settings=_settings(),
        requirement="ship auth",
        mode=WorkMode.full,
        on_work_item_created=created.append,
        register_cancel_target=MagicMock(),
    )

    kwargs = mock_run_remote.call_args.kwargs
    assert kwargs["mode"] == WorkMode.full
    assert kwargs["requirement"] == "ship auth"
    kwargs["on_work_item_created"](99)
    assert created == [99]


@patch("myswat.cli.chat_cmd._run_remote_workflow")
def test_run_workflow_allows_proposer_for_public_mode_without_local_forking(mock_run_remote):
    store = MagicMock()
    dev_agent = _agent_row("developer")
    qa_main = _agent_row("qa_main", "kimi")
    store.get_agent.side_effect = lambda _pid, role: {
        "developer": dev_agent,
        "qa_main": qa_main,
        "qa_vice": None,
    }.get(role)
    proposer_sm = MagicMock()

    _run_workflow(
        store=store,
        proj=_proj(),
        workdir="/tmp",
        settings=_settings(),
        requirement="ship auth",
        mode=WorkMode.develop,
        proposer_sm=proposer_sm,
        should_cancel=lambda: True,
    )

    kwargs = mock_run_remote.call_args.kwargs
    assert kwargs["mode"] == WorkMode.develop
    assert kwargs["should_cancel"]()


@patch("myswat.cli.chat_cmd._run_remote_workflow")
def test_run_workflow_testplan_mode_requires_architect_but_not_qa(mock_run_remote):
    store = MagicMock()
    arch_agent = _agent_row("architect")
    dev_agent = _agent_row("developer")
    store.get_agent.side_effect = lambda _pid, role: {
        "architect": arch_agent,
        "developer": dev_agent,
        "qa_main": None,
        "qa_vice": None,
    }.get(role)

    _run_workflow(
        store=store,
        proj=_proj(),
        workdir="/tmp",
        settings=_settings(),
        requirement="plan tests",
        mode=WorkMode.testplan_design,
        proposer_sm=MagicMock(),
    )

    assert mock_run_remote.call_args.kwargs["mode"] == WorkMode.testplan_design


@patch("myswat.cli.chat_cmd.console.print")
def test_run_design_review_interactive_requires_architect_session(mock_print):
    _run_design_review_interactive(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=None,
        workdir="/tmp",
        settings=_settings(),
        task="design thing",
    )

    assert any("Missing architect session." in str(call) for call in mock_print.call_args_list)


@patch("myswat.cli.chat_cmd.console.print")
def test_run_testplan_review_interactive_requires_qa_session(mock_print):
    _run_testplan_review_interactive(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=None,
        workdir="/tmp",
        settings=_settings(),
        task="plan tests",
    )

    assert any("Missing QA session." in str(call) for call in mock_print.call_args_list)


@patch("myswat.cli.chat_cmd._run_with_task_monitor")
@patch("myswat.cli.chat_cmd._run_workflow")
def test_workflow_interactive_executes_worker_callback(mock_run_workflow, mock_monitor):
    def _run_monitor(**kwargs):
        kwargs["worker_fn"]()
        assert kwargs["work_item_ref"]["id"] == 66
        assert kwargs["cancel_targets"] == []

    def _run_workflow_side_effect(**kwargs):
        kwargs["on_work_item_created"](66)
        assert kwargs["mode"] == WorkMode.full

    mock_monitor.side_effect = _run_monitor
    mock_run_workflow.side_effect = _run_workflow_side_effect

    _run_workflow_interactive(
        store=MagicMock(),
        proj=_proj(),
        workdir="/tmp",
        settings=_settings(),
        requirement="do work",
    )


@patch("myswat.cli.chat_cmd._run_workflow")
def test_run_full_workflow_forwards_full_mode(mock_run_workflow):
    _run_full_workflow(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=MagicMock(),
        workdir="/tmp",
        settings=_settings(),
        requirement="do work",
    )

    assert mock_run_workflow.call_args.kwargs["mode"] == WorkMode.full


@patch("myswat.cli.chat_cmd._run_workflow_interactive")
def test_run_full_workflow_interactive_forwards_label(mock_run_workflow_interactive):
    _run_full_workflow_interactive(
        store=MagicMock(),
        proj=_proj(),
        proposer_sm=MagicMock(),
        workdir="/tmp",
        settings=_settings(),
        requirement="do work",
    )

    assert mock_run_workflow_interactive.call_args.kwargs["label"] == "Running architect-led full workflow"

@patch("myswat.cli.chat_cmd.console.print")
@patch("myswat.cli.chat_cmd._build_prompt_session")
@patch("myswat.cli.chat_cmd.preload_model")
@patch("myswat.cli.chat_cmd.MySwatSettings")
@patch("myswat.cli.chat_cmd.TiDBPool")
@patch("myswat.cli.chat_cmd.ensure_schema")
@patch("myswat.cli.chat_cmd.MemoryStore")
@patch("myswat.cli.chat_cmd.SessionManager")
def test_run_chat_missing_initial_agent_exits(
    mock_sm_cls,
    mock_store_cls,
    mock_mig,
    mock_pool_cls,
    mock_settings_cls,
    mock_preload,
    mock_build_prompt,
    mock_print,
):
    from myswat.cli.chat_cmd import run_chat

    settings = _settings()
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = _proj()
    store.get_agent.return_value = None
    mock_store_cls.return_value = store

    with pytest.raises(ClickExit):
        run_chat("proj")

    assert any("Agent role 'developer' not found." in str(call) for call in mock_print.call_args_list)


@patch("myswat.cli.chat_cmd._build_prompt_session")
@patch("myswat.cli.chat_cmd.preload_model")
@patch("myswat.cli.chat_cmd.MySwatSettings")
@patch("myswat.cli.chat_cmd.TiDBPool")
@patch("myswat.cli.chat_cmd.ensure_schema")
@patch("myswat.cli.chat_cmd.MemoryStore")
@patch("myswat.cli.chat_cmd.SessionManager")
def test_run_chat_history_without_active_session(
    mock_sm_cls,
    mock_store_cls,
    mock_mig,
    mock_pool_cls,
    mock_settings_cls,
    mock_preload,
    mock_build_prompt,
):
    from myswat.cli.chat_cmd import run_chat

    settings = _settings()
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = _proj()
    store.get_agent.return_value = _agent_row()
    store.list_agents.return_value = [_agent_row()]
    mock_store_cls.return_value = store
    sm = MagicMock()
    sm.session = SimpleNamespace(session_uuid="uuid-1234")
    mock_sm_cls.return_value = sm
    prompt_session = MagicMock()
    def _prompt(_text, multiline=False):
        if sm.session is not None:
            sm.session = None
            return "/history"
        return "/quit"
    prompt_session.prompt.side_effect = _prompt
    mock_build_prompt.return_value = prompt_session

    with patch("myswat.cli.chat_cmd.console.print") as mock_print:
        run_chat("proj")

    assert any("No active session." in str(call) for call in mock_print.call_args_list)


@patch("myswat.cli.chat_cmd._send_with_timer")
@patch("myswat.cli.chat_cmd._build_prompt_session")
@patch("myswat.cli.chat_cmd.preload_model")
@patch("myswat.cli.chat_cmd.MySwatSettings")
@patch("myswat.cli.chat_cmd.TiDBPool")
@patch("myswat.cli.chat_cmd.ensure_schema")
@patch("myswat.cli.chat_cmd.MemoryStore")
@patch("myswat.cli.chat_cmd.SessionManager")
def test_run_chat_new_then_missing_agent_skips_message_send(
    mock_sm_cls,
    mock_store_cls,
    mock_mig,
    mock_pool_cls,
    mock_settings_cls,
    mock_preload,
    mock_build_prompt,
    mock_send_with_timer,
):
    from myswat.cli.chat_cmd import run_chat

    settings = _settings()
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = _proj()
    store.get_agent.side_effect = [_agent_row(), None, None]
    store.list_agents.return_value = [_agent_row()]
    mock_store_cls.return_value = store
    sm = MagicMock()
    sm.session = SimpleNamespace(session_uuid="uuid-1234")
    mock_sm_cls.return_value = sm
    prompt_session = MagicMock()
    prompt_session.prompt.side_effect = ["/new", "hello", "/quit"]
    mock_build_prompt.return_value = prompt_session

    run_chat("proj")

    mock_send_with_timer.assert_not_called()


@patch("myswat.cli.chat_cmd.submit_chat_learn_request", side_effect=RuntimeError("learn failed"))
@patch("myswat.cli.chat_cmd._send_with_timer")
@patch("myswat.cli.chat_cmd._build_prompt_session")
@patch("myswat.cli.chat_cmd.preload_model")
@patch("myswat.cli.chat_cmd.MySwatSettings")
@patch("myswat.cli.chat_cmd.TiDBPool")
@patch("myswat.cli.chat_cmd.ensure_schema")
@patch("myswat.cli.chat_cmd.MemoryStore")
@patch("myswat.cli.chat_cmd.SessionManager")
def test_run_chat_chat_learn_failure_is_best_effort(
    mock_sm_cls,
    mock_store_cls,
    mock_mig,
    mock_pool_cls,
    mock_settings_cls,
    mock_preload,
    mock_build_prompt,
    mock_send_with_timer,
    mock_submit,
):
    from myswat.cli.chat_cmd import run_chat

    settings = _settings()
    mock_settings_cls.return_value = settings
    store = MagicMock()
    store.get_project_by_slug.return_value = _proj()
    store.get_agent.return_value = _agent_row()
    store.list_agents.return_value = [_agent_row()]
    mock_store_cls.return_value = store
    sm = MagicMock()
    sm.session = SimpleNamespace(session_uuid="uuid-1234", id=3)
    mock_sm_cls.return_value = sm
    prompt_session = MagicMock()
    prompt_session.prompt.side_effect = ["hello", "/quit"]
    mock_build_prompt.return_value = prompt_session
    mock_send_with_timer.return_value = (AgentResponse(content="done", exit_code=0), 1.0)

    with patch("sys.stderr") as fake_stderr:
        run_chat("proj")

    mock_submit.assert_called_once()
    assert fake_stderr.write.called


@patch("myswat.cli.chat_cmd.console.print")
@patch("myswat.cli.chat_cmd._send_with_timer")
@patch("myswat.cli.chat_cmd._build_prompt_session")
@patch("myswat.cli.chat_cmd.preload_model")
@patch("myswat.cli.chat_cmd.MySwatSettings")
@patch("myswat.cli.chat_cmd.TiDBPool")
@patch("myswat.cli.chat_cmd.ensure_schema")
@patch("myswat.cli.chat_cmd.MemoryStore")
@patch("myswat.cli.chat_cmd.SessionManager")
def test_run_chat_warns_on_misconfigured_delegation_handler(
    mock_sm_cls,
    mock_store_cls,
    mock_mig,
    mock_pool_cls,
    mock_settings_cls,
    mock_preload,
    mock_build_prompt,
    mock_send_with_timer,
    mock_print,
):
    from myswat.cli.chat_cmd import run_chat
    from myswat.cli.chat_cmd import DELEGATION_MODE_SPECS

    settings = _settings()
    mock_settings_cls.return_value = settings
    architect = _agent_row("architect")
    store = MagicMock()
    store.get_project_by_slug.return_value = _proj()
    store.get_agent.return_value = architect
    store.list_agents.return_value = [architect]
    mock_store_cls.return_value = store
    sm = MagicMock()
    sm.session = SimpleNamespace(session_uuid="uuid-1234")
    mock_sm_cls.return_value = sm
    prompt_session = MagicMock()
    prompt_session.prompt.side_effect = ["hello", "/quit"]
    mock_build_prompt.return_value = prompt_session
    mock_send_with_timer.return_value = (
        AgentResponse(
            content="```delegate\nMODE: develop\nTASK: do the thing\n```",
            exit_code=0,
        ),
        1.0,
    )
    original = DELEGATION_MODE_SPECS["develop"]
    misconfigured = replace(original, chat_handler="missing_handler")

    with patch.dict("myswat.cli.chat_cmd.DELEGATION_MODE_SPECS", {"develop": misconfigured}, clear=False):
        run_chat("proj", role="architect")

    assert any("misconfigured" in str(call) for call in mock_print.call_args_list)
