"""Tests for the stdio MCP dispatcher and framing helpers."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from myswat.server.contracts import ReviewVerdictEnvelope
from myswat.server.mcp_stdio import (
    MySwatMCPDispatcher,
    _read_message,
    _write_message,
    dispatch_rpc_request,
    serve_stdio,
)


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _parse_frames(data: bytes) -> list[dict]:
    messages: list[dict] = []
    cursor = 0
    while cursor < len(data):
        header_end = data.index(b"\r\n\r\n", cursor)
        header_blob = data[cursor:header_end].decode("ascii")
        headers = {}
        for line in header_blob.split("\r\n"):
            key, _, value = line.partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers["content-length"])
        body_start = header_end + 4
        body_end = body_start + length
        messages.append(json.loads(data[body_start:body_end].decode("utf-8")))
        cursor = body_end
    return messages


def test_list_tools_includes_assignment_tools():
    service = Mock()
    dispatcher = MySwatMCPDispatcher(service)

    result = dispatcher.list_tools()
    names = {tool["name"] for tool in result["tools"]}

    assert "register_runtime" in names
    assert "claim_next_assignment" in names
    assert "complete_stage_task" in names
    assert "renew_stage_run_lease" in names
    assert "renew_review_cycle_lease" in names
    assert "append_coordination_event" in names
    assert "fail_review_cycle" in names
    assert "cancel_review_cycles" in names
    assert "open_chat_session" in names
    assert "send_chat_message" in names
    assert "reset_chat_session" in names
    assert "close_chat_session" in names


def test_call_tool_validates_and_dispatches():
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 7}
    )
    dispatcher = MySwatMCPDispatcher(service)

    result = dispatcher.call_tool(
        "register_runtime",
        {
            "project_id": 1,
            "runtime_name": "codex-daemon",
            "runtime_kind": "mcp",
            "agent_role": "developer",
        },
    )

    assert result["structuredContent"]["runtime_registration_id"] == 7
    service.register_runtime.assert_called_once()


def test_call_tool_normalizes_none_result_to_empty_structured_content():
    service = Mock()
    service.heartbeat_runtime.return_value = None
    dispatcher = MySwatMCPDispatcher(service)

    result = dispatcher.call_tool(
        "heartbeat_runtime",
        {
            "runtime_registration_id": 1,
        },
    )

    assert result["structuredContent"] == {}
    service.heartbeat_runtime.assert_called_once()


def test_call_tool_dispatches_stage_lease_renewal():
    service = Mock()
    service.renew_stage_run_lease.return_value = None
    dispatcher = MySwatMCPDispatcher(service)

    result = dispatcher.call_tool(
        "renew_stage_run_lease",
        {
            "stage_run_id": 41,
            "runtime_registration_id": 7,
            "lease_seconds": 120,
        },
    )

    assert result["structuredContent"] == {}
    service.renew_stage_run_lease.assert_called_once()


def test_call_tool_serializes_lists_of_pydantic_models():
    service = Mock()
    service.wait_for_review_verdicts.return_value = [
        ReviewVerdictEnvelope(
            cycle_id=9,
            reviewer_role="qa_main",
            verdict="lgtm",
            summary="Looks good.",
        )
    ]
    dispatcher = MySwatMCPDispatcher(service)

    result = dispatcher.call_tool(
        "wait_for_review_verdicts",
        {
            "cycle_ids": [9],
        },
    )

    assert result["structuredContent"] == [
        {
            "cycle_id": 9,
            "reviewer_role": "qa_main",
            "verdict": "lgtm",
            "issues": [],
            "summary": "Looks good.",
        }
    ]


def test_call_tool_rejects_unknown_tool():
    dispatcher = MySwatMCPDispatcher(Mock())

    with pytest.raises(ValueError, match="Unknown tool: nope"):
        dispatcher.call_tool("nope", {})


def test_dispatch_rpc_request_treats_initialized_as_notification():
    dispatcher = MySwatMCPDispatcher(Mock())

    result = dispatch_rpc_request(dispatcher, "initialized", {})

    assert result is None


def test_read_message_parses_length_prefixed_payload():
    stream = io.BytesIO(_frame({"jsonrpc": "2.0", "id": 1, "method": "ping"}))

    result = _read_message(stream)

    assert result == {"jsonrpc": "2.0", "id": 1, "method": "ping"}


def test_read_message_returns_none_for_missing_content_length():
    stream = io.BytesIO(b"X-Test: 1\r\n\r\n")

    assert _read_message(stream) is None


def test_read_message_treats_blank_decoded_header_line_as_terminator():
    stream = io.BytesIO(b"Content-Length: 0\r\n   \r\n")

    assert _read_message(stream) is None


class _ShortReadStream(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return b""


def test_read_message_returns_none_when_payload_is_truncated():
    stream = _ShortReadStream(b"Content-Length: 10\r\n\r\n")

    assert _read_message(stream) is None


def test_write_message_emits_content_length_frame():
    output = io.BytesIO()

    _write_message(output, {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

    messages = _parse_frames(output.getvalue())
    assert messages == [{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}]


def test_serve_stdio_handles_initialize_list_call_and_ping(monkeypatch):
    service = Mock()
    service.register_runtime.return_value = SimpleNamespace(
        model_dump=lambda: {"runtime_registration_id": 41}
    )
    stdin = io.BytesIO(
        b"".join(
            [
                _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                _frame({"jsonrpc": "2.0", "method": "initialized", "params": {}}),
                _frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "register_runtime",
                            "arguments": {
                                "project_id": 1,
                                "runtime_name": "codex-daemon",
                                "runtime_kind": "managed_worker",
                                "agent_role": "developer",
                            },
                        },
                    }
                ),
                _frame({"jsonrpc": "2.0", "id": 4, "method": "ping", "params": {}}),
            ]
        )
    )
    stdout = io.BytesIO()
    fake_sys = SimpleNamespace(
        stdin=SimpleNamespace(buffer=stdin),
        stdout=SimpleNamespace(buffer=stdout),
    )
    monkeypatch.setattr("myswat.server.mcp_stdio.sys", fake_sys)

    result = serve_stdio(service)

    assert result == 0
    messages = _parse_frames(stdout.getvalue())
    assert [message["id"] for message in messages] == [1, 2, 3, 4]
    assert messages[0]["result"]["serverInfo"]["name"] == "myswat"
    assert "tools" in messages[1]["result"]
    assert messages[2]["result"]["structuredContent"]["runtime_registration_id"] == 41
    assert messages[3]["result"] == {}


def test_serve_stdio_emits_jsonrpc_errors_and_skips_notification_errors(monkeypatch):
    service = Mock()
    stdin = io.BytesIO(
        b"".join(
            [
                _frame({"jsonrpc": "2.0", "method": "totally-unknown", "params": {}}),
                _frame({"jsonrpc": "2.0", "id": 6, "method": "totally-unknown", "params": {}}),
                _frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "tools/call",
                        "params": {"name": "nope", "arguments": {}},
                    }
                ),
            ]
        )
    )
    stdout = io.BytesIO()
    fake_sys = SimpleNamespace(
        stdin=SimpleNamespace(buffer=stdin),
        stdout=SimpleNamespace(buffer=stdout),
    )
    monkeypatch.setattr("myswat.server.mcp_stdio.sys", fake_sys)

    result = serve_stdio(service)

    assert result == 0
    messages = _parse_frames(stdout.getvalue())
    assert [message["id"] for message in messages] == [6, 7]
    assert messages[0]["error"]["message"] == "Unsupported MCP method: totally-unknown"
    assert messages[1]["error"]["message"] == "Unknown tool: nope"


def test_serve_stdio_skips_success_notifications(monkeypatch):
    service = Mock()
    stdin = io.BytesIO(
        b"".join(
            [
                _frame({"jsonrpc": "2.0", "method": "ping", "params": {}}),
                _frame({"jsonrpc": "2.0", "id": 8, "method": "ping", "params": {}}),
            ]
        )
    )
    stdout = io.BytesIO()
    fake_sys = SimpleNamespace(
        stdin=SimpleNamespace(buffer=stdin),
        stdout=SimpleNamespace(buffer=stdout),
    )
    monkeypatch.setattr("myswat.server.mcp_stdio.sys", fake_sys)

    assert serve_stdio(service) == 0
    assert _parse_frames(stdout.getvalue()) == [{"jsonrpc": "2.0", "id": 8, "result": {}}]
