"""Tests for the HTTP MCP client transport."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from myswat.server.mcp_http_client import MCPHTTPClient, MCPHTTPClientError


class _FakeResponse:
    def __init__(self, payload: dict | str) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self._payload, str):
            return self._payload.encode("utf-8")
        return json.dumps(self._payload).encode("utf-8")


def _http_error(payload: dict[str, object], *, code: int = 500) -> HTTPError:
    body = io.BytesIO(json.dumps(payload).encode("utf-8"))
    return HTTPError(
        url="http://127.0.0.1:8765/mcp",
        code=code,
        msg="server error",
        hdrs=None,
        fp=body,
    )


def test_call_tool_returns_structured_content_and_posts_json(monkeypatch):
    captured = {}

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"structuredContent": {"runtime_registration_id": 7}},
            }
        )

    monkeypatch.setattr("myswat.server.mcp_http_client.urlopen", _fake_urlopen)
    client = MCPHTTPClient("http://127.0.0.1:8765", timeout_seconds=9)

    result = client.call_tool("register_runtime", {"project_id": 1})

    assert result == {"runtime_registration_id": 7}
    assert captured["url"] == "http://127.0.0.1:8765/mcp"
    assert captured["timeout"] == 9
    assert captured["body"]["params"]["name"] == "register_runtime"
    assert captured["body"]["params"]["arguments"] == {"project_id": 1}


def test_call_tool_normalizes_missing_structured_content(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.mcp_http_client.urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"structuredContent": None},
            }
        ),
    )
    client = MCPHTTPClient("http://127.0.0.1:8765")

    assert client.call_tool("heartbeat_runtime", {"runtime_registration_id": 1}) == {}


def test_call_tool_raises_for_transport_errors(monkeypatch):
    def _boom(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("myswat.server.mcp_http_client.urlopen", _boom)
    client = MCPHTTPClient("http://127.0.0.1:8765")

    with pytest.raises(MCPHTTPClientError, match="MCP endpoint is unavailable"):
        client.call_tool("ping", {})


def test_call_tool_raises_for_transport_timeouts(monkeypatch):
    def _boom(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("myswat.server.mcp_http_client.urlopen", _boom)
    client = MCPHTTPClient("http://127.0.0.1:8765", timeout_seconds=9)

    with pytest.raises(MCPHTTPClientError, match="timed out after 9s"):
        client.call_tool("send_chat_message", {})


def test_call_tool_raises_for_http_errors(monkeypatch):
    def _boom(request, timeout):
        raise _http_error({"error": {"message": "tool failed hard"}}, code=502)

    monkeypatch.setattr(
        "myswat.server.mcp_http_client.urlopen",
        _boom,
    )
    client = MCPHTTPClient("http://127.0.0.1:8765")

    with pytest.raises(
        MCPHTTPClientError,
        match=r"MCP endpoint returned HTTP 502.*tool failed hard",
    ):
        client.call_tool("send_chat_message", {})


def test_call_tool_raises_for_jsonrpc_error(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.mcp_http_client.urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"message": "tool failed"},
            }
        ),
    )
    client = MCPHTTPClient("http://127.0.0.1:8765")

    with pytest.raises(MCPHTTPClientError, match="tool failed"):
        client.call_tool("complete_stage_task", {})


def test_call_tool_raises_for_non_mapping_result(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.mcp_http_client.urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": ["not", "a", "dict"],
            }
        ),
    )
    client = MCPHTTPClient("http://127.0.0.1:8765")

    with pytest.raises(MCPHTTPClientError, match="Invalid MCP response"):
        client.call_tool("claim_next_assignment", {})


def test_call_tool_raises_for_invalid_json_response(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.mcp_http_client.urlopen",
        lambda request, timeout: _FakeResponse("{not json"),
    )
    client = MCPHTTPClient("http://127.0.0.1:8765")

    with pytest.raises(MCPHTTPClientError, match="Invalid MCP response"):
        client.call_tool("claim_next_assignment", {})


def test_call_tool_raises_for_non_mapping_structured_content(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.mcp_http_client.urlopen",
        lambda request, timeout: _FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"structuredContent": ["bad"]},
            }
        ),
    )
    client = MCPHTTPClient("http://127.0.0.1:8765")

    with pytest.raises(MCPHTTPClientError, match="Missing structuredContent"):
        client.call_tool("claim_next_assignment", {})
