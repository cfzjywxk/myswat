"""Minimal HTTP JSON-RPC client for the local MySwat MCP endpoint."""

from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MCPHTTPClientError(RuntimeError):
    """Raised when the HTTP MCP endpoint cannot be reached or returns an error."""


class MCPHTTPClient:
    def __init__(self, server_url: str, timeout_seconds: int = 30) -> None:
        self._endpoint = server_url.rstrip("/") + "/mcp"
        self._timeout = max(1, int(timeout_seconds))
        self._request_id = 0

    def call_tool(self, name: str, arguments: dict) -> dict:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }
        request = Request(
            url=self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise MCPHTTPClientError(
                f"MCP endpoint returned HTTP {exc.code} for {name}: {body[:200] or exc.reason}"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise MCPHTTPClientError(
                f"MCP request timed out after {self._timeout}s: {name}"
            ) from exc
        except URLError as exc:
            raise MCPHTTPClientError(
                f"MCP endpoint is unavailable at {self._endpoint}: {exc}"
            ) from exc
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise MCPHTTPClientError(f"Invalid MCP response: {body[:200]}") from exc
        if isinstance(parsed, dict) and parsed.get("error"):
            raise MCPHTTPClientError(
                str(parsed["error"].get("message") or parsed["error"])
            )
        result = parsed.get("result", {}) if isinstance(parsed, dict) else {}
        if not isinstance(result, dict):
            raise MCPHTTPClientError(f"Invalid MCP response: {result!r}")
        structured = result.get("structuredContent")
        if structured is None:
            return {}
        if not isinstance(structured, dict):
            raise MCPHTTPClientError(
                f"Missing structuredContent in MCP response: {result!r}"
            )
        return structured
