"""Tests for the daemon control HTTP client."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from myswat.server.control_client import DaemonClient, DaemonClientError


def _http_error(payload: dict[str, object], *, code: int = 400) -> HTTPError:
    body = io.BytesIO(json.dumps(payload).encode("utf-8"))
    return HTTPError(
        url="http://127.0.0.1:8765/api/work",
        code=code,
        msg="bad request",
        hdrs=None,
        fp=body,
    )


def test_request_surfaces_http_error_payload_message(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(_http_error({"error": "project not found"})),
    )
    client = DaemonClient()

    with pytest.raises(DaemonClientError, match="project not found"):
        client.submit_work(
            project="missing",
            requirement="do work",
            workdir=None,
            mode="full",
        )


def test_request_surfaces_nested_http_error_message(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(_http_error({"error": {"message": "internal server error"}}, code=500)),
    )
    client = DaemonClient()

    with pytest.raises(DaemonClientError, match="internal server error"):
        client.cleanup_project(project="fib-demo")
