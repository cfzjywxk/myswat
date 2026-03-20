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


def test_request_wraps_timeout_as_retryable_client_error(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    client = DaemonClient()

    with pytest.raises(DaemonClientError, match="timed out after") as exc_info:
        client.get_work_item(project="fib-demo", work_item_id=41)

    assert exc_info.value.retryable is True


def test_cleanup_project_uses_extended_timeout(monkeypatch):
    observed: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def _urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("myswat.server.control_client.urlopen", _urlopen)
    client = DaemonClient()

    result = client.cleanup_project(project="fib-demo")

    assert result == {"ok": True}
    assert observed["url"].endswith("/api/project-cleanup")
    assert observed["timeout"] == 300


def test_submit_work_includes_skip_ga_test_when_requested(monkeypatch):
    observed: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def _urlopen(request, timeout):
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("myswat.server.control_client.urlopen", _urlopen)
    client = DaemonClient()

    result = client.submit_work(
        project="fib-demo",
        requirement="implement fibonacci",
        workdir=None,
        mode="full",
        skip_ga_test=True,
    )

    assert result == {"ok": True}
    assert observed["payload"]["skip_ga_test"] is True
