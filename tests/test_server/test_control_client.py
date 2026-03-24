"""Tests for the daemon control HTTP client."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

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


class _Response:
    def __init__(self, body: bytes | str = b"", *, status: int = 200) -> None:
        self.status = status
        self._body = body.encode("utf-8") if isinstance(body, str) else body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


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

    with pytest.raises(DaemonClientError, match="MySwat daemon request timed out:") as exc_info:
        client.get_work_item(project="fib-demo", work_item_id=41)

    assert exc_info.value.retryable is True


def test_cleanup_project_is_timeoutless(monkeypatch):
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
    assert observed["timeout"] is None


def test_submit_work_includes_with_ga_test_when_requested(monkeypatch):
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
        with_ga_test=True,
    )

    assert result == {"ok": True}
    assert observed["payload"]["with_ga_test"] is True


def test_parse_error_body_variants():
    assert DaemonClient._parse_error_body("") is None
    assert DaemonClient._parse_error_body("plain text") == "plain text"
    assert DaemonClient._parse_error_body(json.dumps({"error": {"message": "nested"}})) == "nested"
    assert DaemonClient._parse_error_body(json.dumps({"error": "simple"})) == "simple"
    assert "detail" in DaemonClient._parse_error_body(json.dumps({"detail": "missing"}))
    assert DaemonClient._parse_error_body(json.dumps(["bad", "payload"])) == "['bad', 'payload']"


def test_health_uses_get_and_returns_empty_mapping_for_empty_body(monkeypatch):
    observed: dict[str, object] = {}

    def _urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        return _Response(b"")

    monkeypatch.setattr("myswat.server.control_client.urlopen", _urlopen)
    client = DaemonClient()

    assert client.health() == {}
    assert observed["url"] == client.base_url + "/api/health"
    assert observed["method"] == "GET"
    assert observed["timeout"] is None


def test_request_wraps_url_error_as_retryable_client_error(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("connection refused")),
    )
    client = DaemonClient()

    with pytest.raises(DaemonClientError, match="unavailable") as exc_info:
        client.health()

    assert exc_info.value.retryable is True


def test_request_rejects_invalid_json_body(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: _Response("not-json"),
    )
    client = DaemonClient()

    with pytest.raises(DaemonClientError, match="Invalid daemon response"):
        client.health()


def test_request_rejects_embedded_error_payload(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: _Response('{"error": "still running"}'),
    )
    client = DaemonClient()

    with pytest.raises(DaemonClientError, match="still running"):
        client.health()


def test_request_wraps_non_mapping_json_payload(monkeypatch):
    monkeypatch.setattr(
        "myswat.server.control_client.urlopen",
        lambda request, timeout: _Response("7"),
    )
    client = DaemonClient()

    assert client.health() == {"result": 7}


def test_daemon_client_ignores_request_timeout_setting(monkeypatch):
    observed: dict[str, object] = {}

    def _urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _Response('{"ok": true}')

    monkeypatch.setattr("myswat.server.control_client.urlopen", _urlopen)
    settings = SimpleNamespace(
        server=SimpleNamespace(
            host="127.0.0.1",
            port=8765,
            request_timeout_seconds=1,
        )
    )
    client = DaemonClient(settings)

    assert client.health() == {"ok": True}
    assert observed["url"] == client.base_url + "/api/health"
    assert observed["timeout"] is None


def test_init_project_and_control_work_send_expected_payloads_without_timeout(monkeypatch):
    observed: list[dict[str, object]] = []

    def _urlopen(request, timeout):
        observed.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _Response('{"ok": true}')

    monkeypatch.setattr("myswat.server.control_client.urlopen", _urlopen)
    client = DaemonClient()

    assert client.init_project(name="Proj", repo_path="/tmp/repo", description="demo") == {"ok": True}
    assert client.control_work(project="proj", work_item_id=41, action="cancel") == {"ok": True}

    assert observed == [
        {
            "url": client.base_url + "/api/init",
            "payload": {
                "name": "Proj",
                "repo_path": "/tmp/repo",
                "description": "demo",
            },
            "timeout": None,
        },
        {
            "url": client.base_url + "/api/work-control",
            "payload": {
                "project": "proj",
                "work_item_id": 41,
                "action": "cancel",
            },
            "timeout": None,
        },
    ]
