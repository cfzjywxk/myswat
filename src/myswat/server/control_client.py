"""HTTP client helpers for the local MySwat daemon."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from myswat.config.settings import MySwatSettings


class DaemonClientError(RuntimeError):
    """Raised when the local daemon cannot be reached or returns an error."""


class DaemonClient:
    def __init__(self, settings: MySwatSettings | None = None) -> None:
        self._settings = settings or MySwatSettings()
        self._base_url = f"http://{self._settings.server.host}:{self._settings.server.port}"
        self._timeout = max(1, int(self._settings.server.request_timeout_seconds))

    @property
    def base_url(self) -> str:
        return self._base_url

    def _request(self, *, method: str, path: str, payload: dict | None = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            url=self._base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except URLError as exc:
            raise DaemonClientError(f"MySwat daemon is unavailable at {self._base_url}: {exc}") from exc
        if not body:
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DaemonClientError(f"Invalid daemon response: {body[:200]}") from exc
        if response.status >= 400:
            raise DaemonClientError(str(parsed.get("error") or parsed))
        if isinstance(parsed, dict) and parsed.get("error"):
            raise DaemonClientError(str(parsed["error"]))
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    def health(self) -> dict:
        return self._request(method="GET", path="/api/health")

    def init_project(self, *, name: str, repo_path: str | None, description: str | None) -> dict:
        return self._request(
            method="POST",
            path="/api/init",
            payload={
                "name": name,
                "repo_path": repo_path,
                "description": description,
            },
        )

    def submit_work(
        self,
        *,
        project: str,
        requirement: str,
        workdir: str | None,
        mode: str,
    ) -> dict:
        return self._request(
            method="POST",
            path="/api/work",
            payload={
                "project": project,
                "requirement": requirement,
                "workdir": workdir,
                "mode": mode,
            },
        )

    def get_work_item(self, *, project: str, work_item_id: int) -> dict:
        return self._request(
            method="POST",
            path="/api/work-item",
            payload={
                "project": project,
                "work_item_id": work_item_id,
            },
        )

    def control_work(self, *, project: str, work_item_id: int, action: str) -> dict:
        return self._request(
            method="POST",
            path="/api/work-control",
            payload={
                "project": project,
                "work_item_id": work_item_id,
                "action": action,
            },
        )

    def cleanup_project(self, *, project: str) -> dict:
        return self._request(
            method="POST",
            path="/api/project-cleanup",
            payload={
                "project": project,
            },
        )
