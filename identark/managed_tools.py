"""Client for IdentArk-managed MCP/database tools.

This client is intentionally separate from AgentGateway: managed tool
execution is an asynchronous capability workflow, not an LLM gateway method.
"""

# ruff: noqa: ANN101, ANN102

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from identark.exceptions import ConfigurationError, ControlPlaneError

TERMINAL_STATUSES = {"succeeded", "failed", "rejected", "timeout", "cancelled"}


class ManagedToolsError(ControlPlaneError):
    """A managed MCP/database execution request was rejected or failed."""


@dataclass(frozen=True)
class ManagedExecution:
    execution_id: str
    connector_id: str
    tool_name: str
    status: str
    risk_score: int | None
    risk_level: str | None
    approval_id: str | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: str
    updated_at: str
    executed_at: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManagedExecution:
        return cls(
            execution_id=str(data["execution_id"]),
            connector_id=str(data["connector_id"]),
            tool_name=str(data["tool_name"]),
            status=str(data["status"]),
            risk_score=int(data["risk_score"]) if data.get("risk_score") is not None else None,
            risk_level=str(data["risk_level"]) if data.get("risk_level") is not None else None,
            approval_id=str(data["approval_id"]) if data.get("approval_id") is not None else None,
            result=data.get("result") if isinstance(data.get("result"), dict) else None,
            error=data.get("error") if isinstance(data.get("error"), dict) else None,
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            executed_at=str(data["executed_at"]) if data.get("executed_at") is not None else None,
        )


class ManagedToolsClient:
    """Invoke managed tools without receiving the provider credential."""

    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        agent_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("IDENTARK_SESSION_TOKEN")
            or os.environ.get("IDENTARK_API_KEY")
        )
        resolved_url = url or os.environ.get("IDENTARK_CONTROL_PLANE_URL")
        if not resolved_key:
            raise ConfigurationError(
                "Provide api_key or set IDENTARK_API_KEY/IDENTARK_SESSION_TOKEN"
            )
        if not resolved_url:
            raise ConfigurationError("Provide url or set IDENTARK_CONTROL_PLANE_URL")
        self._agent_id = agent_id or os.environ.get("IDENTARK_AGENT_ID")
        self._client = httpx.AsyncClient(
            base_url=resolved_url.rstrip("/"),
            headers={"Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def __aenter__(self) -> ManagedToolsClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def list_tools(self, connector_id: str) -> list[dict[str, Any]]:
        data = await self._request("GET", f"/mcp/managed-databases/{connector_id}/tools")
        tools = data.get("tools")
        return list(tools) if isinstance(tools, list) else []

    async def create_execution(
        self,
        *,
        connector_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        idempotency_key: str,
    ) -> ManagedExecution:
        headers = {"Idempotency-Key": idempotency_key}
        if self._agent_id:
            headers["X-IdentArk-Agent-Id"] = self._agent_id
        data = await self._request(
            "POST",
            "/mcp/managed-databases/executions",
            headers=headers,
            json={"connector_id": connector_id, "tool_name": tool_name, "arguments": arguments},
        )
        return ManagedExecution.from_dict(data)

    async def get_execution(self, execution_id: str) -> ManagedExecution:
        data = await self._request("GET", f"/mcp/managed-databases/executions/{execution_id}")
        return ManagedExecution.from_dict(data)

    async def resume_execution(self, execution_id: str) -> ManagedExecution:
        data = await self._request(
            "POST", f"/mcp/managed-databases/executions/{execution_id}/resume"
        )
        return ManagedExecution.from_dict(data)

    async def cancel_execution(self, execution_id: str) -> ManagedExecution:
        data = await self._request(
            "POST", f"/mcp/managed-databases/executions/{execution_id}/cancel"
        )
        return ManagedExecution.from_dict(data)

    async def wait_for_execution(
        self,
        execution_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> ManagedExecution:
        deadline = time.monotonic() + timeout
        current = await self.get_execution(execution_id)
        while current.status not in TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for managed execution {execution_id}")
            await asyncio.sleep(poll_interval)
            current = await self.resume_execution(execution_id)
        return current

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            raise ManagedToolsError("Unable to reach the IdentArk managed tools API") from exc
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            detail = body.get("detail", body)
            if isinstance(detail, dict):
                code = str(detail.get("code", "managed_tools_error"))
                message = str(detail.get("message", "Managed tool request failed"))
            else:
                code = "managed_tools_error"
                message = str(detail or "Managed tool request failed")
            raise ManagedToolsError(message, status_code=response.status_code, error_code=code)
        result = response.json()
        if not isinstance(result, dict):
            raise ManagedToolsError("Managed tools API returned an invalid response")
        return result
