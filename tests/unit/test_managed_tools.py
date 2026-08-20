"""ManagedToolsClient contract tests; all HTTP is mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from identark.managed_tools import ManagedToolsClient, ManagedToolsError


def response(status: int, body: dict[str, Any]) -> MagicMock:
    value = MagicMock()
    value.status_code = status
    value.json.return_value = body
    return value


def execution(status: str = "pending_approval") -> dict[str, Any]:
    return {
        "execution_id": "exec-1",
        "connector_id": "connector-1",
        "tool_name": "update_customer",
        "status": status,
        "risk_score": 67,
        "risk_level": "medium",
        "approval_id": "approval-1",
        "result": None,
        "error": None,
        "created_at": "2026-08-20T00:00:00Z",
        "updated_at": "2026-08-20T00:00:00Z",
        "executed_at": None,
    }


async def test_create_execution_sends_only_capability_request() -> None:
    client = ManagedToolsClient(
        api_key="csk-test",
        url="https://api.identark.io/v1",
        agent_id="customer-agent",
    )
    client._client.request = AsyncMock(return_value=response(202, execution()))

    result = await client.create_execution(
        connector_id="connector-1",
        tool_name="update_customer",
        arguments={"customer_id": "cus-1", "updates": {"status": "inactive"}},
        idempotency_key="customer-request-1",
    )

    assert result.status == "pending_approval"  # nosec B101
    kwargs = client._client.request.call_args.kwargs
    assert kwargs["headers"]["Idempotency-Key"] == "customer-request-1"  # nosec B101
    assert kwargs["headers"]["X-IdentArk-Agent-Id"] == "customer-agent"  # nosec B101
    assert "password" not in str(kwargs["json"])  # nosec B101


async def test_structured_error_is_preserved() -> None:
    client = ManagedToolsClient(api_key="csk-test", url="https://api.identark.io/v1")
    client._client.request = AsyncMock(
        return_value=response(
            409,
            {"detail": {"code": "execution_in_progress", "message": "Already running"}},
        )
    )

    with pytest.raises(ManagedToolsError) as raised:
        await client.resume_execution("exec-1")

    assert raised.value.status_code == 409  # nosec B101
    assert raised.value.error_code == "execution_in_progress"  # nosec B101
