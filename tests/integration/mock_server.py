"""
Mock control plane server for testing.

Mocks httpx responses using unittest.mock to simulate the control plane API
without requiring a real server or external dependencies like respx.

Usage::

    from tests.integration.mock_server import MockControlPlane

    mock_cp = MockControlPlane()
    mock_cp.inject_error("invoke_llm", status=500)

    with mock_cp.mocked() as mock_client:
        gateway = ControlPlaneGateway(api_key="test", url="...")
        # httpx calls will be intercepted by mock_cp
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


class MockControlPlane:
    """
    Mock control plane API server.

    Intercepts httpx calls and returns configured responses.
    Can inject latency or errors on specific endpoints.
    """

    def __init__(self) -> None:
        self._base_url = "https://api.identark.io/v1"
        self._latency = 0.0
        self._error_overrides: dict[str, tuple[int, dict[str, Any]]] = {}
        self._session_costs: dict[str, float] = {}
        self._request_count = 0

    def set_latency(self, seconds: float) -> None:
        """Set artificial latency for all requests."""
        self._latency = seconds

    def inject_error(
        self,
        endpoint: str,
        status: int,
        body: dict[str, Any] | None = None,
    ) -> None:
        """Inject an error response for a specific endpoint."""
        if body is None:
            body = {"error_code": "test_error", "message": "Injected error"}
        self._error_overrides[endpoint] = (status, body)

    def clear_error(self, endpoint: str) -> None:
        """Clear an injected error."""
        self._error_overrides.pop(endpoint, None)

    def set_session_cost(self, session_id: str, cost: float) -> None:
        """Set the accumulated cost for a session."""
        self._session_costs[session_id] = cost

    def request_count(self) -> int:
        """Number of requests processed."""
        return self._request_count

    async def _apply_latency(self) -> None:
        """Apply configured latency."""
        if self._latency > 0:
            await asyncio.sleep(self._latency)

    def _mock_response(self, status: int, json_data: dict[str, Any] | None = None,
                       text: str | None = None) -> MagicMock:
        """Create a mock httpx Response."""
        resp = AsyncMock()
        resp.status_code = status
        if json_data is not None:
            resp.json = AsyncMock(return_value=json_data)
            resp.text = json.dumps(json_data)
        else:
            resp.json = AsyncMock(return_value={})
            resp.text = text or ""
        return resp

    async def _invoke_llm_handler(self) -> MagicMock:
        """Handle POST /llm/invoke."""
        self._request_count += 1
        await self._apply_latency()

        # Check for error injection
        if "invoke_llm" in self._error_overrides:
            status, body = self._error_overrides["invoke_llm"]
            return self._mock_response(status, body)

        return self._mock_response(
            200,
            {
                "message": {
                    "role": "assistant",
                    "content": "Mock LLM response",
                },
                "cost_usd": 0.0012,
                "model": "gpt-4o",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "cached_tokens": 0,
                },
                "session_id": "sess-" + str(uuid.uuid4())[:8],
            },
        )

    async def _invoke_llm_stream_handler(self) -> MagicMock:
        """Handle POST /llm/stream (SSE)."""
        self._request_count += 1
        await self._apply_latency()

        # Check for error injection
        if "invoke_llm_stream" in self._error_overrides:
            status, body = self._error_overrides["invoke_llm_stream"]
            return self._mock_response(status, body)

        # Stream response as SSE
        chunks = [
            "data: " + json.dumps({"content": "Hello ", "finish_reason": None, "model": "gpt-4o"}),
            "data: " + json.dumps({"content": "world!", "finish_reason": None, "model": "gpt-4o"}),
            "data: " + json.dumps({
                "content": "",
                "finish_reason": "stop",
                "model": "gpt-4o",
                "input_tokens": 10,
                "output_tokens": 20,
            }),
            "data: [DONE]",
        ]
        sse_text = "\n".join(chunks) + "\n"

        resp = AsyncMock()
        resp.status_code = 200

        # Mock async iteration for streaming
        async def aiter_lines_impl() -> Any:
            for line in sse_text.split("\n"):
                if line:
                    yield line

        resp.aiter_lines = aiter_lines_impl

        # Mock context manager
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=None)
        resp.aread = AsyncMock()

        return resp

    async def _persist_messages_handler(self) -> MagicMock:
        """Handle POST /messages/persist."""
        self._request_count += 1
        await self._apply_latency()

        if "persist_messages" in self._error_overrides:
            status, body = self._error_overrides["persist_messages"]
            return self._mock_response(status, body)

        return self._mock_response(200, {"success": True})

    async def _request_file_url_handler(self) -> MagicMock:
        """Handle POST /files/presigned-urls."""
        self._request_count += 1
        await self._apply_latency()

        if "request_file_url" in self._error_overrides:
            status, body = self._error_overrides["request_file_url"]
            return self._mock_response(status, body)

        return self._mock_response(
            200,
            {
                "url": f"https://s3.example.com/presigned/{uuid.uuid4()}",
                "expires_at": "2025-12-31T23:59:59Z",
                "method": "PUT",
                "file_path": "/workspace/test.txt",
            },
        )

    async def _get_session_cost_handler(self) -> MagicMock:
        """Handle GET /sessions/cost."""
        self._request_count += 1
        await self._apply_latency()

        if "get_session_cost" in self._error_overrides:
            status, body = self._error_overrides["get_session_cost"]
            return self._mock_response(status, body)

        return self._mock_response(200, {"cost_usd": 0.05})

    @contextmanager
    def mocked(self) -> Any:
        """Context manager that mocks httpx client requests."""
        original_request = None

        async def mock_request(self_: Any, method: str, path: str, **kwargs: Any) -> Any:
            # Route to appropriate handler based on method and path
            if method == "POST" and path == "/llm/invoke":
                return await self._invoke_llm_handler()
            elif method == "POST" and path == "/llm/stream":
                return await self._invoke_llm_stream_handler()
            elif method == "POST" and path == "/messages/persist":
                return await self._persist_messages_handler()
            elif method == "POST" and path == "/files/presigned-urls":
                return await self._request_file_url_handler()
            elif method == "GET" and path == "/sessions/cost":
                return await self._get_session_cost_handler()
            else:
                # Fallback
                raise ValueError(f"Unexpected request: {method} {path}")

        with patch("httpx.AsyncClient.request", side_effect=mock_request):
            yield self
