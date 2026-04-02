"""
SDK-to-API contract tests.

Verify that the SDK sends correctly formatted requests and correctly
parses responses according to the control plane API specification.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from httpx import Response

from identark.exceptions import ContentPolicyError, ControlPlaneError
from identark.gateways.control_plane import ControlPlaneGateway
from identark.models import Message, Role
from tests.integration.mock_server import MockControlPlane


@pytest.mark.integration
class TestRequestPayloadFormat:
    """Verify request payload format."""

    @pytest.mark.asyncio
    async def test_invoke_llm_request_format(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify invoke_llm sends correctly formatted request."""
        from unittest.mock import patch, AsyncMock, MagicMock

        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
        )

        # Capture the request payload
        captured_requests = []

        async def mock_request(self_: any, method: str, path: str, **kwargs: any) -> MagicMock:
            if "json" in kwargs:
                captured_requests.append(kwargs["json"])
            resp = MagicMock()
            resp.status_code = 200
            resp.json = AsyncMock(return_value={
                "message": {"role": "assistant", "content": "Test"},
                "cost_usd": 0.001,
                "model": "gpt-4o",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
            })
            return resp

        with patch("httpx.AsyncClient.request", side_effect=mock_request):
            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

        # Verify request structure
        assert len(captured_requests) == 1
        request = captured_requests[0]
        assert "new_messages" in request
        assert isinstance(request["new_messages"], list)
        assert request["new_messages"][0]["role"] == "user"
        assert request["new_messages"][0]["content"] == "Test"

    @pytest.mark.asyncio
    async def test_invoke_llm_with_tools_includes_tool_spec(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify tools are included in request payload."""
        from unittest.mock import patch, AsyncMock, MagicMock

        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
        )

        captured_requests = []

        async def mock_request(self_: any, method: str, path: str, **kwargs: any) -> MagicMock:
            if "json" in kwargs:
                captured_requests.append(kwargs["json"])
            resp = MagicMock()
            resp.status_code = 200
            resp.json = AsyncMock(return_value={
                "message": {"role": "assistant", "content": "Test"},
                "cost_usd": 0.001,
                "model": "gpt-4o",
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            })
            return resp

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        with patch("httpx.AsyncClient.request", side_effect=mock_request):
            await gateway.invoke_llm([Message(role=Role.USER, content="Test")], tools=tools)

        request = captured_requests[0]
        assert "tools" in request
        assert len(request["tools"]) == 1
        assert request["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_auth_header_format(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify Authorization header is correctly formatted."""
        from unittest.mock import patch, AsyncMock, MagicMock

        gateway = ControlPlaneGateway(
            api_key="sk-test-12345",
            url="https://api.identark.io/v1",
        )

        captured_headers = []

        async def mock_request(self_: any, method: str, path: str, **kwargs: any) -> MagicMock:
            # Capture headers from the client
            captured_headers.append(dict(self_.headers))
            resp = MagicMock()
            resp.status_code = 200
            resp.json = AsyncMock(return_value={
                "message": {"role": "assistant", "content": "Test"},
                "cost_usd": 0.001,
                "model": "gpt-4o",
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            })
            return resp

        with patch("httpx.AsyncClient.request", side_effect=mock_request):
            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

        headers = captured_headers[0]
        assert "authorization" in headers or "Authorization" in headers
        auth_header = headers.get("authorization") or headers.get("Authorization")
        assert auth_header == "Bearer sk-test-12345"


@pytest.mark.integration
class TestResponseParsing:
    """Verify correct parsing of API responses."""

    @pytest.mark.asyncio
    async def test_llm_response_parsing_all_fields(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify all LLM response fields are correctly parsed."""
        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
        )

        async def mock_response(request: any) -> Response:
            return Response(200, json={
                "message": {
                    "role": "assistant",
                    "content": "Complete answer",
                },
                "cost_usd": 0.0234,
                "model": "gpt-4o-turbo",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 75,
                    "total_tokens": 225,
                    "cached_tokens": 50,
                },
            })

        router = mock_control_plane.mocked()
        router.post("https://api.identark.io/v1/llm/invoke").mock(side_effect=mock_response)

        with router:
            response = await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

        assert response.message.content == "Complete answer"
        assert response.cost_usd == 0.0234
        assert response.model == "gpt-4o-turbo"
        assert response.finish_reason == "stop"
        assert response.usage.input_tokens == 150
        assert response.usage.output_tokens == 75
        assert response.usage.total_tokens == 225
        assert response.usage.cached_tokens == 50

    @pytest.mark.asyncio
    async def test_presigned_url_response_parsing(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify presigned URL response is correctly parsed."""
        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
        )

        async def mock_response(request: any) -> Response:
            return Response(200, json={
                "url": "https://s3.aws.amazon.com/bucket/key?signature=xyz",
                "expires_at": "2025-12-31T23:59:59Z",
                "method": "PUT",
                "file_path": "/workspace/output.txt",
            })

        router = mock_control_plane.mocked()
        router.post("https://api.identark.io/v1/files/presigned-urls").mock(
            side_effect=mock_response
        )

        with router:
            url_obj = await gateway.request_file_url("/workspace/output.txt")

        assert url_obj.url == "https://s3.aws.amazon.com/bucket/key?signature=xyz"
        assert url_obj.expires_at == "2025-12-31T23:59:59Z"
        assert url_obj.method == "PUT"
        assert url_obj.file_path == "/workspace/output.txt"


@pytest.mark.integration
class TestErrorResponseMapping:
    """Verify error codes map to correct exception types."""

    @pytest.mark.asyncio
    async def test_401_maps_to_authentication_error(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify 401 responses raise AuthenticationError."""
        from identark.exceptions import AuthenticationError

        gateway = ControlPlaneGateway(
            api_key="invalid",
            url="https://api.identark.io/v1",
        )

        mock_control_plane.inject_error("invoke_llm", 401, {
            "error_code": "authentication_failed",
            "message": "Invalid token",
            "reason": "expired",
        })

        with mock_control_plane.mocked():
            with pytest.raises(AuthenticationError) as exc:
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_402_maps_to_cost_cap_exceeded_error(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify 402 responses raise CostCapExceededError."""
        from identark.exceptions import CostCapExceededError

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        mock_control_plane.inject_error("invoke_llm", 402, {
            "error_code": "cost_cap_exceeded",
            "message": "Cap exceeded",
            "cap_usd": 10.0,
            "consumed_usd": 10.5,
        })

        with mock_control_plane.mocked():
            with pytest.raises(CostCapExceededError) as exc:
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            assert exc.value.cap_usd == 10.0

    @pytest.mark.asyncio
    async def test_404_maps_to_session_not_found_error(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify 404 responses raise SessionNotFoundError."""
        from identark.exceptions import SessionNotFoundError

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
            session_id="expired",
        )

        mock_control_plane.inject_error("invoke_llm", 404, {
            "error_code": "session_not_found",
            "message": "Not found",
            "session_id": "expired",
        })

        with mock_control_plane.mocked():
            with pytest.raises(SessionNotFoundError) as exc:
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            assert exc.value.session_id == "expired"

    @pytest.mark.asyncio
    async def test_content_policy_error_mapping(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify content policy errors are correctly mapped."""
        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        mock_control_plane.inject_error("invoke_llm", 400, {
            "error_code": "content_policy",
            "message": "Request violates content policy",
        })

        with mock_control_plane.mocked():
            with pytest.raises(ContentPolicyError):
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

    @pytest.mark.asyncio
    async def test_unknown_error_raises_control_plane_error(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify unknown errors raise ControlPlaneError."""
        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        mock_control_plane.inject_error("invoke_llm", 418, {
            "error_code": "teapot",
            "message": "I am a teapot",
        })

        with mock_control_plane.mocked():
            with pytest.raises(ControlPlaneError) as exc:
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            assert exc.value.status_code == 418


@pytest.mark.integration
class TestStreamingSSEFormat:
    """Verify streaming SSE format is correctly parsed."""

    @pytest.mark.asyncio
    async def test_stream_chunk_parsing(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Verify SSE chunks are correctly parsed."""
        from unittest.mock import patch, AsyncMock, MagicMock

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        async def mock_stream_request(self_: any, method: str, path: str, **kwargs: any) -> MagicMock:
            sse_events = "\n".join([
                'data: {"content": "Hello", "finish_reason": null, "model": "gpt-4o"}',
                'data: {"content": " ", "finish_reason": null, "model": "gpt-4o"}',
                'data: {"content": "world", "finish_reason": null, "model": "gpt-4o"}',
                'data: {"content": "", "finish_reason": "stop", "model": "gpt-4o", "input_tokens": 10, "output_tokens": 15}',
                "data: [DONE]",
            ])

            resp = MagicMock()
            resp.status_code = 200

            async def aiter_lines_impl() -> any:
                for line in sse_events.split("\n"):
                    if line:
                        yield line

            resp.aiter_lines = aiter_lines_impl
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=None)
            resp.aread = AsyncMock()
            return resp

        with patch("httpx.AsyncClient.stream", side_effect=mock_stream_request):
            chunks = []
            async for chunk in gateway.invoke_llm_stream([Message(role=Role.USER, content="Test")]):
                chunks.append(chunk)

        assert len(chunks) >= 4
        assert chunks[0].content == "Hello"
        assert chunks[0].finish_reason is None
        assert chunks[1].content == " "
        assert chunks[2].content == "world"

        # Final chunk should have token counts
        final = chunks[-1]
        assert final.finish_reason == "stop"
        assert final.input_tokens == 10
        assert final.output_tokens == 15
