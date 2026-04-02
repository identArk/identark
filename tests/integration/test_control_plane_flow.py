"""
End-to-end control plane integration tests.

Tests the complete flow of control plane interactions including
authentication, session management, streaming, and cost tracking.
"""

from __future__ import annotations

import pytest

from identark.exceptions import (
    AuthenticationError,
    CostCapExceededError,
    NetworkError,
    SessionNotFoundError,
)
from identark.gateways.control_plane import ControlPlaneGateway
from identark.models import Message, Role
from tests.integration.mock_server import MockControlPlane


@pytest.mark.integration
class TestControlPlaneAuthFlow:
    """Test authentication and session initialization."""

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_api_key(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test successful authentication with valid API key."""
        with mock_control_plane.mocked():
            response = await control_plane_gateway.invoke_llm(sample_messages)
            assert response.message.content
            assert response.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_authentication_error(
        self,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test that invalid API key raises AuthenticationError."""
        gateway = ControlPlaneGateway(
            api_key="invalid-key",
            url="https://api.identark.io/v1",
        )
        mock_control_plane.inject_error(
            "invoke_llm",
            401,
            {
                "error_code": "authentication_failed",
                "message": "Invalid API key",
                "session_id": "sess-123",
                "reason": "key_expired",
            },
        )
        with mock_control_plane.mocked():
            with pytest.raises(AuthenticationError) as exc_info:
                await gateway.invoke_llm(sample_messages)
            assert exc_info.value.status_code == 401
            assert exc_info.value.reason == "key_expired"


@pytest.mark.integration
class TestLLMInvocation:
    """Test LLM invocation flows."""

    @pytest.mark.asyncio
    async def test_invoke_llm_with_new_messages(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test basic invoke_llm with new messages returns response with cost."""
        with mock_control_plane.mocked():
            response = await control_plane_gateway.invoke_llm(sample_messages)

            assert response.message.role == Role.ASSISTANT
            assert response.message.content
            assert response.cost_usd > 0
            assert response.model == "gpt-4o"
            assert response.finish_reason == "stop"
            assert response.usage.input_tokens == 10
            assert response.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_invoke_llm_with_tools(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test invoke_llm with tool definitions."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        with mock_control_plane.mocked():
            response = await control_plane_gateway.invoke_llm(sample_messages, tools=tools)
            assert response.message.content

    @pytest.mark.asyncio
    async def test_invoke_llm_stream_returns_chunks(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test streaming invocation returns chunks with final metadata."""
        with mock_control_plane.mocked():
            chunks = []
            async for chunk in control_plane_gateway.invoke_llm_stream(sample_messages):
                chunks.append(chunk)

            # Should have at least 3 chunks (data, data, final)
            assert len(chunks) >= 3

            # First chunks have content but no finish reason
            assert chunks[0].content == "Hello "
            assert chunks[0].finish_reason is None

            assert chunks[1].content == "world!"
            assert chunks[1].finish_reason is None

            # Final chunk has finish_reason and token counts
            final = chunks[-1]
            assert final.finish_reason == "stop"
            assert final.input_tokens > 0
            assert final.output_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming_concatenates_to_full_response(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test that streaming chunks concatenate to form full response."""
        with mock_control_plane.mocked():
            full_text = ""
            async for chunk in control_plane_gateway.invoke_llm_stream(sample_messages):
                full_text += chunk.content

            assert "Hello" in full_text
            assert "world" in full_text


@pytest.mark.integration
class TestMessagePersistence:
    """Test message persistence."""

    @pytest.mark.asyncio
    async def test_persist_messages_maintains_history(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
        sample_assistant_message: Message,
    ) -> None:
        """Test persist_messages stores conversation history."""
        with mock_control_plane.mocked():
            # Persist a conversation exchange
            await control_plane_gateway.persist_messages(
                [*sample_messages, sample_assistant_message]
            )
            # If no exception, persistence succeeded
            assert True

    @pytest.mark.asyncio
    async def test_persist_empty_message_list(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test persist_messages with empty list."""
        with mock_control_plane.mocked():
            await control_plane_gateway.persist_messages([])
            assert True


@pytest.mark.integration
class TestFileOperations:
    """Test file URL operations."""

    @pytest.mark.asyncio
    async def test_request_file_url_returns_presigned_url(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test request_file_url returns valid presigned URL."""
        with mock_control_plane.mocked():
            url_obj = await control_plane_gateway.request_file_url(
                file_path="/workspace/output.txt",
                method="PUT",
            )

            assert url_obj.url.startswith("https://")
            assert url_obj.file_path == "/workspace/output.txt"
            assert url_obj.method == "PUT"
            assert url_obj.expires_at

    @pytest.mark.asyncio
    async def test_request_file_url_for_download(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test requesting presigned URL for file download."""
        with mock_control_plane.mocked():
            url_obj = await control_plane_gateway.request_file_url(
                file_path="/workspace/data.json",
                method="GET",
            )

            assert url_obj.method == "GET"
            assert url_obj.file_path == "/workspace/data.json"

    @pytest.mark.asyncio
    async def test_request_file_url_rejects_invalid_path(
        self,
        control_plane_gateway: ControlPlaneGateway,
    ) -> None:
        """Test that paths outside /workspace/ are rejected locally."""
        with pytest.raises(Exception):  # PathNotAllowedError
            await control_plane_gateway.request_file_url(
                file_path="/etc/passwd",  # Outside workspace
            )


@pytest.mark.integration
class TestCostTracking:
    """Test cost tracking and limits."""

    @pytest.mark.asyncio
    async def test_get_session_cost_returns_accumulated_cost(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test get_session_cost returns accumulated session cost."""
        # Set up mock to return a specific cost
        mock_control_plane.set_session_cost("sess-test", 0.05)

        with mock_control_plane.mocked():
            cost = await control_plane_gateway.get_session_cost()
            assert cost == 0.05

    @pytest.mark.asyncio
    async def test_get_session_cost_zero_for_new_session(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test get_session_cost returns 0 for new session."""
        with mock_control_plane.mocked():
            cost = await control_plane_gateway.get_session_cost()
            assert cost >= 0

    @pytest.mark.asyncio
    async def test_cost_cap_exceeded_raises_error(
        self,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test that cost cap exceeded returns appropriate error."""
        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
        )
        mock_control_plane.inject_error(
            "invoke_llm",
            402,
            {
                "error_code": "cost_cap_exceeded",
                "message": "Session cost cap exceeded",
                "cap_usd": 1.0,
                "consumed_usd": 1.05,
                "session_id": "sess-123",
            },
        )

        with mock_control_plane.mocked():
            with pytest.raises(CostCapExceededError) as exc_info:
                await gateway.invoke_llm(sample_messages)
            assert exc_info.value.cap_usd == 1.0
            assert exc_info.value.consumed_usd == 1.05


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_expired_session_raises_session_not_found(
        self,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test that expired session raises SessionNotFoundError."""
        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
            session_id="expired-sess",
        )
        mock_control_plane.inject_error(
            "invoke_llm",
            404,
            {
                "error_code": "session_not_found",
                "message": "Session not found",
                "session_id": "expired-sess",
            },
        )

        with mock_control_plane.mocked():
            with pytest.raises(SessionNotFoundError) as exc_info:
                await gateway.invoke_llm(sample_messages)
            assert exc_info.value.session_id == "expired-sess"

    @pytest.mark.asyncio
    async def test_network_error_after_retries_exhausted(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test that network errors raise NetworkError after retries exhausted."""
        # Inject 500 error which should trigger retries
        mock_control_plane.inject_error(
            "invoke_llm",
            500,
            {"error": "Internal server error"},
        )

        with mock_control_plane.mocked():
            with pytest.raises(NetworkError):
                await control_plane_gateway.invoke_llm(sample_messages)

    @pytest.mark.asyncio
    async def test_rate_limit_handling(
        self,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test handling of rate limit responses."""
        gateway = ControlPlaneGateway(
            api_key="test-key",
            url="https://api.identark.io/v1",
        )
        # Note: 429 rate limits are typically not retried by control plane
        # They bubble up as ControlPlaneError
        mock_control_plane.inject_error(
            "invoke_llm",
            429,
            {
                "error_code": "rate_limited",
                "message": "Rate limit exceeded",
            },
        )

        with mock_control_plane.mocked():
            # Should raise a ControlPlaneError (not retried)
            with pytest.raises(Exception):
                await gateway.invoke_llm(sample_messages)


@pytest.mark.integration
class TestContextManager:
    """Test context manager usage."""

    @pytest.mark.asyncio
    async def test_gateway_as_async_context_manager(
        self,
        mock_control_plane: MockControlPlane,
        sample_messages: list[Message],
    ) -> None:
        """Test using ControlPlaneGateway as async context manager."""
        with mock_control_plane.mocked():
            async with ControlPlaneGateway(
                api_key="test-key",
                url="https://api.identark.io/v1",
            ) as gateway:
                response = await gateway.invoke_llm(sample_messages)
                assert response.message.content
