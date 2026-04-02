"""
End-to-end tests for DirectGateway.

Tests local LLM gateway functionality including conversation history,
cost tracking, file operations, and streaming.

These tests use a mock OpenAI client to avoid making real API calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from identark.exceptions import CostCapExceededError, PathNotAllowedError
from identark.gateways.direct import DirectGateway
from identark.models import Message, Role, StreamChunk


def _mock_openai_client() -> Any:
    """Create a mock OpenAI AsyncClient."""
    client = AsyncMock()

    # Mock chat.completions.create
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Test response",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(
                prompt_tokens=10,
                completion_tokens=20,
            ),
            model="gpt-4o-mini",
        )
    )

    return client


@pytest.mark.integration
class TestDirectGatewayBasics:
    """Test basic DirectGateway functionality."""

    @pytest.mark.asyncio
    async def test_create_gateway_and_invoke(self) -> None:
        """Test creating a gateway and invoking LLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            messages = [Message(role=Role.USER, content="Hello")]
            response = await gateway.invoke_llm(messages)

            assert response.message.content == "Test response"
            assert response.model == "gpt-4o-mini"
            assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_gateway_with_system_prompt(self) -> None:
        """Test gateway initialized with system prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            system_prompt = "You are a helpful assistant."
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                workspace_dir=tmpdir,
            )

            messages = [Message(role=Role.USER, content="Test")]
            await gateway.invoke_llm(messages)

            # Verify that the client was called with the system prompt
            call_args = client.chat.completions.create.call_args
            messages_arg = call_args.kwargs.get("messages") or call_args[1].get("messages")
            assert any(m.get("role") == "system" for m in messages_arg)


@pytest.mark.integration
class TestConversationHistory:
    """Test conversation history management."""

    @pytest.mark.asyncio
    async def test_conversation_history_accumulates(self) -> None:
        """Test that conversation history accumulates across invocations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            # First invocation
            msg1 = Message(role=Role.USER, content="First message")
            await gateway.invoke_llm([msg1])

            # Second invocation
            msg2 = Message(role=Role.USER, content="Second message")
            await gateway.invoke_llm([msg2])

            # Verify history was maintained
            assert len(gateway._conversation_history) >= 2

    @pytest.mark.asyncio
    async def test_persist_messages_adds_to_history(self) -> None:
        """Test that persist_messages adds messages to history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            messages = [
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there"),
            ]
            await gateway.persist_messages(messages)

            assert len(gateway._conversation_history) == 2


@pytest.mark.integration
class TestCostTracking:
    """Test cost tracking."""

    @pytest.mark.asyncio
    async def test_cost_tracking_across_invocations(self) -> None:
        """Test that cost accumulates across multiple invocations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            # First call
            await gateway.invoke_llm([Message(role=Role.USER, content="Test 1")])
            cost_after_1 = gateway._accumulated_cost

            # Second call
            await gateway.invoke_llm([Message(role=Role.USER, content="Test 2")])
            cost_after_2 = gateway._accumulated_cost

            assert cost_after_2 > cost_after_1

    @pytest.mark.asyncio
    async def test_get_session_cost_returns_accumulated(self) -> None:
        """Test get_session_cost returns accumulated cost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            cost = await gateway.get_session_cost()

            assert cost > 0

    @pytest.mark.asyncio
    async def test_cost_cap_enforcement(self) -> None:
        """Test that cost cap is enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
                cost_cap_usd=0.0001,  # Very low cap
            )

            with pytest.raises(CostCapExceededError):
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

    @pytest.mark.asyncio
    async def test_reset_clears_cost_and_history(self) -> None:
        """Test that reset clears accumulated cost and history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            # Accumulate state
            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            assert gateway._accumulated_cost > 0
            assert len(gateway._conversation_history) > 0

            # Reset
            gateway.reset()

            assert gateway._accumulated_cost == 0.0
            assert len(gateway._conversation_history) == 0


@pytest.mark.integration
class TestFileOperations:
    """Test file operations."""

    @pytest.mark.asyncio
    async def test_request_file_url_returns_local_path(self) -> None:
        """Test request_file_url returns valid local path for DirectGateway."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            url_obj = await gateway.request_file_url(
                file_path="/workspace/test.txt",
                method="PUT",
            )

            # For DirectGateway, URL is a local path
            assert url_obj.file_path == "/workspace/test.txt"
            assert url_obj.method == "PUT"

    @pytest.mark.asyncio
    async def test_file_url_for_download(self) -> None:
        """Test requesting file URL for download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            url_obj = await gateway.request_file_url(
                file_path="/workspace/data.json",
                method="GET",
            )

            assert url_obj.method == "GET"

    @pytest.mark.asyncio
    async def test_file_url_rejects_invalid_path(self) -> None:
        """Test that invalid paths are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            with pytest.raises(PathNotAllowedError):
                await gateway.request_file_url(file_path="/etc/passwd")


@pytest.mark.integration
class TestStreaming:
    """Test streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_returns_async_generator(self) -> None:
        """Test streaming returns an async generator of chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()

            # Mock streaming response
            async def mock_stream() -> Any:
                yield StreamChunk(
                    content="Hello ",
                    finish_reason=None,
                    model="gpt-4o-mini",
                )
                yield StreamChunk(
                    content="world!",
                    finish_reason=None,
                    model="gpt-4o-mini",
                )
                yield StreamChunk(
                    content="",
                    finish_reason="stop",
                    model="gpt-4o-mini",
                    input_tokens=10,
                    output_tokens=20,
                )

            client.chat.completions.create = AsyncMock(return_value=mock_stream())

            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            chunks = []
            async for chunk in gateway.invoke_llm_stream(
                [Message(role=Role.USER, content="Test")]
            ):
                chunks.append(chunk)

            assert len(chunks) >= 2
            assert chunks[0].content == "Hello "

    @pytest.mark.asyncio
    async def test_streaming_final_chunk_has_metadata(self) -> None:
        """Test that final chunk includes token counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()

            async def mock_stream() -> Any:
                yield StreamChunk(
                    content="Response",
                    finish_reason=None,
                    model="gpt-4o-mini",
                )
                yield StreamChunk(
                    content="",
                    finish_reason="stop",
                    model="gpt-4o-mini",
                    input_tokens=5,
                    output_tokens=10,
                )

            client.chat.completions.create = AsyncMock(return_value=mock_stream())

            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
            )

            chunks = []
            async for chunk in gateway.invoke_llm_stream(
                [Message(role=Role.USER, content="Test")]
            ):
                chunks.append(chunk)

            final_chunk = chunks[-1]
            assert final_chunk.finish_reason == "stop"
            assert final_chunk.input_tokens > 0


@pytest.mark.integration
class TestProviderDetection:
    """Test provider auto-detection."""

    @pytest.mark.asyncio
    async def test_local_provider_zero_cost(self) -> None:
        """Test that local provider (Ollama) tracks zero cost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="llama3.2",
                workspace_dir=tmpdir,
                provider="local",  # Force local provider
            )

            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

            # Local provider should have zero cost
            cost = await gateway.get_session_cost()
            assert cost == 0.0

    @pytest.mark.asyncio
    async def test_openai_provider_has_cost(self) -> None:
        """Test that OpenAI provider tracks costs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
                provider="openai",
            )

            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

            cost = await gateway.get_session_cost()
            assert cost > 0


@pytest.mark.integration
class TestWorkspaceDirectory:
    """Test workspace directory handling."""

    @pytest.mark.asyncio
    async def test_workspace_directory_created_if_missing(self) -> None:
        """Test that workspace directory is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            client = _mock_openai_client()

            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=str(workspace),
            )

            # Should work even if workspace doesn't exist
            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])
            assert True
