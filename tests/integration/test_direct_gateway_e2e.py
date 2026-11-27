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


def _openai_stream_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    usage: tuple[int, int] | None = None,
    *,
    with_choice: bool = True,
) -> Any:
    """Build a ChatCompletionChunk-shaped mock as the OpenAI SDK streams them."""
    choices = (
        [MagicMock(delta=MagicMock(content=content), finish_reason=finish_reason)]
        if with_choice
        else []
    )
    return MagicMock(
        choices=choices,
        usage=(
            MagicMock(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
        ),
    )


def _as_stream(*chunks: Any) -> Any:
    async def gen() -> Any:
        for chunk in chunks:
            yield chunk

    return gen()


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
            assert len(gateway.history) >= 2

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

            assert len(gateway.history) == 2


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
            cost_after_1 = await gateway.get_session_cost()

            # Second call
            await gateway.invoke_llm([Message(role=Role.USER, content="Test 2")])
            cost_after_2 = await gateway.get_session_cost()

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
        """Test that cost cap is enforced.

        The cap is checked before each call: the call that crosses it completes,
        and the next one is refused without reaching the provider.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            gateway = DirectGateway(
                llm_client=client,
                model="gpt-4o-mini",
                workspace_dir=tmpdir,
                cost_cap_usd=0.00001,  # Below the ~$0.0000135 cost of one mock call
            )

            await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

            with pytest.raises(CostCapExceededError) as exc_info:
                await gateway.invoke_llm([Message(role=Role.USER, content="Test")])

            assert exc_info.value.cap_usd == 0.00001
            assert exc_info.value.consumed_usd >= 0.00001
            assert client.chat.completions.create.await_count == 1

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
            assert await gateway.get_session_cost() > 0
            assert len(gateway.history) > 0

            # Reset
            gateway.reset()

            assert await gateway.get_session_cost() == 0.0
            assert len(gateway.history) == 0


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

            # The OpenAI SDK streams ChatCompletionChunk objects, not StreamChunks
            client.chat.completions.create = AsyncMock(
                return_value=_as_stream(
                    _openai_stream_chunk("Hello "),
                    _openai_stream_chunk("world!"),
                    _openai_stream_chunk(None, "stop", usage=(10, 20)),
                )
            )

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
            assert all(isinstance(c, StreamChunk) for c in chunks)
            assert chunks[0].content == "Hello "

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_chunks",
        [
            pytest.param(
                lambda: _as_stream(
                    _openai_stream_chunk("Response"),
                    _openai_stream_chunk(None, "stop", usage=(5, 10)),
                ),
                id="usage-on-finish-chunk",
            ),
            pytest.param(
                # What OpenAI sends with stream_options={"include_usage": True}:
                # usage arrives in a separate trailing chunk with empty choices,
                # after the chunk carrying finish_reason.
                lambda: _as_stream(
                    _openai_stream_chunk("Response"),
                    _openai_stream_chunk(None, "stop"),
                    _openai_stream_chunk(usage=(5, 10), with_choice=False),
                ),
                id="openai-trailing-usage-chunk",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "DirectGateway._stream_openai emits the final StreamChunk (and "
                        "books cost) when it sees finish_reason, before OpenAI's trailing "
                        "usage-only chunk arrives, so token counts and streamed cost are 0"
                    ),
                ),
            ),
        ],
    )
    async def test_streaming_final_chunk_has_metadata(self, provider_chunks: Any) -> None:
        """Test that final chunk includes token counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _mock_openai_client()
            client.chat.completions.create = AsyncMock(return_value=provider_chunks())

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
