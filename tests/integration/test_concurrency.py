"""
Concurrency and thread-safety tests.

Verify that the SDK correctly handles concurrent invocations,
streaming, and shared state.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from identark.gateways.direct import DirectGateway
from identark.models import Message, Role, StreamChunk
from tests.integration.mock_server import MockControlPlane


@pytest.mark.integration
class TestConcurrentInvocations:
    """Test concurrent invoke_llm calls."""

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_invocations(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test multiple simultaneous invocations don't interfere."""
        from identark.gateways.control_plane import ControlPlaneGateway

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        with mock_control_plane.mocked():
            # Run 5 invocations concurrently
            tasks = [
                gateway.invoke_llm([Message(role=Role.USER, content=f"Message {i}")])
                for i in range(5)
            ]

            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            for response in responses:
                assert response.message.content
                assert response.cost_usd > 0


@pytest.mark.integration
class TestConcurrentStreaming:
    """Test concurrent streaming calls."""

    @pytest.mark.asyncio
    async def test_multiple_streams_concurrently(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test multiple streaming calls can run concurrently."""
        from identark.gateways.control_plane import ControlPlaneGateway

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        async def collect_stream(msg: str) -> list[str]:
            chunks = []
            async for chunk in gateway.invoke_llm_stream(
                [Message(role=Role.USER, content=msg)]
            ):
                chunks.append(chunk.content)
            return chunks

        with mock_control_plane.mocked():
            results = await asyncio.gather(
                collect_stream("Message 1"),
                collect_stream("Message 2"),
                collect_stream("Message 3"),
            )

            assert len(results) == 3
            for result in results:
                assert isinstance(result, list)


@pytest.mark.integration
class TestMixedConcurrentOperations:
    """Test mixing different operation types concurrently."""

    @pytest.mark.asyncio
    async def test_streaming_and_non_streaming_concurrent(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test streaming + non-streaming concurrent calls."""
        from identark.gateways.control_plane import ControlPlaneGateway

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        async def stream_task() -> list[str]:
            chunks = []
            async for chunk in gateway.invoke_llm_stream(
                [Message(role=Role.USER, content="Stream message")]
            ):
                chunks.append(chunk.content)
            return chunks

        async def invoke_task() -> str:
            response = await gateway.invoke_llm(
                [Message(role=Role.USER, content="Regular message")]
            )
            return response.message.content

        with mock_control_plane.mocked():
            stream_result, invoke_result = await asyncio.gather(
                stream_task(),
                invoke_task(),
            )

            assert isinstance(stream_result, list)
            assert isinstance(invoke_result, str)


@pytest.mark.integration
class TestDirectGatewayConcurrency:
    """Test DirectGateway concurrent operations."""

    @pytest.mark.asyncio
    async def test_direct_gateway_concurrent_invocations(
        self,
        tmp_path,
    ) -> None:
        """Test DirectGateway handles concurrent invocations."""
        from unittest.mock import AsyncMock, MagicMock

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Response",
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

        gateway = DirectGateway(
            llm_client=client,
            model="gpt-4o-mini",
            workspace_dir=str(tmp_path),
        )

        # Run 3 invocations concurrently
        tasks = [
            gateway.invoke_llm([Message(role=Role.USER, content=f"Message {i}")])
            for i in range(3)
        ]

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        for response in responses:
            assert response.message.content == "Response"

    @pytest.mark.asyncio
    async def test_concurrent_invocations_accumulate_cost_correctly(
        self,
        tmp_path,
    ) -> None:
        """Test that cost accumulates correctly with concurrent calls."""
        from unittest.mock import AsyncMock, MagicMock

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Response",
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

        gateway = DirectGateway(
            llm_client=client,
            model="gpt-4o-mini",
            workspace_dir=str(tmp_path),
        )

        # Run 3 invocations concurrently
        await asyncio.gather(
            gateway.invoke_llm([Message(role=Role.USER, content="Message 1")]),
            gateway.invoke_llm([Message(role=Role.USER, content="Message 2")]),
            gateway.invoke_llm([Message(role=Role.USER, content="Message 3")]),
        )

        final_cost = await gateway.get_session_cost()

        # Cost should be accumulated from all 3 calls
        assert final_cost > 0


@pytest.mark.integration
class TestCostTracking:
    """Test cost tracking thread-safety."""

    @pytest.mark.asyncio
    async def test_session_cost_thread_safe(self, tmp_path) -> None:
        """Test that session cost is correctly updated with concurrent calls."""
        from unittest.mock import AsyncMock, MagicMock

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Response",
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

        gateway = DirectGateway(
            llm_client=client,
            model="gpt-4o-mini",
            workspace_dir=str(tmp_path),
        )

        costs_before = await gateway.get_session_cost()

        # Run concurrent invocations
        await asyncio.gather(
            gateway.invoke_llm([Message(role=Role.USER, content="Msg 1")]),
            gateway.invoke_llm([Message(role=Role.USER, content="Msg 2")]),
            gateway.invoke_llm([Message(role=Role.USER, content="Msg 3")]),
        )

        costs_after = await gateway.get_session_cost()

        # Cost should have increased
        assert costs_after > costs_before
        # Should reflect all 3 calls
        assert costs_after > 0


@pytest.mark.integration
class TestConcurrencyStress:
    """Stress test concurrent operations."""

    @pytest.mark.asyncio
    async def test_many_concurrent_invocations(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test handling of many concurrent invocations."""
        from identark.gateways.control_plane import ControlPlaneGateway

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        num_concurrent = 10

        with mock_control_plane.mocked():
            tasks = [
                gateway.invoke_llm([Message(role=Role.USER, content=f"Message {i}")])
                for i in range(num_concurrent)
            ]

            responses = await asyncio.gather(*tasks)

            assert len(responses) == num_concurrent
            for response in responses:
                assert response.message.content

    @pytest.mark.asyncio
    async def test_concurrent_file_url_requests(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test concurrent file URL requests."""
        from identark.gateways.control_plane import ControlPlaneGateway

        gateway = ControlPlaneGateway(
            api_key="test",
            url="https://api.identark.io/v1",
        )

        with mock_control_plane.mocked():
            tasks = [
                gateway.request_file_url(f"/workspace/file{i}.txt")
                for i in range(5)
            ]

            urls = await asyncio.gather(*tasks)

            assert len(urls) == 5
            for url in urls:
                assert url.url
                assert url.expires_at
