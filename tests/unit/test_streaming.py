"""
Tests for invoke_llm_stream across all gateway implementations.
"""

from __future__ import annotations

import pytest

from identark.models import LLMResponse, Message, Role, StreamChunk, TokenUsage
from identark.testing import MockGateway

# ── MockGateway streaming ─────────────────────────────────────────────────────

class TestMockGatewayStream:
    async def test_yields_chunks_then_final(self) -> None:
        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content="Hello world"),
            cost_usd=0.001,
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7),
        ))

        chunks: list[StreamChunk] = []
        async for chunk in mock.invoke_llm_stream(
            new_messages=[Message(role=Role.USER, content="Hi")]
        ):
            chunks.append(chunk)

        # All intermediate chunks have no finish_reason
        mid_chunks = [c for c in chunks if c.finish_reason is None]
        assert len(mid_chunks) >= 1

        # Final chunk has finish_reason set
        final = chunks[-1]
        assert final.finish_reason == "stop"
        assert final.input_tokens == 5
        assert final.output_tokens == 2

    async def test_full_content_reconstructed(self) -> None:
        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content="The answer is 42"),
            cost_usd=0.001,
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=3, output_tokens=4, total_tokens=7),
        ))

        text = ""
        async for chunk in mock.invoke_llm_stream(
            new_messages=[Message(role=Role.USER, content="What is the answer?")]
        ):
            text += chunk.content

        assert text == "The answer is 42"

    async def test_records_call(self) -> None:
        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content="Hi"),
            cost_usd=0.0,
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        ))

        async for _ in mock.invoke_llm_stream(
            new_messages=[Message(role=Role.USER, content="Hello")]
        ):
            pass

        assert mock.invoke_llm_call_count == 1

    async def test_accumulates_cost(self) -> None:
        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content="Hi"),
            cost_usd=0.005,
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        ))

        async for _ in mock.invoke_llm_stream(
            new_messages=[Message(role=Role.USER, content="Hello")]
        ):
            pass

        assert await mock.get_session_cost() == pytest.approx(0.005)

    async def test_empty_content_yields_final_chunk_only(self) -> None:
        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content=""),
            cost_usd=0.0,
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        ))

        chunks: list[StreamChunk] = []
        async for chunk in mock.invoke_llm_stream(
            new_messages=[Message(role=Role.USER, content="Hi")]
        ):
            chunks.append(chunk)

        # Empty content: split("") gives [""], so 1 mid chunk + 1 final
        assert chunks[-1].finish_reason == "stop"

    async def test_stream_chunk_model_field(self) -> None:
        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content="Ok"),
            cost_usd=0.0,
            model="gpt-4o",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        ))

        final: StreamChunk | None = None
        async for chunk in mock.invoke_llm_stream(
            new_messages=[Message(role=Role.USER, content="Hi")]
        ):
            final = chunk

        assert final is not None
        assert final.model == "gpt-4o"


# ── StreamChunk model ─────────────────────────────────────────────────────────

class TestStreamChunkModel:
    def test_basic_fields(self) -> None:
        chunk = StreamChunk(content="hello", finish_reason=None, model="gpt-4o")
        assert chunk.content == "hello"
        assert chunk.finish_reason is None
        assert chunk.model == "gpt-4o"
        assert chunk.input_tokens == 0
        assert chunk.output_tokens == 0

    def test_final_chunk(self) -> None:
        chunk = StreamChunk(
            content="",
            finish_reason="stop",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        assert chunk.finish_reason == "stop"
        assert chunk.input_tokens == 100
        assert chunk.output_tokens == 50

    def test_is_final(self) -> None:
        mid = StreamChunk(content="hello", finish_reason=None, model="m")
        final = StreamChunk(content="", finish_reason="stop", model="m")
        assert mid.finish_reason is None
        assert final.finish_reason is not None
