"""
Tests for invoke_llm_stream across all gateway implementations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from identark import DirectGateway
from identark.exceptions import CostCapExceededError
from identark.models import LLMResponse, Message, Role, StreamChunk, TokenUsage
from identark.pricing import estimate_cost
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


# ── DirectGateway OpenAI streaming ────────────────────────────────────────────


def _chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
    *,
    no_choices: bool = False,
) -> SimpleNamespace:
    """Build an OpenAI ``ChatCompletionChunk``-shaped object."""
    choices = (
        []
        if no_choices
        else [
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )
    return SimpleNamespace(choices=choices, usage=usage)


def _usage(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _FakeOpenAIStreamClient:
    """Minimal ``AsyncOpenAI`` stand-in whose ``create`` returns a chunk stream."""

    def __init__(self, *streams: list[SimpleNamespace]) -> None:
        self._streams = list(streams)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        self.calls += 1
        chunks = self._streams.pop(0)

        async def gen() -> AsyncIterator[SimpleNamespace]:
            for c in chunks:
                yield c

        return gen()


# Usage arrives in a trailing chunk with empty ``choices``, after the chunk that
# carries ``finish_reason``. This is what OpenAI sends with include_usage=True.
_TRAILING_USAGE = [
    _chunk("Hel"),
    _chunk("lo"),
    _chunk(finish_reason="stop"),
    _chunk(usage=_usage(1000, 500), no_choices=True),
]

# Usage attached to the finish chunk itself, as some providers do.
_USAGE_ON_FINISH = [
    _chunk("Hel"),
    _chunk("lo"),
    _chunk(finish_reason="stop", usage=_usage(1000, 500)),
]


class TestDirectGatewayOpenAIStream:
    async def _drain(self, gw: DirectGateway) -> list[StreamChunk]:
        return [
            c
            async for c in gw.invoke_llm_stream(
                new_messages=[Message(role=Role.USER, content="Hi")]
            )
        ]

    @pytest.mark.parametrize(
        "stream",
        [_TRAILING_USAGE, _USAGE_ON_FINISH],
        ids=["trailing-usage-chunk", "usage-on-finish-chunk"],
    )
    async def test_final_chunk_has_usage_and_cost_is_booked(
        self, stream: list[SimpleNamespace]
    ) -> None:
        gw = DirectGateway(
            llm_client=_FakeOpenAIStreamClient(stream), model="gpt-4o", provider="openai"
        )
        chunks = await self._drain(gw)

        assert "".join(c.content for c in chunks) == "Hello"
        finals = [c for c in chunks if c.finish_reason is not None]
        assert len(finals) == 1
        assert chunks[-1] is finals[0]
        assert finals[0].finish_reason == "stop"
        assert finals[0].input_tokens == 1000
        assert finals[0].output_tokens == 500

        expected = estimate_cost("gpt-4o", 1000, 500, "openai")
        assert expected > 0
        assert await gw.get_session_cost() == pytest.approx(expected)

    async def test_cost_cap_blocks_the_next_stream(self) -> None:
        one_call = estimate_cost("gpt-4o", 1000, 500, "openai")
        client = _FakeOpenAIStreamClient(_TRAILING_USAGE, _TRAILING_USAGE)
        gw = DirectGateway(
            llm_client=client, model="gpt-4o", provider="openai", cost_cap_usd=one_call / 2
        )

        await self._drain(gw)  # soft cap: checked before the call, so this one runs

        with pytest.raises(CostCapExceededError):
            await self._drain(gw)
        assert client.calls == 1

    async def test_no_final_chunk_without_finish_reason(self) -> None:
        gw = DirectGateway(
            llm_client=_FakeOpenAIStreamClient([_chunk("partial")]),
            model="gpt-4o",
            provider="openai",
        )
        chunks = await self._drain(gw)

        assert [c.content for c in chunks] == ["partial"]
        assert all(c.finish_reason is None for c in chunks)
