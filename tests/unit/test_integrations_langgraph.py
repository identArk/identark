"""
Tests for LangGraph integration — IdentArkNode and IdentArkStreamNode.
"""

from __future__ import annotations

import pytest

from identark.models import LLMResponse, Message, Role, TokenUsage
from identark.testing import MockGateway


def _make_response(content: str = "Hello!", model: str = "mock") -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        cost_usd=0.001,
        model=model,
        finish_reason="stop",
        usage=TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8),
    )


# ── _normalise_messages ───────────────────────────────────────────────────────

class TestNormaliseMessages:
    def test_passthrough_langchain_messages(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import _normalise_messages

        msg = HumanMessage(content="Hello")
        result = _normalise_messages([msg])
        assert result[0] is msg

    def test_converts_user_dict(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import _normalise_messages

        result = _normalise_messages([{"role": "user", "content": "Hi"}])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hi"

    def test_converts_assistant_dict(self) -> None:
        from langchain_core.messages import AIMessage

        from identark.integrations.langgraph import _normalise_messages

        result = _normalise_messages([{"role": "assistant", "content": "Hi back"}])
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "Hi back"

    def test_converts_system_dict(self) -> None:
        from langchain_core.messages import SystemMessage

        from identark.integrations.langgraph import _normalise_messages

        result = _normalise_messages([{"role": "system", "content": "Be helpful"}])
        assert isinstance(result[0], SystemMessage)

    def test_converts_tool_dict(self) -> None:
        from langchain_core.messages import ToolMessage

        from identark.integrations.langgraph import _normalise_messages

        result = _normalise_messages([
            {"role": "tool", "content": "result", "tool_call_id": "tc-1"}
        ])
        assert isinstance(result[0], ToolMessage)
        assert result[0].tool_call_id == "tc-1"

    def test_unknown_role_falls_back_to_human(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import _normalise_messages

        result = _normalise_messages([{"role": "unknown_role", "content": "x"}])
        assert isinstance(result[0], HumanMessage)

    def test_empty_list(self) -> None:
        from identark.integrations.langgraph import _normalise_messages

        assert _normalise_messages([]) == []


# ── IdentArkNode ──────────────────────────────────────────────────────────────

class TestIdentArkNode:
    async def test_returns_updated_state(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response("The answer is 42."))
        node = IdentArkNode(gateway=mock)

        state = {"messages": [HumanMessage(content="What is the answer?")]}
        result = await node(state)

        assert "messages" in result
        msgs = result["messages"]
        assert len(msgs) == 2
        assert isinstance(msgs[-1], AIMessage)
        assert msgs[-1].content == "The answer is 42."

    async def test_empty_state_returns_empty(self) -> None:
        from identark.integrations.langgraph import IdentArkNode

        node = IdentArkNode(gateway=MockGateway())
        result = await node({"messages": []})
        assert result["messages"] == []

    async def test_dict_messages_are_normalised(self) -> None:
        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response("Got it"))
        node = IdentArkNode(gateway=mock)

        result = await node({"messages": [{"role": "user", "content": "Hi"}]})
        assert len(result["messages"]) == 2

    async def test_uses_custom_messages_key(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response("Hi"))
        node = IdentArkNode(gateway=mock, messages_key="chat")

        state = {"chat": [HumanMessage(content="Hello")]}
        result = await node(state)
        assert "chat" in result
        assert len(result["chat"]) == 2

    async def test_records_gateway_call(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response())
        node = IdentArkNode(gateway=mock)

        await node({"messages": [HumanMessage(content="Hello")]})
        assert mock.invoke_llm_call_count == 1

    async def test_response_metadata_populated(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response("Hi", model="gpt-4o"))
        node = IdentArkNode(gateway=mock)

        result = await node({"messages": [HumanMessage(content="Hi")]})
        ai_msg: AIMessage = result["messages"][-1]
        assert ai_msg.response_metadata["model"] == "gpt-4o"
        assert ai_msg.response_metadata["cost_usd"] == pytest.approx(0.001)

    async def test_tools_passed_to_gateway(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response())
        tools = [{"type": "function", "function": {"name": "search"}}]
        node = IdentArkNode(gateway=mock, tools=tools)

        await node({"messages": [HumanMessage(content="Search for cats")]})
        last_call = mock.last_request
        assert last_call is not None
        assert last_call.get("tools") == tools

    def test_sync_invoke(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import IdentArkNode

        mock = MockGateway()
        mock.queue_response(_make_response("Sync works"))
        node = IdentArkNode(gateway=mock)

        result = node.invoke({"messages": [HumanMessage(content="Hello")]})
        assert result["messages"][-1].content == "Sync works"


# ── IdentArkStreamNode ────────────────────────────────────────────────────────

class TestIdentArkStreamNode:
    async def test_returns_accumulated_content(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from identark.integrations.langgraph import IdentArkStreamNode

        mock = MockGateway()
        mock.queue_response(_make_response("Hello world"))
        node = IdentArkStreamNode(gateway=mock)

        result = await node({"messages": [HumanMessage(content="Hi")]})

        msgs = result["messages"]
        assert len(msgs) == 2
        ai_msg: AIMessage = msgs[-1]
        assert "Hello world" in ai_msg.content

    async def test_empty_state_returns_empty(self) -> None:
        from identark.integrations.langgraph import IdentArkStreamNode

        node = IdentArkStreamNode(gateway=MockGateway())
        result = await node({"messages": []})
        assert result["messages"] == []

    async def test_final_chunk_metadata_captured(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from identark.integrations.langgraph import IdentArkStreamNode

        mock = MockGateway()
        mock.queue_response(LLMResponse(
            message=Message(role=Role.ASSISTANT, content="Done"),
            cost_usd=0.002,
            model="gpt-4o",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        ))
        node = IdentArkStreamNode(gateway=mock)

        result = await node({"messages": [HumanMessage(content="Go")]})
        ai_msg: AIMessage = result["messages"][-1]

        assert ai_msg.response_metadata["model"] == "gpt-4o"
        assert ai_msg.response_metadata["finish_reason"] == "stop"
        assert ai_msg.response_metadata["input_tokens"] == 10
        assert ai_msg.response_metadata["output_tokens"] == 5

    async def test_records_gateway_call(self) -> None:
        from langchain_core.messages import HumanMessage

        from identark.integrations.langgraph import IdentArkStreamNode

        mock = MockGateway()
        mock.queue_response(_make_response())
        node = IdentArkStreamNode(gateway=mock)

        await node({"messages": [HumanMessage(content="Hi")]})
        assert mock.invoke_llm_call_count == 1
