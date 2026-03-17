"""Unit tests for CredSealChatModel (LangChain integration)."""


import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from credseal.integrations.langchain import (
    CredSealChatModel,
    credseal_to_ai_message,
    lc_to_credseal,
)
from credseal.models import Function, LLMResponse, Message, Role, TokenUsage, ToolCall
from credseal.testing import MockGateway

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_response(
    content: str = "Hello!",
    finish_reason: str = "stop",
    cost: float = 0.001,
) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        cost_usd=cost,
        model="mock-gpt-4o",
        finish_reason=finish_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_tool_response() -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=""),
        cost_usd=0.001,
        model="mock-gpt-4o",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="call_abc123",
                function=Function(name="get_weather", arguments='{"city": "London"}'),
            )
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=8, total_tokens=18),
    )


def _make_llm(mock: MockGateway) -> CredSealChatModel:
    return CredSealChatModel(gateway=mock)


# ── lc_to_credseal ────────────────────────────────────────────────────────────


class TestLcToCredseal:
    def test_human_message(self) -> None:
        msgs = lc_to_credseal([HumanMessage(content="Hi")])
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "Hi"

    def test_ai_message(self) -> None:
        msgs = lc_to_credseal([AIMessage(content="Hello")])
        assert msgs[0].role == Role.ASSISTANT

    def test_system_message(self) -> None:
        msgs = lc_to_credseal([SystemMessage(content="You are helpful.")])
        assert msgs[0].role == Role.SYSTEM

    def test_tool_message_preserves_call_id(self) -> None:
        msg = ToolMessage(content='{"result": 42}', tool_call_id="call_xyz")
        msgs = lc_to_credseal([msg])
        assert msgs[0].role == Role.TOOL
        assert msgs[0].tool_call_id == "call_xyz"
        assert msgs[0].content == '{"result": 42}'

    def test_mixed_conversation(self) -> None:
        lc_msgs = [
            SystemMessage(content="Be concise."),
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="4"),
            HumanMessage(content="Thanks"),
        ]
        cs = lc_to_credseal(lc_msgs)
        assert [m.role for m in cs] == [
            Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER
        ]

    def test_multimodal_list_content(self) -> None:
        msg = HumanMessage(content=[{"type": "text", "text": "Describe this image"}])
        msgs = lc_to_credseal([msg])
        assert isinstance(msgs[0].content, list)
        assert msgs[0].content[0]["type"] == "text"  # type: ignore[index]


# ── credseal_to_ai_message ────────────────────────────────────────────────────


class TestCredSealToAiMessage:
    def test_basic_text_response(self) -> None:
        ai_msg = credseal_to_ai_message(_make_response("The answer is 42."))
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "The answer is 42."
        assert ai_msg.tool_calls == []

    def test_tool_call_response(self) -> None:
        ai_msg = credseal_to_ai_message(_make_tool_response())
        assert ai_msg.content == ""
        assert len(ai_msg.tool_calls) == 1
        tc = ai_msg.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"city": "London"}

    def test_response_metadata_populated(self) -> None:
        response = _make_response(cost=0.005)
        ai_msg = credseal_to_ai_message(response)
        assert ai_msg.response_metadata["cost_usd"] == 0.005
        assert ai_msg.response_metadata["model"] == "mock-gpt-4o"
        assert ai_msg.response_metadata["finish_reason"] == "stop"
        assert ai_msg.response_metadata["input_tokens"] == 10
        assert ai_msg.response_metadata["output_tokens"] == 5

    def test_malformed_tool_arguments_handled(self) -> None:
        response = LLMResponse(
            message=Message(role=Role.ASSISTANT, content=""),
            cost_usd=0.001,
            model="mock",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="c1", function=Function(name="fn", arguments="not-json"))
            ],
            usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        ai_msg = credseal_to_ai_message(response)
        assert ai_msg.tool_calls[0]["args"] == {"_raw": "not-json"}


# ── CredSealChatModel ─────────────────────────────────────────────────────────


class TestCredSealChatModel:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_ai_message(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Hi there!"))
        llm = _make_llm(mock)

        result = await llm.ainvoke([HumanMessage(content="Hello")])

        assert isinstance(result, AIMessage)
        assert result.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_gateway_receives_correct_messages(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response())
        llm = _make_llm(mock)

        await llm.ainvoke([
            SystemMessage(content="Be brief."),
            HumanMessage(content="What is AI?"),
        ])

        assert mock.invoke_llm_call_count == 1
        sent = mock.last_request["new_messages"]
        assert sent[0].role == Role.SYSTEM
        assert sent[1].role == Role.USER
        assert sent[1].content == "What is AI?"

    @pytest.mark.asyncio
    async def test_tools_passed_through_to_gateway(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_tool_response())
        llm = _make_llm(mock)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        await llm.ainvoke([HumanMessage(content="Weather in London?")], tools=tools)

        assert mock.last_request["tools"] == tools

    @pytest.mark.asyncio
    async def test_tool_call_in_response(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_tool_response())
        llm = _make_llm(mock)

        result = await llm.ainvoke([HumanMessage(content="Weather?")])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"city": "London"}

    @pytest.mark.asyncio
    async def test_cost_in_response_metadata(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response(cost=0.0042))
        llm = _make_llm(mock)

        result = await llm.ainvoke([HumanMessage(content="x")])

        assert result.response_metadata["cost_usd"] == pytest.approx(0.0042)

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate_in_gateway(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("First"))
        mock.queue_response(_make_response("Second"))
        llm = _make_llm(mock)

        await llm.ainvoke([HumanMessage(content="Turn 1")])
        await llm.ainvoke([HumanMessage(content="Turn 2")])

        assert mock.invoke_llm_call_count == 2

    def test_llm_type_is_credseal(self) -> None:
        mock = MockGateway()
        llm = _make_llm(mock)
        assert llm._llm_type == "credseal"

    def test_identifying_params_include_gateway_type(self) -> None:
        mock = MockGateway()
        llm = _make_llm(mock)
        params = llm._identifying_params
        assert params["gateway_type"] == "MockGateway"

    def test_sync_invoke(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Sync works!"))
        llm = _make_llm(mock)

        result = llm.invoke([HumanMessage(content="Hello")])

        assert isinstance(result, AIMessage)
        assert result.content == "Sync works!"
