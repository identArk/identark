"""Unit tests for LangChain and LlamaIndex integrations."""


import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

from identark.integrations.langchain import (
    IdentArkChatModel,
    identark_to_ai_message,
    lc_to_identark,
)
from identark.integrations.llamaindex import (
    IdentArkLLM,
    identark_to_chat_response,
    li_to_identark,
)
from identark.models import Function, LLMResponse, Message, Role, TokenUsage, ToolCall
from identark.testing import MockGateway

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


def _make_llm(mock: MockGateway) -> IdentArkChatModel:
    return IdentArkChatModel(gateway=mock)


# ── lc_to_identark ────────────────────────────────────────────────────────────


class TestLcToCredseal:
    def test_human_message(self) -> None:
        msgs = lc_to_identark([HumanMessage(content="Hi")])
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "Hi"

    def test_ai_message(self) -> None:
        msgs = lc_to_identark([AIMessage(content="Hello")])
        assert msgs[0].role == Role.ASSISTANT

    def test_system_message(self) -> None:
        msgs = lc_to_identark([SystemMessage(content="You are helpful.")])
        assert msgs[0].role == Role.SYSTEM

    def test_tool_message_preserves_call_id(self) -> None:
        msg = ToolMessage(content='{"result": 42}', tool_call_id="call_xyz")
        msgs = lc_to_identark([msg])
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
        cs = lc_to_identark(lc_msgs)
        assert [m.role for m in cs] == [
            Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER
        ]

    def test_multimodal_list_content(self) -> None:
        msg = HumanMessage(content=[{"type": "text", "text": "Describe this image"}])
        msgs = lc_to_identark([msg])
        assert isinstance(msgs[0].content, list)
        assert msgs[0].content[0]["type"] == "text"  # type: ignore[index]


# ── identark_to_ai_message ────────────────────────────────────────────────────


class TestIdentArkToAiMessage:
    def test_basic_text_response(self) -> None:
        ai_msg = identark_to_ai_message(_make_response("The answer is 42."))
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "The answer is 42."
        assert ai_msg.tool_calls == []

    def test_tool_call_response(self) -> None:
        ai_msg = identark_to_ai_message(_make_tool_response())
        assert ai_msg.content == ""
        assert len(ai_msg.tool_calls) == 1
        tc = ai_msg.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"city": "London"}

    def test_response_metadata_populated(self) -> None:
        response = _make_response(cost=0.005)
        ai_msg = identark_to_ai_message(response)
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
        ai_msg = identark_to_ai_message(response)
        assert ai_msg.tool_calls[0]["args"] == {"_raw": "not-json"}


# ── IdentArkChatModel ─────────────────────────────────────────────────────────


class TestIdentArkChatModel:
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

    def test_llm_type_is_identark(self) -> None:
        mock = MockGateway()
        llm = _make_llm(mock)
        assert llm._llm_type == "identark"

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


# ── LlamaIndex: li_to_identark ────────────────────────────────────────────────


class TestLiToCredseal:
    def test_user_message(self) -> None:
        msgs = li_to_identark([ChatMessage(role=MessageRole.USER, content="Hi")])
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "Hi"

    def test_assistant_message(self) -> None:
        msgs = li_to_identark([ChatMessage(role=MessageRole.ASSISTANT, content="Hello")])
        assert msgs[0].role == Role.ASSISTANT

    def test_system_message(self) -> None:
        msgs = li_to_identark([ChatMessage(role=MessageRole.SYSTEM, content="Be helpful.")])
        assert msgs[0].role == Role.SYSTEM

    def test_tool_message(self) -> None:
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content='{"result": 42}',
            additional_kwargs={"tool_call_id": "call_abc"},
        )
        msgs = li_to_identark([msg])
        assert msgs[0].role == Role.TOOL
        assert msgs[0].tool_call_id == "call_abc"

    def test_chatbot_maps_to_assistant(self) -> None:
        msgs = li_to_identark([ChatMessage(role=MessageRole.CHATBOT, content="Hey")])
        assert msgs[0].role == Role.ASSISTANT

    def test_developer_maps_to_system(self) -> None:
        msgs = li_to_identark([ChatMessage(role=MessageRole.DEVELOPER, content="Rules")])
        assert msgs[0].role == Role.SYSTEM

    def test_mixed_conversation(self) -> None:
        li_msgs = [
            ChatMessage(role=MessageRole.SYSTEM, content="Be concise."),
            ChatMessage(role=MessageRole.USER, content="What is 2+2?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="4"),
        ]
        cs = li_to_identark(li_msgs)
        assert [m.role for m in cs] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]


# ── LlamaIndex: identark_to_chat_response ─────────────────────────────────────


class TestIdentArkToChatResponse:
    def test_basic_text_response(self) -> None:
        resp = identark_to_chat_response(_make_response("Hello!"))
        assert isinstance(resp, ChatResponse)
        assert resp.message.role == MessageRole.ASSISTANT
        assert resp.message.content == "Hello!"

    def test_raw_metadata_populated(self) -> None:
        resp = identark_to_chat_response(_make_response(cost=0.007))
        assert resp.raw["cost_usd"] == pytest.approx(0.007)
        assert resp.raw["model"] == "mock-gpt-4o"
        assert resp.raw["input_tokens"] == 10
        assert resp.raw["output_tokens"] == 5

    def test_tool_calls_in_additional_kwargs(self) -> None:
        resp = identark_to_chat_response(_make_tool_response())
        tcs = resp.message.additional_kwargs["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["function"]["name"] == "get_weather"
        assert tcs[0]["id"] == "call_abc123"

    def test_no_tool_calls_when_absent(self) -> None:
        resp = identark_to_chat_response(_make_response())
        assert "tool_calls" not in resp.message.additional_kwargs


# ── IdentArkLLM ───────────────────────────────────────────────────────────────


class TestIdentArkLLM:
    @pytest.mark.asyncio
    async def test_achat_returns_chat_response(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Hi there!"))
        llm = IdentArkLLM(gateway=mock)

        result = await llm.achat([ChatMessage(role=MessageRole.USER, content="Hello")])

        assert isinstance(result, ChatResponse)
        assert result.message.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_gateway_receives_correct_messages(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response())
        llm = IdentArkLLM(gateway=mock)

        await llm.achat([
            ChatMessage(role=MessageRole.SYSTEM, content="Be brief."),
            ChatMessage(role=MessageRole.USER, content="What is AI?"),
        ])

        sent = mock.last_request["new_messages"]
        assert sent[0].role == Role.SYSTEM
        assert sent[1].role == Role.USER
        assert sent[1].content == "What is AI?"

    @pytest.mark.asyncio
    async def test_acomplete_wraps_as_user_message(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Completion result"))
        llm = IdentArkLLM(gateway=mock)

        result = await llm.acomplete("Finish this sentence:")

        assert result.text == "Completion result"
        sent = mock.last_request["new_messages"]
        assert sent[0].role == Role.USER
        assert sent[0].content == "Finish this sentence:"

    @pytest.mark.asyncio
    async def test_tool_calls_in_response(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_tool_response())
        llm = IdentArkLLM(gateway=mock)

        result = await llm.achat([ChatMessage(role=MessageRole.USER, content="Weather?")])

        tcs = result.message.additional_kwargs["tool_calls"]
        assert tcs[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_tools_passed_to_gateway(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_tool_response())
        llm = IdentArkLLM(gateway=mock)

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        await llm.achat([ChatMessage(role=MessageRole.USER, content="x")], tools=tools)

        assert mock.last_request["tools"] == tools

    def test_metadata_has_model_name(self) -> None:
        # MockGateway has no .model attr — falls back to "identark"
        llm = IdentArkLLM(gateway=MockGateway())
        assert llm.metadata.model_name == "identark"

    def test_metadata_uses_gateway_model_attr(self) -> None:
        from identark import DirectGateway
        gw = DirectGateway(llm_client=object(), model="gpt-4o")
        llm = IdentArkLLM(gateway=gw)
        assert llm.metadata.model_name == "gpt-4o"

    def test_metadata_is_chat_model(self) -> None:
        llm = IdentArkLLM(gateway=MockGateway())
        assert llm.metadata.is_chat_model is True

    def test_sync_chat(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Sync OK"))
        llm = IdentArkLLM(gateway=mock)

        result = llm.chat([ChatMessage(role=MessageRole.USER, content="Hello")])

        assert result.message.content == "Sync OK"

    def test_sync_complete(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Complete OK"))
        llm = IdentArkLLM(gateway=mock)

        result = llm.complete("Hello")

        assert result.text == "Complete OK"

    def test_stream_complete_yields_chunks(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Hello world"))
        llm = IdentArkLLM(gateway=mock)

        chunks = list(llm.stream_complete("Say hi"))

        assert len(chunks) >= 1
        assert chunks[-1].text == "Hello world"

    def test_stream_chat_yields_chunks(self) -> None:
        mock = MockGateway()
        mock.queue_response(_make_response("Hi there"))
        llm = IdentArkLLM(gateway=mock)

        chunks = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Hello")]))

        assert len(chunks) >= 1
        assert chunks[-1].message.content == "Hi there"
