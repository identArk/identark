import pytest

from identark.models import Function, LLMResponse, Message, Role, TokenUsage, ToolCall
from identark.testing import MockGateway


def _crewai_installed() -> bool:
    try:
        import crewai  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _crewai_installed(), reason="crewai not installed")


def _make_tool_response() -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=""),
        cost_usd=0.001,
        model="mock",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="call_1",
                function=Function(name="add", arguments='{"a": 2, "b": 3}'),
            )
        ],
        usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def _make_text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        cost_usd=0.001,
        model="mock",
        finish_reason="stop",
        usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class TestIdentArkCrewAIIntegration:
    def test_basic_string_prompt(self) -> None:
        from identark.integrations.crewai import IdentArkCrewAILLM

        mock = MockGateway()
        mock.queue_response(_make_text_response("Hello from gateway"))
        llm = IdentArkCrewAILLM(gateway=mock)

        out = llm.call("Hi")

        assert out == "Hello from gateway"
        assert mock.invoke_llm_call_count == 1
        sent = mock.last_request["new_messages"]
        assert sent[0].role == Role.USER
        assert sent[0].content == "Hi"

    def test_message_delta_only_sends_tail(self) -> None:
        from identark.integrations.crewai import IdentArkCrewAILLM

        mock = MockGateway()
        mock.queue_response(_make_text_response("First"))
        mock.queue_response(_make_text_response("Second"))
        llm = IdentArkCrewAILLM(gateway=mock)

        m1 = [{"role": "user", "content": "Turn 1"}]
        m2 = [{"role": "user", "content": "Turn 1"}, {"role": "user", "content": "Turn 2"}]

        out1 = llm.call(m1)
        out2 = llm.call(m2)

        assert out1 == "First"
        assert out2 == "Second"
        assert mock.invoke_llm_call_count == 2
        # Second call should only send the delta (Turn 2).
        sent = mock.last_request["new_messages"]
        assert len(sent) == 1
        assert sent[0].content == "Turn 2"

    @pytest.mark.asyncio
    async def test_tool_calls_execute_available_functions(self) -> None:
        from identark.integrations.crewai import IdentArkCrewAILLM

        def add(a: int, b: int) -> int:
            return a + b

        mock = MockGateway()
        mock.queue_response(_make_tool_response())
        mock.queue_response(_make_text_response("Result is 5"))
        llm = IdentArkCrewAILLM(gateway=mock)

        # Use call() (sync) even in async test; it will run in a thread.
        out = llm.call(
            [{"role": "user", "content": "Add 2 and 3"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            available_functions={"add": add},
        )

        assert out == "Result is 5"
        assert mock.invoke_llm_call_count == 2
