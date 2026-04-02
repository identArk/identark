"""
LlamaIndex agent — IdentArkLLM with tool calling.

Run:
    pip install identark-sdk[llamaindex] openai
    export OPENAI_API_KEY=sk-...
    python examples/llamaindex_agent.py
"""

import asyncio

from llama_index.core.llms import ChatMessage, MessageRole
from openai import AsyncOpenAI

from identark import DirectGateway
from identark.integrations.llamaindex import IdentArkLLM

# ── Tool definition (OpenAI function-calling format) ─────────────────────────

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'London'"},
            },
            "required": ["city"],
        },
    },
}


def get_weather(city: str) -> str:
    """Stub — replace with a real weather API call."""
    return f"It is sunny and 18°C in {city}."


# ── Agent loop ────────────────────────────────────────────────────────────────


async def main() -> None:
    gateway = DirectGateway(
        llm_client=AsyncOpenAI(),
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant. Use tools when appropriate.",
    )
    llm = IdentArkLLM(gateway=gateway)

    messages = [ChatMessage(role=MessageRole.USER, content="What is the weather in London?")]

    # First turn — model decides to call the tool
    response = await llm.achat(messages, tools=[GET_WEATHER_TOOL])
    print(f"Finish reason : {response.raw.get('finish_reason')}")

    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            fn = tc["function"]
            import json
            args = json.loads(fn["arguments"])
            print(f"Tool call     : {fn['name']}({args})")
            result = get_weather(**args)

            # Feed the tool result back
            messages.append(response.message)
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=result,
                    additional_kwargs={"tool_call_id": tc["id"]},
                )
            )

        # Second turn — model summarises the tool result
        final = await llm.achat(messages)
        print(f"\nAssistant     : {final.message.content}")
    else:
        print(f"\nAssistant     : {response.message.content}")

    cost = await gateway.get_session_cost()
    print(f"\nSession cost  : ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
