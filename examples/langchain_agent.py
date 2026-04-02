"""
LangChain agent — IdentArkChatModel with tool calling.

Run:
    pip install identark[langchain] openai
    export OPENAI_API_KEY=sk-...
    python examples/langchain_agent.py
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, ToolMessage
from openai import AsyncOpenAI

from identark import DirectGateway
from identark.integrations.langchain import IdentArkChatModel

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
    llm = IdentArkChatModel(gateway=gateway)

    messages = [HumanMessage(content="What is the weather like in London right now?")]

    # First LLM turn — model decides to call the tool
    response = await llm.ainvoke(messages, tools=[GET_WEATHER_TOOL])
    print(f"Finish reason : {response.response_metadata['finish_reason']}")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"Tool call     : {tc['name']}({tc['args']})")
            result = get_weather(**tc["args"])

            # Feed the tool result back
            messages.append(response)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # Second LLM turn — model summarises the tool result
        final = await llm.ainvoke(messages)
        print(f"\nAssistant     : {final.content}")
    else:
        print(f"\nAssistant     : {response.content}")

    cost = await gateway.get_session_cost()
    print(f"\nSession cost  : ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
