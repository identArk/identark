#!/usr/bin/env python3
"""
Gemini Quickstart Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates using IdentArk with Google's Gemini models.

Setup:
    1. Get an API key from https://aistudio.google.com/apikey
    2. Set the environment variable: export GEMINI_API_KEY=your-key
    3. Install: pip install identark[gemini]
    4. Run: python examples/gemini_quickstart.py
"""

import asyncio
import os

from identark import Message, Role
from identark.integrations.gemini import GeminiGateway


async def basic_chat():
    """Simple chat with Gemini."""
    print("=" * 60)
    print("Basic Chat with Gemini")
    print("=" * 60)

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash",  # Fast and cheap
        system_prompt="You are a helpful assistant. Be concise.",
    )

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="What is the capital of France?")]
    )

    print(f"Response: {response.message.content}")
    print(f"Model: {response.model}")
    print(f"Cost: ${response.cost_usd:.6f}")
    print(f"Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
    print()


async def multi_turn_conversation():
    """Multi-turn conversation with history."""
    print("=" * 60)
    print("Multi-turn Conversation")
    print("=" * 60)

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
    )

    # First turn
    response1 = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="My name is Alex and I work in finance.")]
    )
    print(f"User: My name is Alex and I work in finance.")
    print(f"Gemini: {response1.message.content}")
    print()

    # Second turn - Gemini should remember the context
    response2 = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="What's my name and profession?")]
    )
    print(f"User: What's my name and profession?")
    print(f"Gemini: {response2.message.content}")
    print()

    total_cost = await gateway.get_session_cost()
    print(f"Total session cost: ${total_cost:.6f}")
    print()


async def tool_calling():
    """Function/tool calling with Gemini."""
    print("=" * 60)
    print("Tool Calling")
    print("=" * 60)

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-pro",  # Better at tool calling
    )

    # Define a tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol, e.g., AAPL, GOOGL"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="What's Apple's stock price?")],
        tools=tools,
    )

    print(f"User: What's Apple's stock price?")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")

        # Simulate tool response
        tool_result = Message(
            role=Role.TOOL,
            content='{"price": 178.50, "currency": "USD"}',
            tool_call_id=response.tool_calls[0].id,
        )

        # Send tool result back
        final_response = await gateway.invoke_llm(new_messages=[tool_result])
        print(f"Gemini: {final_response.message.content}")
    else:
        print(f"Gemini: {response.message.content}")

    print()


async def streaming():
    """Streaming response from Gemini."""
    print("=" * 60)
    print("Streaming Response")
    print("=" * 60)

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
    )

    print("User: Write a haiku about AI.")
    print("Gemini: ", end="", flush=True)

    async for chunk in gateway.invoke_llm_stream(
        new_messages=[Message(role=Role.USER, content="Write a haiku about AI.")]
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.finish_reason:
            print()
            print(f"\n[Finished: {chunk.finish_reason}, {chunk.output_tokens} tokens]")

    print()


async def cost_tracking():
    """Demonstrate cost cap enforcement."""
    print("=" * 60)
    print("Cost Tracking with Cap")
    print("=" * 60)

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
        cost_cap_usd=0.001,  # Very low cap for demo
    )

    try:
        for i in range(10):
            response = await gateway.invoke_llm(
                new_messages=[Message(role=Role.USER, content=f"Say 'Hello {i}'")]
            )
            cost = await gateway.get_session_cost()
            print(f"Request {i+1}: cost so far ${cost:.6f}")
            gateway.reset()  # Reset history but keep cost
    except Exception as e:
        print(f"Cost cap reached: {e}")

    print()


async def main():
    """Run all examples."""
    if "GEMINI_API_KEY" not in os.environ:
        print("Please set GEMINI_API_KEY environment variable")
        print("Get your key from: https://aistudio.google.com/apikey")
        return

    await basic_chat()
    await multi_turn_conversation()
    await tool_calling()
    await streaming()
    await cost_tracking()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
