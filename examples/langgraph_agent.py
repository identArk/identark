#!/usr/bin/env python3
"""
LangGraph ReAct agent — IdentArk credential isolation with tool calling.

Demonstrates a multi-node LangGraph where the LLM node routes through
IdentArk's gateway so API keys never enter the agent graph state.

Run::

    pip install identark[langgraph] openai
    export OPENAI_API_KEY=sk-...
    python examples/langgraph_agent.py

The graph structure::

        ┌─────────┐
        │  start  │
        └────┬────┘
             ▼
        ┌─────────┐     tool_calls?     ┌─────────┐
        │  agent  │ ──────────────────► │  tools  │
        │(IdentArk│ ◄────────────────── │         │
        │ Node)   │    tool results     └─────────┘
        └────┬────┘
             │ no tool_calls
             ▼
        ┌─────────┐
        │   END   │
        └─────────┘
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from openai import AsyncOpenAI

from identark import DirectGateway
from identark.integrations.langgraph import IdentArkNode

# ── Simulated tools (replace with real APIs in production) ────────────────────


def get_weather(city: str) -> str:
    """Return current weather for a city."""
    # In production this would call OpenWeatherMap, etc.
    # The API key would live in IdentArk vault, not in code.
    return f"Sunny and 22°C in {city}."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Very restricted eval for demo purposes
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# OpenAI-format tool schemas
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'London' or 'Tokyo'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '15 * 23' or '2**10'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# ── Custom tool node ──────────────────────────────────────────────────────────


def tool_node(state: MessagesState) -> dict[str, list[ToolMessage]]:
    """
    Execute tool calls found in the last AIMessage.

    In production you might use LangGraph's prebuilt ``ToolNode``,
    but this explicit version shows exactly how tool results flow
    back into the graph state.
    """
    messages = state["messages"]
    last_msg = messages[-1]

    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {"messages": []}

    tool_results: list[ToolMessage] = []
    for tc in last_msg.tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        tool_id = tc.get("id", "")

        if name == "get_weather":
            output = get_weather(**args)
        elif name == "calculate":
            output = calculate(**args)
        else:
            output = f"Unknown tool: {name}"

        tool_results.append(ToolMessage(content=output, tool_call_id=tool_id))
        print(f"   🔧 {name}({args}) → {output}")

    return {"messages": tool_results}


# ── Conditional routing ───────────────────────────────────────────────────────


def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Route to tools if the last AI message requested tool calls."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_react_agent(gateway: DirectGateway) -> StateGraph:
    """Compile a ReAct agent backed by IdentArk."""
    agent_node = IdentArkNode(gateway=gateway, tools=TOOL_DEFINITIONS)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        print("   export OPENAI_API_KEY=sk-...")
        raise SystemExit(1)

    # IdentArk isolates the credential. The graph never sees the API key.
    gateway = DirectGateway(
        llm_client=AsyncOpenAI(api_key=api_key),
        model="gpt-4o-mini",
        system_prompt=(
            "You are a helpful assistant with access to weather and calculator tools. "
            "Use tools when they help answer the user's question. "
            "Be concise."
        ),
    )

    app = build_react_agent(gateway)

    # Example query that triggers both tools
    query = "What's the weather in Tokyo and what is 847 divided by 13?"

    print("=" * 60)
    print("🌐 IdentArk + LangGraph ReAct Agent")
    print("=" * 60)
    print(f"Query: {query}")
    print("-" * 60)

    result = await app.ainvoke({"messages": [{"role": "user", "content": query}]})

    # Print the conversation
    print("\n📜 Conversation transcript:")
    for msg in result["messages"]:
        if msg.type == "human":
            print(f"   👤 {msg.content}")
        elif msg.type == "ai":
            print(f"   🤖 {msg.content}")

    # IdentArk cost tracking
    cost = await gateway.get_session_cost()
    print(f"\n💰 Session cost : ${cost:.6f}")
    print(f"📊 Turns taken   : {len([m for m in result['messages'] if m.type == 'human'])}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
