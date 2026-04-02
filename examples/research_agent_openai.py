#!/usr/bin/env python3
"""
IdentArk Research Agent — Deep product research with OpenAI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same research agent but using DirectGateway with OpenAI.
Demonstrates the AgentGateway Protocol's provider-agnostic design.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/research_agent_openai.py "research Tesla Model 3"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identark import DirectGateway, Message, Role

# ── Research Tools ─────────────────────────────────────────────────────────────

RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_product",
            "description": "Get detailed product specifications",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product"}
                },
                "required": ["product_name"]
            }
        }
    }
]


def execute_tool(tool_name: str, args: dict) -> str:
    """Execute research tools (simulated for demo)."""
    if tool_name == "web_search":
        query = args.get("query", "")
        if "tesla" in query.lower():
            return json.dumps({
                "results": [
                    {"title": "Tesla Model 3 2024 Review", "snippet": "Starting at £39,990. 341mi range. 0-60 in 3.1s."},
                    {"title": "Model 3 vs BMW i4", "snippet": "Tesla wins on range, BMW on interior quality."},
                    {"title": "Is Model 3 Worth It?", "snippet": "After price cuts, compelling value proposition."},
                ]
            })
        return json.dumps({"results": [{"title": f"Results for {query}", "snippet": "..."}]})

    elif tool_name == "analyze_product":
        product = args.get("product_name", "")
        if "tesla" in product.lower() or "model 3" in product.lower():
            return json.dumps({
                "product": "Tesla Model 3",
                "pricing": {"base": "£39,990", "long_range": "£47,990", "performance": "£52,990"},
                "features": ["341mi range", "0-60 in 3.1s", "Autopilot included"],
                "rating": 4.7,
                "competitors": ["BMW i4", "Polestar 2", "Hyundai Ioniq 6"]
            })
        return json.dumps({"product": product, "status": "analysis_available"})

    return json.dumps({"error": "Unknown tool"})


async def run_research_agent(query: str, gateway: DirectGateway, max_iterations: int = 5) -> str:
    """Run research agent with tool calling."""
    system = """You are a research analyst. When given a topic:
1. Use web_search to find current information
2. Use analyze_product for detailed specs
3. Synthesize into a clear report with: Summary, Key Findings, Recommendations"""

    messages = [Message(role=Role.USER, content=f"{system}\n\nResearch: {query}")]
    full_response = ""

    for i in range(max_iterations):
        print(f"\n{'─'*50}\nIteration {i+1}/{max_iterations}\n{'─'*50}")

        response = await gateway.invoke_llm(new_messages=messages, tools=RESEARCH_TOOLS)

        if response.tool_calls:
            print(f"🔧 Using {len(response.tool_calls)} tool(s):")
            tool_results = []
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"   → {tc.function.name}({args})")
                result = execute_tool(tc.function.name, args)
                tool_results.append(Message(role=Role.TOOL, content=result, tool_call_id=tc.id))
            messages = tool_results
        else:
            full_response = response.message.content
            print("✅ Research complete!")
            break

    cost = await gateway.get_session_cost()
    print(f"\n{'═'*50}\n📊 Cost: ${cost:.6f} | Model: {gateway.model}\n{'═'*50}")
    return full_response


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Research Tesla Model 3 pricing and features")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    from openai import AsyncOpenAI
    gateway = DirectGateway(llm_client=AsyncOpenAI(api_key=api_key), model=args.model)

    print(f"🔬 IdentArk Research Agent\nQuery: {args.query}\nModel: {args.model}")

    report = await run_research_agent(args.query, gateway)
    print(f"\n📋 REPORT:\n{report}")


if __name__ == "__main__":
    asyncio.run(main())
