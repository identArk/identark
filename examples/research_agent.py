#!/usr/bin/env python3
"""
IdentArk Research Agent — Deep product research with Gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A production-ready AI research agent built with IdentArk SDK.
Demonstrates the full AgentGateway Protocol workflow:

1. Zero-secret agent design (API key isolated in gateway)
2. Tool calling for web search
3. Multi-turn conversation with memory
4. Cost tracking and caps
5. Streaming output

Usage:
    export GEMINI_API_KEY=AIza...
    python examples/research_agent.py "research Tesla Model 3"

Or run with --demo for a pre-defined research task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identark import Message, Role
from identark.integrations.gemini import GeminiGateway

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("research_agent")


# ── Research Tools ─────────────────────────────────────────────────────────────

RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for finding recent news, pricing, reviews, and facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and extract text content from a webpage URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_product",
            "description": "Get detailed product specifications and analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product to analyze"
                    },
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Aspects to analyze: pricing, features, competitors, reviews, etc."
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Compare multiple products side by side",
            "parameters": {
                "type": "object",
                "properties": {
                    "products": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product names to compare"
                    },
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Comparison criteria: price, features, quality, etc."
                    }
                },
                "required": ["products"]
            }
        }
    }
]


# ── Simulated Tool Execution (replace with real APIs in production) ───────────

def execute_web_search(query: str, num_results: int = 5) -> str:
    """Simulate web search results."""
    # In production, use Google Search API, Serper, or SerpAPI
    results = [
        {"title": f"Result for: {query}", "snippet": f"Comprehensive information about {query}...", "url": f"https://example.com/{i}"}
        for i in range(min(num_results, 5))
    ]

    # Add realistic mock data based on query
    if "tesla" in query.lower():
        results = [
            {"title": "Tesla Model 3 | Electric Cars | Tesla UK", "snippet": "Starting at £39,990. 0-60 mph in 3.1 seconds. Up to 341 miles range.", "url": "https://tesla.com/model3"},
            {"title": "Tesla Model 3 Review 2024 | Top Gear", "snippet": "The Model 3 remains the benchmark electric saloon. Excellent range, superb tech, and genuinely fun to drive.", "url": "https://topgear.com/tesla-model-3"},
            {"title": "Tesla Model 3 vs BMW i4 Comparison", "snippet": "Head-to-head comparison of two premium electric sedans. Tesla wins on range, BMW on interior quality.", "url": "https://carwow.co.uk/tesla-vs-bmw"},
            {"title": "Is Tesla Model 3 Worth It in 2024?", "snippet": "After price cuts, the Model 3 offers compelling value. Here's our full analysis.", "url": "https://electrek.co/model3-review"},
            {"title": "Tesla Model 3 Highland Update", "snippet": "2024 refresh brings new design, better range, and improved interior. Full breakdown inside.", "url": "https://insideevs.com/highland"},
        ]
    elif "ai agent" in query.lower() or "framework" in query.lower():
        results = [
            {"title": "Best AI Agent Frameworks 2024", "snippet": "LangChain, CrewAI, AutoGPT comparison. Which is right for your use case?", "url": "https://aiframework.com/comparison"},
            {"title": "Building Production AI Agents", "snippet": "Key considerations: security, cost control, observability. IdentArk solves credential isolation.", "url": "https://identark.ai/docs"},
            {"title": "LangGraph vs CrewAI", "snippet": "Two approaches to multi-agent systems. LangGraph for complex workflows, CrewAI for team collaboration.", "url": "https://langchain.dev/comparison"},
        ]

    return json.dumps({"results": results, "query": query})


def execute_fetch_webpage(url: str) -> str:
    """Simulate webpage fetch."""
    # In production, use httpx + BeautifulSoup or a headless browser
    return json.dumps({
        "url": url,
        "title": f"Page content from {url}",
        "content": f"Extracted content from {url}. This would contain the actual page text in production.",
        "word_count": 1500
    })


def execute_analyze_product(product_name: str, aspects: list[str] | None = None) -> str:
    """Simulate product analysis."""
    aspects = aspects or ["pricing", "features", "reviews"]

    if "tesla" in product_name.lower() or "model 3" in product_name.lower():
        analysis = {
            "product": "Tesla Model 3",
            "category": "Electric Vehicle",
            "pricing": {
                "base": "£39,990 (Rear-Wheel Drive)",
                "mid": "£47,990 (Long Range)",
                "top": "£52,990 (Performance)",
                "note": "Prices as of March 2024, may vary"
            },
            "features": [
                "Up to 341 miles range (Long Range)",
                "0-60 mph in 3.1 seconds (Performance)",
                "15-inch touchscreen with Netflix/YouTube",
                "Autopilot included, Full Self-Driving £6,800 extra",
                "Supercharger network access",
                "Over-the-air updates"
            ],
            "reviews": {
                "average_rating": 4.7,
                "total_reviews": 12500,
                "pros": ["Best-in-class range", "Fast charging", "Low running costs", "Fun to drive"],
                "cons": ["Panel gaps on some units", "No CarPlay/Android Auto", "Service centre availability"]
            },
            "competitors": ["BMW i4", "Polestar 2", "Hyundai Ioniq 6", "Mercedes EQE"]
        }
    else:
        analysis = {
            "product": product_name,
            "analysis": "Product analysis data would be fetched from product databases and review aggregators.",
            "aspects_analyzed": aspects
        }

    return json.dumps(analysis)


def execute_compare_products(products: list[str], criteria: list[str] | None = None) -> str:
    """Simulate product comparison."""
    criteria = criteria or ["price", "features", "rating"]

    comparison = {
        "products": products,
        "criteria": criteria,
        "comparison_table": {
            product: {
                "price_range": "£35,000 - £55,000",
                "rating": 4.5,
                "key_strength": "Varies by product"
            }
            for product in products
        },
        "recommendation": f"Based on {', '.join(criteria)}, detailed comparison follows..."
    }

    return json.dumps(comparison)


def execute_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Route tool calls to appropriate handler."""
    handlers = {
        "web_search": lambda: execute_web_search(args["query"], args.get("num_results", 5)),
        "fetch_webpage": lambda: execute_fetch_webpage(args["url"]),
        "analyze_product": lambda: execute_analyze_product(args["product_name"], args.get("aspects")),
        "compare_products": lambda: execute_compare_products(args["products"], args.get("criteria")),
    }

    handler = handlers.get(tool_name)
    if handler:
        return handler()
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Research Agent ─────────────────────────────────────────────────────────────

async def run_research_agent(
    query: str,
    gateway: GeminiGateway,
    max_iterations: int = 5,
    stream: bool = True,
) -> str:
    """
    Run a research agent that uses tools to gather information.

    The agent will:
    1. Plan research approach
    2. Use tools to gather data
    3. Synthesize findings
    4. Return comprehensive report
    """

    system_context = """You are an expert research analyst. Your job is to provide thorough, accurate research reports.

When given a research task:
1. First, use web_search to find current information
2. Use analyze_product for detailed product data
3. Use compare_products when comparisons are needed
4. Synthesize all findings into a clear, actionable report

Always cite your sources and be specific about data. If information seems outdated or uncertain, note that clearly.

Format your final report with:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Detailed Analysis
- Recommendations
- Sources Used"""

    # Start with research request
    messages = [
        Message(role=Role.USER, content=f"{system_context}\n\nResearch task: {query}")
    ]

    full_response = ""
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'─' * 60}")
        print(f"Research iteration {iteration}/{max_iterations}")
        print(f"{'─' * 60}")

        # Call LLM with tools
        response = await gateway.invoke_llm(
            new_messages=messages,
            tools=RESEARCH_TOOLS,
        )

        # Check for tool calls
        if response.tool_calls:
            print(f"\n🔧 Agent using {len(response.tool_calls)} tool(s):")

            tool_results = []
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"   → {tc.function.name}({json.dumps(args, indent=2)[:100]}...)")

                # Execute tool
                result = execute_tool(tc.function.name, args)
                tool_results.append(Message(
                    role=Role.TOOL,
                    content=result,
                    tool_call_id=tc.id,
                ))

            # Add tool results for next iteration
            messages = tool_results

        else:
            # No tool calls — agent is done researching
            full_response = response.message.content
            print(f"\n✅ Research complete!")
            break

    # Print cost summary
    cost = await gateway.get_session_cost()
    print(f"\n{'═' * 60}")
    print(f"📊 Session Summary")
    print(f"{'═' * 60}")
    print(f"Total cost: ${cost:.6f}")
    print(f"Model: {gateway.model}")
    print(f"Messages in history: {len(gateway.history)}")
    print(f"Iterations: {iteration}")

    return full_response


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="IdentArk Research Agent")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--demo", action="store_true", help="Run demo research")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model to use")
    parser.add_argument("--cost-cap", type=float, default=0.50, help="Max cost in USD")
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Get your key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    # Determine query
    if args.demo:
        query = "Research Tesla Model 3: current pricing, features, comparison with BMW i4, and whether it's worth buying in 2024"
    elif args.query:
        query = args.query
    else:
        print("Usage: python research_agent.py 'your research query'")
        print("       python research_agent.py --demo")
        sys.exit(1)

    print("=" * 60)
    print("🔬 IdentArk Research Agent")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Model: {args.model}")
    print(f"Cost cap: ${args.cost_cap}")
    print("=" * 60)

    # Initialize gateway (credentials isolated here, not in agent logic)
    gateway = GeminiGateway(
        api_key=api_key,
        model=args.model,
        cost_cap_usd=args.cost_cap,
    )

    # Run research
    try:
        report = await run_research_agent(query, gateway)

        print("\n" + "=" * 60)
        print("📋 RESEARCH REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
