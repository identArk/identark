"""
OpenAI quickstart — DirectGateway with GPT-4o.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/openai_quickstart.py
"""

import asyncio

from openai import AsyncOpenAI

from sandcastle import DirectGateway, Message, Role


async def main() -> None:
    gateway = DirectGateway(
        llm_client=AsyncOpenAI(),
        model="gpt-4o-mini",
        system_prompt="You are a concise assistant. Reply in one sentence.",
        cost_cap_usd=0.05,
    )

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="What is the AgentGateway Protocol?")]
    )

    print(response.message.content)
    print(f"\nCost: ${response.cost_usd:.6f}")
    print(f"Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    cost = await gateway.get_session_cost()
    print(f"Session total: ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
