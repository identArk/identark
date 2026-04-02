"""
Mistral AI (EU) — DirectGateway with data processed inside the European Union.

Mistral AI is a French company. All inference runs in EU data centres.
Use this when your data governance, GDPR obligations, or UK/EU sovereignty
requirements prohibit sending data to US-based cloud providers.

Run:
    export MISTRAL_API_KEY=...
    python examples/mistral_eu.py
"""

import asyncio
import os

from openai import AsyncOpenAI

from identark import DirectGateway, Message, Role


async def main() -> None:
    # Mistral exposes an OpenAI-compatible API — no mistralai package needed.
    # DirectGateway auto-detects "mistral" from the base_url.
    gateway = DirectGateway(
        llm_client=AsyncOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=os.environ["MISTRAL_API_KEY"],
        ),
        model="mistral-small-latest",
        system_prompt="You are a helpful assistant. Reply concisely.",
    )

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="What is data sovereignty in the context of AI?")]
    )

    print(response.message.content)
    print(f"\nProvider : {gateway.provider}")   # "mistral"
    print(f"Model    : {gateway.model}")
    print(f"Cost     : ${response.cost_usd:.6f}")
    print(f"Tokens   : {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    cost = await gateway.get_session_cost()
    print(f"Session  : ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
