"""
Ollama (fully local) — DirectGateway with zero data egress, zero cost.

Requirements:
    brew install ollama          # macOS
    ollama pull llama3.2         # download the model once
    ollama serve                 # start the local server

Run:
    python examples/ollama_local.py

No API key required. All inference runs on your machine.
Data never leaves your hardware — ideal for sovereign / air-gapped deployments.
"""

import asyncio

from openai import AsyncOpenAI

from identark import DirectGateway, Message, Role


async def main() -> None:
    # Ollama exposes an OpenAI-compatible API at localhost:11434
    # provider="local" tells DirectGateway that cost is always £0/$0
    gateway = DirectGateway(
        llm_client=AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required by the openai SDK; ignored by Ollama
        ),
        model="llama3.2",
        provider="local",  # forces $0 cost tracking; no data leaves your machine
        system_prompt="You are a helpful assistant.",
    )

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Explain the AgentGateway Protocol in one paragraph.")]
    )

    print(response.message.content)
    print(f"\nProvider : {gateway.provider}")
    print(f"Cost     : ${response.cost_usd:.6f}  (always $0.00 for local models)")

    cost = await gateway.get_session_cost()
    print(f"Session  : ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
