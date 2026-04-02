"""
CrewAI agent — IdentArkCrewAILLM (routes calls through IdentArk gateway).

Run:
    pip install identark-sdk crewai openai
    export OPENAI_API_KEY=sk-...
    python examples/crewai_agent.py
"""

from __future__ import annotations

from openai import AsyncOpenAI

from identark import DirectGateway
from identark.integrations.crewai import IdentArkCrewAILLM


def main() -> None:
    # Import CrewAI lazily so this example fails with a clean error message
    # if the user didn't install crewai.
    from crewai import Agent, Crew, Task

    gateway = DirectGateway(
        llm_client=AsyncOpenAI(),
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant. Be concise.",
        cost_cap_usd=0.05,
    )

    llm = IdentArkCrewAILLM(gateway=gateway)

    agent = Agent(
        role="Research assistant",
        goal="Answer questions accurately and concisely.",
        backstory="You are a careful assistant who cites assumptions explicitly.",
        llm=llm,
        allow_delegation=False,
    )

    task = Task(
        description="In one sentence, explain what the AgentGateway Protocol is.",
        expected_output="A single concise sentence.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()

