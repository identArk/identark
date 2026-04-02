# CrewAI Integration

IdentArk provides `IdentArkCrewAILLM` for use with CrewAI agents and crews.

## Installation

```bash
pip install identark-sdk[all]
pip install crewai
```

## Quick Start

```python
from crewai import Agent, Task, Crew
from identark.integrations.crewai import IdentArkCrewAILLM
from identark import DirectGateway
from openai import AsyncOpenAI

# Create gateway
gateway = DirectGateway(
    llm_client=AsyncOpenAI(),
    model="gpt-4o",
)

# Create CrewAI-compatible LLM
llm = IdentArkCrewAILLM(gateway=gateway)

# Create agent with IdentArk LLM
researcher = Agent(
    role="Researcher",
    goal="Research and summarize topics",
    backstory="You are an expert researcher.",
    llm=llm,
)

# Create task
task = Task(
    description="Research the history of artificial intelligence",
    agent=researcher,
    expected_output="A brief summary of AI history",
)

# Create and run crew
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

## Production Setup

```python
from identark import ControlPlaneGateway

gateway = ControlPlaneGateway()
llm = IdentArkCrewAILLM(gateway=gateway)

# All agents use the secure gateway
agent = Agent(
    role="Assistant",
    goal="Help users",
    backstory="You are helpful.",
    llm=llm,
)
```

## Multiple Agents

```python
# Each agent can share the same gateway (shared session/cost tracking)
# Or use separate gateways for isolated sessions

gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
llm = IdentArkCrewAILLM(gateway=gateway)

researcher = Agent(role="Researcher", goal="Research", backstory="...", llm=llm)
writer = Agent(role="Writer", goal="Write", backstory="...", llm=llm)
editor = Agent(role="Editor", goal="Edit", backstory="...", llm=llm)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
)
```

## Data Residency

For UK/EU compliance:

```python
from identark import DirectGateway
from openai import AsyncOpenAI

# Use Mistral AI (EU data centres)
gateway = DirectGateway(
    llm_client=AsyncOpenAI(
        base_url="https://api.mistral.ai/v1",
        api_key="your-mistral-key",
    ),
    model="mistral-large-latest",
)

llm = IdentArkCrewAILLM(gateway=gateway)
```

## Cost Tracking

```python
# After crew execution
import asyncio

async def get_cost():
    return await gateway.get_session_cost()

cost = asyncio.run(get_cost())
print(f"Crew execution cost: ${cost:.6f}")
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `gateway` | `AgentGateway` | IdentArk gateway instance |

Model and provider configuration is done on the gateway level.
