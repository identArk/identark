# LangChain Integration

IdentArk provides `IdentArkChatModel`, a drop-in replacement for LangChain chat models.

## Installation

```bash
pip install identark-sdk[langchain]
```

## Quick Start

```python
from identark.integrations.langchain import IdentArkChatModel
from identark import DirectGateway
from openai import AsyncOpenAI

# Create gateway
gateway = DirectGateway(
    llm_client=AsyncOpenAI(),
    model="gpt-4o",
)

# Create LangChain-compatible model
model = IdentArkChatModel(gateway=gateway)

# Use like any LangChain chat model
response = model.invoke("What is the capital of France?")
print(response.content)
```

## With ControlPlaneGateway (Production)

```python
from identark.integrations.langchain import IdentArkChatModel
from identark import ControlPlaneGateway

# Zero credentials in agent — all fetched from control plane
gateway = ControlPlaneGateway()
model = IdentArkChatModel(gateway=gateway)

response = model.invoke("Hello!")
```

## Streaming

```python
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## Async Support

```python
import asyncio

async def main():
    response = await model.ainvoke("Async hello!")
    print(response.content)

asyncio.run(main())
```

## Tool Calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in London?")
```

## In a Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | model | StrOutputParser()
result = chain.invoke({"input": "Hello!"})
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `gateway` | `AgentGateway` | IdentArk gateway instance |
| `temperature` | `float` | Sampling temperature (optional) |
| `max_tokens` | `int` | Maximum response tokens (optional) |

## Cost Tracking

```python
# After invocations
cost = await gateway.get_session_cost()
print(f"Total session cost: ${cost:.6f}")
```
