# LangGraph Integration

IdentArk provides `IdentArkNode` and `IdentArkStreamNode` for use in LangGraph state machines.

## Installation

```bash
pip install identark[langgraph]
```

## Quick Start

```python
from langgraph.graph import StateGraph, END
from identark.integrations.langgraph import IdentArkNode
from identark import DirectGateway
from openai import AsyncOpenAI
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Create gateway and node
gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
llm_node = IdentArkNode(gateway=gateway)

# Build graph
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

# Compile and run
app = graph.compile()
result = app.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

## Streaming Node

```python
from identark.integrations.langgraph import IdentArkStreamNode

stream_node = IdentArkStreamNode(gateway=gateway)

# In a graph with streaming
async for chunk in app.astream({
    "messages": [{"role": "user", "content": "Tell me a story"}]
}):
    print(chunk)
```

## With Tool Routing

```python
from langgraph.prebuilt import ToolNode

def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")

app = graph.compile()
```

## Production Setup

```python
from identark import ControlPlaneGateway

# Zero credentials in the agent
gateway = ControlPlaneGateway()
llm_node = IdentArkNode(gateway=gateway)
```

## Node Classes

### IdentArkNode

Standard node that returns complete responses.

```python
IdentArkNode(
    gateway: AgentGateway,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
)
```

### IdentArkStreamNode

Streaming node for real-time token output.

```python
IdentArkStreamNode(
    gateway: AgentGateway,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
)
```

## State Requirements

Your state must include a `messages` key with a list of message dicts:

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
```

Messages follow the standard format:
```python
{"role": "user", "content": "Hello"}
{"role": "assistant", "content": "Hi there!"}
```
