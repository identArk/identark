# LlamaIndex Integration

IdentArk provides `IdentArkLLM`, a custom LLM class for LlamaIndex.

## Installation

```bash
pip install identark[llamaindex]
```

## Quick Start

```python
from identark.integrations.llamaindex import IdentArkLLM
from identark import DirectGateway
from openai import AsyncOpenAI

# Create gateway
gateway = DirectGateway(
    llm_client=AsyncOpenAI(),
    model="gpt-4o",
)

# Create LlamaIndex LLM
llm = IdentArkLLM(gateway=gateway)

# Use directly
response = llm.complete("What is the meaning of life?")
print(response.text)
```

## Chat Interface

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Hello!"),
]

response = llm.chat(messages)
print(response.message.content)
```

## Streaming

```python
# Stream completion
for chunk in llm.stream_complete("Tell me a story"):
    print(chunk.delta, end="", flush=True)

# Stream chat
for chunk in llm.stream_chat(messages):
    print(chunk.delta, end="", flush=True)
```

## With Index/Query Engine

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index with IdentArk LLM
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
```

## Production Setup

```python
from identark import ControlPlaneGateway

gateway = ControlPlaneGateway()
llm = IdentArkLLM(gateway=gateway)
```

## Async Support

```python
import asyncio

async def main():
    response = await llm.acomplete("Async query")
    print(response.text)

    # Async chat
    response = await llm.achat(messages)
    print(response.message.content)

asyncio.run(main())
```

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `gateway` | `AgentGateway` | IdentArk gateway instance |

The model and provider are configured on the gateway, not the LLM wrapper.

## Supported Methods

| Method | Sync | Async | Streaming |
|--------|------|-------|-----------|
| `complete` | ✓ | ✓ | ✓ |
| `chat` | ✓ | ✓ | ✓ |

## Cost Tracking

```python
cost = await gateway.get_session_cost()
print(f"Query cost: ${cost:.6f}")
```
