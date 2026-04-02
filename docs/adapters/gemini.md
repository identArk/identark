# Gemini Integration

Native Google Gemini integration for IdentArk using the `google-generativeai` SDK.

## Installation

```bash
pip install identark[gemini]
```

## Quick Start

```python
from identark.integrations.gemini import GeminiGateway
from identark import Message, Role

gateway = GeminiGateway(
    api_key="your-gemini-api-key",  # From https://aistudio.google.com/apikey
    model="gemini-1.5-pro",
    system_prompt="You are a helpful assistant.",
)

response = await gateway.invoke_llm(
    new_messages=[Message(role=Role.USER, content="Hello, Gemini!")]
)

print(response.message.content)
print(f"Cost: ${response.cost_usd:.6f}")
```

## Supported Models

| Model | Best For | Cost (per 1M tokens) |
|-------|----------|---------------------|
| `gemini-1.5-pro` | Complex reasoning, long context | $1.25 in / $5.00 out |
| `gemini-1.5-flash` | Fast responses, general use | $0.075 in / $0.30 out |
| `gemini-1.5-flash-8b` | Cheapest, simple tasks | $0.0375 in / $0.15 out |
| `gemini-2.0-flash-exp` | Experimental features | $0.10 in / $0.40 out |

## Configuration Options

```python
gateway = GeminiGateway(
    api_key="your-api-key",
    model="gemini-1.5-pro",

    # Optional parameters
    system_prompt="You are a financial advisor.",
    cost_cap_usd=1.00,  # Stop if cost exceeds $1
    workspace_dir="/workspace",  # For file operations

    # Gemini-specific settings
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    },
    generation_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 4096,
    },
)
```

## Tool Calling (Function Calling)

GeminiGateway supports the OpenAI tool format for compatibility:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., AAPL)"
                    }
                },
                "required": ["symbol"]
            }
        }
    }
]

response = await gateway.invoke_llm(
    new_messages=[Message(role=Role.USER, content="What's Apple's stock price?")],
    tools=tools,
)

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Call: {tc.function.name}({tc.function.arguments})")

        # Execute the tool and send result back
        result = execute_my_tool(tc.function.name, tc.function.arguments)

        tool_response = await gateway.invoke_llm(
            new_messages=[Message(
                role=Role.TOOL,
                content=result,
                tool_call_id=tc.id,
            )],
            tools=tools,
        )
```

## Streaming

```python
async for chunk in gateway.invoke_llm_stream(
    new_messages=[Message(role=Role.USER, content="Write a story")]
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.finish_reason:
        print(f"\n\n[Done: {chunk.output_tokens} tokens]")
```

## Multi-Turn Conversations

The gateway automatically maintains conversation history:

```python
# First turn
await gateway.invoke_llm(
    new_messages=[Message(role=Role.USER, content="My name is Alex.")]
)

# Second turn - Gemini remembers the context
response = await gateway.invoke_llm(
    new_messages=[Message(role=Role.USER, content="What's my name?")]
)
# Response: "Your name is Alex."

# Check history
print(len(gateway.history))  # 4 messages

# Reset when needed
gateway.reset()
```

## Cost Tracking

```python
# Check current session cost
cost = await gateway.get_session_cost()
print(f"Session cost: ${cost:.6f}")

# Set a cost cap
gateway = GeminiGateway(
    api_key="...",
    model="gemini-1.5-flash",
    cost_cap_usd=0.50,  # Raises CostCapExceededError if exceeded
)

try:
    response = await gateway.invoke_llm(...)
except CostCapExceededError as e:
    print(f"Cost cap of ${e.cap_usd} reached!")
```

## Multimodal Input

Gemini supports images and other media:

```python
import base64

# Load image
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = await gateway.invoke_llm(
    new_messages=[Message(
        role=Role.USER,
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            }
        ]
    )]
)
```

## Error Handling

```python
from identark.exceptions import (
    ConfigurationError,
    CostCapExceededError,
    ContentPolicyError,
    RateLimitError,
    ProviderError,
)

try:
    response = await gateway.invoke_llm(...)
except CostCapExceededError:
    print("Cost cap reached!")
except RateLimitError:
    print("Rate limited - try again later")
except ContentPolicyError:
    print("Content blocked by safety filters")
except ProviderError as e:
    print(f"Gemini API error: {e}")
```

## Comparison with OpenAI-Compatible Mode

You can also use Gemini via the OpenAI-compatible API:

```python
# Native GeminiGateway (recommended)
from identark.integrations.gemini import GeminiGateway
gateway = GeminiGateway(api_key="...", model="gemini-1.5-pro")

# OR via OpenAI-compatible endpoint (alternative)
from openai import AsyncOpenAI
from identark import DirectGateway

gateway = DirectGateway(
    llm_client=AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="your-gemini-key",
    ),
    model="gemini-1.5-pro",
)
```

**When to use GeminiGateway:**
- Access to Gemini-specific features (grounding, multimodal)
- Native SDK performance
- Better error messages

**When to use DirectGateway with OpenAI endpoint:**
- Consistent code across providers
- Already using OpenAI SDK patterns

## Full Example: Finance Agent

See `examples/gemini_finance_agent.py` for a complete banking assistant example.

```bash
export GEMINI_API_KEY=your-key
python examples/gemini_finance_agent.py --demo
```
