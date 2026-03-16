# sandcastle-sdk

**The AgentGateway Protocol — secure, scalable AI agent execution infrastructure.**

[![CI](https://github.com/Goldokpa/sandcastle-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Goldokpa/sandcastle-sdk/actions)
[![PyPI](https://img.shields.io/pypi/v/sandcastle-sdk)](https://pypi.org/project/sandcastle-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/sandcastle-sdk)](https://pypi.org/project/sandcastle-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The problem

When an AI agent can execute code, call APIs, or access files, it runs in a process. That process has an environment. That environment typically contains everything that can cause serious damage: LLM API keys, database credentials, AWS tokens.

The naive solution — run your agent on the same backend as your REST API — creates two problems at once:

1. **Security**: The agent can access every secret on the machine.
2. **Reliability**: A memory-hungry agent degrades your API. Redeploying your API kills all running agents.

`sandcastle-sdk` solves both.

---

## How it works

The SDK implements the **AgentGateway Protocol** — a clean interface between your agent logic and the outside world. Two implementations ship out of the box:

| Gateway | When to use | Credentials | History |
|---|---|---|---|
| `DirectGateway` | Local development, CI evals | Your API key | In-memory |
| `ControlPlaneGateway` | Production on Sandcastle | **Zero** — none in the agent | Control plane DB |

Your agent code is **identical** in both environments. The switch is two lines.

---

## Quick start

```bash
pip install sandcastle-sdk[openai]
```

```python
import asyncio
from openai import AsyncOpenAI
from sandcastle import DirectGateway, Message, Role

async def main():
    gateway = DirectGateway(
        llm_client=AsyncOpenAI(),   # Your API key — not in the agent loop
        model="gpt-4o",
    )

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello, Sandcastle!")]
    )

    print(response.message.content)
    print(f"Cost: ${response.cost_usd:.6f}")

asyncio.run(main())
```

### Moving to production

Change **two lines**. Your agent logic is untouched.

```python
# Before (local)
from sandcastle import DirectGateway
gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")

# After (production — agent holds zero secrets)
from sandcastle import ControlPlaneGateway
gateway = ControlPlaneGateway()  # auto-detects env vars inside a Sandcastle sandbox
```

---

## Installation

```bash
# Core SDK only
pip install sandcastle-sdk

# With OpenAI support
pip install sandcastle-sdk[openai]

# With Anthropic support
pip install sandcastle-sdk[anthropic]

# Both providers
pip install sandcastle-sdk[all]
```

**Requirements:** Python 3.10+

---

## The AgentGateway Protocol

Any class implementing these four async methods is a valid gateway:

```python
class AgentGateway(Protocol):
    async def invoke_llm(self, new_messages, tools=None, tool_choice="auto") -> LLMResponse: ...
    async def persist_messages(self, messages) -> None: ...
    async def request_file_url(self, file_path, method="PUT") -> PresignedURL: ...
    async def get_session_cost(self) -> float: ...
```

Write your agent against the protocol. The implementation — local or production — is a runtime detail.

---

## Features

- **Zero-secret agents** — `ControlPlaneGateway` holds no API keys, database credentials, or cloud tokens
- **Stateless by design** — conversation history owned by the gateway, not the agent; kill and restart without data loss  
- **Framework-agnostic** — works with LangChain, LlamaIndex, raw API calls, or any custom agent framework
- **Built-in cost tracking** — every `invoke_llm` call returns `cost_usd`; `get_session_cost()` returns the running total
- **OpenAI + Anthropic** — both providers supported in `DirectGateway` out of the box
- **MockGateway for testing** — no LLM calls in your test suite; full call recording for assertions
- **Full type annotations** — `py.typed` marker; works with mypy strict mode

---

## Testing your agents

```python
from sandcastle.testing import MockGateway
from sandcastle.models import LLMResponse, Message, Role

async def test_my_agent():
    mock = MockGateway()
    mock.queue_response(LLMResponse(
        message=Message(role=Role.ASSISTANT, content="The answer is 42."),
        cost_usd=0.001,
        model="mock",
        finish_reason="stop",
    ))

    result = await my_agent(gateway=mock)

    assert mock.invoke_llm_call_count == 1
    assert mock.total_messages_sent == 1
```

---

## Supported providers

| Provider | DirectGateway | ControlPlaneGateway |
|---|---|---|
| OpenAI (gpt-4o, gpt-4o-mini, …) | ✓ | ✓ (via control plane) |
| Anthropic (claude-3-5-sonnet, …) | ✓ | ✓ (via control plane) |
| OpenAI-compatible (Ollama, Groq, …) | ✓ | Roadmap |

---

## Error handling

```python
from sandcastle.exceptions import CostCapExceededError, RateLimitError, SandcastleError

try:
    response = await gateway.invoke_llm(new_messages=[...])
except CostCapExceededError as e:
    print(f"Cost cap of ${e.cap_usd} reached. Spent: ${e.consumed_usd}")
except RateLimitError as e:
    await asyncio.sleep(e.retry_after_seconds)
except SandcastleError as e:
    # Catch-all for any SDK error
    raise
```

Full exception hierarchy: `SandcastleError > GatewayError > ControlPlaneError > AuthenticationError | CostCapExceededError | SessionNotFoundError`

---

## Architecture

```
┌─────────────────────────────────────┐
│            Your Agent Code          │
│   (depends only on AgentGateway)    │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │    AgentGateway      │  ← Protocol (interface)
    │      Protocol        │
    └──────┬────────┬──────┘
           │        │
  ┌────────▼─┐  ┌───▼──────────────┐
  │  Direct  │  │  ControlPlane    │
  │ Gateway  │  │    Gateway       │
  │          │  │                  │
  │ Local /  │  │   Production     │
  │  Evals   │  │  (zero secrets)  │
  └──────────┘  └────────┬─────────┘
                         │ HTTP
                ┌────────▼─────────┐
                │  Sandcastle      │
                │  Control Plane   │
                │  (holds creds)   │
                └──────────────────┘
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting significant changes.

```bash
git clone https://github.com/Goldokpa/sandcastle-sdk.git
cd sandcastle-sdk
pip install -e ".[dev]"
pre-commit install
pytest tests/unit/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Roadmap

- [ ] LangChain adapter (`SandcastleChatModel`)
- [ ] LlamaIndex adapter (`SandcastleLLM`)
- [ ] Streaming support (`invoke_llm_stream`)
- [ ] CrewAI integration
- [ ] Pluggable inference backends (distributed compute)
- [ ] `sandcastle-cli` for one-command control plane deployment

---

## License

MIT © [Gold Okpa](https://github.com/Goldokpa/sandcastle-sdk)

---

*Built on the control plane pattern described in [How We Built Secure, Scalable Agent Sandbox Infrastructure](https://github.com/Goldokpa/Sandcastle).*
