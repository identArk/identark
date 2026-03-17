---
paths:
  - "credseal/integrations/**/*.py"
  - "adapters/**"
---

# Adapter Development Rules

## The Contract Every Adapter Must Fulfil

Every adapter wraps an external framework's tool/agent primitive and routes execution
through `CredSealGateway`. The adapter:

1. Accepts a `gateway: CredSealGateway` instance at construction time
2. NEVER accepts raw credentials — only the gateway
3. Translates the framework's native tool call format into `GatewayRequest`
4. Returns the framework's expected response format from `GatewayResponse`
5. Propagates `CredSealError` subclasses without swallowing them

## LangChain Adapter Pattern

```python
from langchain.tools import BaseTool
from credseal import CredSealGateway

class CredSealTool(BaseTool):
    gateway: CredSealGateway
    tool_name: str  # e.g. "openai.chat"

    async def _arun(self, **kwargs) -> str:
        response = await self.gateway.execute(tool=self.tool_name, params=kwargs)
        return response.result
```

## n8n Adapter (TypeScript — adapters/n8n/)

- The n8n node is a separate TypeScript package in `adapters/n8n/`
- It calls the credseal-cloud REST API, not the Python SDK directly
- Node name: `credsealGateway` (camelCase, as required by n8n)
- Credentials type: `credsealApi` with fields: `gatewayUrl`, `apiKey`
- The node must appear in the n8n community registry under `n8n-nodes-credseal`

## Adding a New Adapter

1. Create `credseal/integrations/<framework>.py`
2. Add tests in `tests/unit/test_integrations.py`
3. Add a working example in `examples/<framework>/`
4. Add a docs page at `docs/adapters/<framework>.md`
5. Export from `credseal/integrations/__init__.py`
6. Update README.md integrations table
