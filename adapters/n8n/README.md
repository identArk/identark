# n8n-nodes-credseal

This is an n8n community node for [CredSeal](https://github.com/credseal/sdk) — the secure AI agent gateway.

**CredSeal lets your n8n workflows invoke LLMs without exposing API keys.** Your credentials stay in a secure vault; n8n only holds a session reference.

## Installation

### Community Nodes (Recommended)

1. Go to **Settings > Community Nodes**
2. Select **Install**
3. Enter `n8n-nodes-credseal`
4. Agree to the risks and click **Install**

### Manual Installation

```bash
cd ~/.n8n/custom
npm install n8n-nodes-credseal
```

## Prerequisites

1. A CredSeal account — sign up at your control plane (e.g., `https://credseal-cloud.fly.dev`)
2. An API key from `/v1/orgs/signup`
3. LLM credentials registered via `/v1/credentials`

## Operations

### Invoke LLM

Send a message to an LLM and receive a response. Supports:
- Continuing conversations via `session_id`
- System, user, and assistant roles
- Automatic cost tracking

### Create Session

Create a new agent session with explicit configuration:
- Model selection (GPT-4o, Claude, Mistral, etc.)
- Provider selection (OpenAI, Anthropic, Azure, Bedrock, Mistral)
- Cost cap enforcement
- Credential reference (vault path)

### Get Session Cost

Retrieve the running cost for a session — useful for monitoring and billing workflows.

## Example Workflow

1. **Create Session** → Configure model, provider, cost cap
2. **Invoke LLM** → Send user message, receive response
3. **Get Session Cost** → Check spend before next invocation

## Credentials

| Field | Description |
|-------|-------------|
| API Key | Your CredSeal API key (starts with `csk_`) |
| Control Plane URL | Your CredSeal instance URL |

## Data Residency

CredSeal supports UK/EU data residency:
- **Azure OpenAI UK South** — GPT-4o inference stays in UK
- **AWS Bedrock eu-west-2** — Claude in London
- **Mistral AI** — EU data centres (French company)

Configure your session with the appropriate provider to ensure compliance.

## License

AGPL-3.0 — see [LICENSE](../../LICENSE)

## Links

- [CredSeal SDK](https://github.com/credseal/sdk)
- [n8n Community Nodes](https://docs.n8n.io/integrations/community-nodes/)
