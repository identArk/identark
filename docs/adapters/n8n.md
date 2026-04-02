# n8n Integration

IdentArk provides a community node for [n8n](https://n8n.io), the workflow automation platform.

## Installation

```bash
# Via n8n Community Nodes UI
Settings > Community Nodes > Install > n8n-nodes-identark

# Or manually
cd ~/.n8n/custom
npm install n8n-nodes-identark
```

## Quick Start

### 1. Get API Credentials

```bash
curl -X POST https://identark-cloud.fly.dev/v1/orgs/signup \
  -H "Content-Type: application/json" \
  -d '{"name": "My Org", "email": "me@example.com"}'
```

Save the returned `api_key` — you'll need it for n8n credentials.

### 2. Register LLM Credentials

```bash
curl -X POST https://identark-cloud.fly.dev/v1/credentials \
  -H "Authorization: Bearer csk_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "credential": "sk-your-openai-key"}'
```

Save the `credential_ref` for session creation.

### 3. Configure n8n Credentials

1. Open n8n
2. Go to **Credentials > Add Credential**
3. Search for **IdentArk API**
4. Enter your API key and control plane URL

### 4. Build a Workflow

**Simple LLM Call:**

```
[Trigger] → [IdentArk: Invoke LLM] → [Output]
```

**With Session Management:**

```
[Trigger] → [IdentArk: Create Session] → [IdentArk: Invoke LLM] → [IdentArk: Get Session Cost] → [Output]
```

## Operations

### Invoke LLM

Send messages to an LLM via the IdentArk gateway.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Message | string | Yes | The message content |
| Role | enum | No | `user`, `system`, or `assistant` (default: `user`) |
| Session ID | string | No | Continue an existing conversation |

**Response:**

```json
{
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "cost_usd": 0.000123,
  "model": "gpt-4o",
  "finish_reason": "stop",
  "session_id": "abc123"
}
```

### Create Session

Create a new session with explicit configuration.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Agent ID | string | Yes | Identifier for this agent |
| Model | enum | Yes | LLM model (gpt-4o, claude-sonnet-4-6, etc.) |
| Provider | enum | Yes | Provider (openai, anthropic, mistral, etc.) |
| Credential Ref | string | Yes | Vault path to the LLM credential |
| Cost Cap USD | number | No | Maximum session cost (default: 5.00) |

### Get Session Cost

Retrieve the running cost for a session.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Session ID | string | Yes | The session to query |

## Example: Customer Support Bot

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐
│  Webhook    │───▶│ IdentArk: Invoke │───▶│ Slack: Send   │
│  (incoming) │    │ LLM              │    │ Message       │
└─────────────┘    └──────────────────┘    └───────────────┘
```

This workflow:
1. Receives a customer message via webhook
2. Sends it to an LLM through IdentArk (credentials stay secure)
3. Forwards the response to Slack

## Data Residency

For UK/EU compliance, configure your sessions with:

| Provider | Region | Data Residency |
|----------|--------|----------------|
| `azure_openai` | UK South | UK 🇬🇧 |
| `bedrock` | eu-west-2 | UK 🇬🇧 |
| `mistral` | EU | EU 🇪🇺 |

## Troubleshooting

### "Invalid API key"

- Ensure your API key starts with `csk_`
- Check the control plane URL is correct
- Verify the key hasn't been revoked

### "Session not found"

- Session IDs are scoped to your organisation
- Sessions may expire after inactivity
- Create a new session if needed

### "Cost cap exceeded"

- The session has hit its cost limit
- Create a new session with a higher cap
- Or implement cost-aware logic in your workflow
