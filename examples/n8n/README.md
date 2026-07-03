# IdentArk + n8n — Enterprise AI Support Agent

A production-ready workflow that demonstrates how to build an **enterprise-grade AI agent** in n8n with IdentArk credential isolation, cost control, risk-based routing, and full audit logging.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Webhook        │────▶│ IdentArk Gateway     │────▶│  Risk-Based Router  │
│  (Ticket In)    │     │  (Secure LLM)        │     │  (Classification)   │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
                                                              │
                                    ┌─────────────────────────┴─────────────────────────┐
                                    │                                                   │
                                    ▼                                                   ▼
                         ┌─────────────────────┐                           ┌─────────────────────┐
                         │ HIGH RISK           │                           │ LOW RISK            │
                         │ (critical/negative) │                           │ (standard)          │
                         └──────────┬──────────┘                           └──────────┬──────────┘
                                    │                                                   │
                                    ▼                                                   ▼
                         ┌─────────────────────┐                           ┌─────────────────────┐
                         │ Draft escalation    │                           │ Draft auto-response │
                         │ + Slack alert       │                           │ + Send email        │
                         └──────────┬──────────┘                           └──────────┬──────────┘
                                    │                                                   │
                                    └───────────────────────┬───────────────────────────┘
                                                            │
                                                            ▼
                                               ┌─────────────────────┐
                                               │ IdentArk: Get Cost  │
                                               └──────────┬──────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │ Build Audit Log     │
                                               └──────────┬──────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │ POST to Audit API   │
                                               └──────────┬──────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │ Respond to Webhook  │
                                               └─────────────────────┘
```

## Enterprise Features

| Feature | Implementation | Value |
|---------|---------------|-------|
| **Credential Isolation** | IdentArk vault — n8n never stores API keys | Zero secrets in workflow exports |
| **Cost Control** | Per-session cost cap ($2.50) + real-time cost query | Prevents runaway LLM spend |
| **Risk Routing** | LLM classifies sentiment/urgency; IF node routes | Critical tickets get human attention |
| **Audit Trail** | Every ticket logged with classification, cost, timestamp | Compliance-ready (SOC 2, GDPR) |
| **Data Residency** | Choose EU/UK providers (Mistral, Azure UK, Bedrock EU) | Meet regulatory requirements |
| **Session Isolation** | Each run gets a unique IdentArk session | Conversations don't leak between customers |

## Prerequisites

1. **n8n** — Self-hosted or cloud ([n8n.io](https://n8n.io))
2. **IdentArk control plane** — Running at `https://api.identark.io` (or your instance)
3. **IdentArk n8n node** — Install via Community Nodes: `n8n-nodes-identark`
4. **Slack workspace** — For escalation alerts
5. **SMTP server** — For auto-responses

## Setup

### 1. Install the IdentArk n8n Node

In n8n:
1. **Settings** → **Community Nodes**
2. Click **Install**
3. Enter: `n8n-nodes-identark`
4. Restart n8n

Or manually:
```bash
cd ~/.n8n/custom
npm install n8n-nodes-identark
```

### 2. Create IdentArk Credentials

In n8n:
1. **Credentials** → **Add Credential**
2. Search **IdentArk API**
3. Fill in:
   - **API Key**: `csk_...` (from your IdentArk org)
   - **Control Plane URL**: `https://api.identark.io`

### 3. Import the Workflow

1. Download [`enterprise_support_agent.json`](./enterprise_support_agent.json)
2. In n8n: **Workflows** → **Import from File**
3. Select the JSON
4. Re-map credentials (the import uses placeholder IDs)

### 4. Configure External Credentials

| Node | Credential Type | Purpose |
|------|----------------|---------|
| Slack: Alert Team | Slack API | Escalation channel |
| Send Auto-Response | SMTP | Customer email replies |
| POST Audit Log | HTTP Request (optional) | Your audit system endpoint |

### 5. Register LLM Credentials in IdentArk

```bash
curl -X POST https://api.identark.io/v1/credentials \
  -H "Authorization: Bearer csk_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "credential": "sk-your-openai-key",
    "label": "prod-support-openai"
  }'
```

Save the returned `credential_ref` (e.g., `vault://prod/support/openai`) and update the **IdentArk: Create Session** node.

## Workflow Nodes Explained

### 1. Support Ticket Webhook
Receives POST requests with:
```json
{
  "customer_email": "alice@example.com",
  "message": "I'm being double-billed and your dashboard is broken!"
}
```

### 2. IdentArk: Create Session
Spawns an isolated session with:
- **Model**: `gpt-4o`
- **Cost cap**: $2.50 USD
- **Credential ref**: Points to your vault-stored OpenAI key

### 3. IdentArk: Classify Ticket
Prompts the LLM to classify sentiment, urgency, and category. Returns structured JSON.

### 4. High-Risk Router
n8n IF node checks:
- `urgency == "critical"` OR
- `sentiment == "negative"`

**True branch** → Human escalation
**False branch** → Auto-response

### 5. IdentArk: Draft Escalation / Draft Response
LLM generates context-aware replies. For escalations, the prompt emphasises empathy and clear next steps.

### 6. Slack: Alert Team
Posts to `#support-escalations` with:
- Customer email
- Classification
- Draft response for human review

### 7. IdentArk: Get Cost
Queries the session's running cost. Use this for:
- Real-time budget dashboards
- Per-ticket cost attribution
- Alerting when a session nears its cap

### 8. Build Audit Log + POST Audit Log
Constructs a compliance record:
```json
{
  "ticket_id": 42,
  "customer_email": "alice@example.com",
  "classification": { "sentiment": "negative", "urgency": "critical", "category": "billing" },
  "session_id": "sess_abc123",
  "cost_usd": 0.0042,
  "processed_at": "2026-04-30T12:00:00Z",
  "agent_type": "escalation"
}
```

## Testing

### Local Test
```bash
curl -X POST https://your-n8n-instance/webhook/support-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "customer_email": "test@example.com",
    "message": "I love your product but my invoice is wrong."
  }'
```

### Expected Response
```json
{
  "status": "ok",
  "ticket_id": 0,
  "classification": {
    "sentiment": "neutral",
    "urgency": "medium",
    "category": "billing"
  },
  "cost_usd": 0.0008,
  "agent_type": "auto-response"
}
```

## Security Hardening

1. **Webhook Authentication** — Add an API key header to the webhook node
2. **Rate Limiting** — Use n8n's built-in rate limiting or a reverse proxy
3. **Input Validation** — Add a Function node to sanitise customer_email
4. **Cost Caps** — Lower the IdentArk session cap for untrusted sources
5. **Audit Retention** — Ensure your audit API retains logs for 7+ years

## Cost Optimisation

| Strategy | Implementation |
|----------|---------------|
| Model tiering | Use `gpt-4o-mini` for classification, `gpt-4o` only for drafting |
| Session reuse | Cache session IDs for follow-up messages |
| Batch classification | Classify 10 tickets in one LLM call |
| Caching | Cache identical queries with n8n's built-in cache |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Invalid API key" | IdentArk cred mismatch | Re-check `csk_` key in n8n credentials |
| "Cost cap exceeded" | Session hit $2.50 | Increase cap or implement retry logic |
| "Session not found" | Session expired | Reduce idle time or create new sessions |
| Slack not sending | Missing OAuth scopes | Add `chat:write:bot` scope |
| Email not sending | SMTP auth error | Verify SMTP credentials |

## Human-in-the-Loop (HITL) Approval

The workflow includes a **Wait for Approval** node in the high-risk branch. This pauses execution until a human approves or rejects the AI-drafted response.

### How it works

```
High-Risk Ticket
    → IdentArk drafts escalation
    → Slack alerts team
    → WAIT ⏸️ (workflow paused)
         ↓
    Human clicks Approve/Reject in Slack
         ↓
    Resume → Approved? → Yes → Send email
                    → No  → Send to human queue
```

### Setting up the approval webhook

1. **Configure the Wait node**
   - Mode: `Webhook`
   - Path: `support-approval`
   - Method: `POST`

2. **Add Slack interactive buttons**
   Update the Slack node to include a Block Kit message with buttons:
   ```json
   {
     "type": "section",
     "text": { "type": "mrkdwn", "text": "Approve this AI response?" },
     "accessory": {
       "type": "button",
       "text": { "type": "plain_text", "text": "Approve" },
       "value": "approve",
       "action_id": "approve_response"
     }
   }
   ```

3. **Route Slack interactions to the webhook**
   In your Slack app settings, set the **Interactive Components** request URL to:
   ```
   https://your-n8n-instance/webhook/support-approval
   ```

4. **Format the webhook payload**
   Use a Function node before the Wait node to pre-format the expected payload:
   ```javascript
   return [{
     json: {
       approval_url: "https://your-n8n-instance/webhook/support-approval",
       ticket_id: $runIndex,
       draft: $('IdentArk: Draft Escalation').item.json.message.content
     }
   }];
   ```

### Alternative: IdentArk-native HITL

If your control plane has HITL policies enabled, you can use the **IdentArk node** directly:

| Operation | Purpose |
|-----------|---------|
| **List Pending Approvals** | Fetch all HITL requests awaiting review |
| **Get Approval** | Check status of a specific approval |
| **Submit Decision** | Approve or reject with optional MFA |

This integrates with IdentArk's risk-scoring engine — high-risk tool calls automatically create approval requests that block until a human decides.

### Approval decision via CLI

```bash
# List pending approvals
identark approvals list

# Inspect details
identark approvals inspect <approval-id>

# Approve (MFA required for risk >70)
identark approvals approve <approval-id> --mfa 123456

# Reject
identark approvals reject <approval-id> --reason "Incorrect refund amount"
```

## Extending the Workflow

### Multi-Language Support
Add a **Switch** node on `classification.category` and route to language-specific IdentArk sessions:
- `provider: mistral` for French/German (EU data residency)
- `provider: azure_openai` with UK South for English

### CRM Integration
Add nodes for:
- **HubSpot**: Create ticket, update contact property
- **Salesforce**: Create Case object
- **Zendesk**: Create ticket with AI-generated response as internal note

### Add a Re-Draft Loop
If approval is rejected, route back to **IdentArk: Draft Escalation** with human feedback:
```
Rejected → Function: Add feedback → IdentArk: Draft Escalation → Wait for Approval
```

## Compliance Notes

- **GDPR**: Use EU providers (`mistral`, `azure_openai` UK) and include data-processing agreements
- **SOC 2**: The audit log node captures all LLM interactions with cost attribution
- **ISO 27001**: IdentArk vault ensures API keys are never in workflow code or exports
- **Human oversight**: HITL approvals provide the human-in-the-loop required by EU AI Act for high-risk automated decisions

## Support

- **IdentArk Docs**: [docs.identark.io](https://docs.identark.io)
- **n8n Docs**: [docs.n8n.io](https://docs.n8n.io)
- **Issues**: [github.com/identArk/identark/issues](https://github.com/identArk/identark/issues)
