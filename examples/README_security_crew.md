# Security Incident Response Crew

A multi-agent security operations demo showcasing IdentArk's core capabilities with CrewAI.

## What This Demonstrates

| Feature | Description |
|---------|-------------|
| **Multi-Agent Collaboration** | Triage → Forensics → Remediation pipeline |
| **Human-in-the-Loop (HITL)** | High-risk actions require operator approval |
| **Audit Trail** | Every action logged with timestamp, agent, risk level |
| **Cost Tracking** | LLM spend tracked across all agents |
| **Capabilities, Not Credentials** | Agents execute without holding secrets |

## Quick Start

### Option 1: Groq (Free, Fast — Recommended)

```bash
# Get free API key at console.groq.com
export GROQ_API_KEY=gsk_...

cd identark-sdk
pip install -e ".[openai]" crewai
python examples/security_incident_crew.py --groq
```

### Option 2: OpenAI

```bash
export OPENAI_API_KEY=sk-...

python examples/security_incident_crew.py --openai
```

### Option 3: Ollama (Local, Slow)

```bash
# Install and run Ollama
brew install ollama
ollama serve &
ollama pull llama3.2

python examples/security_incident_crew.py --model llama3.2
```

## The Workflow

```
┌─────────────────┐
│  Security Alert │
│  (VPN login)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Triage Agent   │  Classifies severity: HIGH
│  (LOW risk)     │  "VPN + failed attempts = suspicious"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Forensics Agent │  Investigates IP, login history
│  (MEDIUM risk)  │  "IP has 47 abuse reports"
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Remediation Agent                    │
│              (HIGH risk)                     │
├─────────────────────────────────────────────┤
│  Action: block_ip(203.0.113.42)             │
│  ════════════════════════════════════════   │
│  🚨 HITL APPROVAL REQUIRED                  │
│  [A]pprove  [R]eject  > _                   │
└─────────────────────────────────────────────┘
```

## Interactive HITL Prompts

When a HIGH-risk action is proposed, you'll see:

```
══════════════════════════════════════════════════════════════════════
  🚨 HITL APPROVAL REQUIRED — HITL-0001
══════════════════════════════════════════════════════════════════════
  Agent:         Remediation
  Action:        block_ip
  Risk Level:    HIGH
  Justification: Block malicious IP 203.0.113.42 for 24 hours

  Parameters:
    ip_address: 203.0.113.42
    duration_hours: 24
──────────────────────────────────────────────────────────────────────
  [A]pprove  [R]eject  > _
```

- Press **A** to approve the action
- Press **R** to reject (optionally with a reason)
- Press **Ctrl+C** to skip

## Audit Log Output

```
══════════════════════════════════════════════════════════════════════
  📋 AUDIT LOG — Security Incident Response
══════════════════════════════════════════════════════════════════════
  2026-04-29T14:35:22
    Agent:  Forensics
    Action: analyze_ip (LOW)
    Status: ✅ auto-approved (low risk)

  2026-04-29T14:35:25
    Agent:  Forensics
    Action: check_login_history (MEDIUM)
    Status: ✅ approved by operator

  2026-04-29T14:35:30
    Agent:  Remediation
    Action: block_ip (HIGH)
    Status: ✅ approved by operator

  2026-04-29T14:35:32
    Agent:  Remediation
    Action: force_password_reset (HIGH)
    Status: ❌ rejected by operator
```

## CI/Non-Interactive Mode

For CI pipelines or automated testing:

```bash
python examples/security_incident_crew.py --groq --non-interactive
```

This auto-approves all actions (useful for demos and testing).

## In Production

With IdentArk's ControlPlaneGateway, this becomes:

| Demo | Production |
|------|------------|
| Local HITL prompts | WebSocket push to reviewers |
| Manual approval | Mobile/Slack/Teams notifications |
| In-memory audit log | Immutable database log |
| Local cost tracking | Org-wide spend limits |
| Hardcoded tools | Real integrations (firewalls, IAM) |

## Deploy to Streamlit Cloud

1. **Push to GitHub** (if not already)
   ```bash
   git add examples/
   git commit -m "feat: add security crew Streamlit demo"
   git push
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repo
   - Set main file path: `examples/security_crew_app.py`
   - Add secret: `GROQ_API_KEY` = your Groq key

3. **Done!** Share the URL with customers.

## Files

| File | Description |
|------|-------------|
| `security_crew_app.py` | Streamlit web app (deploy this) |
| `security_incident_crew.py` | CLI version |
| `.streamlit/config.toml` | Dark theme config |
| `requirements-streamlit.txt` | Dependencies |

## Cost

| Provider | Typical Cost |
|----------|--------------|
| Groq | $0.02 - $0.05 |
| OpenAI (gpt-4o-mini) | $0.01 - $0.03 |
| Ollama | $0.00 (local) |
