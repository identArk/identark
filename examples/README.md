# Example index

Choose an example by provider or framework. Run commands from the repository root with Python 3.10+ and install the relevant optional SDK extra. Scripts using `DirectGateway` or `GeminiGateway` do **not** need an IdentArk control plane; the n8n workflow does.

The research, banking, weather, and incident-response examples contain simulated tools or data. They are demonstrations, not connected search, banking, weather, or security operations. Running cloud-model examples can incur provider charges.

## Start here

| File | What it shows | What you need to run it |
| --- | --- | --- |
| [`openai_quickstart.py`](openai_quickstart.py) | A single GPT-4o-mini call and session cost tracking. | `pip install identark[openai]`; `OPENAI_API_KEY`. No control plane. |
| [`gemini_quickstart.py`](gemini_quickstart.py) | Gemini chat, conversation history, streaming, and usage. | `pip install identark[gemini]`; `GEMINI_API_KEY`. No control plane. |

## Frameworks

| File | What it shows | What you need to run it |
| --- | --- | --- |
| [`langchain_agent.py`](langchain_agent.py) | LangChain chat model and a simulated weather tool call. | `pip install identark[langchain] openai`; `OPENAI_API_KEY`. No control plane. |
| [`langgraph_agent.py`](langgraph_agent.py) | LangGraph agent and tool nodes using the IdentArk adapter. | `pip install identark[langgraph] openai`; `OPENAI_API_KEY`. No control plane. |
| [`llamaindex_agent.py`](llamaindex_agent.py) | LlamaIndex LLM adapter and a simulated weather tool. | `pip install identark[llamaindex] openai`; `OPENAI_API_KEY`. No control plane. |
| [`crewai_agent.py`](crewai_agent.py) | A CrewAI task routed through the IdentArk LLM adapter. | `pip install identark crewai openai`; `OPENAI_API_KEY`. No control plane. |

## Local inference and data residency

| File | What it shows | What you need to run it |
| --- | --- | --- |
| [`ollama_local.py`](ollama_local.py) | Local inference and zero-cost tracking with Ollama's OpenAI-compatible API. | `pip install identark[openai]`; Ollama running locally with `llama3.2` pulled. No cloud key or control plane. |
| [`langgraph_agent_ollama.py`](langgraph_agent_ollama.py) | The LangGraph tool loop with a local Ollama model. | `pip install identark[langgraph] openai httpx`; Ollama running locally with `llama3.1` pulled. No cloud key or control plane. |
| [`mistral_eu.py`](mistral_eu.py) | Mistral's OpenAI-compatible API via `DirectGateway`. | `pip install identark[openai]`; `MISTRAL_API_KEY`. No control plane. |

## Worked demonstrations

| File | What it shows | What you need to run it |
| --- | --- | --- |
| [`research_agent.py`](research_agent.py) | Gemini research-agent loop with **stubbed** web and product tools. | `pip install identark[gemini]`; `GEMINI_API_KEY`. No control plane. |
| [`research_agent_openai.py`](research_agent_openai.py) | The research-agent pattern with OpenAI and **stubbed** tools. | `pip install identark[openai]`; `OPENAI_API_KEY`. No control plane. |
| [`gemini_finance_agent.py`](gemini_finance_agent.py) | Multi-turn finance example with **simulated** accounts and transactions. | `pip install identark[gemini]`; `GEMINI_API_KEY`. No banking connection or control plane. |
| [`security_incident_crew.py`](security_incident_crew.py) | CrewAI triage, forensics, remediation, and a **demo** approval/audit flow. | `pip install identark crewai openai`; `GROQ_API_KEY` with `--groq`, `OPENAI_API_KEY` with `--openai`, or a local Ollama server/model. No control plane. |
| [`security_crew_app.py`](security_crew_app.py) | Streamlit interface for the incident-response demonstration. | `pip install -r examples/requirements-streamlit.txt` and install the SDK from this repo; `GROQ_API_KEY` in the environment or Streamlit secrets. Run `streamlit run examples/security_crew_app.py`. No control plane. |

## Supporting files and workflow

| File | What it shows | What you need to run it |
| --- | --- | --- |
| [`README_security_crew.md`](README_security_crew.md) | Setup and walkthrough for the CLI security crew. | Documentation only; follow its provider-specific setup for the CLI example. |
| [`requirements-streamlit.txt`](requirements-streamlit.txt) | Streamlit demo dependencies. | Dependency list, not an executable example; install the SDK separately as noted above. |
| [`.streamlit/config.toml`](.streamlit/config.toml) | Theme and headless-server defaults for Streamlit. | Configuration only; used when running the Streamlit app. |
| [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) | Example Streamlit secret format. | Template only; supply your own `GROQ_API_KEY` privately. Do not commit a filled-in secrets file. |
| [`n8n/README.md`](n8n/README.md) | Setup for the enterprise support-agent workflow. | n8n, an IdentArk control plane, the `n8n-nodes-identark` community node, Slack, and SMTP. |
| [`n8n/enterprise_support_agent.json`](n8n/enterprise_support_agent.json) | Importable n8n support-agent workflow. | Same requirements as the n8n guide; remap credentials after import. |
