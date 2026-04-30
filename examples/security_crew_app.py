#!/usr/bin/env python3
"""
Security Incident Response Crew — Streamlit Demo
IdentArk + CrewAI + Human-in-the-Loop

Deploy to Streamlit Cloud:
    1. Push to GitHub
    2. Go to streamlit.io/cloud
    3. Connect repo, set examples/security_crew_app.py as entry point
    4. Add GROQ_API_KEY to secrets
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import streamlit as st

# Page config
st.set_page_config(
    page_title="Security Incident Response | IdentArk Demo",
    page_icon="🛡️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Risk Level & Audit
# ══════════════════════════════════════════════════════════════════════════════


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def color(self) -> str:
        return {
            "low": "green",
            "medium": "orange",
            "high": "red",
            "critical": "purple",
        }[self.value]

    @property
    def emoji(self) -> str:
        return {
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴",
            "critical": "🟣",
        }[self.value]


@dataclass
class AuditEntry:
    timestamp: str
    agent: str
    action: str
    risk_level: RiskLevel
    approved: bool | None
    result: str


@dataclass
class HITLRequest:
    request_id: str
    action: str
    params: dict[str, Any]
    risk_level: RiskLevel
    agent: str
    justification: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))


# ══════════════════════════════════════════════════════════════════════════════
# Session State Init
# ══════════════════════════════════════════════════════════════════════════════


def init_session_state() -> None:
    if "stage" not in st.session_state:
        st.session_state.stage = "intro"  # intro, running, hitl, complete
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    if "crew_output" not in st.session_state:
        st.session_state.crew_output = ""
    if "hitl_queue" not in st.session_state:
        st.session_state.hitl_queue = []
    if "current_hitl" not in st.session_state:
        st.session_state.current_hitl = 0
    if "cost" not in st.session_state:
        st.session_state.cost = 0.0
    if "start_time" not in st.session_state:
        st.session_state.start_time = None


init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════


def render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/identark/identark/main/docs/logo.png",
            width=200,
        )
        st.markdown("## IdentArk Demo")
        st.markdown(
            """
        This demo shows:
        - **Multi-Agent Collaboration**
        - **Human-in-the-Loop Approval**
        - **Audit Trail**
        - **Cost Tracking**
        """
        )

        st.divider()

        st.markdown("### How It Works")
        st.markdown(
            """
        1. **Triage Agent** classifies the alert
        2. **Forensics Agent** investigates
        3. **Remediation Agent** proposes actions
        4. **You** approve or reject HIGH-risk actions
        """
        )

        st.divider()

        if st.session_state.audit_log:
            st.markdown("### Audit Log")
            for entry in st.session_state.audit_log:
                icon = "✅" if entry.approved else "❌"
                st.markdown(f"{entry.risk_level.emoji} **{entry.action}** {icon}")

        st.divider()
        st.markdown(
            f"**Cost:** ${st.session_state.cost:.4f}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# CrewAI Integration
# ══════════════════════════════════════════════════════════════════════════════


def run_crew() -> tuple[str, float]:
    """Run the security incident crew and return (output, cost)."""
    try:
        from crewai import Agent, Crew, Process, Task
        from openai import AsyncOpenAI

        from identark import DirectGateway
        from identark.integrations.crewai import IdentArkCrewAILLM
    except ImportError as e:
        return f"Import error: {e}", 0.0

    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return "ERROR: GROQ_API_KEY not configured", 0.0

    gateway = DirectGateway(
        llm_client=AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        ),
        model="llama-3.3-70b-versatile",
        system_prompt="You are a security operations expert. Be concise.",
    )

    llm = IdentArkCrewAILLM(gateway=gateway)

    triage_agent = Agent(
        role="Security Triage Analyst",
        goal="Classify security alerts by severity",
        backstory="You are the SOC's first responder who quickly assesses alerts.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    forensics_agent = Agent(
        role="Security Forensics Investigator",
        goal="Investigate incidents to determine root cause",
        backstory="You analyze threat intelligence and login patterns.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    remediation_agent = Agent(
        role="Security Remediation Specialist",
        goal="Recommend remediation actions",
        backstory="You recommend actions to stop threats. Be specific.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    triage_task = Task(
        description="""Analyze this security alert:

ALERT: Unusual login detected
- User: john.doe@company.com
- Source IP: 203.0.113.42
- Location: Unknown (VPN/Proxy)
- Previous location: London, UK (yesterday)

Classify as LOW, MEDIUM, HIGH, or CRITICAL with brief reasoning.""",
        expected_output="Severity classification with reasoning",
        agent=triage_agent,
    )

    forensics_task = Task(
        description="""Investigate this incident briefly:
1. Is the IP suspicious?
2. Is the login pattern anomalous?
3. Is this a real threat or false positive?""",
        expected_output="Brief investigation findings",
        agent=forensics_agent,
        context=[triage_task],
    )

    remediation_task = Task(
        description="""Based on findings, recommend specific actions:
- Should we block the IP?
- Should we reset the password?
- Should we enable MFA?

Be specific about what actions to take.""",
        expected_output="Specific remediation recommendations",
        agent=remediation_agent,
        context=[forensics_task],
    )

    crew = Crew(
        agents=[triage_agent, forensics_agent, remediation_agent],
        tasks=[triage_task, forensics_task, remediation_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    async def get_cost() -> float:
        return await gateway.get_session_cost()

    cost = asyncio.run(get_cost())

    return str(result), cost


# ══════════════════════════════════════════════════════════════════════════════
# HITL Actions
# ══════════════════════════════════════════════════════════════════════════════


def get_hitl_actions() -> list[HITLRequest]:
    """Return the remediation actions requiring approval."""
    return [
        HITLRequest(
            request_id="HITL-0001",
            action="block_ip",
            params={"ip_address": "203.0.113.42", "duration_hours": 24},
            risk_level=RiskLevel.HIGH,
            agent="Remediation Agent",
            justification="Block malicious IP associated with suspicious login attempt",
        ),
        HITLRequest(
            request_id="HITL-0002",
            action="force_password_reset",
            params={"username": "john.doe@company.com"},
            risk_level=RiskLevel.HIGH,
            agent="Remediation Agent",
            justification="Reset password for potentially compromised account",
        ),
        HITLRequest(
            request_id="HITL-0003",
            action="enable_mfa",
            params={"username": "john.doe@company.com"},
            risk_level=RiskLevel.MEDIUM,
            agent="Remediation Agent",
            justification="Enforce MFA to prevent future unauthorized access",
        ),
    ]


def log_action(request: HITLRequest, approved: bool, result: str) -> None:
    """Log an action to the audit trail."""
    st.session_state.audit_log.append(
        AuditEntry(
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            agent=request.agent,
            action=request.action,
            risk_level=request.risk_level,
            approved=approved,
            result=result,
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# UI Stages
# ══════════════════════════════════════════════════════════════════════════════


def render_intro() -> None:
    """Render the intro/start screen."""
    st.markdown(
        """
    # 🛡️ Security Incident Response Crew

    **IdentArk + CrewAI — Human-in-the-Loop Demo**

    This demo shows how AI agents can collaborate on security incidents while
    keeping humans in control of high-risk actions.

    ---

    ### The Scenario

    ```
    ALERT: Unusual login detected
    - User: john.doe@company.com
    - Source IP: 203.0.113.42
    - Location: Unknown (VPN/Proxy detected)
    - Previous location: London, UK (yesterday)
    ```

    ---

    ### What Happens

    1. **Triage Agent** — Classifies the alert severity
    2. **Forensics Agent** — Investigates the threat
    3. **Remediation Agent** — Proposes actions
    4. **You** — Approve or reject high-risk actions

    ---
    """
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Start Incident Response",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.stage = "running"
            st.session_state.start_time = time.time()
            st.rerun()


def render_running() -> None:
    """Render the crew execution screen."""
    st.markdown("# 🔄 Agents Working...")
    st.markdown("The security crew is analyzing the incident.")

    progress = st.progress(0)
    status = st.empty()

    stages = [
        ("🔍 Triage Agent analyzing alert...", 0.2),
        ("🔬 Forensics Agent investigating...", 0.5),
        ("🛠️ Remediation Agent planning response...", 0.8),
        ("✅ Analysis complete!", 1.0),
    ]

    # Run crew
    with st.spinner("Running security crew..."):
        output, cost = run_crew()

    st.session_state.crew_output = output
    st.session_state.cost = cost

    # Animate progress
    for msg, pct in stages:
        status.markdown(f"### {msg}")
        progress.progress(pct)
        time.sleep(0.3)

    # Setup HITL queue
    st.session_state.hitl_queue = get_hitl_actions()
    st.session_state.current_hitl = 0
    st.session_state.stage = "hitl"
    time.sleep(0.5)
    st.rerun()


def render_hitl() -> None:
    """Render the HITL approval screen."""
    queue = st.session_state.hitl_queue
    idx = st.session_state.current_hitl

    if idx >= len(queue):
        st.session_state.stage = "complete"
        st.rerun()
        return

    request = queue[idx]

    st.markdown("# 🚨 Approval Required")

    # Show crew output
    with st.expander("📋 Agent Analysis", expanded=False):
        st.markdown(st.session_state.crew_output)

    st.divider()

    # HITL Card
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {request.request_id}")
        st.markdown(f"**Agent:** {request.agent}")
        st.markdown(
            f"**Action:** `{request.action}` {request.risk_level.emoji} "
            f":{request.risk_level.color}[{request.risk_level.value.upper()}]"
        )
        st.markdown(f"**Justification:** {request.justification}")

        st.markdown("**Parameters:**")
        for k, v in request.params.items():
            st.code(f"{k}: {v}")

    with col2:
        st.markdown("### Your Decision")
        st.markdown(f"Action {idx + 1} of {len(queue)}")

        approve_col, reject_col = st.columns(2)

        with approve_col:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                log_action(request, True, "approved by operator")
                st.session_state.current_hitl += 1
                st.rerun()

        with reject_col:
            if st.button("❌ Reject", type="secondary", use_container_width=True):
                log_action(request, False, "rejected by operator")
                st.session_state.current_hitl += 1
                st.rerun()

    # Progress indicator
    st.divider()
    st.progress((idx + 1) / len(queue))


def render_complete() -> None:
    """Render the completion screen."""
    st.markdown("# ✅ Incident Response Complete")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    approved = sum(1 for e in st.session_state.audit_log if e.approved)
    rejected = len(st.session_state.audit_log) - approved

    with col1:
        st.metric("Actions Reviewed", len(st.session_state.audit_log))
    with col2:
        st.metric("Approved", approved)
    with col3:
        st.metric("Rejected", rejected)
    with col4:
        st.metric("LLM Cost", f"${st.session_state.cost:.4f}")

    st.divider()

    # Agent output
    st.markdown("### 📋 Agent Analysis")
    st.markdown(st.session_state.crew_output)

    st.divider()

    # Audit log
    st.markdown("### 📜 Audit Trail")

    for entry in st.session_state.audit_log:
        icon = "✅" if entry.approved else "❌"
        color = entry.risk_level.color

        st.markdown(
            f"""
        **{entry.timestamp}** — {entry.agent}

        - Action: `{entry.action}` :{color}[{entry.risk_level.value.upper()}]
        - Status: {icon} {entry.result}
        """
        )

    st.divider()

    # Key takeaways
    st.markdown("### 📌 Key Takeaways")

    st.info(
        """
    **1. CAPABILITIES, NOT CREDENTIALS**
    Agents execute actions without holding API keys or secrets.

    **2. HUMAN-IN-THE-LOOP**
    High-risk actions required your approval before execution.

    **3. AUDIT TRAIL**
    Every action logged with timestamp, agent, and outcome.

    **4. COST TRACKING**
    LLM spend tracked for budgeting and alerts.
    """
    )

    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Run Again", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    render_sidebar()

    stage = st.session_state.stage

    if stage == "intro":
        render_intro()
    elif stage == "running":
        render_running()
    elif stage == "hitl":
        render_hitl()
    elif stage == "complete":
        render_complete()


if __name__ == "__main__":
    main()
