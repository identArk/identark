#!/usr/bin/env python3
"""
Security Incident Response Crew — IdentArk + CrewAI Demo

A multi-agent security operations center that demonstrates:
  - Multi-agent collaboration (Triage → Forensics → Remediation)
  - Human-in-the-loop approval for high-risk actions
  - Full audit trail of all agent actions
  - Cost tracking across the entire workflow

Run with Groq (free, recommended):
    export GROQ_API_KEY=gsk_...  # Get free key at console.groq.com
    python examples/security_incident_crew.py --groq

Run with OpenAI:
    export OPENAI_API_KEY=sk-...
    python examples/security_incident_crew.py --openai

Run with Ollama (local, slow):
    ollama serve && ollama pull llama3.2
    python examples/security_incident_crew.py --model llama3.2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# HITL Approval System
# ══════════════════════════════════════════════════════════════════════════════


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def color(self) -> str:
        return {
            "low": "\033[32m",
            "medium": "\033[33m",
            "high": "\033[91m",
            "critical": "\033[95m",
        }[self.value]


@dataclass
class AuditEntry:
    """Immutable audit log entry."""

    timestamp: str
    agent: str
    action: str
    risk_level: RiskLevel
    approved: bool | None
    result: str
    cost_usd: float = 0.0


@dataclass
class HITLRequest:
    """A pending action awaiting human approval."""

    request_id: str
    action: str
    params: dict[str, Any]
    risk_level: RiskLevel
    agent: str
    justification: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HITLApprovalQueue:
    """
    Human-in-the-Loop approval queue.

    In production with IdentArk, this would be:
    - WebSocket push notifications to reviewers
    - Mobile/Slack/Teams alerts
    - Timeout-to-deny (fail-safe)
    - MFA for critical actions
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __init__(self, interactive: bool = True) -> None:
        self.audit_log: list[AuditEntry] = []
        self.interactive = interactive
        self._counter = 0

    def submit(
        self,
        action: str,
        params: dict[str, Any],
        risk_level: RiskLevel,
        agent: str,
        justification: str,
    ) -> tuple[bool, str]:
        """
        Submit an action for approval.

        Returns (approved: bool, result: str)
        """
        self._counter += 1
        request = HITLRequest(
            request_id=f"HITL-{self._counter:04d}",
            action=action,
            params=params,
            risk_level=risk_level,
            agent=agent,
            justification=justification,
        )

        # Auto-approve low-risk actions
        if risk_level == RiskLevel.LOW:
            self._log(request, True, "auto-approved (low risk)")
            return True, "auto-approved"

        # Display approval prompt
        self._display_approval_request(request)

        if self.interactive:
            approved, result = self._prompt_user(request)
        else:
            # Non-interactive mode: auto-approve for demo
            approved, result = True, "auto-approved (non-interactive)"
            print(f"  {self.DIM}[Auto-approved for demo]{self.RESET}\n")

        self._log(request, approved, result)
        return approved, result

    def _display_approval_request(self, request: HITLRequest) -> None:
        """Display a formatted approval request."""
        color = request.risk_level.color
        print()
        print(f"{'═' * 70}")
        print(f"  {self.BOLD}🚨 HITL APPROVAL REQUIRED — {request.request_id}{self.RESET}")
        print(f"{'═' * 70}")
        print(f"  Agent:         {request.agent}")
        print(f"  Action:        {self.BOLD}{request.action}{self.RESET}")
        print(f"  Risk Level:    {color}{request.risk_level.value.upper()}{self.RESET}")
        print(f"  Justification: {request.justification}")
        print()
        print("  Parameters:")
        for k, v in request.params.items():
            print(f"    {k}: {self.BOLD}{v}{self.RESET}")
        print(f"{'─' * 70}")

    def _prompt_user(self, request: HITLRequest) -> tuple[bool, str]:
        """Prompt user for approval decision."""
        while True:
            try:
                choice = (
                    input(
                        f"  [{self.BOLD}A{self.RESET}]pprove  [{self.BOLD}R{self.RESET}]eject  > "
                    )
                    .strip()
                    .lower()
                )
                if choice in ("a", "approve", "y", "yes"):
                    print(f"  ✅ {self.BOLD}APPROVED{self.RESET}\n")
                    return True, "approved by operator"
                elif choice in ("r", "reject", "n", "no"):
                    reason = (
                        input("  Rejection reason (optional): ").strip() or "rejected by operator"
                    )
                    print(f"  ❌ {self.BOLD}REJECTED{self.RESET}: {reason}\n")
                    return False, reason
                else:
                    print("  Please enter 'A' to approve or 'R' to reject.")
            except (KeyboardInterrupt, EOFError):
                print(f"\n  ⏹️  {self.BOLD}SKIPPED{self.RESET}\n")
                return False, "skipped by operator"

    def _log(self, request: HITLRequest, approved: bool, result: str) -> None:
        """Log the action to the audit trail."""
        self.audit_log.append(
            AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent=request.agent,
                action=request.action,
                risk_level=request.risk_level,
                approved=approved,
                result=result,
            )
        )

    def print_audit_log(self) -> None:
        """Print the formatted audit trail."""
        print()
        print(f"{'═' * 70}")
        print(f"  {self.BOLD}📋 AUDIT LOG — Security Incident Response{self.RESET}")
        print(f"{'═' * 70}")

        if not self.audit_log:
            print("  No actions recorded.")
            return

        for entry in self.audit_log:
            status_icon = "✅" if entry.approved else "❌"
            color = entry.risk_level.color
            print(f"  {entry.timestamp[:19]}")
            print(f"    Agent:  {entry.agent}")
            print(
                f"    Action: {entry.action} ({color}{entry.risk_level.value.upper()}{self.RESET})"
            )
            print(f"    Status: {status_icon} {entry.result}")
            print()


# Global HITL queue
hitl_queue: HITLApprovalQueue


# ══════════════════════════════════════════════════════════════════════════════
# HITL Action Demonstration
# ══════════════════════════════════════════════════════════════════════════════


def demonstrate_hitl_actions() -> None:
    """
    Demonstrate the HITL approval flow for remediation actions.

    In production with IdentArk, these would be actual tool calls made by
    agents, with the ControlPlaneGateway routing them through the MCP
    approval queue.
    """
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🔐 EXECUTING REMEDIATION ACTIONS                                    ║")
    print("║                                                                      ║")
    print("║  The following actions were recommended by the agents.               ║")
    print("║  Each HIGH-RISK action requires your approval.                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Action 1: Block IP
    approved, _ = hitl_queue.submit(
        action="block_ip",
        params={"ip_address": "203.0.113.42", "duration_hours": 24},
        risk_level=RiskLevel.HIGH,
        agent="Remediation Agent",
        justification="Block malicious IP associated with credential stuffing attack",
    )

    # Action 2: Force password reset
    approved, _ = hitl_queue.submit(
        action="force_password_reset",
        params={"username": "john.doe@company.com"},
        risk_level=RiskLevel.HIGH,
        agent="Remediation Agent",
        justification="Reset password for potentially compromised account",
    )

    # Action 3: Enable MFA
    approved, _ = hitl_queue.submit(
        action="enable_mfa",
        params={"username": "john.doe@company.com"},
        risk_level=RiskLevel.MEDIUM,
        agent="Remediation Agent",
        justification="Enforce MFA to prevent future unauthorized access",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CrewAI Setup
# ══════════════════════════════════════════════════════════════════════════════


def create_crew(llm: Any) -> Any:
    """Create the Security Incident Response Crew."""
    from crewai import Agent, Crew, Process, Task

    triage_agent = Agent(
        role="Security Triage Analyst",
        goal="Classify security alerts by severity and determine investigation priority",
        backstory="You are the SOC's first responder. You quickly assess alerts and classify "
        "them as LOW, MEDIUM, HIGH, or CRITICAL based on threat indicators.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    forensics_agent = Agent(
        role="Security Forensics Investigator",
        goal="Investigate incidents to determine root cause and scope",
        backstory="You are a skilled investigator who analyzes threat intelligence, "
        "login patterns, and system logs to determine if an incident is real.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    remediation_agent = Agent(
        role="Security Remediation Specialist",
        goal="Execute approved remediation actions to contain threats",
        backstory="You take action to stop active threats. Your actions are HIGH RISK "
        "and require human approval. Always explain your reasoning clearly.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    triage_task = Task(
        description="""Analyze this security alert and classify its severity:

ALERT: Unusual login detected
- User: john.doe@company.com
- Source IP: 203.0.113.42
- Time: 2026-04-29 14:32:00 UTC
- Location: Unknown (VPN/Proxy detected)
- Previous location: London, UK (yesterday)

Classify as LOW, MEDIUM, HIGH, or CRITICAL. Explain your reasoning.""",
        expected_output="Severity classification with reasoning",
        agent=triage_agent,
    )

    forensics_task = Task(
        description="""Investigate this incident:
1. Analyze the IP address for threat indicators
2. Check the user's login history for anomalies
3. Determine if this is a real compromise or false positive
4. Recommend specific remediation actions""",
        expected_output="Investigation findings with recommended actions",
        agent=forensics_agent,
        context=[triage_task],
    )

    remediation_task = Task(
        description="""Based on the investigation, execute remediation:

If compromised:
1. Block the malicious IP
2. Force password reset
3. Enable MFA

If false positive:
- Document findings and recommend monitoring

Each high-risk action requires human approval.""",
        expected_output="Remediation report with action status",
        agent=remediation_agent,
        context=[forensics_task],
    )

    return Crew(
        agents=[triage_agent, forensics_agent, remediation_agent],
        tasks=[triage_task, forensics_task, remediation_task],
        process=Process.sequential,
        verbose=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def print_banner() -> None:
    """Print the demo banner."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║   🛡️  SECURITY INCIDENT RESPONSE CREW                                ║")
    print("║                                                                      ║")
    print("║   IdentArk + CrewAI — Human-in-the-Loop Demo                         ║")
    print("║                                                                      ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                                                                      ║")
    print("║   This demo shows:                                                   ║")
    print("║   • Multi-agent collaboration (Triage → Forensics → Remediation)     ║")
    print("║   • Human-in-the-loop approval for high-risk actions                 ║")
    print("║   • Complete audit trail of all agent actions                        ║")
    print("║   • Cost tracking via IdentArk gateway                               ║")
    print("║                                                                      ║")
    print("║   HIGH-RISK actions will prompt for your approval.                   ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def print_summary(result: str, cost: float) -> None:
    """Print the final summary."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ INCIDENT RESPONSE COMPLETE                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(result)
    print()
    print(f"{'─' * 70}")
    print(f"  💰 Total LLM Cost: ${cost:.4f}")
    print(f"{'─' * 70}")


def print_takeaways() -> None:
    """Print the key takeaways."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📌 KEY TAKEAWAYS                                                    ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║                                                                      ║")
    print("║  1. CAPABILITIES, NOT CREDENTIALS                                    ║")
    print("║     Agents execute actions without holding API keys or secrets.      ║")
    print("║     In production, ControlPlaneGateway fetches credentials per-call. ║")
    print("║                                                                      ║")
    print("║  2. HUMAN-IN-THE-LOOP                                                ║")
    print("║     High-risk actions (block_ip, password_reset) require approval.   ║")
    print("║     Production: WebSocket push, mobile alerts, timeout-to-deny.      ║")
    print("║                                                                      ║")
    print("║  3. AUDIT TRAIL                                                      ║")
    print("║     Every action logged with timestamp, agent, risk, and outcome.    ║")
    print("║     Immutable log for compliance and post-incident review.           ║")
    print("║                                                                      ║")
    print("║  4. COST TRACKING                                                    ║")
    print("║     LLM spend tracked across all agents for budgeting & alerts.      ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def main() -> None:
    global hitl_queue

    parser = argparse.ArgumentParser(
        description="Security Incident Response Crew — IdentArk + CrewAI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--openai", action="store_true", help="Use OpenAI (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--groq", action="store_true", help="Use Groq (free, requires GROQ_API_KEY)"
    )
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument(
        "--non-interactive", action="store_true", help="Auto-approve all actions (for CI)"
    )
    args = parser.parse_args()

    # Check dependencies
    try:
        import crewai  # noqa: F401
    except ImportError:
        print("ERROR: CrewAI not installed. Run: pip install crewai")
        sys.exit(1)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: OpenAI SDK not installed. Run: pip install openai")
        sys.exit(1)

    from identark import DirectGateway
    from identark.integrations.crewai import IdentArkCrewAILLM

    # Create gateway based on provider
    if args.openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set")
            sys.exit(1)
        print("Using OpenAI (gpt-4o-mini)")
        gateway = DirectGateway(
            llm_client=AsyncOpenAI(),
            model=args.model or "gpt-4o-mini",
            system_prompt="You are a security operations expert. Be concise.",
        )
    elif args.groq:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY not set. Get a free key at console.groq.com")
            sys.exit(1)
        print("Using Groq (llama-3.3-70b-versatile)")
        gateway = DirectGateway(
            llm_client=AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            ),
            model=args.model or "llama-3.3-70b-versatile",
            system_prompt="You are a security operations expert. Be concise.",
        )
    else:
        print("Using Ollama (local) — ensure 'ollama serve' is running")
        gateway = DirectGateway(
            llm_client=AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            model=args.model or "llama3.2",
            provider="local",
            system_prompt="You are a security operations expert. Be concise.",
        )

    # Initialize HITL queue
    hitl_queue = HITLApprovalQueue(interactive=not args.non_interactive)

    # Create LLM wrapper and crew
    llm = IdentArkCrewAILLM(gateway=gateway)
    crew = create_crew(llm)

    # Run
    print_banner()
    print("Starting incident response...\n")

    try:
        result = crew.kickoff()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        hitl_queue.print_audit_log()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        hitl_queue.print_audit_log()
        sys.exit(1)

    # Execute remediation actions with HITL approval
    demonstrate_hitl_actions()

    # Get final cost
    async def get_cost() -> float:
        return await gateway.get_session_cost()

    cost = asyncio.run(get_cost())

    # Print results
    print_summary(str(result), cost)
    hitl_queue.print_audit_log()
    print_takeaways()


if __name__ == "__main__":
    main()
