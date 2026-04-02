#!/usr/bin/env python3
"""
Gemini Finance Agent Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A complete example of a finance AI agent for Lloyds banking
using IdentArk with Google Gemini.

This demonstrates:
- Tool calling for banking operations
- Multi-turn conversations
- Cost tracking
- Session management

Setup:
    export GEMINI_API_KEY=your-key
    pip install identark[gemini]
    python examples/gemini_finance_agent.py
"""

import asyncio
import json
import os
from datetime import datetime

from identark import Message, Role
from identark.integrations.gemini import GeminiGateway


# Simulated banking data
ACCOUNTS = {
    "current": {"balance": 2547.83, "currency": "GBP", "name": "Current Account"},
    "savings": {"balance": 15420.00, "currency": "GBP", "name": "Savings Account"},
    "isa": {"balance": 8750.50, "currency": "GBP", "name": "Stocks & Shares ISA"},
}

TRANSACTIONS = [
    {"date": "2026-03-22", "description": "Tesco", "amount": -45.67, "category": "groceries"},
    {"date": "2026-03-21", "description": "Salary - ACME Corp", "amount": 3200.00, "category": "income"},
    {"date": "2026-03-20", "description": "Netflix", "amount": -15.99, "category": "entertainment"},
    {"date": "2026-03-19", "description": "TfL", "amount": -8.50, "category": "transport"},
    {"date": "2026-03-18", "description": "Amazon", "amount": -127.45, "category": "shopping"},
]


def get_balance(account_id: str) -> str:
    """Simulate getting account balance."""
    if account_id in ACCOUNTS:
        acc = ACCOUNTS[account_id]
        return json.dumps({
            "account": acc["name"],
            "balance": acc["balance"],
            "currency": acc["currency"],
        })
    return json.dumps({"error": f"Account {account_id} not found"})


def get_transactions(account_id: str, limit: int = 5) -> str:
    """Simulate getting recent transactions."""
    return json.dumps({
        "account": account_id,
        "transactions": TRANSACTIONS[:limit],
    })


def transfer_money(from_account: str, to_account: str, amount: float) -> str:
    """Simulate a money transfer."""
    if from_account not in ACCOUNTS or to_account not in ACCOUNTS:
        return json.dumps({"error": "Invalid account"})
    if ACCOUNTS[from_account]["balance"] < amount:
        return json.dumps({"error": "Insufficient funds"})

    return json.dumps({
        "status": "success",
        "reference": f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "from": from_account,
        "to": to_account,
        "amount": amount,
        "new_balance": ACCOUNTS[from_account]["balance"] - amount,
    })


# Tool definitions
BANKING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Get the current balance of a bank account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account ID: 'current', 'savings', or 'isa'",
                        "enum": ["current", "savings", "isa"]
                    }
                },
                "required": ["account_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "Get recent transactions for an account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account ID",
                        "enum": ["current", "savings", "isa"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of transactions to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["account_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_money",
            "description": "Transfer money between accounts",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account": {
                        "type": "string",
                        "description": "Source account ID",
                        "enum": ["current", "savings", "isa"]
                    },
                    "to_account": {
                        "type": "string",
                        "description": "Destination account ID",
                        "enum": ["current", "savings", "isa"]
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to transfer in GBP"
                    }
                },
                "required": ["from_account", "to_account", "amount"]
            }
        }
    }
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a banking tool and return the result."""
    if name == "get_balance":
        return get_balance(args["account_id"])
    elif name == "get_transactions":
        return get_transactions(args["account_id"], args.get("limit", 5))
    elif name == "transfer_money":
        return transfer_money(args["from_account"], args["to_account"], args["amount"])
    return json.dumps({"error": f"Unknown tool: {name}"})


async def run_finance_agent():
    """Run the interactive finance agent."""
    print("=" * 60)
    print("Lloyds Finance Agent (powered by Gemini + IdentArk)")
    print("=" * 60)
    print()

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-pro",
        system_prompt="""You are a helpful Lloyds banking assistant. You can help customers:
- Check their account balances
- View recent transactions
- Transfer money between accounts

Always be polite, professional, and follow FCA guidelines.
When performing operations, explain what you're doing.
Format currency amounts properly (e.g., GBP 1,234.56).""",
        cost_cap_usd=0.50,  # Set a reasonable cost cap
    )

    print("Welcome to Lloyds Banking! How can I help you today?")
    print("(Type 'quit' to exit, 'cost' to see session cost)")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "cost":
            cost = await gateway.get_session_cost()
            print(f"[Session cost: ${cost:.6f}]")
            continue

        # Send message to agent
        response = await gateway.invoke_llm(
            new_messages=[Message(role=Role.USER, content=user_input)],
            tools=BANKING_TOOLS,
        )

        # Handle tool calls
        while response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"[Calling {tc.function.name}...]")
                result = execute_tool(tc.function.name, args)
                tool_results.append(Message(
                    role=Role.TOOL,
                    content=result,
                    tool_call_id=tc.id,
                ))

            # Send tool results back
            response = await gateway.invoke_llm(
                new_messages=tool_results,
                tools=BANKING_TOOLS,
            )

        print(f"Agent: {response.message.content}")
        print()

    # Final summary
    cost = await gateway.get_session_cost()
    print()
    print("=" * 60)
    print(f"Session ended. Total cost: ${cost:.6f}")
    print("Thank you for banking with Lloyds!")
    print("=" * 60)


async def demo_queries():
    """Run some demo queries non-interactively."""
    print("=" * 60)
    print("Finance Agent Demo")
    print("=" * 60)

    gateway = GeminiGateway(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash",  # Use flash for demo
        system_prompt="You are a Lloyds banking assistant. Be concise.",
    )

    queries = [
        "What's my current account balance?",
        "Show me my recent transactions",
        "Transfer GBP 100 from savings to current account",
    ]

    for query in queries:
        print(f"\nUser: {query}")

        response = await gateway.invoke_llm(
            new_messages=[Message(role=Role.USER, content=query)],
            tools=BANKING_TOOLS,
        )

        # Handle tool calls
        while response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  [Tool: {tc.function.name}({json.dumps(args)})]")
                result = execute_tool(tc.function.name, args)
                tool_results.append(Message(
                    role=Role.TOOL,
                    content=result,
                    tool_call_id=tc.id,
                ))

            response = await gateway.invoke_llm(
                new_messages=tool_results,
                tools=BANKING_TOOLS,
            )

        print(f"Agent: {response.message.content}")

    cost = await gateway.get_session_cost()
    print(f"\n[Total cost: ${cost:.6f}]")


async def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Please set GEMINI_API_KEY environment variable")
        print("Get your key from: https://aistudio.google.com/apikey")
        return

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        await demo_queries()
    else:
        await run_finance_agent()


if __name__ == "__main__":
    asyncio.run(main())
