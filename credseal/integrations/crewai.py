"""
credseal.integrations.crewai
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CrewAI integration — CredSealCrewAILLM.

Wraps any AgentGateway as a CrewAI BaseLLM so you can run CrewAI agents
through CredSeal gateways (DirectGateway / ControlPlaneGateway / MockGateway).

Install::

    pip install credseal-sdk crewai

Usage::

    from crewai import Agent
    from credseal import DirectGateway
    from credseal.integrations.crewai import CredSealCrewAILLM
    from openai import AsyncOpenAI

    gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
    llm = CredSealCrewAILLM(gateway=gateway)

    agent = Agent(
        role="Researcher",
        goal="Find and summarize information",
        backstory="You are a helpful research assistant.",
        llm=llm,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from credseal.models import Message, Role

logger = logging.getLogger("credseal.integrations.crewai")

try:
    # CrewAI exposes BaseLLM for custom model integrations.
    from crewai import BaseLLM  # type: ignore[import-not-found]

    _CREWAI_AVAILABLE = True
except Exception:  # pragma: no cover
    BaseLLM = object  # type: ignore[assignment]
    _CREWAI_AVAILABLE = False


CrewAIMessages = str | list[dict[str, Any]]


def _ensure_crewai_available() -> None:
    if _CREWAI_AVAILABLE:
        return
    raise ImportError(
        "CrewAI is not installed. Install it with `pip install crewai` "
        "or `pip install credseal-sdk[all] crewai`."
    )


def crewai_to_credseal(messages: CrewAIMessages) -> list[Message]:
    """
    Convert CrewAI-style messages into CredSeal Message objects.

    CrewAI passes either a prompt string or a list of dict messages with
    at least: {'role': 'user'|'assistant'|'system'|'tool', 'content': ...}
    """
    if isinstance(messages, str):
        return [Message(role=Role.USER, content=messages)]

    result: list[Message] = []
    for m in messages:
        role_raw = str(m.get("role", "user"))
        content = m.get("content", "")

        try:
            role = Role(role_raw)
        except ValueError:
            role = Role.USER

        tool_call_id = m.get("tool_call_id")
        name = m.get("name")

        result.append(
            Message(
                role=role,
                content=cast(str | list[dict[str, Any]], content),
                tool_call_id=cast(str | None, tool_call_id),
                name=cast(str | None, name),
            )
        )
    return result


def _messages_prefix_len(
    prev: Sequence[dict[str, Any]],
    curr: Sequence[dict[str, Any]],
) -> int:
    """Return the shared prefix length for two message lists."""
    n = min(len(prev), len(curr))
    for i in range(n):
        if prev[i] != curr[i]:
            return i
    return n


class CredSealCrewAILLM(BaseLLM):  # type: ignore[misc]
    """
    CrewAI BaseLLM backed by a CredSeal AgentGateway.

    CrewAI typically provides the *full* message list each call, while
    CredSeal gateways expect only the *new* messages for this turn. This
    adapter tracks the last message list it saw and sends only the delta.
    """

    def __init__(
        self,
        gateway: Any,
        model: str | None = None,
        temperature: float | None = None,
        context_window_size: int = 8192,
    ) -> None:
        _ensure_crewai_available()

        super().__init__(
            model=model or getattr(gateway, "model", "credseal"),
            temperature=temperature,
        )
        self._gateway = gateway
        self._context_window_size = context_window_size

        # CrewAI passes full message history on each call; track the last
        # one so we can compute an incremental "new_messages" delta.
        self._last_messages: list[dict[str, Any]] = []

    # ── CrewAI optional capabilities ──────────────────────────────────────────

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        # Gateways don't universally support native stop sequences.
        # We handle stop words by truncating the final text response.
        return False

    def get_context_window_size(self) -> int:
        return self._context_window_size

    # ── Core call implementation ──────────────────────────────────────────────

    def call(
        self,
        messages: CrewAIMessages,
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
    ) -> str | Any:
        """
        Synchronous CrewAI entry point.

        CrewAI's BaseLLM expects a sync call() method. We run the async
        gateway in an event loop (or in an isolated thread if already in one).
        """
        _ensure_crewai_available()

        coro = self._call_async(
            messages=messages,
            tools=tools,
            available_functions=available_functions,
        )
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _call_async(
        self,
        messages: CrewAIMessages,
        tools: list[dict] | None,
        available_functions: dict[str, Any] | None,
    ) -> str:
        # Normalize CrewAI inputs to the list[dict] representation so we can
        # compute a delta against the last call.
        if isinstance(messages, str):
            curr_msgs: list[dict[str, Any]] = [{"role": "user", "content": messages}]
        else:
            curr_msgs = list(messages)

        shared = _messages_prefix_len(self._last_messages, curr_msgs)
        delta = curr_msgs[shared:]
        self._last_messages = curr_msgs

        new_messages = crewai_to_credseal(delta)

        logger.debug(
            "CredSealCrewAILLM call total=%d delta=%d tools=%s",
            len(curr_msgs),
            len(new_messages),
            len(tools) if tools else 0,
        )

        response = await self._gateway.invoke_llm(
            new_messages=new_messages,
            tools=tools,
            tool_choice="auto",
        )

        # Tool calling loop: if the gateway requested tool calls and CrewAI
        # provided available_functions, execute and continue until a final answer.
        #
        # We only persist tool outputs; the gateway is expected to persist its
        # own assistant tool-call message as part of invoke_llm().
        while response.tool_calls and available_functions:
            tool_messages: list[Message] = []
            for tc in response.tool_calls:
                fn_name = tc.function.name
                fn = available_functions.get(fn_name)
                if fn is None:
                    tool_messages.append(
                        Message(
                            role=Role.TOOL,
                            content=f"Tool '{fn_name}' not found in available_functions.",
                            tool_call_id=tc.id,
                            name=fn_name,
                        )
                    )
                    continue

                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}

                try:
                    result = fn(**args) if isinstance(args, dict) else fn(args)
                except Exception as e:  # noqa: BLE001
                    result = f"Tool '{fn_name}' raised: {type(e).__name__}: {e}"

                tool_messages.append(
                    Message(
                        role=Role.TOOL,
                        content=str(result),
                        tool_call_id=tc.id,
                        name=fn_name,
                    )
                )

            await self._gateway.persist_messages(tool_messages)
            response = await self._gateway.invoke_llm(
                new_messages=tool_messages,
                tools=tools,
                tool_choice="auto",
            )

        content = response.message.content
        text: str = content if isinstance(content, str) else ""

        # Manual stop-word handling if CrewAI configured stop sequences.
        stops = getattr(self, "stop", None)
        if stops:
            for s in stops:
                if s and s in text:
                    text = text.split(s)[0]
                    break

        return text

