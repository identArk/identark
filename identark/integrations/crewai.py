"""
identark.integrations.crewai
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CrewAI integration — IdentArkCrewAILLM.

Wraps any AgentGateway as a CrewAI BaseLLM so you can run CrewAI agents
through IdentArk gateways (DirectGateway / ControlPlaneGateway / MockGateway).

Install::

    pip install identark crewai

Usage::

    from crewai import Agent
    from identark import DirectGateway
    from identark.integrations.crewai import IdentArkCrewAILLM
    from openai import AsyncOpenAI

    gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
    llm = IdentArkCrewAILLM(gateway=gateway)

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from identark.models import Message, Role

logger = logging.getLogger("identark.integrations.crewai")

try:
    from crewai import BaseLLM

    _CREWAI_AVAILABLE = True
except Exception:  # pragma: no cover
    BaseLLM = object
    _CREWAI_AVAILABLE = False


CrewAIMessages = str | list[dict[str, Any]]


def _ensure_crewai_available() -> None:
    if _CREWAI_AVAILABLE:
        return
    raise ImportError(
        "CrewAI is not installed. Install it with `pip install crewai` "
        "or `pip install identark[all] crewai`."
    )


def _crewai_to_identark(messages: CrewAIMessages) -> list[Message]:
    """Convert CrewAI-style messages into IdentArk Message objects."""
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

        result.append(
            Message(
                role=role,
                content=cast(str | list[dict[str, Any]], content),
                tool_call_id=cast(str | None, m.get("tool_call_id")),
                name=cast(str | None, m.get("name")),
            )
        )
    return result


class IdentArkCrewAILLM(BaseLLM):  # type: ignore[misc]
    """
    CrewAI BaseLLM backed by an IdentArk AgentGateway.

    CrewAI sends the full message history on each call. This adapter clears
    the gateway's internal history before each call to avoid duplication,
    then forwards the complete message list to the LLM.

    Args:
        gateway: An IdentArk gateway (DirectGateway, ControlPlaneGateway, etc.)
        model: Optional model name override for CrewAI's internal tracking.
        temperature: Optional temperature setting.
        context_window_size: Context window size for CrewAI's token management.
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
            model=model or getattr(gateway, "model", "identark"),
            temperature=temperature,
        )
        self._gateway = gateway
        self._context_window_size = context_window_size

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return self._context_window_size

    def call(
        self,
        messages: CrewAIMessages,
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | Any:
        """
        Synchronous CrewAI entry point.

        Runs the async gateway in an event loop, handling the case where
        we're already inside an async context.
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
        tools: list[dict[str, Any]] | None,
        available_functions: dict[str, Any] | None,
    ) -> str:
        # Normalize to list format
        if isinstance(messages, str):
            curr_msgs: list[dict[str, Any]] = [{"role": "user", "content": messages}]
        else:
            curr_msgs = list(messages)

        # CrewAI sends full history each call. Clear gateway's internal history
        # to avoid duplication, then send the complete message list.
        if hasattr(self._gateway, "_history"):
            self._gateway._history = []

        new_messages = _crewai_to_identark(curr_msgs)

        logger.debug(
            "IdentArkCrewAILLM: %d messages, %d tools",
            len(new_messages),
            len(tools) if tools else 0,
        )

        response = await self._gateway.invoke_llm(
            new_messages=new_messages,
            tools=tools,
            tool_choice="auto",
        )

        # Tool calling loop
        while response.tool_calls and available_functions:
            tool_messages: list[Message] = []
            for tc in response.tool_calls:
                fn_name = tc.function.name
                fn = available_functions.get(fn_name)
                if fn is None:
                    tool_messages.append(
                        Message(
                            role=Role.TOOL,
                            content=f"Tool '{fn_name}' not found.",
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
                    result = f"Tool error: {type(e).__name__}: {e}"

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

        # Handle stop sequences if configured
        stops = getattr(self, "stop", None)
        if stops:
            for s in stops:
                if s and s in text:
                    text = text.split(s)[0]
                    break

        return text
