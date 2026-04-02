"""
identark.integrations.langchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangChain integration — IdentArkChatModel.

Wraps any AgentGateway as a LangChain BaseChatModel so you can use
IdentArk's credential-isolated gateway inside any LangChain chain,
agent, or pipeline. Conversation history is maintained by the gateway,
not by LangChain memory objects.

Install::

    pip install identark[langchain]

Usage::

    from identark import DirectGateway
    from identark.integrations.langchain import IdentArkChatModel
    from openai import AsyncOpenAI
    from langchain_core.messages import HumanMessage

    gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
    llm = IdentArkChatModel(gateway=gateway)

    # Async (recommended)
    response = await llm.ainvoke([HumanMessage(content="Hello!")])

    # Sync
    response = llm.invoke([HumanMessage(content="Hello!")])
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from identark.models import LLMResponse, Message, Role

logger = logging.getLogger("identark.integrations.langchain")


# ── Message conversion helpers ────────────────────────────────────────────────


def _extract_content(msg: BaseMessage) -> str | list[dict[str, Any]]:
    """Return message content in a form compatible with identark.models.Message."""
    if isinstance(msg.content, str):
        return msg.content
    # Multimodal / structured content blocks
    result: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, dict):
            result.append(block)
        else:
            result.append({"type": "text", "text": str(block)})
    return result


def lc_to_identark(messages: list[BaseMessage]) -> list[Message]:
    """Convert a list of LangChain messages to IdentArk Message objects."""
    result: list[Message] = []
    for msg in messages:
        content = _extract_content(msg)
        if isinstance(msg, HumanMessage):
            result.append(Message(role=Role.USER, content=content))
        elif isinstance(msg, AIMessage):
            result.append(Message(role=Role.ASSISTANT, content=content))
        elif isinstance(msg, SystemMessage):
            result.append(Message(role=Role.SYSTEM, content=content))
        elif isinstance(msg, ToolMessage):
            result.append(
                Message(role=Role.TOOL, content=content, tool_call_id=msg.tool_call_id)
            )
        else:
            # ChatMessage or other custom types — infer role from .role attribute
            raw_role: str = getattr(msg, "role", "user")
            try:
                role = Role(raw_role)
            except ValueError:
                role = Role.USER
            result.append(Message(role=role, content=content))
    return result


def identark_to_ai_message(response: LLMResponse) -> AIMessage:
    """Convert a IdentArk LLMResponse to a LangChain AIMessage."""
    tool_calls: list[dict[str, Any]] = []
    if response.tool_calls:
        for tc in response.tool_calls:
            try:
                args: dict[str, Any] = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "args": args, "type": "tool_call"}
            )

    content = response.message.content
    text: str = content if isinstance(content, str) else ""

    return AIMessage(
        content=text,
        tool_calls=tool_calls,
        response_metadata={
            "model": response.model,
            "finish_reason": response.finish_reason,
            "cost_usd": response.cost_usd,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )


# ── IdentArkChatModel ─────────────────────────────────────────────────────────


class IdentArkChatModel(BaseChatModel):
    """
    LangChain ``BaseChatModel`` backed by a IdentArk ``AgentGateway``.

    Drop-in replacement for any LangChain chat model. Routes all LLM
    calls through the gateway so credentials never enter the agent loop.

    Args:
        gateway: Any :class:`~identark.gateway.AgentGateway` implementation
                 (``DirectGateway``, ``ControlPlaneGateway``, ``MockGateway``, …).

    Example::

        from identark import DirectGateway
        from identark.integrations.langchain import IdentArkChatModel
        from openai import AsyncOpenAI

        llm = IdentArkChatModel(
            gateway=DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
        )
        chain = prompt | llm | StrOutputParser()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway: Any = Field(..., exclude=True)

    # ── LangChain required interface ──────────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "identark"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": getattr(self.gateway, "model", "unknown"),
            "provider": getattr(self.gateway, "provider", "unknown"),
            "gateway_type": type(self.gateway).__name__,
        }

    # ── Core implementation ───────────────────────────────────────────────────

    async def _agenerate_impl(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
    ) -> ChatResult:
        """Shared async core used by both _generate and _agenerate."""
        cs_messages = lc_to_identark(messages)
        logger.debug(
            "IdentArkChatModel invoke messages=%d tools=%s",
            len(cs_messages),
            len(tools) if tools else 0,
        )
        response = await self.gateway.invoke_llm(
            new_messages=cs_messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return ChatResult(generations=[ChatGeneration(message=identark_to_ai_message(response))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous path — runs the async gateway in an isolated thread."""
        tools: list[dict[str, Any]] | None = kwargs.get("tools")
        raw_choice = kwargs.get("tool_choice", "auto")
        tool_choice: str | dict[str, Any] = (
            raw_choice if isinstance(raw_choice, (str, dict)) else "auto"
        )
        coro = self._agenerate_impl(messages, tools=tools, tool_choice=tool_choice)
        try:
            asyncio.get_running_loop()
            # Already inside an event loop (e.g. Jupyter) — run in a thread
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async path — preferred entry point."""
        tools: list[dict[str, Any]] | None = kwargs.get("tools")
        raw_choice = kwargs.get("tool_choice", "auto")
        tool_choice: str | dict[str, Any] = (
            raw_choice if isinstance(raw_choice, (str, dict)) else "auto"
        )
        return await self._agenerate_impl(messages, tools=tools, tool_choice=tool_choice)
