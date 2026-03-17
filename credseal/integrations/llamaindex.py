"""
credseal.integrations.llamaindex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LlamaIndex integration — CredSealLLM.

Wraps any AgentGateway as a LlamaIndex ``CustomLLM`` so you can use
CredSeal's credential-isolated gateway inside any LlamaIndex query
engine, agent, or pipeline. Conversation history is maintained by
the gateway, not by LlamaIndex's chat store.

Install::

    pip install credseal-sdk[llamaindex]

Usage::

    from credseal import DirectGateway
    from credseal.integrations.llamaindex import CredSealLLM
    from openai import AsyncOpenAI
    from llama_index.core.llms import ChatMessage, MessageRole

    gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
    llm = CredSealLLM(gateway=gateway)

    # Async (recommended)
    response = await llm.achat([ChatMessage(role=MessageRole.USER, content="Hello!")])

    # Sync
    response = llm.chat([ChatMessage(role=MessageRole.USER, content="Hello!")])
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llama_index.core.base.llms.types import (
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
)
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from pydantic import ConfigDict, Field

from credseal.models import LLMResponse, Message, Role

logger = logging.getLogger("credseal.integrations.llamaindex")

# ── Role mapping ──────────────────────────────────────────────────────────────

_TO_CREDSEAL_ROLE: dict[MessageRole, Role] = {
    MessageRole.USER: Role.USER,
    MessageRole.ASSISTANT: Role.ASSISTANT,
    MessageRole.CHATBOT: Role.ASSISTANT,
    MessageRole.MODEL: Role.ASSISTANT,
    MessageRole.SYSTEM: Role.SYSTEM,
    MessageRole.DEVELOPER: Role.SYSTEM,
    MessageRole.TOOL: Role.TOOL,
    MessageRole.FUNCTION: Role.TOOL,
}

_FROM_CREDSEAL_ROLE: dict[Role, MessageRole] = {
    Role.USER: MessageRole.USER,
    Role.ASSISTANT: MessageRole.ASSISTANT,
    Role.SYSTEM: MessageRole.SYSTEM,
    Role.TOOL: MessageRole.TOOL,
}

# ── Conversion helpers ────────────────────────────────────────────────────────


def li_to_credseal(messages: list[ChatMessage]) -> list[Message]:
    """Convert LlamaIndex ChatMessages to CredSeal Message objects."""
    result: list[Message] = []
    for msg in messages:
        role = _TO_CREDSEAL_ROLE.get(msg.role, Role.USER)
        content: str = msg.content or ""
        tool_call_id: str | None = msg.additional_kwargs.get("tool_call_id")
        result.append(Message(role=role, content=content, tool_call_id=tool_call_id))
    return result


def credseal_to_chat_response(response: LLMResponse) -> ChatResponse:
    """Convert a CredSeal LLMResponse to a LlamaIndex ChatResponse."""
    additional_kwargs: dict[str, Any] = {}
    if response.tool_calls:
        additional_kwargs["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in response.tool_calls
        ]

    content = response.message.content
    text: str = content if isinstance(content, str) else ""

    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            additional_kwargs=additional_kwargs,
        ),
        raw={
            "model": response.model,
            "finish_reason": response.finish_reason,
            "cost_usd": response.cost_usd,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )


# ── CredSealLLM ───────────────────────────────────────────────────────────────


class CredSealLLM(CustomLLM):
    """
    LlamaIndex ``CustomLLM`` backed by a CredSeal ``AgentGateway``.

    Drop-in replacement for any LlamaIndex LLM. Routes all inference
    calls through the gateway so credentials never enter the agent loop.

    Args:
        gateway: Any :class:`~credseal.gateway.AgentGateway` implementation
                 (``DirectGateway``, ``ControlPlaneGateway``, ``MockGateway``, …).

    Example::

        from credseal import DirectGateway
        from credseal.integrations.llamaindex import CredSealLLM
        from openai import AsyncOpenAI

        llm = CredSealLLM(
            gateway=DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
        )
        engine = index.as_query_engine(llm=llm)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway: Any = Field(..., exclude=True)

    # ── LlamaIndex required interface ─────────────────────────────────────────

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=getattr(self.gateway, "model", "credseal"),
            is_chat_model=True,
            is_function_calling_model=True,
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Wrap prompt as a user message and delegate to chat."""
        chat_resp = self.chat([ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs)
        return CompletionResponse(text=chat_resp.message.content or "")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming not yet supported."""
        raise NotImplementedError(
            "CredSealLLM does not yet support streaming. "
            "Use complete() or chat() instead. Streaming is on the roadmap."
        )
        yield  # pragma: no cover  # makes this a generator function for the type checker

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming not yet supported."""
        raise NotImplementedError(
            "CredSealLLM does not yet support streaming. "
            "Use chat() or achat() instead. Streaming is on the roadmap."
        )
        yield  # pragma: no cover

    # ── Chat (primary) ────────────────────────────────────────────────────────

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Synchronous chat — runs the async gateway in an isolated thread."""
        coro = self.achat(messages, **kwargs)
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat — primary implementation."""
        cs_messages = li_to_credseal(list(messages))
        tools: list[dict[str, Any]] | None = kwargs.get("tools")
        raw_choice = kwargs.get("tool_choice", "auto")
        tool_choice: str | dict[str, Any] = (
            raw_choice if isinstance(raw_choice, (str, dict)) else "auto"
        )
        logger.debug(
            "CredSealLLM.achat messages=%d tools=%s",
            len(cs_messages),
            len(tools) if tools else 0,
        )
        response = await self.gateway.invoke_llm(
            new_messages=cs_messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return credseal_to_chat_response(response)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Async completion — wraps prompt as a user message."""
        chat_resp = await self.achat(
            [ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs
        )
        return CompletionResponse(text=chat_resp.message.content or "")
