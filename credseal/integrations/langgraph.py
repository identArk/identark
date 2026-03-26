"""
credseal.integrations.langgraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph integration — CredSealNode and CredSealStreamNode.

Wrap any AgentGateway as a LangGraph node so you can use CredSeal's
credential-isolated gateway inside any LangGraph StateGraph. Conversation
history is maintained by the gateway, not by LangGraph state.

Install::

    pip install credseal-sdk[langgraph]

Usage::

    from langgraph.graph import StateGraph, MessagesState
    from credseal import DirectGateway
    from credseal.integrations.langgraph import CredSealNode
    from openai import AsyncOpenAI

    gateway = DirectGateway(llm_client=AsyncOpenAI(), model="gpt-4o")
    node = CredSealNode(gateway=gateway)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    app = graph.compile()

    result = await app.ainvoke({"messages": [{"role": "user", "content": "Hello!"}]})
"""

from __future__ import annotations

import logging
from typing import Any

from credseal.integrations.langchain import credseal_to_ai_message, lc_to_credseal

logger = logging.getLogger("credseal.integrations.langgraph")


def _normalise_messages(raw: list[Any]) -> list[Any]:
    """
    Accept either LangChain BaseMessage objects or plain dicts
    (LangGraph sometimes passes dicts when state is serialised).
    Returns LangChain messages in either case.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    result = []
    for msg in raw:
        if hasattr(msg, "role") or hasattr(msg, "content"):
            result.append(msg)
            continue
        # Plain dict — reconstruct
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        elif role == "tool":
            result.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
        else:
            result.append(HumanMessage(content=content))
    return result


class CredSealNode:
    """
    A LangGraph node backed by a CredSeal ``AgentGateway``.

    Reads ``state["messages"]``, sends the *last* message (new turn only)
    to the gateway, and appends the assistant reply to ``state["messages"]``.

    The gateway maintains full conversation history internally — the node
    only passes the latest user message on each invocation, not the full
    history. This avoids double-counting history that the gateway already holds.

    Args:
        gateway:    Any :class:`~credseal.gateway.AgentGateway` implementation.
        tools:      Optional list of OpenAI-format tool definitions.
        tool_choice: Tool selection mode. Default ``'auto'``.
        messages_key: The state key that holds the message list. Default ``'messages'``.

    Example::

        node = CredSealNode(gateway=gateway)
        graph.add_node("agent", node)
    """

    def __init__(
        self,
        gateway: Any,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        messages_key: str = "messages",
    ) -> None:
        self._gateway = gateway
        self._tools = tools
        self._tool_choice = tool_choice
        self._messages_key = messages_key

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Async node invocation — preferred in LangGraph async graphs."""
        messages = _normalise_messages(state.get(self._messages_key, []))
        if not messages:
            return {self._messages_key: []}

        # Send only the last message as the new turn
        last_msg = messages[-1]
        cs_messages = lc_to_credseal([last_msg])

        logger.debug("CredSealNode invoke tools=%s", bool(self._tools))
        response = await self._gateway.invoke_llm(
            new_messages=cs_messages,
            tools=self._tools,
            tool_choice=self._tool_choice,
        )
        ai_message = credseal_to_ai_message(response)
        return {self._messages_key: messages + [ai_message]}

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Sync node invocation for non-async graphs."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        coro = self.__call__(state)
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)


class CredSealStreamNode:
    """
    A LangGraph node that streams the assistant response token by token.

    Identical to :class:`CredSealNode` but uses ``invoke_llm_stream``
    and accumulates chunks into a single ``AIMessage`` before returning
    the updated state. Useful when you want to display partial output
    via LangGraph's streaming callbacks.

    Args:
        gateway:      Any :class:`~credseal.gateway.AgentGateway` implementation.
        tools:        Optional list of OpenAI-format tool definitions.
        messages_key: The state key that holds the message list. Default ``'messages'``.
    """

    def __init__(
        self,
        gateway: Any,
        tools: list[dict[str, Any]] | None = None,
        messages_key: str = "messages",
    ) -> None:
        self._gateway = gateway
        self._tools = tools
        self._messages_key = messages_key

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import AIMessage

        messages = _normalise_messages(state.get(self._messages_key, []))
        if not messages:
            return {self._messages_key: []}

        last_msg = messages[-1]
        cs_messages = lc_to_credseal([last_msg])

        full_content = ""
        finish_reason = "stop"
        model = "unknown"
        input_tokens = output_tokens = 0

        async for chunk in self._gateway.invoke_llm_stream(
            new_messages=cs_messages,
            tools=self._tools,
        ):
            full_content += chunk.content
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
                model = chunk.model
                input_tokens = chunk.input_tokens
                output_tokens = chunk.output_tokens

        ai_message = AIMessage(
            content=full_content,
            response_metadata={
                "model": model,
                "finish_reason": finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        return {self._messages_key: messages + [ai_message]}
