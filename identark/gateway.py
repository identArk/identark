"""
identark.gateway
~~~~~~~~~~~~~~~~~~
The AgentGateway Protocol — the core interface of the SDK.

Any class that implements the four methods below is a valid gateway,
whether it comes from this SDK or not. Python's structural subtyping
(typing.Protocol) means no explicit inheritance is required.

Usage::

    from identark import AgentGateway

    def run_agent(gateway: AgentGateway) -> None:
        # Works with DirectGateway, ControlPlaneGateway, MockGateway,
        # or any custom implementation.
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

from identark.models import LLMResponse, Message, PresignedURL, StreamChunk


@runtime_checkable
class AgentGateway(Protocol):
    """
    The AgentGateway protocol defines how an agent communicates
    with the outside world.

    Implement this protocol to create a custom gateway for any backend.
    All four methods must be ``async``.

    The gateway is the single boundary between your agent logic and
    everything external: LLM providers, file storage, cost tracking.
    Agents built against this protocol hold no secrets and maintain
    no persistent state themselves.
    """

    async def invoke_llm(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """
        Send new messages to the LLM and receive a response.

        The gateway is responsible for maintaining and reconstructing
        the full conversation history. Callers should pass only the
        *new* messages for this turn — do not include prior history.

        Args:
            new_messages: New messages to send this turn.
            tools:        OpenAI-format tool/function definitions.
                          Pass ``None`` if no tools are available.
            tool_choice:  Tool selection mode. One of ``'auto'``,
                          ``'none'``, ``'required'``, or a specific
                          tool dict ``{"type": "function", "function": {"name": "…"}}``.

        Returns:
            An :class:`~identark.models.LLMResponse` containing the
            assistant message, cost, finish reason, and token usage.

        Raises:
            CostCapExceededError: If the session cost cap has been reached.
            RateLimitError:       If the provider rate-limits the request.
            LLMError:             For any other provider-level error.
            NetworkError:         If all retry attempts to the control
                                  plane are exhausted.
        """
        ...

    async def persist_messages(self, messages: list[Message]) -> None:
        """
        Persist messages to conversation history without invoking the LLM.

        Use this to store tool call results, system context, or any
        messages you want the agent to remember on future turns without
        generating a new LLM response.

        Args:
            messages: Messages to persist. Can include any role.

        Raises:
            GatewayError: If persistence fails.
        """
        ...

    async def request_file_url(
        self,
        file_path: str,
        method: str = "PUT",
    ) -> PresignedURL:
        """
        Request a presigned URL for reading or writing a workspace file.

        The agent never holds cloud storage credentials. The gateway
        generates a time-limited, path-scoped URL on demand.

        Args:
            file_path: Absolute path to the file in the sandbox workspace.
                       Must start with ``/workspace/``.
            method:    ``'PUT'`` for upload, ``'GET'`` for download.

        Returns:
            A :class:`~identark.models.PresignedURL` with the URL,
            expiry timestamp, method, and resolved file path.

        Raises:
            PathNotAllowedError: If ``file_path`` is outside ``/workspace/``.
            FileError:           For any other file-related error.
        """
        ...

    async def get_session_cost(self) -> float:
        """
        Return the total cost in USD consumed by this session so far.

        Reflects all :meth:`invoke_llm` calls made through this gateway
        instance. With :class:`~identark.gateways.ControlPlaneGateway`,
        this queries the control plane for the authoritative total.

        Returns:
            Total accumulated cost in USD as a ``float``.
        """
        ...

    async def invoke_llm_stream(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a response from the LLM token by token.

        Yields :class:`~identark.models.StreamChunk` objects as they arrive.
        The final chunk has ``finish_reason`` set and ``input_tokens`` /
        ``output_tokens`` populated. All prior chunks have ``finish_reason=None``.

        Args:
            new_messages: New messages to send this turn.
            tools:        OpenAI-format tool/function definitions.
            tool_choice:  Tool selection mode.

        Yields:
            :class:`~identark.models.StreamChunk` — one per token delta.

        Raises:
            CostCapExceededError: If the session cost cap has been reached.
            RateLimitError:       If the provider rate-limits the request.
            ContentPolicyError:   If the output is blocked by content filtering.
        """
        ...
