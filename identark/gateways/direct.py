"""
identark.gateways.direct
~~~~~~~~~~~~~~~~~~~~~~~~~~
DirectGateway — local development implementation of AgentGateway.

Calls LLM providers directly using your own API keys, keeps
conversation history in memory, and resolves file paths to the
local filesystem. No IdentArk account or control plane required.

Supports OpenAI, Anthropic, Mistral (EU), and any OpenAI-compatible
endpoint including Ollama for fully local, zero-egress inference.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from identark.exceptions import (
    ConfigurationError,
    ContentPolicyError,
    CostCapExceededError,
    PathNotAllowedError,
    ProviderError,
    RateLimitError,
)
from identark.models import (
    Function,
    LLMResponse,
    Message,
    PresignedURL,
    Role,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from identark.pricing import estimate_cost as _estimate_cost
from identark.validation import validate_tool_definitions

logger = logging.getLogger("identark.direct")


class DirectGateway:
    """
    Local development implementation of :class:`~identark.gateway.AgentGateway`.

    Calls LLM providers directly. Keeps conversation history in memory.
    Resolves ``/workspace/`` file paths to the local filesystem.

    Supports OpenAI, Anthropic, Mistral (EU), and any OpenAI-compatible
    endpoint. Use ``provider='local'`` with Ollama for fully on-machine
    inference with zero data egress and zero cost.

    Args:
        llm_client:    An initialised async LLM client (``AsyncOpenAI``,
                       ``AsyncAnthropic``, or any OpenAI-compatible client).
        model:         Model identifier e.g. ``'gpt-4o'``, ``'mistral-large-latest'``,
                       ``'llama3.2'``.
        system_prompt: Optional system prompt prepended to every conversation.
        cost_cap_usd:  Optional soft cost cap. Raises
                       :exc:`~identark.exceptions.CostCapExceededError`
                       when exceeded. Not enforced server-side.
        workspace_dir: Local directory for file operations.
                       Defaults to ``'/workspace'``.
        provider:      Optional explicit provider override. Recognised values:
                       ``'openai'``, ``'anthropic'``, ``'mistral'``, ``'local'``.
                       If omitted, auto-detected from the client class and
                       ``base_url``. Set to ``'local'`` to force £0/$0 cost
                       tracking (e.g. when using Ollama on your own hardware).

    OpenAI example::

        from openai import AsyncOpenAI
        from identark import DirectGateway

        gateway = DirectGateway(
            llm_client=AsyncOpenAI(),
            model="gpt-4o",
            system_prompt="You are a helpful assistant.",
        )

    Ollama (fully local, zero cost, zero data egress)::

        from openai import AsyncOpenAI
        from identark import DirectGateway

        gateway = DirectGateway(
            llm_client=AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",   # required by openai SDK, ignored by Ollama
            ),
            model="llama3.2",
            provider="local",       # forces £0/$0 cost tracking
        )

    Mistral (EU/French provider, data stays in the EU)::

        from openai import AsyncOpenAI
        from identark import DirectGateway

        gateway = DirectGateway(
            llm_client=AsyncOpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key="your-mistral-api-key",
            ),
            model="mistral-large-latest",
        )
    """

    def __init__(
        self,
        llm_client: Any,
        model: str,
        system_prompt: str | None = None,
        cost_cap_usd: float | None = None,
        workspace_dir: str = "/workspace",
        provider: str | None = None,
    ) -> None:
        if llm_client is None:
            raise ConfigurationError("llm_client must not be None.")
        if not model:
            raise ConfigurationError("model must be a non-empty string.")

        self._client = llm_client
        self._model = model
        self._system_prompt = system_prompt
        self._cost_cap = cost_cap_usd
        self._workspace = Path(workspace_dir)
        self._history: list[Message] = []
        self._total_cost: float = 0.0

        # Determine provider — explicit override wins, then class name, then base_url
        if provider is not None:
            self._provider = provider
        else:
            client_cls = type(llm_client).__name__
            if "Anthropic" in client_cls:
                self._provider = "anthropic"
            elif "Mistral" in client_cls:
                self._provider = "mistral"
            elif hasattr(llm_client, "base_url"):
                base = str(getattr(llm_client, "base_url", ""))
                if "localhost" in base or "127.0.0.1" in base or "::1" in base:
                    self._provider = "local"
                elif "mistral.ai" in base:
                    self._provider = "mistral"
                else:
                    self._provider = "openai"
            else:
                self._provider = "openai"

        logger.debug(
            "DirectGateway initialised provider=%s model=%s workspace=%s",
            self._provider,
            self._model,
            self._workspace,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def invoke_llm(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Send new messages to the LLM and receive a response."""
        self._check_cost_cap()
        validate_tool_definitions(tools)

        # Build full message list: system + history + new
        messages = self._build_messages(new_messages)

        logger.debug(
            "invoke_llm model=%s provider=%s history_len=%d new_messages=%d",
            self._model,
            self._provider,
            len(self._history),
            len(new_messages),
        )

        if self._provider == "anthropic":
            response = await self._call_anthropic(messages, tools, tool_choice)
        else:
            # openai, mistral, local, and any OpenAI-compatible endpoint
            response = await self._call_openai(messages, tools, tool_choice)

        # Accumulate cost
        self._total_cost += response.cost_usd

        # Persist new messages + assistant response to history
        self._history.extend(new_messages)
        self._history.append(response.message)

        logger.debug(
            "invoke_llm complete cost_usd=%.6f total_cost=%.6f finish=%s",
            response.cost_usd,
            self._total_cost,
            response.finish_reason,
        )

        return response

    async def persist_messages(self, messages: list[Message]) -> None:
        """Persist messages to conversation history without calling the LLM."""
        self._history.extend(messages)
        logger.debug("persist_messages count=%d", len(messages))

    async def request_file_url(
        self,
        file_path: str,
        method: str = "PUT",
    ) -> PresignedURL:
        """Return a local file:// URL for the given workspace path."""
        resolved = self._resolve_workspace_path(file_path)
        if method == "PUT":
            resolved.parent.mkdir(parents=True, exist_ok=True)

        expiry = datetime.now(timezone.utc).replace(hour=23, minute=59).isoformat()
        url = resolved.as_uri()

        logger.debug("request_file_url path=%s method=%s url=%s", file_path, method, url)
        return PresignedURL(url=url, expires_at=expiry, method=method, file_path=file_path)

    async def get_session_cost(self) -> float:
        """Return total accumulated cost in USD for this gateway instance."""
        return self._total_cost

    def reset(self) -> None:
        """Clear conversation history and reset cost counter."""
        self._history.clear()
        self._total_cost = 0.0
        logger.debug("DirectGateway reset")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def history(self) -> list[Message]:
        """Read-only view of the conversation history."""
        return list(self._history)

    @property
    def model(self) -> str:
        """The model identifier this gateway is configured to use."""
        return self._model

    @property
    def provider(self) -> str:
        """The resolved provider string (e.g. 'openai', 'mistral', 'local')."""
        return self._provider

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_cost_cap(self) -> None:
        if self._cost_cap is not None and self._total_cost >= self._cost_cap:
            raise CostCapExceededError(
                f"DirectGateway cost cap of ${self._cost_cap:.4f} reached. "
                f"Accumulated: ${self._total_cost:.4f}. Call gateway.reset() to start fresh.",
                cap_usd=self._cost_cap,
                consumed_usd=self._total_cost,
            )

    def _build_messages(self, new_messages: list[Message]) -> list[dict[str, Any]]:
        """Assemble full message list in OpenAI dict format."""
        msgs: list[dict[str, Any]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(m.to_openai_dict() for m in self._history)
        msgs.extend(m.to_openai_dict() for m in new_messages)
        return msgs

    def _resolve_workspace_path(self, file_path: str) -> Path:
        """Validate and resolve a file path to the local workspace."""
        if not file_path.startswith("/workspace/"):
            raise PathNotAllowedError(file_path)
        relative = file_path.removeprefix("/workspace/")
        return self._workspace / relative

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
    ) -> LLMResponse:
        """Make an OpenAI-compatible chat completion call.

        Used for OpenAI, Mistral, Ollama, and any OpenAI-compatible endpoint.
        """
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            self._classify_openai_error(exc)

        choice = completion.choices[0]
        raw_msg = choice.message
        usage = completion.usage

        # Parse tool calls if present
        tool_calls = None
        if raw_msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in raw_msg.tool_calls
            ]

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = _estimate_cost(self._model, input_tokens, output_tokens, self._provider)

        return LLMResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=raw_msg.content or "",
                tokens=output_tokens,
            ),
            cost_usd=cost,
            model=self._model,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=getattr(usage, "prompt_tokens_details", None)
                and getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0,
            ),
        )

    async def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
    ) -> LLMResponse:
        """Make an Anthropic messages API call."""
        # Separate system message from the rest
        system = None
        filtered: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system
        if tools:
            # Convert OpenAI tool format to Anthropic format
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            self._classify_anthropic_error(exc)

        # Extract text content
        content = ""
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                import json
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function=Function(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = _estimate_cost(self._model, input_tokens, output_tokens, "anthropic")

        finish_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
        finish_reason = finish_map.get(response.stop_reason or "end_turn", "stop")

        return LLMResponse(
            message=Message(role=Role.ASSISTANT, content=content, tokens=output_tokens),
            cost_usd=cost,
            model=self._model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    def _classify_openai_error(self, exc: Exception) -> NoReturn:
        """Re-raise an OpenAI-compatible SDK exception as a IdentArk exception.

        Detection priority:
        1. Exception class hierarchy (most reliable)
        2. HTTP status code attribute (if present)
        3. Error code attribute (if present)
        4. String matching (fallback)
        """
        # Check exception class hierarchy first (most reliable)
        exc_class_name = type(exc).__name__
        exc_mro = [c.__name__ for c in type(exc).__mro__]

        # Rate limit: class name or 429 status
        if "RateLimitError" in exc_mro:
            retry_after: int | None = getattr(exc, "retry_after", None)
            raise RateLimitError(
                str(exc), provider=self._provider, retry_after_seconds=retry_after or 60
            ) from exc

        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            raise RateLimitError(str(exc), provider=self._provider) from exc

        # Content policy: class name, error code, or status 400 with specific message
        if "ContentFilterFinishReasonError" in exc_mro or "ContentFilter" in exc_class_name:
            raise ContentPolicyError(str(exc)) from exc

        error_code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
        if error_code in ("content_filter", "content_policy_violation", "output_blocked"):
            raise ContentPolicyError(str(exc)) from exc

        # String fallback for edge cases (provider SDK variations)
        exc_str = str(exc).lower()
        if any(
            kw in exc_str
            for kw in ("content_filter", "content filtering policy", "output blocked")
        ):
            raise ContentPolicyError(str(exc)) from exc

        # Authentication errors
        if "AuthenticationError" in exc_mro or status_code == 401:
            raise ProviderError(f"Authentication failed: {exc}") from exc

        # Default: generic provider error
        raise ProviderError(f"{self._provider.capitalize()} API error: {exc}") from exc

    def _classify_anthropic_error(self, exc: Exception) -> NoReturn:
        """Re-raise an Anthropic SDK exception as a IdentArk exception."""
        exc_mro = [c.__name__ for c in type(exc).__mro__]

        # Rate limit
        if "RateLimitError" in exc_mro:
            raise RateLimitError(str(exc), provider="anthropic") from exc

        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            raise RateLimitError(str(exc), provider="anthropic") from exc

        # Content policy (Anthropic uses "content_policy_violation" error type)
        error_type = getattr(exc, "type", None) or getattr(exc, "error_type", None)
        if error_type in ("content_policy_violation", "content_filter"):
            raise ContentPolicyError(str(exc)) from exc

        # String fallback
        exc_str = str(exc).lower()
        if any(
            kw in exc_str
            for kw in ("content filtering policy", "output blocked", "content policy")
        ):
            raise ContentPolicyError(str(exc)) from exc

        # Authentication
        if "AuthenticationError" in exc_mro or status_code == 401:
            raise ProviderError(f"Anthropic authentication failed: {exc}") from exc

        raise ProviderError(f"Anthropic API error: {exc}") from exc

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def invoke_llm_stream(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream the LLM response token by token.

        Yields :class:`~identark.models.StreamChunk` objects as they arrive.
        The final chunk has ``finish_reason`` set and token counts populated.
        """
        validate_tool_definitions(tools)
        if self._cost_cap is not None and self._total_cost >= self._cost_cap:
            raise CostCapExceededError(
                f"Cost cap of ${self._cost_cap:.4f} reached.",
                cap_usd=self._cost_cap,
                consumed_usd=self._total_cost,
            )

        messages = self._build_messages(new_messages)

        delegate = (
            self._stream_anthropic(messages, tools, tool_choice)
            if self._provider == "anthropic"
            else self._stream_openai(messages, tools, tool_choice)
        )
        async for chunk in delegate:
            yield chunk

    async def _stream_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream using the OpenAI-compatible API (OpenAI, Mistral, Ollama, etc.)."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            input_tokens = output_tokens = 0
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                # Usage is populated on the final chunk by some providers
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0

                if choice is None:
                    continue

                delta_content = choice.delta.content or ""
                finish_reason = choice.finish_reason

                if delta_content:
                    yield StreamChunk(
                        content=delta_content,
                        finish_reason=None,
                        model=self._model,
                    )

                if finish_reason:
                    cost = _estimate_cost(self._model, input_tokens, output_tokens, self._provider)
                    self._total_cost += cost
                    yield StreamChunk(
                        content="",
                        finish_reason=finish_reason,
                        model=self._model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
        except Exception as exc:
            self._classify_openai_error(exc)

    async def _stream_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream using the Anthropic messages streaming API."""
        system: str | None = None
        filtered = [m for m in messages if m["role"] != "system"]
        for m in messages:
            if m["role"] == "system":
                system = str(m["content"])

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {}),
                }
                for t in tools
            ]

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(content=text, finish_reason=None, model=self._model)

                final = await stream.get_final_message()
                input_tokens = final.usage.input_tokens
                output_tokens = final.usage.output_tokens
                cost = _estimate_cost(self._model, input_tokens, output_tokens, "anthropic")
                self._total_cost += cost

                finish_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}
                finish_reason = finish_map.get(final.stop_reason or "end_turn", "stop")
                yield StreamChunk(
                    content="",
                    finish_reason=finish_reason,
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
        except Exception as exc:
            self._classify_anthropic_error(exc)
