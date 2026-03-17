"""
sandcastle.gateways.direct
~~~~~~~~~~~~~~~~~~~~~~~~~~
DirectGateway — local development implementation of AgentGateway.

Calls LLM providers directly using your own API keys, keeps
conversation history in memory, and resolves file paths to the
local filesystem. No Sandcastle account or control plane required.

Supports OpenAI, Anthropic, Mistral (EU), and any OpenAI-compatible
endpoint including Ollama for fully local, zero-egress inference.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from sandcastle.exceptions import (
    ConfigurationError,
    ContentPolicyError,
    CostCapExceededError,
    PathNotAllowedError,
    ProviderError,
    RateLimitError,
)
from sandcastle.models import (
    Function,
    LLMResponse,
    Message,
    PresignedURL,
    Role,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger("sandcastle.direct")

# Cost per 1M tokens (USD) — approximate, update as providers change pricing
_OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":       {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
}

_ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-20241022":  {"input": 0.80,  "output": 4.00},
    "claude-3-opus-20240229":     {"input": 15.00, "output": 75.00},
}

# Mistral AI — EU/French provider (mistral.ai). Prices in USD per 1M tokens.
_MISTRAL_PRICING: dict[str, dict[str, float]] = {
    "mistral-large-latest":  {"input": 2.00, "output": 6.00},
    "mistral-small-latest":  {"input": 0.20, "output": 0.60},
    "open-mistral-nemo":     {"input": 0.15, "output": 0.15},
    "codestral-latest":      {"input": 0.20, "output": 0.60},
}


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "openai",
) -> float:
    """Estimate cost in USD for a given model and token counts.

    Returns 0.0 for local providers (Ollama, self-hosted models).
    """
    if provider == "local":
        return 0.0
    pricing = {**_OPENAI_PRICING, **_ANTHROPIC_PRICING, **_MISTRAL_PRICING}
    if model not in pricing:
        # Unknown model — use a conservative estimate
        return (input_tokens + output_tokens) * 0.000_010
    rates = pricing[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


class DirectGateway:
    """
    Local development implementation of :class:`~sandcastle.gateway.AgentGateway`.

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
                       :exc:`~sandcastle.exceptions.CostCapExceededError`
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
        from sandcastle import DirectGateway

        gateway = DirectGateway(
            llm_client=AsyncOpenAI(),
            model="gpt-4o",
            system_prompt="You are a helpful assistant.",
        )

    Ollama (fully local, zero cost, zero data egress)::

        from openai import AsyncOpenAI
        from sandcastle import DirectGateway

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
        from sandcastle import DirectGateway

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
            raise ProviderError(f"Anthropic API error: {exc}") from exc

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
        """Re-raise an OpenAI-compatible SDK exception as a Sandcastle exception."""
        exc_type = type(exc).__name__
        if "RateLimitError" in exc_type:
            raise RateLimitError(str(exc), provider=self._provider) from exc
        if "ContentFilter" in exc_type or "content_filter" in str(exc).lower():
            raise ContentPolicyError(str(exc)) from exc
        raise ProviderError(f"{self._provider.capitalize()} API error: {exc}") from exc
