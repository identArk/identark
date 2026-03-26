"""
credseal.integrations.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Google Gemini integration — CredSealGeminiGateway.

Native integration with Google's Gemini API using the google-generativeai SDK.
Supports multimodal inputs, function calling, and proper cost tracking.

Install::

    pip install credseal-sdk[gemini]

Usage::

    from credseal.integrations.gemini import GeminiGateway
    from credseal import Message, Role

    gateway = GeminiGateway(
        api_key="your-gemini-api-key",
        model="gemini-1.5-pro",
    )

    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello, Gemini!")]
    )
    print(response.message.content)
    print(f"Cost: ${response.cost_usd:.6f}")
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credseal.exceptions import (
    ConfigurationError,
    ContentPolicyError,
    CostCapExceededError,
    PathNotAllowedError,
    ProviderError,
    RateLimitError,
)
from credseal.models import (
    Function,
    LLMResponse,
    Message,
    PresignedURL,
    Role,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger("credseal.integrations.gemini")

# Gemini pricing per 1M tokens (USD) — as of 2024
# See: https://ai.google.dev/pricing
_GEMINI_PRICING: dict[str, dict[str, float]] = {
    # Gemini 1.5 Pro
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro-002": {"input": 1.25, "output": 5.00},
    # Gemini 1.5 Flash
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
    # Gemini 1.5 Flash-8B (cheapest)
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-flash-8b-latest": {"input": 0.0375, "output": 0.15},
    # Gemini 2.0 Flash (experimental)
    "gemini-2.0-flash-exp": {"input": 0.10, "output": 0.40},
    # Gemini 1.0 Pro (legacy)
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    "gemini-pro": {"input": 0.50, "output": 1.50},
}


def _estimate_gemini_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost in USD for a Gemini model."""
    # Normalize model name
    model_key = model.lower()
    if model_key not in _GEMINI_PRICING:
        # Try prefix matching
        for key in _GEMINI_PRICING:
            if model_key.startswith(key.replace("-latest", "").replace("-002", "")):
                model_key = key
                break
        else:
            # Unknown model — use flash pricing as conservative estimate
            logger.warning("Unknown Gemini model %s, using flash pricing", model)
            model_key = "gemini-1.5-flash"

    rates = _GEMINI_PRICING[model_key]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def _convert_role_to_gemini(role: Role) -> str:
    """Convert CredSeal role to Gemini role."""
    if role == Role.USER:
        return "user"
    elif role == Role.ASSISTANT:
        return "model"
    elif role == Role.SYSTEM:
        return "user"  # Gemini handles system via system_instruction
    elif role == Role.TOOL:
        return "function"
    return "user"


def _convert_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tools to Gemini function declarations."""
    gemini_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            gemini_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            })
    return gemini_tools


class GeminiGateway:
    """
    Native Google Gemini implementation of :class:`~credseal.gateway.AgentGateway`.

    Uses the google-generativeai SDK directly for optimal performance and
    access to Gemini-specific features like multimodal inputs and grounding.

    Args:
        api_key:        Your Google AI API key (from https://aistudio.google.com).
        model:          Gemini model identifier e.g. ``'gemini-1.5-pro'``,
                        ``'gemini-1.5-flash'``, ``'gemini-2.0-flash-exp'``.
        system_prompt:  Optional system instruction prepended to every conversation.
        cost_cap_usd:   Optional soft cost cap. Raises
                        :exc:`~credseal.exceptions.CostCapExceededError`
                        when exceeded.
        workspace_dir:  Local directory for file operations.
                        Defaults to ``'/workspace'``.
        safety_settings: Optional Gemini safety settings dict.
        generation_config: Optional Gemini generation config dict.

    Example::

        from credseal.integrations.gemini import GeminiGateway
        from credseal import Message, Role

        gateway = GeminiGateway(
            api_key="your-api-key",
            model="gemini-1.5-pro",
            system_prompt="You are a helpful financial assistant.",
        )

        response = await gateway.invoke_llm(
            new_messages=[Message(role=Role.USER, content="Analyze my portfolio")]
        )

    With tool calling::

        tools = [{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock symbol"}
                    },
                    "required": ["symbol"]
                }
            }
        }]

        response = await gateway.invoke_llm(
            new_messages=[Message(role=Role.USER, content="What's AAPL trading at?")],
            tools=tools,
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        system_prompt: str | None = None,
        cost_cap_usd: float | None = None,
        workspace_dir: str = "/workspace",
        safety_settings: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError("api_key must be provided for GeminiGateway.")
        if not model:
            raise ConfigurationError("model must be a non-empty string.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ConfigurationError(
                "google-generativeai package not installed. "
                "Run: pip install credseal-sdk[gemini]"
            ) from exc

        self._api_key = api_key
        self._model_name = model
        self._system_prompt = system_prompt
        self._cost_cap = cost_cap_usd
        self._workspace = Path(workspace_dir)
        self._safety_settings = safety_settings
        self._generation_config = generation_config or {}
        self._history: list[Message] = []
        self._total_cost: float = 0.0

        # Configure the SDK
        genai.configure(api_key=api_key)
        self._genai = genai

        # Create model instance
        model_kwargs: dict[str, Any] = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt
        if safety_settings:
            model_kwargs["safety_settings"] = safety_settings
        if generation_config:
            model_kwargs["generation_config"] = generation_config

        self._model = genai.GenerativeModel(model, **model_kwargs)

        logger.debug(
            "GeminiGateway initialised model=%s workspace=%s",
            self._model_name,
            self._workspace,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def invoke_llm(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Send new messages to Gemini and receive a response."""
        self._check_cost_cap()

        # Build conversation history
        gemini_history = self._build_gemini_history()
        gemini_content = self._messages_to_gemini_content(new_messages)

        logger.debug(
            "invoke_llm model=%s history_len=%d new_messages=%d tools=%s",
            self._model_name,
            len(self._history),
            len(new_messages),
            len(tools) if tools else 0,
        )

        try:
            # Create a chat session with history
            chat = self._model.start_chat(history=gemini_history)

            # Prepare generation kwargs
            gen_kwargs: dict[str, Any] = {}
            if tools:
                gemini_tools = _convert_tools_to_gemini(tools)
                gen_kwargs["tools"] = [{"function_declarations": gemini_tools}]

            # Send message
            response = await chat.send_message_async(gemini_content, **gen_kwargs)

        except Exception as exc:
            self._classify_gemini_error(exc)

        # Parse response
        result = self._parse_gemini_response(response)

        # Accumulate cost
        self._total_cost += result.cost_usd

        # Persist new messages + assistant response to history
        self._history.extend(new_messages)
        self._history.append(result.message)

        logger.debug(
            "invoke_llm complete cost_usd=%.6f total_cost=%.6f finish=%s",
            result.cost_usd,
            self._total_cost,
            result.finish_reason,
        )

        return result

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
        logger.debug("GeminiGateway reset")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def history(self) -> list[Message]:
        """Read-only view of the conversation history."""
        return list(self._history)

    @property
    def model(self) -> str:
        """The model identifier this gateway is configured to use."""
        return self._model_name

    @property
    def provider(self) -> str:
        """The provider string."""
        return "google"

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_cost_cap(self) -> None:
        if self._cost_cap is not None and self._total_cost >= self._cost_cap:
            raise CostCapExceededError(
                f"GeminiGateway cost cap of ${self._cost_cap:.4f} reached. "
                f"Accumulated: ${self._total_cost:.4f}. Call gateway.reset() to start fresh.",
                cap_usd=self._cost_cap,
                consumed_usd=self._total_cost,
            )

    def _resolve_workspace_path(self, file_path: str) -> Path:
        """Validate and resolve a file path to the local workspace."""
        if not file_path.startswith("/workspace/"):
            raise PathNotAllowedError(file_path)
        relative = file_path.removeprefix("/workspace/")
        return self._workspace / relative

    def _build_gemini_history(self) -> list[dict[str, Any]]:
        """Convert internal history to Gemini format."""
        history = []
        for msg in self._history:
            role = _convert_role_to_gemini(msg.role)
            if role == "function":
                # Tool result
                history.append({
                    "role": "function",
                    "parts": [{"function_response": {
                        "name": msg.tool_call_id or "tool",
                        "response": {"result": msg.content}
                    }}]
                })
            else:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history.append({
                    "role": role,
                    "parts": [{"text": content}]
                })
        return history

    def _messages_to_gemini_content(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert new messages to Gemini content format."""
        parts = []
        for msg in messages:
            if msg.role == Role.TOOL:
                # Tool result
                parts.append({
                    "function_response": {
                        "name": msg.tool_call_id or "tool",
                        "response": {"result": msg.content}
                    }
                })
            elif isinstance(msg.content, str):
                parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                # Multimodal content
                for block in msg.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})
                        elif block.get("type") == "image_url":
                            # Handle image URL
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Base64 image
                                header, data = url.split(",", 1)
                                mime_type = header.split(";")[0].split(":")[1]
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": data
                                    }
                                })
                            else:
                                parts.append({"text": f"[Image: {url}]"})
                    else:
                        parts.append({"text": str(block)})
            else:
                parts.append({"text": str(msg.content)})
        return parts

    def _parse_gemini_response(self, response: Any) -> LLMResponse:
        """Parse a Gemini response into LLMResponse."""
        candidate = response.candidates[0]

        # Extract text content
        content = ""
        tool_calls = None

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                content += part.text
            elif hasattr(part, "function_call"):
                fc = part.function_call
                if tool_calls is None:
                    tool_calls = []
                # Generate a unique ID for the tool call
                tool_id = f"call_{len(tool_calls)}_{fc.name}"
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        function=Function(
                            name=fc.name,
                            arguments=json.dumps(dict(fc.args)),
                        ),
                    )
                )

        # Get token counts from usage metadata
        usage_meta = response.usage_metadata
        input_tokens = getattr(usage_meta, "prompt_token_count", 0)
        output_tokens = getattr(usage_meta, "candidates_token_count", 0)
        total_tokens = getattr(usage_meta, "total_token_count", input_tokens + output_tokens)

        # Calculate cost
        cost = _estimate_gemini_cost(self._model_name, input_tokens, output_tokens)

        # Determine finish reason
        finish_reason_map = {
            1: "stop",           # STOP
            2: "length",         # MAX_TOKENS
            3: "tool_calls",     # TOOL
            4: "content_filter", # SAFETY
            5: "recitation",     # RECITATION
            0: "stop",           # FINISH_REASON_UNSPECIFIED
        }
        finish_reason = finish_reason_map.get(
            candidate.finish_reason,
            "stop" if tool_calls else "stop"
        )
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        return LLMResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=content,
                tokens=output_tokens,
            ),
            cost_usd=cost,
            model=self._model_name,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
        )

    def _classify_gemini_error(self, exc: Exception) -> None:
        """Re-raise a Gemini SDK exception as a CredSeal exception."""
        exc_str = str(exc).lower()

        if "quota" in exc_str or "rate" in exc_str or "429" in exc_str:
            raise RateLimitError(str(exc), provider="google") from exc
        if (
            "safety" in exc_str
            or "blocked" in exc_str
            or "harm" in exc_str
            or "content" in exc_str
        ):
            raise ContentPolicyError(str(exc)) from exc
        if "invalid" in exc_str and "api" in exc_str:
            raise ConfigurationError(f"Invalid Gemini API key: {exc}") from exc

        raise ProviderError(f"Gemini API error: {exc}") from exc

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def invoke_llm_stream(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream the LLM response token by token.

        Yields :class:`~credseal.models.StreamChunk` objects as they arrive.
        The final chunk has ``finish_reason`` set and token counts populated.
        """
        self._check_cost_cap()

        gemini_history = self._build_gemini_history()
        gemini_content = self._messages_to_gemini_content(new_messages)

        try:
            chat = self._model.start_chat(history=gemini_history)

            gen_kwargs: dict[str, Any] = {}
            if tools:
                gemini_tools = _convert_tools_to_gemini(tools)
                gen_kwargs["tools"] = [{"function_declarations": gemini_tools}]

            response = await chat.send_message_async(
                gemini_content,
                stream=True,
                **gen_kwargs,
            )

            full_content = ""
            input_tokens = 0
            output_tokens = 0

            async for chunk in response:
                # Extract text from chunk
                for part in chunk.parts:
                    if hasattr(part, "text") and part.text:
                        full_content += part.text
                        yield StreamChunk(
                            content=part.text,
                            finish_reason=None,
                            model=self._model_name,
                        )

                # Check for usage metadata on final chunk
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    input_tokens = getattr(chunk.usage_metadata, "prompt_token_count", 0)
                    output_tokens = getattr(chunk.usage_metadata, "candidates_token_count", 0)

            # Emit final chunk with usage info
            cost = _estimate_gemini_cost(self._model_name, input_tokens, output_tokens)
            self._total_cost += cost

            # Persist to history
            self._history.extend(new_messages)
            self._history.append(Message(role=Role.ASSISTANT, content=full_content))

            yield StreamChunk(
                content="",
                finish_reason="stop",
                model=self._model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as exc:
            self._classify_gemini_error(exc)


# Convenience alias
CredSealGeminiGateway = GeminiGateway
