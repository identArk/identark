"""
credseal.models
~~~~~~~~~~~~~~~~~
Core data types used throughout the SDK.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class Role(StrEnum):
    """Message role in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class Message:
    """
    A single message in a conversation.

    Args:
        role:        Who authored this message.
        content:     Text content, or a list of content blocks for
                     multimodal / structured tool-call messages.
        tool_call_id: Required when role is Role.TOOL. Must match the
                      ``id`` of the tool call in the preceding assistant message.
        name:        Optional display name. Useful in multi-agent systems.
        tokens:      Token count. Populated automatically by the gateway
                     after ``invoke_llm`` calls.
    """

    role: Role
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None
    name: str | None = None
    tokens: int = 0

    def to_openai_dict(self) -> dict[str, Any]:
        """Serialise to the OpenAI messages API format."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class Function:
    """The function portion of a tool call."""

    name: str
    arguments: str  # JSON-encoded string


@dataclass
class ToolCall:
    """A tool/function call requested by the assistant."""

    id: str
    function: Function
    type: str = "function"


@dataclass
class TokenUsage:
    """Token consumption for a single LLM call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int = 0


@dataclass
class LLMResponse:
    """
    The result of an ``invoke_llm`` call.

    Attributes:
        message:       The assistant's response message.
        cost_usd:      Cost of this specific call in USD.
        model:         The model that generated the response.
        finish_reason: ``'stop'``, ``'tool_calls'``, ``'length'``,
                       or ``'content_filter'``.
        tool_calls:    Populated when ``finish_reason == 'tool_calls'``.
        usage:         Token usage breakdown.
    """

    message: Message
    cost_usd: float
    model: str
    finish_reason: str
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage = field(
        default_factory=lambda: TokenUsage(
            input_tokens=0, output_tokens=0, total_tokens=0
        )
    )


@dataclass
class PresignedURL:
    """
    A time-limited, scoped URL for reading or writing a workspace file.

    Attributes:
        url:        The presigned URL. Use immediately — it is short-lived.
        expires_at: ISO 8601 expiry timestamp.
        method:     ``'PUT'`` for upload, ``'GET'`` for download.
        file_path:  The ``/workspace/`` path this URL corresponds to.
    """

    url: str
    expires_at: str
    method: str
    file_path: str
