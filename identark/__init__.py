"""
identark
~~~~~~~~~~~~~~
The AgentGateway Protocol — secure, scalable agent execution infrastructure.

Quick start::

    # Local development
    from openai import AsyncOpenAI
    from identark import DirectGateway, Message, Role

    gateway = DirectGateway(
        llm_client=AsyncOpenAI(),
        model="gpt-4o",
    )
    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello!")]
    )

    # Production — two line change, agent code identical
    from identark import ControlPlaneGateway
    gateway = ControlPlaneGateway()  # auto-detects env vars in sandbox

Full documentation: https://github.com/identark/identark#readme
GitHub: https://github.com/identark/identark
"""

from identark.gateway import AgentGateway
from identark.gateways.control_plane import ControlPlaneGateway
from identark.gateways.direct import DirectGateway
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

__version__ = "1.2.0"
__author__ = "Gold Okpa"
__license__ = "AGPL-3.0-only"

__all__ = [
    # Protocol
    "AgentGateway",
    # Implementations
    "DirectGateway",
    "ControlPlaneGateway",
    # Models
    "Message",
    "Role",
    "LLMResponse",
    "StreamChunk",
    "PresignedURL",
    "TokenUsage",
    "ToolCall",
    "Function",
    # Meta
    "__version__",
]
