"""
sandcastle-sdk
~~~~~~~~~~~~~~
The AgentGateway Protocol — secure, scalable agent execution infrastructure.

Quick start::

    # Local development
    from openai import AsyncOpenAI
    from sandcastle import DirectGateway, Message, Role

    gateway = DirectGateway(
        llm_client=AsyncOpenAI(),
        model="gpt-4o",
    )
    response = await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello!")]
    )

    # Production — two line change, agent code identical
    from sandcastle import ControlPlaneGateway
    gateway = ControlPlaneGateway()  # auto-detects env vars in sandbox

Full documentation: https://github.com/Goldokpa/Sandcastle#readme
GitHub: https://github.com/Goldokpa/Sandcastle
"""

from sandcastle.gateway import AgentGateway
from sandcastle.gateways.control_plane import ControlPlaneGateway
from sandcastle.gateways.direct import DirectGateway
from sandcastle.models import (
    Function,
    LLMResponse,
    Message,
    PresignedURL,
    Role,
    TokenUsage,
    ToolCall,
)

__version__ = "1.0.0"
__author__ = "Gold Okpa"
__license__ = "MIT"

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
    "PresignedURL",
    "TokenUsage",
    "ToolCall",
    "Function",
    # Meta
    "__version__",
]
