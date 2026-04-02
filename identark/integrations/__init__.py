"""
identark.integrations
~~~~~~~~~~~~~~~~~~~~~~~
Framework adapters for LangChain, LlamaIndex, Gemini, and others.

These are optional — install the relevant extras to use them:

    pip install identark-sdk[langchain]
    pip install identark-sdk[llamaindex]
    pip install identark-sdk[gemini]
    pip install identark-sdk crewai
"""

# Lazy imports to avoid requiring all dependencies
__all__ = [
    "GeminiGateway",
    "IdentArkGeminiGateway",
]


def __getattr__(name: str) -> type:
    if name in ("GeminiGateway", "IdentArkGeminiGateway"):
        from identark.integrations.gemini import GeminiGateway
        return GeminiGateway
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
