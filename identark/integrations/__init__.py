"""
identark.integrations
~~~~~~~~~~~~~~~~~~~~~~~
Framework adapters for LangChain, LlamaIndex, Gemini, and others.

These are optional — install the relevant extras to use them:

    pip install identark[langchain]
    pip install identark[llamaindex]
    pip install identark[gemini]
    pip install identark crewai
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
