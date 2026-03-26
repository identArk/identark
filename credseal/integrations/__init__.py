"""
credseal.integrations
~~~~~~~~~~~~~~~~~~~~~~~
Framework adapters for LangChain, LlamaIndex, Gemini, and others.

These are optional — install the relevant extras to use them:

    pip install credseal-sdk[langchain]
    pip install credseal-sdk[llamaindex]
    pip install credseal-sdk[gemini]
    pip install credseal-sdk crewai
"""

# Lazy imports to avoid requiring all dependencies
__all__ = [
    "GeminiGateway",
    "CredSealGeminiGateway",
]


def __getattr__(name: str) -> type:
    if name in ("GeminiGateway", "CredSealGeminiGateway"):
        from credseal.integrations.gemini import GeminiGateway
        return GeminiGateway
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
