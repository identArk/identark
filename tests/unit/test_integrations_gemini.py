"""
Tests for the Gemini integration.

These tests use mocking to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from identark.models import Message, Role


@pytest.fixture
def mock_genai():
    """Mock the google.generativeai module."""
    with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
        import google.generativeai as genai

        # Mock the configure function
        genai.configure = MagicMock()

        # Create mock model
        mock_model = MagicMock()
        mock_chat = MagicMock()

        # Mock response
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Hello! How can I help you?"
        mock_part.function_call = None
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = 1  # STOP
        mock_response.candidates = [mock_candidate]

        # Mock usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 15
        mock_usage.total_token_count = 25
        mock_response.usage_metadata = mock_usage

        # Setup async mock
        mock_chat.send_message_async = AsyncMock(return_value=mock_response)
        mock_model.start_chat = MagicMock(return_value=mock_chat)

        genai.GenerativeModel = MagicMock(return_value=mock_model)

        yield genai


@pytest.fixture
def gemini_gateway(mock_genai, tmp_path):
    """Create a GeminiGateway with mocked SDK."""
    from identark.integrations.gemini import GeminiGateway

    return GeminiGateway(
        api_key="test-api-key",
        model="gemini-1.5-flash",
        system_prompt="You are a helpful assistant.",
        workspace_dir=str(tmp_path),
    )


async def test_basic_invoke(gemini_gateway):
    """Test basic invoke_llm call."""
    response = await gemini_gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello!")]
    )

    assert response.message.role == Role.ASSISTANT
    assert response.message.content == "Hello! How can I help you?"
    assert response.model == "gemini-1.5-flash"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 15


async def test_history_persistence(gemini_gateway):
    """Test that history is maintained across calls."""
    await gemini_gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="First message")]
    )

    assert len(gemini_gateway.history) == 2  # User + Assistant

    await gemini_gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Second message")]
    )

    assert len(gemini_gateway.history) == 4  # 2 more messages


async def test_cost_tracking(gemini_gateway):
    """Test cost accumulation."""
    initial_cost = await gemini_gateway.get_session_cost()
    assert initial_cost == 0.0

    await gemini_gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello")]
    )

    cost_after = await gemini_gateway.get_session_cost()
    assert cost_after > 0  # Should have some cost


async def test_reset(gemini_gateway):
    """Test gateway reset."""
    await gemini_gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello")]
    )

    assert len(gemini_gateway.history) > 0
    assert await gemini_gateway.get_session_cost() > 0

    gemini_gateway.reset()

    assert len(gemini_gateway.history) == 0
    assert await gemini_gateway.get_session_cost() == 0


async def test_persist_messages(gemini_gateway):
    """Test persist_messages without LLM call."""
    await gemini_gateway.persist_messages([
        Message(role=Role.USER, content="Test"),
        Message(role=Role.ASSISTANT, content="Response"),
    ])

    assert len(gemini_gateway.history) == 2


def test_provider_property(gemini_gateway):
    """Test provider property returns 'google'."""
    assert gemini_gateway.provider == "google"


def test_model_property(gemini_gateway):
    """Test model property returns configured model."""
    assert gemini_gateway.model == "gemini-1.5-flash"


async def test_request_file_url(gemini_gateway):
    """Test file URL generation."""
    url = await gemini_gateway.request_file_url("/workspace/test.txt", "PUT")

    assert url.file_path == "/workspace/test.txt"
    assert url.method == "PUT"
    assert url.url.startswith("file://")


async def test_invalid_file_path(gemini_gateway):
    """Test rejection of paths outside workspace."""
    from identark.exceptions import PathNotAllowedError

    with pytest.raises(PathNotAllowedError):
        await gemini_gateway.request_file_url("/etc/passwd", "GET")


def test_cost_estimation():
    """Test cost estimation for various models."""
    from identark.integrations.gemini import _estimate_gemini_cost

    # Flash model (cheap)
    flash_cost = _estimate_gemini_cost("gemini-1.5-flash", 1000, 1000)
    assert flash_cost > 0
    assert flash_cost < 0.001  # Very cheap

    # Pro model (more expensive)
    pro_cost = _estimate_gemini_cost("gemini-1.5-pro", 1000, 1000)
    assert pro_cost > flash_cost


def test_missing_api_key():
    """Test error when API key is missing."""
    from identark.exceptions import ConfigurationError

    with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
        from identark.integrations.gemini import GeminiGateway

        with pytest.raises(ConfigurationError):
            GeminiGateway(api_key="", model="gemini-1.5-flash")


def test_missing_model():
    """Test error when model is missing."""
    from identark.exceptions import ConfigurationError

    with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
        from identark.integrations.gemini import GeminiGateway

        with pytest.raises(ConfigurationError):
            GeminiGateway(api_key="test-key", model="")


async def test_cost_cap_enforcement(mock_genai):
    """Test cost cap raises exception."""
    from identark.exceptions import CostCapExceededError
    from identark.integrations.gemini import GeminiGateway

    gateway = GeminiGateway(
        api_key="test-key",
        model="gemini-1.5-flash",
        cost_cap_usd=0.0001,  # Very low cap
    )

    # First call should work
    await gateway.invoke_llm(
        new_messages=[Message(role=Role.USER, content="Hello")]
    )

    # Force cost to exceed cap
    gateway._total_cost = 0.001

    # Second call should raise
    with pytest.raises(CostCapExceededError):
        await gateway.invoke_llm(
            new_messages=[Message(role=Role.USER, content="Hello again")]
        )


def test_tool_conversion():
    """Test OpenAI tool format to Gemini conversion."""
    from identark.integrations.gemini import _convert_tools_to_gemini

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    gemini_tools = _convert_tools_to_gemini(openai_tools)

    assert len(gemini_tools) == 1
    assert gemini_tools[0]["name"] == "get_weather"
    assert gemini_tools[0]["description"] == "Get current weather"
    assert "location" in gemini_tools[0]["parameters"]["properties"]
