"""Unit tests for credseal.models."""

from credseal.models import (
    Function,
    LLMResponse,
    Message,
    PresignedURL,
    Role,
    TokenUsage,
    ToolCall,
)


class TestMessage:
    def test_basic_user_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tokens == 0
        assert msg.tool_call_id is None

    def test_to_openai_dict_basic(self):
        msg = Message(role=Role.USER, content="Hello")
        d = msg.to_openai_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_openai_dict_with_tool_call_id(self):
        msg = Message(role=Role.TOOL, content='{"result": 42}', tool_call_id="call_123")
        d = msg.to_openai_dict()
        assert d["tool_call_id"] == "call_123"
        assert d["role"] == "tool"

    def test_to_openai_dict_with_name(self):
        msg = Message(role=Role.USER, content="Hi", name="agent_1")
        d = msg.to_openai_dict()
        assert d["name"] == "agent_1"

    def test_to_openai_dict_excludes_none_fields(self):
        msg = Message(role=Role.USER, content="Hi")
        d = msg.to_openai_dict()
        assert "tool_call_id" not in d
        assert "name" not in d

    def test_multimodal_content(self):
        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = Message(role=Role.USER, content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestRole:
    def test_role_values(self):
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"
        assert Role.SYSTEM == "system"

    def test_role_is_str(self):
        assert isinstance(Role.USER, str)


class TestTokenUsage:
    def test_basic(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150
        assert usage.cached_tokens == 0

    def test_with_cached(self):
        usage = TokenUsage(
            input_tokens=100, output_tokens=50, total_tokens=150, cached_tokens=40
        )
        assert usage.cached_tokens == 40


class TestLLMResponse:
    def test_basic_response(self):
        msg = Message(role=Role.ASSISTANT, content="Hello!")
        usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        response = LLMResponse(
            message=msg,
            cost_usd=0.000_025,
            model="gpt-4o",
            finish_reason="stop",
            usage=usage,
        )
        assert response.message.content == "Hello!"
        assert response.cost_usd == 0.000_025
        assert response.finish_reason == "stop"
        assert response.tool_calls is None

    def test_response_with_tool_calls(self):
        msg = Message(role=Role.ASSISTANT, content="")
        tc = ToolCall(id="call_1", function=Function(name="search", arguments='{"q":"test"}'))
        response = LLMResponse(
            message=msg,
            cost_usd=0.001,
            model="gpt-4o",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "search"


class TestPresignedURL:
    def test_basic(self):
        url = PresignedURL(
            url="https://s3.amazonaws.com/bucket/file.txt?sig=abc",
            expires_at="2026-12-31T23:59:00+00:00",
            method="PUT",
            file_path="/workspace/file.txt",
        )
        assert url.method == "PUT"
        assert url.file_path == "/workspace/file.txt"
