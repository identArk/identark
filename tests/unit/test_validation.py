"""Tests for identark.validation module."""

import pytest

from identark.exceptions import ConfigurationError
from identark.validation import (
    validate_message_content,
    validate_tool_definitions,
    validate_tool_result_json,
)


class TestValidateToolDefinitions:
    """Test tool definition validation."""

    def test_none_tools_is_valid(self) -> None:
        """None tools list is valid."""
        validate_tool_definitions(None)  # Should not raise

    def test_empty_list_is_valid(self) -> None:
        """Empty tools list is valid."""
        validate_tool_definitions([])  # Should not raise

    def test_valid_tool_definition(self) -> None:
        """Well-formed tool definition passes validation."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        validate_tool_definitions(tools)  # Should not raise

    def test_minimal_valid_tool(self) -> None:
        """Minimal tool definition (just type and name) is valid."""
        tools = [{"type": "function", "function": {"name": "do_something"}}]
        validate_tool_definitions(tools)  # Should not raise

    def test_tools_must_be_list(self) -> None:
        """Tools must be a list."""
        with pytest.raises(ConfigurationError, match="must be a list"):
            validate_tool_definitions({"type": "function"})  # type: ignore

    def test_tool_must_be_dict(self) -> None:
        """Each tool must be a dict."""
        with pytest.raises(ConfigurationError, match="must be a dict"):
            validate_tool_definitions(["not a dict"])  # type: ignore

    def test_missing_type_field(self) -> None:
        """Tool missing type field raises error."""
        with pytest.raises(ConfigurationError, match="missing required 'type'"):
            validate_tool_definitions([{"function": {"name": "test"}}])

    def test_invalid_type_value(self) -> None:
        """Tool with non-function type raises error."""
        with pytest.raises(ConfigurationError, match="must be 'function'"):
            validate_tool_definitions([{"type": "retrieval", "function": {"name": "test"}}])

    def test_missing_function_field(self) -> None:
        """Tool missing function field raises error."""
        with pytest.raises(ConfigurationError, match="missing required 'function'"):
            validate_tool_definitions([{"type": "function"}])

    def test_function_must_be_dict(self) -> None:
        """Function field must be a dict."""
        with pytest.raises(ConfigurationError, match="function must be a dict"):
            validate_tool_definitions([{"type": "function", "function": "not a dict"}])

    def test_missing_function_name(self) -> None:
        """Function missing name raises error."""
        with pytest.raises(ConfigurationError, match="missing required 'name'"):
            validate_tool_definitions([{"type": "function", "function": {}}])

    def test_empty_function_name(self) -> None:
        """Empty function name raises error."""
        with pytest.raises(ConfigurationError, match="non-empty string"):
            validate_tool_definitions([{"type": "function", "function": {"name": ""}}])

    def test_invalid_description_type(self) -> None:
        """Non-string description raises error."""
        with pytest.raises(ConfigurationError, match="description must be a string"):
            validate_tool_definitions([
                {"type": "function", "function": {"name": "test", "description": 123}}
            ])

    def test_parameters_must_be_dict(self) -> None:
        """Parameters must be a dict (JSON Schema)."""
        with pytest.raises(ConfigurationError, match="parameters must be a dict"):
            validate_tool_definitions([
                {"type": "function", "function": {"name": "test", "parameters": "invalid"}}
            ])


class TestValidateToolResultJson:
    """Test tool result JSON validation."""

    def test_valid_json(self) -> None:
        """Valid JSON passes silently."""
        validate_tool_result_json('{"result": "success"}', "test_tool")

    def test_empty_content(self) -> None:
        """Empty content passes silently."""
        validate_tool_result_json("", "test_tool")

    def test_invalid_json_warns(self, caplog) -> None:
        """Invalid JSON logs a warning but doesn't raise."""
        validate_tool_result_json("not json", "test_tool")
        assert "not valid JSON" in caplog.text


class TestValidateMessageContent:
    """Test message content validation."""

    def test_string_content(self) -> None:
        """String content is valid."""
        validate_message_content("Hello, world!")

    def test_none_content(self) -> None:
        """None content is valid (for tool calls)."""
        validate_message_content(None)

    def test_multimodal_content(self) -> None:
        """List of content blocks is valid."""
        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        validate_message_content(content)

    def test_invalid_content_type(self) -> None:
        """Invalid content type raises error."""
        with pytest.raises(ConfigurationError, match="must be str, list, or None"):
            validate_message_content(123)  # type: ignore

    def test_content_block_must_be_dict(self) -> None:
        """Content blocks must be dicts."""
        with pytest.raises(ConfigurationError, match="must be a dict"):
            validate_message_content(["not a dict"])  # type: ignore

    def test_content_block_missing_type(self) -> None:
        """Content blocks must have type field."""
        with pytest.raises(ConfigurationError, match="missing 'type'"):
            validate_message_content([{"text": "no type field"}])
