"""
identark.validation
~~~~~~~~~~~~~~~~~~~~
Input validation utilities for tool definitions and messages.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from identark.exceptions import ConfigurationError

logger = logging.getLogger("identark.validation")


def validate_tool_definitions(tools: list[dict[str, Any]] | None) -> None:
    """Validate tool definitions follow the OpenAI function calling schema.

    Raises ConfigurationError if tools are malformed.
    """
    if tools is None:
        return

    if not isinstance(tools, list):
        raise ConfigurationError(f"tools must be a list, got {type(tools).__name__}")

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ConfigurationError(f"tools[{i}] must be a dict, got {type(tool).__name__}")

        # Required: type field
        tool_type = tool.get("type")
        if tool_type is None:
            raise ConfigurationError(f"tools[{i}] missing required 'type' field")
        if tool_type != "function":
            raise ConfigurationError(f"tools[{i}].type must be 'function', got '{tool_type}'")

        # Required: function field
        func = tool.get("function")
        if func is None:
            raise ConfigurationError(f"tools[{i}] missing required 'function' field")
        if not isinstance(func, dict):
            raise ConfigurationError(
                f"tools[{i}].function must be a dict, got {type(func).__name__}"
            )

        # Required: function.name
        name = func.get("name")
        if name is None:
            raise ConfigurationError(f"tools[{i}].function missing required 'name' field")
        if not isinstance(name, str) or not name:
            raise ConfigurationError(f"tools[{i}].function.name must be a non-empty string")

        # Optional but common: function.description
        if "description" in func and not isinstance(func["description"], str):
            raise ConfigurationError(
                f"tools[{i}].function.description must be a string"
            )

        # Optional: function.parameters (JSON Schema)
        params = func.get("parameters")
        if params is not None:
            if not isinstance(params, dict):
                raise ConfigurationError(
                    f"tools[{i}].function.parameters must be a dict (JSON Schema)"
                )
            # Basic JSON Schema validation
            if "type" not in params:
                logger.debug(
                    "tools[%d].function.parameters missing 'type', assuming 'object'", i
                )


def validate_tool_result_json(content: str, tool_name: str) -> None:
    """Validate that a tool result is valid JSON.

    Warns but does not raise if content is not valid JSON.
    Tool results should be JSON-serializable for best LLM understanding.
    """
    if not content:
        return

    try:
        json.loads(content)
    except json.JSONDecodeError:
        logger.warning(
            "Tool result for '%s' is not valid JSON. "
            "LLMs work best when tool results are JSON-formatted.",
            tool_name,
        )


def validate_message_content(content: str | list[dict[str, Any]] | None) -> None:
    """Validate message content structure.

    Content can be:
    - str: Plain text
    - list[dict]: Multimodal content blocks (images, etc.)
    - None: Empty content (valid for tool calls)
    """
    if content is None or isinstance(content, str):
        return

    if not isinstance(content, list):
        raise ConfigurationError(
            f"Message content must be str, list, or None, got {type(content).__name__}"
        )

    for i, block in enumerate(content):
        if not isinstance(block, dict):
            raise ConfigurationError(
                f"Message content[{i}] must be a dict, got {type(block).__name__}"
            )
        if "type" not in block:
            raise ConfigurationError(f"Message content[{i}] missing 'type' field")
