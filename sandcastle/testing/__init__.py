"""
sandcastle.testing
~~~~~~~~~~~~~~~~~~
Testing utilities for agents built with the SDK.

MockGateway
-----------
A test implementation of AgentGateway that returns configurable
responses without making any network or LLM calls. Records all
calls for assertion in tests.

Usage::

    from sandcastle.testing import MockGateway
    from sandcastle.models import LLMResponse, Message, Role

    mock = MockGateway()
    mock.queue_response(LLMResponse(
        message=Message(role=Role.ASSISTANT, content="Hello!"),
        cost_usd=0.0,
        model="mock",
        finish_reason="stop",
    ))

    response = await my_agent(gateway=mock)
    assert mock.invoke_llm_call_count == 1
"""

from sandcastle.testing.mock_gateway import MockGateway

__all__ = ["MockGateway"]
