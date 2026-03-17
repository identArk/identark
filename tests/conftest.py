"""
Shared pytest fixtures for credseal-sdk tests.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from credseal.models import LLMResponse, Message, Role, TokenUsage
from credseal.testing import MockGateway

# ── Response factory ─────────────────────────────────────────────────────────


def make_response(
    content: str = "Test response",
    cost: float = 0.001,
    finish_reason: str = "stop",
    model: str = "mock-gpt-4o",
) -> LLMResponse:
    """Factory for LLMResponse objects in tests."""
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        cost_usd=cost,
        model=model,
        finish_reason=finish_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_gateway() -> MockGateway:
    """Empty MockGateway. Add responses with mock_gateway.queue_response()."""
    return MockGateway()


@pytest.fixture
def mock_gateway_with_response(mock_gateway: MockGateway) -> MockGateway:
    """MockGateway pre-loaded with a single canned response."""
    mock_gateway.queue_response(make_response())
    return mock_gateway


@pytest.fixture
def tmp_workspace() -> str:
    """Temporary directory to use as a workspace in DirectGateway tests."""
    with tempfile.TemporaryDirectory(prefix="credseal-test-") as tmp:
        yield tmp


@pytest.fixture
def direct_gateway(tmp_workspace: str):
    """
    DirectGateway configured for tests.

    Requires OPENAI_API_KEY in the environment — skips if not set.
    Use mock_gateway for pure unit tests.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping DirectGateway integration test")

    from openai import AsyncOpenAI

    from credseal import DirectGateway

    return DirectGateway(
        llm_client=AsyncOpenAI(api_key=api_key),
        model="gpt-4o-mini",
        workspace_dir=tmp_workspace,
        cost_cap_usd=0.10,  # Hard cap for tests — never spend more than $0.10
    )
