"""
Fixtures for integration tests.

Provides:
- mock_control_plane: MockControlPlane instance
- control_plane_gateway: ControlPlaneGateway pre-configured for testing
- sample_messages: Common test messages
"""

from __future__ import annotations

import pytest

from identark.gateways.control_plane import ControlPlaneGateway
from identark.models import Message, Role
from tests.integration.mock_server import MockControlPlane


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: integration test requiring mock server")


@pytest.fixture
def mock_control_plane() -> MockControlPlane:
    """Create a MockControlPlane instance for mocking httpx calls."""
    return MockControlPlane()


@pytest.fixture
def control_plane_gateway() -> ControlPlaneGateway:
    """Create a ControlPlaneGateway for testing."""
    return ControlPlaneGateway(
        api_key="test-api-key",
        url="https://api.identark.io/v1",
        session_id="sess-test",
    )


@pytest.fixture
def sample_messages() -> list[Message]:
    """Sample messages for testing conversations."""
    return [
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is 2+2?"),
    ]


@pytest.fixture
def sample_assistant_message() -> Message:
    """Sample assistant response."""
    return Message(role=Role.ASSISTANT, content="2+2 equals 4.")
