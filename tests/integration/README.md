# IdentArk SDK Integration Test Suite

Comprehensive integration tests for the IdentArk SDK, testing against a mock control plane API and real gateway implementations.

## Overview

The integration test suite validates:

- **Control Plane Gateway**: End-to-end flows including authentication, LLM invocation, streaming, file operations, and cost tracking
- **Direct Gateway**: Local LLM execution with conversation history and cost management
- **API Contracts**: Request/response format validation and error code mapping
- **Concurrency**: Thread-safe concurrent operations and state management
- **Error Handling**: Proper exception raising and error recovery

## Structure

```
tests/integration/
├── __init__.py                  # Package marker
├── conftest.py                  # Shared pytest fixtures
├── mock_server.py               # Mock control plane API implementation
├── test_control_plane_flow.py   # Control plane end-to-end tests
├── test_direct_gateway_e2e.py   # DirectGateway smoke tests
├── test_contract.py             # SDK-to-API contract tests
└── test_concurrency.py          # Concurrent usage tests
```

## Running Tests

### Run all integration tests

```bash
pytest tests/integration/ -v
```

### Run a specific test file

```bash
pytest tests/integration/test_control_plane_flow.py -v
```

### Run a specific test class

```bash
pytest tests/integration/test_control_plane_flow.py::TestLLMInvocation -v
```

### Run a specific test

```bash
pytest tests/integration/test_control_plane_flow.py::TestLLMInvocation::test_invoke_llm_with_new_messages -v
```

### Run with detailed output

```bash
pytest tests/integration/ -vv -s
```

### Run only integration markers

```bash
pytest -m integration
```

## Mock Control Plane

The `MockControlPlane` class simulates the IdentArk control plane API without requiring network access or external dependencies.

### Features

- **Configurable Latency**: Inject artificial delays to test timeout handling
- **Error Injection**: Simulate 401, 402, 404, 500 responses
- **Request Tracking**: Count requests and verify call patterns
- **Session Cost Tracking**: Set and retrieve mock session costs

### Example

```python
from tests.integration.mock_server import MockControlPlane

def test_something(mock_control_plane: MockControlPlane):
    # Inject a 500 error to test retry logic
    mock_control_plane.inject_error("invoke_llm", 500)

    # Add latency to test timeout handling
    mock_control_plane.set_latency(0.5)

    with mock_control_plane.mocked():
        # Your test code here
        pass
```

## Test Categories

### test_control_plane_flow.py

End-to-end control plane flows:

- **Authentication**: Valid/invalid API keys, session management
- **LLM Invocation**: Basic invocation, streaming, tool definitions
- **Message Persistence**: Storing conversation history
- **File Operations**: Requesting presigned URLs for upload/download
- **Cost Tracking**: Accumulating and capping costs
- **Error Handling**: 401/402/404/500 errors, network failures, rate limits
- **Context Manager**: Using gateway as async context manager

### test_direct_gateway_e2e.py

DirectGateway functionality:

- **Basic Invocation**: Creating and using the gateway
- **Conversation History**: Accumulating messages across calls
- **Cost Tracking**: Accumulating costs and enforcing caps
- **File Operations**: Local file path resolution
- **Streaming**: Async generator of chunks with final metadata
- **Provider Detection**: OpenAI vs local (Ollama) cost tracking
- **Reset**: Clearing history and cost

### test_contract.py

API contract compliance:

- **Request Format**: Verifying payload structure matches API spec
- **Response Parsing**: Ensuring all response fields are parsed correctly
- **Error Codes**: Mapping HTTP status/error codes to exception types
- **Streaming SSE**: Verifying SSE format parsing
- **Auth Headers**: Validating Bearer token format

### test_concurrency.py

Concurrent usage patterns:

- **Simultaneous Invocations**: Multiple concurrent `invoke_llm` calls
- **Streaming Concurrency**: Multiple concurrent streaming calls
- **Mixed Operations**: Streaming + non-streaming concurrent calls
- **DirectGateway Concurrency**: Concurrent calls without interference
- **Cost Safety**: Thread-safe cost accumulation
- **Stress Testing**: 10+ concurrent requests

## Fixtures

### mock_control_plane

Returns a `MockControlPlane` instance for use in tests.

```python
def test_something(mock_control_plane: MockControlPlane):
    with mock_control_plane.mocked():
        # Your test code
        pass
```

### control_plane_gateway

Pre-configured `ControlPlaneGateway` instance.

```python
async def test_something(control_plane_gateway: ControlPlaneGateway):
    with mock_control_plane.mocked():
        response = await control_plane_gateway.invoke_llm(messages)
```

### sample_messages

List of sample `Message` objects for testing conversations.

```python
def test_something(sample_messages: list[Message]):
    # sample_messages = [
    #     Message(role=Role.SYSTEM, content="You are helpful..."),
    #     Message(role=Role.USER, content="What is 2+2?"),
    # ]
    pass
```

### sample_assistant_message

Sample assistant response message.

```python
def test_something(sample_assistant_message: Message):
    # Message(role=Role.ASSISTANT, content="2+2 equals 4.")
    pass
```

## Writing New Integration Tests

### Test Structure

```python
import pytest
from tests.integration.mock_server import MockControlPlane
from identark.gateways.control_plane import ControlPlaneGateway
from identark.models import Message, Role

@pytest.mark.integration
class TestNewFeature:
    """Test suite for new feature."""

    @pytest.mark.asyncio
    async def test_something(
        self,
        mock_control_plane: MockControlPlane,
        control_plane_gateway: ControlPlaneGateway,
    ) -> None:
        """Test description."""
        # Inject error if needed
        mock_control_plane.inject_error("invoke_llm", 500)

        # Use context manager to enable mocking
        with mock_control_plane.mocked():
            # Your test code
            response = await control_plane_gateway.invoke_llm(
                [Message(role=Role.USER, content="Test")]
            )
            assert response.message.content
```

### Best Practices

1. **Use the `@pytest.mark.integration` marker** to clearly identify integration tests
2. **Use the mock context manager** to enable HTTP mocking
3. **Test both success and failure paths** (happy path + error injection)
4. **Use descriptive test names** that explain what's being tested
5. **Add docstrings** explaining the test purpose
6. **Assert on relevant fields** — don't just check that something doesn't error
7. **Clean up resources** (especially async operations) in fixtures

## Markers

Tests are marked with `@pytest.mark.integration` for easy filtering.

```bash
# Run only integration tests
pytest -m integration

# Run excluding integration tests
pytest -m "not integration"
```

## Error Injection Examples

### Simulate authentication failure

```python
mock_control_plane.inject_error("invoke_llm", 401, {
    "error_code": "authentication_failed",
    "message": "Invalid token",
    "reason": "key_expired",
})
```

### Simulate cost cap exceeded

```python
mock_control_plane.inject_error("invoke_llm", 402, {
    "error_code": "cost_cap_exceeded",
    "message": "Cost cap exceeded",
    "cap_usd": 1.0,
    "consumed_usd": 1.05,
})
```

### Simulate session not found

```python
mock_control_plane.inject_error("invoke_llm", 404, {
    "error_code": "session_not_found",
    "message": "Session not found",
    "session_id": "sess-expired",
})
```

### Simulate server error (triggers retries)

```python
mock_control_plane.inject_error("invoke_llm", 500, {
    "error": "Internal server error"
})
```

### Simulate high latency

```python
mock_control_plane.set_latency(2.0)  # 2 seconds
```

## Mock Implementation Details

The `MockControlPlane` uses `unittest.mock` to intercept `httpx.AsyncClient.request` calls, avoiding the need for external dependencies like `respx`.

### Supported Endpoints

| Method | Path | Handler |
|--------|------|---------|
| POST | /llm/invoke | `_invoke_llm_handler` |
| POST | /llm/stream | `_invoke_llm_stream_handler` |
| POST | /messages/persist | `_persist_messages_handler` |
| POST | /files/presigned-urls | `_request_file_url_handler` |
| GET | /sessions/cost | `_get_session_cost_handler` |

## Limitations and TODOs

Current limitations:

- **No Real Network**: Tests don't validate actual network behavior (timeouts, connection resets, etc.)
- **Streaming Mock**: Streaming SSE parsing is partially mocked; full protocol testing recommended with real endpoint
- **File I/O**: File URL operations return mock paths; actual S3/GCS operations not tested

## Debugging

### Enable verbose logging

```bash
pytest tests/integration/ -vv -s --log-cli-level=DEBUG
```

### Print mock request/response details

```python
def test_something(mock_control_plane: MockControlPlane):
    with mock_control_plane.mocked():
        # Your test code
        print(f"Request count: {mock_control_plane.request_count()}")
```

## Contributing

When adding new tests:

1. Place in appropriate file based on component tested
2. Add `@pytest.mark.integration` and `@pytest.mark.asyncio` markers
3. Use descriptive test names: `test_<function>_<scenario>`
4. Document test purpose in docstring
5. Use fixtures for common setup
6. Test both success and failure paths
7. Clean up resources in fixtures or context managers

## See Also

- [Unit Tests](../unit/README.md)
- [IdentArk SDK Documentation](../../README.md)
- [Control Plane API Spec](../../docs/api.md)
