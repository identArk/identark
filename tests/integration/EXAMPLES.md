# Integration Test Examples

Practical examples of writing and running integration tests for the IdentArk SDK.

## Basic Control Plane Test

Test a simple invoke_llm call:

```python
@pytest.mark.integration
class TestBasicInvocation:
    @pytest.mark.asyncio
    async def test_simple_message(
        self,
        mock_control_plane: MockControlPlane,
        control_plane_gateway: ControlPlaneGateway,
    ) -> None:
        """Test invoking LLM with a single message."""
        with mock_control_plane.mocked():
            response = await control_plane_gateway.invoke_llm(
                [Message(role=Role.USER, content="Hello")]
            )

            assert response.message.content
            assert response.cost_usd > 0
            assert response.model == "gpt-4o"
```

## Testing Error Handling

Test how the gateway handles authentication errors:

```python
@pytest.mark.integration
class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_auth_failure(
        self,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test authentication error handling."""
        gateway = ControlPlaneGateway(
            api_key="invalid-key",
            url="https://api.identark.io/v1",
        )

        # Inject 401 error
        mock_control_plane.inject_error(
            "invoke_llm",
            401,
            {
                "error_code": "authentication_failed",
                "message": "Invalid API key",
                "reason": "key_not_found",
            }
        )

        with mock_control_plane.mocked():
            with pytest.raises(AuthenticationError) as exc_info:
                await gateway.invoke_llm(
                    [Message(role=Role.USER, content="Test")]
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.reason == "key_not_found"
```

## Testing Streaming

Test that streaming returns chunks correctly:

```python
@pytest.mark.integration
class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_chunks(
        self,
        mock_control_plane: MockControlPlane,
        control_plane_gateway: ControlPlaneGateway,
    ) -> None:
        """Test streaming returns chunks with metadata."""
        with mock_control_plane.mocked():
            chunks = []
            full_response = ""

            async for chunk in control_plane_gateway.invoke_llm_stream(
                [Message(role=Role.USER, content="Say hello")]
            ):
                chunks.append(chunk)
                if chunk.content:
                    full_response += chunk.content

            # Verify we got multiple chunks
            assert len(chunks) >= 2

            # Verify final chunk has metadata
            final_chunk = chunks[-1]
            assert final_chunk.finish_reason is not None
            assert final_chunk.input_tokens > 0
            assert final_chunk.output_tokens > 0

            # Verify content concatenates correctly
            assert "Hello" in full_response or "world" in full_response
```

## Testing Conversation History

Test that messages persist across calls:

```python
@pytest.mark.integration
class TestConversation:
    @pytest.mark.asyncio
    async def test_conversation_history(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test conversation history is maintained."""
        with mock_control_plane.mocked():
            # First exchange
            messages_1 = [
                Message(role=Role.USER, content="What is 2+2?")
            ]
            response_1 = await control_plane_gateway.invoke_llm(messages_1)

            # Persist to history
            await control_plane_gateway.persist_messages(
                [
                    *messages_1,
                    response_1.message,
                ]
            )

            # Follow-up question
            messages_2 = [
                Message(role=Role.USER, content="What about 3+3?")
            ]
            response_2 = await control_plane_gateway.invoke_llm(messages_2)

            assert response_1.message.content
            assert response_2.message.content
```

## Testing Cost Tracking

Test cost accumulation and limits:

```python
@pytest.mark.integration
class TestCostManagement:
    @pytest.mark.asyncio
    async def test_cost_accumulation(
        self,
        tmp_path,
    ) -> None:
        """Test that costs accumulate correctly."""
        from unittest.mock import AsyncMock, MagicMock

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Response",
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=MagicMock(
                    prompt_tokens=10,
                    completion_tokens=20,
                ),
                model="gpt-4o-mini",
            )
        )

        gateway = DirectGateway(
            llm_client=client,
            model="gpt-4o-mini",
            workspace_dir=str(tmp_path),
        )

        # Initial cost
        cost_before = await gateway.get_session_cost()

        # Make a call
        await gateway.invoke_llm(
            [Message(role=Role.USER, content="Hello")]
        )

        # Cost should have increased
        cost_after = await gateway.get_session_cost()
        assert cost_after > cost_before
```

## Testing Cost Caps

Test that cost caps are enforced:

```python
@pytest.mark.integration
class TestCostCaps:
    @pytest.mark.asyncio
    async def test_cost_cap_exceeded(
        self,
        tmp_path,
    ) -> None:
        """Test that cost cap prevents expensive operations."""
        from unittest.mock import AsyncMock, MagicMock

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="Response",
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=MagicMock(
                    prompt_tokens=10,
                    completion_tokens=20,
                ),
                model="gpt-4o-mini",
            )
        )

        # Set very low cap (0.0001 USD = 0.1 cents)
        gateway = DirectGateway(
            llm_client=client,
            model="gpt-4o-mini",
            workspace_dir=str(tmp_path),
            cost_cap_usd=0.0001,
        )

        # This should exceed the cap
        with pytest.raises(CostCapExceededError) as exc_info:
            await gateway.invoke_llm(
                [Message(role=Role.USER, content="Hello")]
            )

        assert exc_info.value.cap_usd == 0.0001
```

## Testing File Operations

Test file URL generation:

```python
@pytest.mark.integration
class TestFileOperations:
    @pytest.mark.asyncio
    async def test_presigned_urls(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test requesting presigned URLs for file access."""
        with mock_control_plane.mocked():
            # Request upload URL
            upload_url = await control_plane_gateway.request_file_url(
                file_path="/workspace/output.json",
                method="PUT",
            )

            assert upload_url.url.startswith("https://")
            assert upload_url.method == "PUT"
            assert upload_url.file_path == "/workspace/output.json"
            assert upload_url.expires_at

            # Request download URL
            download_url = await control_plane_gateway.request_file_url(
                file_path="/workspace/data.csv",
                method="GET",
            )

            assert download_url.method == "GET"
            assert download_url.file_path == "/workspace/data.csv"
```

## Testing Concurrent Operations

Test multiple simultaneous operations:

```python
@pytest.mark.integration
class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_invocations(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test multiple concurrent invocations."""
        import asyncio

        with mock_control_plane.mocked():
            # Create 5 concurrent tasks
            tasks = [
                control_plane_gateway.invoke_llm(
                    [Message(role=Role.USER, content=f"Message {i}")]
                )
                for i in range(5)
            ]

            # Wait for all to complete
            responses = await asyncio.gather(*tasks)

            # Verify all succeeded
            assert len(responses) == 5
            for response in responses:
                assert response.message.content
                assert response.cost_usd > 0
```

## Testing with Custom Latency

Test timeout handling with latency:

```python
@pytest.mark.integration
class TestLatency:
    @pytest.mark.asyncio
    async def test_with_latency(
        self,
        control_plane_gateway: ControlPlaneGateway,
        mock_control_plane: MockControlPlane,
    ) -> None:
        """Test behavior with artificial latency."""
        # Add 100ms latency
        mock_control_plane.set_latency(0.1)

        with mock_control_plane.mocked():
            import time
            start = time.time()

            await control_plane_gateway.invoke_llm(
                [Message(role=Role.USER, content="Quick test")]
            )

            elapsed = time.time() - start

            # Should have taken at least 100ms
            assert elapsed >= 0.1
```

## Running Examples

Run a specific example:

```bash
pytest tests/integration/test_control_plane_flow.py::TestBasicInvocation::test_simple_message -v
```

Run all control plane tests:

```bash
pytest tests/integration/test_control_plane_flow.py -v
```

Run with output:

```bash
pytest tests/integration/ -v -s
```

## Tips and Tricks

### Mock request inspection

Use `mock_control_plane.request_count()` to verify the mock was called:

```python
with mock_control_plane.mocked():
    await gateway.invoke_llm(messages)

    # Verify one request was made
    assert mock_control_plane.request_count() == 1
```

### Multiple error scenarios

Test multiple error paths in one test class:

```python
@pytest.mark.integration
class TestErrorScenarios:
    @pytest.mark.asyncio
    async def test_404_error(self, mock_control_plane, gateway):
        mock_control_plane.inject_error("invoke_llm", 404, {...})
        with mock_control_plane.mocked():
            with pytest.raises(SessionNotFoundError):
                await gateway.invoke_llm(messages)

    @pytest.mark.asyncio
    async def test_500_error(self, mock_control_plane, gateway):
        mock_control_plane.inject_error("invoke_llm", 500, {...})
        with mock_control_plane.mocked():
            with pytest.raises(NetworkError):
                await gateway.invoke_llm(messages)
```

### Fixture cleanup

Use `tmp_path` fixture for automatic cleanup of temporary files:

```python
@pytest.mark.asyncio
async def test_with_temp_workspace(tmp_path):
    gateway = DirectGateway(
        llm_client=client,
        model="gpt-4o-mini",
        workspace_dir=str(tmp_path),  # Auto-cleaned up
    )
    # test code
```

## See Also

- [Integration Test README](./README.md)
- [Test Coverage Report](../../docs/coverage.md)
- [Unit Test Examples](../unit/README.md)
