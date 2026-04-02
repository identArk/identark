"""Unit tests for MockGateway and DirectGateway."""

import pytest

from identark.exceptions import (
    ConfigurationError,
    PathNotAllowedError,
)
from identark.models import Function, LLMResponse, Message, Role, TokenUsage, ToolCall
from identark.testing import MockGateway

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_response(content: str = "Test response", cost: float = 0.001) -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=content),
        cost_usd=cost,
        model="mock-gpt-4o",
        finish_reason="stop",
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_tool_response() -> LLMResponse:
    return LLMResponse(
        message=Message(role=Role.ASSISTANT, content=""),
        cost_usd=0.001,
        model="mock-gpt-4o",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="call_1", function=Function(name="search", arguments='{"q":"ai"}'))
        ],
    )


# ── MockGateway ───────────────────────────────────────────────────────────────

class TestMockGateway:
    @pytest.mark.asyncio
    async def test_returns_queued_response(self):
        mock = MockGateway()
        mock.queue_response(_make_response("Hello!"))

        response = await mock.invoke_llm(
            new_messages=[Message(role=Role.USER, content="Hi")]
        )
        assert response.message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_responses_consumed_in_order(self):
        mock = MockGateway()
        mock.queue_response(_make_response("First"))
        mock.queue_response(_make_response("Second"))

        r1 = await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="q1")])
        r2 = await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="q2")])

        assert r1.message.content == "First"
        assert r2.message.content == "Second"

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_queue_empty(self):
        mock = MockGateway()
        with pytest.raises(RuntimeError, match="queue is empty"):
            await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="hi")])

    @pytest.mark.asyncio
    async def test_default_response_when_queue_exhausted(self):
        default = _make_response("default")
        mock = MockGateway(default_response=default)

        r1 = await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="a")])
        r2 = await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="b")])
        assert r1.message.content == "default"
        assert r2.message.content == "default"

    @pytest.mark.asyncio
    async def test_call_count_tracking(self):
        mock = MockGateway()
        mock.queue_responses([_make_response(), _make_response(), _make_response()])

        for _ in range(3):
            await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="x")])

        assert mock.invoke_llm_call_count == 3

    @pytest.mark.asyncio
    async def test_total_messages_sent(self):
        mock = MockGateway()
        mock.queue_response(_make_response())
        mock.queue_response(_make_response())

        await mock.invoke_llm(new_messages=[
            Message(role=Role.USER, content="a"),
            Message(role=Role.USER, content="b"),
        ])
        await mock.invoke_llm(new_messages=[
            Message(role=Role.USER, content="c"),
        ])

        assert mock.total_messages_sent == 3

    @pytest.mark.asyncio
    async def test_last_request(self):
        mock = MockGateway()
        mock.queue_response(_make_response())

        msg = Message(role=Role.USER, content="check me")
        await mock.invoke_llm(new_messages=[msg])

        assert mock.last_request is not None
        assert mock.last_request["new_messages"][0].content == "check me"

    @pytest.mark.asyncio
    async def test_persist_messages_recorded(self):
        mock = MockGateway()
        msgs = [Message(role=Role.TOOL, content='{"result": 42}', tool_call_id="c1")]
        await mock.persist_messages(msgs)

        assert mock.persist_messages_call_count == 1
        assert mock.all_persisted_messages[0].content == '{"result": 42}'

    @pytest.mark.asyncio
    async def test_cost_accumulation(self):
        mock = MockGateway()
        mock.queue_response(_make_response(cost=0.001))
        mock.queue_response(_make_response(cost=0.002))

        await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="a")])
        await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="b")])

        cost = await mock.get_session_cost()
        assert abs(cost - 0.003) < 1e-9

    @pytest.mark.asyncio
    async def test_file_url_returns_presigned(self):
        mock = MockGateway(workspace_dir="/tmp/test-workspace")
        url = await mock.request_file_url("/workspace/output.json", method="PUT")

        assert url.method == "PUT"
        assert url.file_path == "/workspace/output.json"
        assert "output.json" in url.url

    @pytest.mark.asyncio
    async def test_reset_clears_calls(self):
        mock = MockGateway()
        mock.queue_response(_make_response())
        await mock.invoke_llm(new_messages=[Message(role=Role.USER, content="test")])

        mock.reset()
        assert mock.invoke_llm_call_count == 0
        assert mock.last_request is None

    def test_init_with_responses_list(self):
        mock = MockGateway(responses=[_make_response("pre-loaded")])
        assert mock.invoke_llm_call_count == 0  # not called yet


# ── DirectGateway init ────────────────────────────────────────────────────────

class TestDirectGatewayInit:
    def test_raises_on_none_client(self):
        from identark import DirectGateway
        with pytest.raises(ConfigurationError):
            DirectGateway(llm_client=None, model="gpt-4o")

    def test_raises_on_empty_model(self):
        from identark import DirectGateway
        with pytest.raises(ConfigurationError):
            DirectGateway(llm_client=object(), model="")

    def test_workspace_path_resolution(self):
        import tempfile

        from identark import DirectGateway

        with tempfile.TemporaryDirectory() as tmp:
            gw = DirectGateway(llm_client=object(), model="gpt-4o", workspace_dir=tmp)
            assert gw._workspace.as_posix() == tmp

    @pytest.mark.asyncio
    async def test_path_not_allowed_raises(self):
        import tempfile

        from identark import DirectGateway

        with tempfile.TemporaryDirectory() as tmp:
            gw = DirectGateway(llm_client=object(), model="gpt-4o", workspace_dir=tmp)
            with pytest.raises(PathNotAllowedError):
                await gw.request_file_url("/etc/passwd", method="GET")

    @pytest.mark.asyncio
    async def test_file_url_local_dev(self):
        import tempfile

        from identark import DirectGateway

        with tempfile.TemporaryDirectory() as tmp:
            gw = DirectGateway(llm_client=object(), model="gpt-4o", workspace_dir=tmp)
            url = await gw.request_file_url("/workspace/output.txt", method="PUT")
            assert url.method == "PUT"
            assert "output.txt" in url.url

    def test_initial_cost_is_zero(self):
        import asyncio
        import tempfile

        from identark import DirectGateway

        with tempfile.TemporaryDirectory() as tmp:
            gw = DirectGateway(llm_client=object(), model="gpt-4o", workspace_dir=tmp)
            cost = asyncio.run(gw.get_session_cost())
            assert cost == 0.0

    def test_reset_clears_history_and_cost(self):
        import tempfile

        from identark import DirectGateway

        with tempfile.TemporaryDirectory() as tmp:
            gw = DirectGateway(llm_client=object(), model="gpt-4o", workspace_dir=tmp)
            gw._history.append(Message(role=Role.USER, content="test"))
            gw._total_cost = 1.5
            gw.reset()
            assert gw.history == []
            assert gw._total_cost == 0.0


# ── DirectGateway provider detection ─────────────────────────────────────────

class TestDirectGatewayProviderDetection:
    """Tests for automatic and explicit provider detection."""

    def test_explicit_provider_overrides_all(self):
        from identark import DirectGateway

        gw = DirectGateway(llm_client=object(), model="gpt-4o", provider="local")
        assert gw.provider == "local"

    def test_explicit_mistral_provider(self):
        from identark import DirectGateway

        gw = DirectGateway(llm_client=object(), model="mistral-small-latest", provider="mistral")
        assert gw.provider == "mistral"

    def test_explicit_openai_provider(self):
        from identark import DirectGateway

        gw = DirectGateway(llm_client=object(), model="gpt-4o", provider="openai")
        assert gw.provider == "openai"

    def test_autodetect_anthropic_from_class_name(self):
        from identark import DirectGateway

        class FakeAsyncAnthropic:
            pass

        gw = DirectGateway(llm_client=FakeAsyncAnthropic(), model="claude-3-5-sonnet-20241022")
        assert gw.provider == "anthropic"

    def test_autodetect_local_from_localhost_base_url(self):
        from identark import DirectGateway

        class FakeLocalClient:
            base_url = "http://localhost:11434/v1"

        gw = DirectGateway(llm_client=FakeLocalClient(), model="llama3.2")
        assert gw.provider == "local"

    def test_autodetect_local_from_127_base_url(self):
        from identark import DirectGateway

        class FakeLocalClient:
            base_url = "http://127.0.0.1:11434/v1"

        gw = DirectGateway(llm_client=FakeLocalClient(), model="llama3.2")
        assert gw.provider == "local"

    def test_autodetect_mistral_from_base_url(self):
        from identark import DirectGateway

        class FakeMistralClient:
            base_url = "https://api.mistral.ai/v1"

        gw = DirectGateway(llm_client=FakeMistralClient(), model="mistral-large-latest")
        assert gw.provider == "mistral"

    def test_autodetect_openai_fallback(self):
        from identark import DirectGateway

        class FakeOpenAIClient:
            base_url = "https://api.openai.com/v1"

        gw = DirectGateway(llm_client=FakeOpenAIClient(), model="gpt-4o")
        assert gw.provider == "openai"

    def test_autodetect_openai_no_base_url(self):
        from identark import DirectGateway

        gw = DirectGateway(llm_client=object(), model="gpt-4o")
        assert gw.provider == "openai"

    def test_provider_property_readonly(self):
        from identark import DirectGateway

        gw = DirectGateway(llm_client=object(), model="gpt-4o", provider="local")
        assert isinstance(gw.provider, str)
