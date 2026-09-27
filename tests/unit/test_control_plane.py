"""
Tests for ControlPlaneGateway.

All HTTP calls are mocked — no real network or control plane required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from identark.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ContentPolicyError,
    CostCapExceededError,
    NetworkError,
    PathNotAllowedError,
    SessionNotFoundError,
)
from identark.gateways.control_plane import ControlPlaneGateway
from identark.models import Message, Role

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_gateway(**kwargs: Any) -> ControlPlaneGateway:
    defaults = {"api_key": "test-key", "url": "https://api.identark.io/v1"}
    return ControlPlaneGateway(**{**defaults, **kwargs})


def _mock_response(status: int, body: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _llm_response_body() -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": "Hello!"},
        "cost_usd": 0.0012,
        "model": "gpt-4o",
        "finish_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "session_id": "sess-123",
    }


# ── Initialisation ────────────────────────────────────────────────────────────


class TestControlPlaneGatewayInit:
    def test_explicit_args(self) -> None:
        gw = ControlPlaneGateway(api_key="k", url="https://example.com")
        assert gw._api_key == "k"
        assert gw._url == "https://example.com"

    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IDENTARK_API_KEY", raising=False)
        monkeypatch.delenv("IDENTARK_SESSION_TOKEN", raising=False)
        with pytest.raises(ConfigurationError, match="No API key"):
            ControlPlaneGateway(url="https://example.com")

    def test_raises_without_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IDENTARK_CONTROL_PLANE_URL", raising=False)
        with pytest.raises(ConfigurationError, match="No control plane URL"):
            ControlPlaneGateway(api_key="k")

    def test_api_key_from_env_identark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDENTARK_API_KEY", "env-key")
        monkeypatch.setenv("IDENTARK_CONTROL_PLANE_URL", "https://example.com")
        gw = ControlPlaneGateway()
        assert gw._api_key == "env-key"

    def test_session_token_takes_priority_over_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IDENTARK_SESSION_TOKEN", "session-tok")
        monkeypatch.setenv("IDENTARK_API_KEY", "api-key")
        monkeypatch.setenv("IDENTARK_CONTROL_PLANE_URL", "https://example.com")
        gw = ControlPlaneGateway()
        assert gw._api_key == "session-tok"

    def test_session_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDENTARK_SESSION_ID", "sess-abc")
        monkeypatch.setenv("IDENTARK_API_KEY", "k")
        monkeypatch.setenv("IDENTARK_CONTROL_PLANE_URL", "https://example.com")
        gw = ControlPlaneGateway()
        assert gw._session_id == "sess-abc"

    def test_trailing_slash_stripped_from_client_base_url(self) -> None:
        gw = ControlPlaneGateway(api_key="k", url="https://example.com/v1/")
        # The httpx client's base_url has the slash stripped; _url stores the raw value
        assert not str(gw._client.base_url).rstrip("/").endswith("/v1/")


# ── Context manager ───────────────────────────────────────────────────────────


class TestContextManager:
    async def test_aenter_returns_self(self) -> None:
        gw = _make_gateway()
        async with gw as ctx:
            assert ctx is gw
        # close is called — no exception

    async def test_close_called_on_exit(self) -> None:
        gw = _make_gateway()
        gw._client.aclose = AsyncMock()
        async with gw:
            pass
        gw._client.aclose.assert_called_once()


# ── invoke_llm ────────────────────────────────────────────────────────────────


class TestInvokeLLM:
    async def test_returns_llm_response(self) -> None:
        gw = _make_gateway(session_id="sess-1")
        mock_resp = _mock_response(200, _llm_response_body())
        gw._client.request = AsyncMock(return_value=mock_resp)

        result = await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hello")])

        assert result.message.content == "Hello!"
        assert result.cost_usd == pytest.approx(0.0012)
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    async def test_session_id_included_in_payload(self) -> None:
        gw = _make_gateway(session_id="sess-xyz")
        mock_resp = _mock_response(200, _llm_response_body())
        gw._client.request = AsyncMock(return_value=mock_resp)

        await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

        call_kwargs = gw._client.request.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.args[2]
        assert payload["session_id"] == "sess-xyz"

    async def test_tools_included_when_provided(self) -> None:
        gw = _make_gateway()
        mock_resp = _mock_response(200, _llm_response_body())
        gw._client.request = AsyncMock(return_value=mock_resp)
        tools = [{"type": "function", "function": {"name": "search"}}]

        await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")], tools=tools)

        call_kwargs = gw._client.request.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.args[2]
        assert payload["tools"] == tools

    async def test_parses_tool_calls(self) -> None:
        body = _llm_response_body()
        body["tool_calls"] = [
            {"id": "tc-1", "function": {"name": "search", "arguments": '{"q":"test"}'}}
        ]
        gw = _make_gateway()
        gw._client.request = AsyncMock(return_value=_mock_response(200, body))

        result = await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"


# ── persist_messages ──────────────────────────────────────────────────────────


class TestPersistMessages:
    async def test_posts_correct_payload(self) -> None:
        gw = _make_gateway(session_id="sess-1")
        gw._client.request = AsyncMock(
            return_value=_mock_response(200, {"persisted": 2, "session_id": "sess-1"})
        )
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi"),
        ]

        await gw.persist_messages(messages)

        call_kwargs = gw._client.request.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.args[2]
        assert len(payload["messages"]) == 2
        assert payload["session_id"] == "sess-1"


# ── request_file_url ──────────────────────────────────────────────────────────


class TestRequestFileUrl:
    async def test_returns_presigned_url(self) -> None:
        gw = _make_gateway()
        gw._client.request = AsyncMock(
            return_value=_mock_response(
                200,
                {
                    "url": "https://r2.example.com/file",
                    "expires_at": "2026-01-01T00:00:00Z",
                    "method": "PUT",
                    "file_path": "/workspace/out.txt",
                },
            )
        )

        result = await gw.request_file_url("/workspace/out.txt")

        assert result.url == "https://r2.example.com/file"
        assert result.method == "PUT"

    async def test_raises_path_not_allowed(self) -> None:
        gw = _make_gateway()
        with pytest.raises(PathNotAllowedError):
            await gw.request_file_url("/etc/passwd")


# ── get_session_cost ──────────────────────────────────────────────────────────


class TestGetSessionCost:
    async def test_returns_cost(self) -> None:
        gw = _make_gateway(session_id="sess-1")
        gw._client.request = AsyncMock(
            return_value=_mock_response(200, {"cost_usd": 0.0456, "session_id": "sess-1"})
        )

        cost = await gw.get_session_cost()
        assert cost == pytest.approx(0.0456)


class TestResolveCredential:
    async def test_structured_credential_exposes_fields(self) -> None:
        gw = _make_gateway()
        gw._client.request = AsyncMock(
            return_value=_mock_response(
                200,
                {
                    "name": "postgres",
                    "path": "prod/postgres",
                    "value": '{"password":"not-logged"}',
                    "type": "postgres",
                    "kind": "database",
                    "category": "database",
                    "fields": {
                        "host": "db.internal",
                        "username": "agent",
                        "password": "not-logged",
                        "database": "prod",
                    },
                },
            )
        )

        session = await gw.resolve_credential("prod/postgres")

        assert session.kind == "database"
        assert session.fields == {
            "host": "db.internal",
            "username": "agent",
            "password": "not-logged",
            "database": "prod",
        }


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    async def test_401_raises_authentication_error(self) -> None:
        gw = _make_gateway()
        gw._client.request = AsyncMock(
            return_value=_mock_response(
                401, {"error_code": "authentication_failed", "message": "Bad token"}
            )
        )
        with pytest.raises(AuthenticationError):
            await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

    async def test_402_raises_cost_cap_exceeded(self) -> None:
        gw = _make_gateway()
        gw._client.request = AsyncMock(
            return_value=_mock_response(
                402,
                {
                    "error_code": "cost_cap_exceeded",
                    "message": "Cap hit",
                    "cap_usd": 5.0,
                    "consumed_usd": 5.01,
                    "session_id": "s",
                },
            )
        )
        with pytest.raises(CostCapExceededError) as exc_info:
            await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])
        assert exc_info.value.cap_usd == pytest.approx(5.0)

    async def test_404_raises_session_not_found(self) -> None:
        gw = _make_gateway()
        gw._client.request = AsyncMock(
            return_value=_mock_response(
                404, {"error_code": "session_not_found", "session_id": "missing"}
            )
        )
        with pytest.raises(SessionNotFoundError):
            await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

    async def test_400_content_policy_raises_content_policy_error(self) -> None:
        gw = _make_gateway()
        gw._client.request = AsyncMock(
            return_value=_mock_response(
                400, {"error_code": "content_policy", "message": "Output blocked."}
            )
        )
        with pytest.raises(ContentPolicyError):
            await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

    async def test_5xx_retries_then_raises_network_error(self) -> None:
        gw = _make_gateway(max_retries=2)
        gw._client.request = AsyncMock(
            return_value=_mock_response(503, {"message": "Service unavailable"})
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NetworkError):
                await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

        assert gw._client.request.call_count == 2

    async def test_network_exception_retries(self) -> None:
        import httpx

        gw = _make_gateway(max_retries=2)
        gw._client.request = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NetworkError):
                await gw.invoke_llm(new_messages=[Message(role=Role.USER, content="Hi")])

        assert gw._client.request.call_count == 2


# ── History modes (server vs zero-retention client) ───────────────────────────


def _sent_payload(gw: ControlPlaneGateway, call: int = -1) -> dict[str, Any]:
    call_kwargs = gw._client.request.call_args_list[call]  # type: ignore[attr-defined]
    payload: dict[str, Any] = call_kwargs.kwargs.get("json") or call_kwargs.args[2]
    return payload


class TestHistoryModes:
    async def test_server_mode_sends_only_the_new_turn(self) -> None:
        gw = _make_gateway(session_id="sess-1")
        gw._client.request = AsyncMock(return_value=_mock_response(200, _llm_response_body()))

        await gw.invoke_llm([Message(role=Role.USER, content="Hi")])

        payload = _sent_payload(gw)
        assert payload["new_messages"] == [{"role": Role.USER, "content": "Hi"}]
        assert "store_history" not in payload
        assert gw.history_mode == "server"
        assert gw.history == []

    async def test_client_mode_holds_and_replays_the_conversation(self) -> None:
        gw = _make_gateway(session_id="sess-1", history="client")
        gw._client.request = AsyncMock(return_value=_mock_response(200, _llm_response_body()))

        await gw.invoke_llm([Message(role=Role.USER, content="Hi")])
        await gw.invoke_llm([Message(role=Role.USER, content="And then?")])

        first, second = _sent_payload(gw, 0), _sent_payload(gw, 1)
        assert first["store_history"] is False and second["store_history"] is False
        assert [m["content"] for m in first["new_messages"]] == ["Hi"]
        assert [m["content"] for m in second["new_messages"]] == ["Hi", "Hello!", "And then?"]
        assert [m["content"] for m in gw.history] == ["Hi", "Hello!", "And then?", "Hello!"]

    async def test_client_mode_keeps_assistant_tool_calls(self) -> None:
        gw = _make_gateway(history="client")
        body = {
            **_llm_response_body(),
            "message": {"role": "assistant", "content": ""},
            "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
        }
        gw._client.request = AsyncMock(return_value=_mock_response(200, body))

        await gw.invoke_llm([Message(role=Role.USER, content="Find it")])

        assert gw.history[-1]["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]

    async def test_client_mode_persist_is_local(self) -> None:
        gw = _make_gateway(history="client")
        gw._client.request = AsyncMock(return_value=_mock_response(200, _llm_response_body()))

        await gw.persist_messages([Message(role=Role.SYSTEM, content="Be brief.")])
        assert gw._client.request.await_count == 0
        await gw.invoke_llm([Message(role=Role.USER, content="Hi")])

        assert [m["content"] for m in _sent_payload(gw)["new_messages"]] == ["Be brief.", "Hi"]
        gw.clear_history()
        assert gw.history == []

    def test_history_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IDENTARK_HISTORY_MODE", "client")
        assert _make_gateway().history_mode == "client"

    def test_invalid_history_mode_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            _make_gateway(history="browser")  # type: ignore[arg-type]

    async def test_client_mode_stream_records_the_streamed_reply(self) -> None:
        import httpx

        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            body = (
                'data: {"delta": "Hello "}\n\n'
                'data: {"delta": "world"}\n\n'
                'data: {"delta": "", "finish_reason": "stop"}\n\n'
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        gw = _make_gateway(history="client")
        gw._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.identark.io/v1"
        )

        chunks = [
            c.content async for c in gw.invoke_llm_stream([Message(role=Role.USER, content="Hi")])
        ]

        assert "".join(chunks) == "Hello world"
        assert seen[0]["store_history"] is False
        assert gw.history == [
            {"role": Role.USER, "content": "Hi"},
            {"role": "assistant", "content": "Hello world"},
        ]

    async def test_stream_error_event_raises(self) -> None:
        import httpx

        from identark.exceptions import ControlPlaneError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text='data: {"error": "provider down"}\n\n')

        gw = _make_gateway(history="client")
        gw._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.identark.io/v1"
        )

        with pytest.raises(ControlPlaneError):
            async for _ in gw.invoke_llm_stream([Message(role=Role.USER, content="Hi")]):
                pass
        assert gw.history == []
