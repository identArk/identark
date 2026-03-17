"""
credseal.testing.mock_gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MockGateway — a test double for AgentGateway.

Returns configurable responses without any network or LLM calls.
Records every call so tests can assert on what was sent.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credseal.models import LLMResponse, Message, PresignedURL


class MockGateway:
    """
    Test implementation of :class:`~credseal.gateway.AgentGateway`.

    Returns queued responses without any network or LLM calls.
    Records every call for assertion.

    Args:
        responses: Optional initial list of
                   :class:`~credseal.models.LLMResponse` objects
                   to return in order.
        default_response: Returned when the queue is exhausted.
                          If ``None``, raises ``StopIteration``.
        workspace_dir:    Local directory for file URL resolution.
                          Defaults to ``'/tmp/credseal-mock-workspace'``.

    Example::

        mock = MockGateway(responses=[
            LLMResponse(
                message=Message(role=Role.ASSISTANT, content="Done."),
                cost_usd=0.001,
                model="mock",
                finish_reason="stop",
            )
        ])
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        default_response: LLMResponse | None = None,
        workspace_dir: str = "/tmp/credseal-mock-workspace",
    ) -> None:
        self._queue: deque[LLMResponse] = deque(responses or [])
        self._default = default_response
        self._workspace = workspace_dir

        # Call recording
        self._invoke_calls: list[dict[str, Any]] = []
        self._persist_calls: list[list[Message]] = []
        self._file_url_calls: list[dict[str, Any]] = []

        self._total_cost: float = 0.0

    # ── Response management ───────────────────────────────────────────────────

    def queue_response(self, response: LLMResponse) -> None:
        """Add a response to the end of the queue."""
        self._queue.append(response)

    def queue_responses(self, responses: list[LLMResponse]) -> None:
        """Add multiple responses to the end of the queue."""
        self._queue.extend(responses)

    def _next_response(self) -> LLMResponse:
        if self._queue:
            return self._queue.popleft()
        if self._default is not None:
            return self._default
        raise StopIteration(
            "MockGateway response queue is empty and no default_response was set. "
            "Call mock.queue_response() to add more responses."
        )

    # ── AgentGateway interface ────────────────────────────────────────────────

    async def invoke_llm(
        self,
        new_messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Return the next queued response."""
        self._invoke_calls.append({
            "new_messages": new_messages,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        response = self._next_response()
        self._total_cost += response.cost_usd
        return response

    async def persist_messages(self, messages: list[Message]) -> None:
        """Record the persist call."""
        self._persist_calls.append(list(messages))

    async def request_file_url(
        self,
        file_path: str,
        method: str = "PUT",
    ) -> PresignedURL:
        """Return a mock file:// presigned URL."""
        self._file_url_calls.append({"file_path": file_path, "method": method})
        resolved = Path(self._workspace) / file_path.removeprefix("/workspace/")
        expiry = datetime.now(timezone.utc).replace(hour=23, minute=59).isoformat()
        return PresignedURL(
            url=resolved.as_uri(),
            expires_at=expiry,
            method=method,
            file_path=file_path,
        )

    async def get_session_cost(self) -> float:
        """Return the total accumulated mock cost."""
        return self._total_cost

    # ── Assertions ────────────────────────────────────────────────────────────

    @property
    def invoke_llm_call_count(self) -> int:
        """Number of times ``invoke_llm`` was called."""
        return len(self._invoke_calls)

    @property
    def persist_messages_call_count(self) -> int:
        """Number of times ``persist_messages`` was called."""
        return len(self._persist_calls)

    @property
    def file_url_request_count(self) -> int:
        """Number of times ``request_file_url`` was called."""
        return len(self._file_url_calls)

    @property
    def total_messages_sent(self) -> int:
        """Total number of messages passed across all ``invoke_llm`` calls."""
        return sum(len(c["new_messages"]) for c in self._invoke_calls)

    @property
    def last_request(self) -> dict[str, Any] | None:
        """The most recent ``invoke_llm`` call arguments, or ``None``."""
        return self._invoke_calls[-1] if self._invoke_calls else None

    @property
    def all_invoke_calls(self) -> list[dict[str, Any]]:
        """All recorded ``invoke_llm`` call arguments in order."""
        return list(self._invoke_calls)

    @property
    def all_persisted_messages(self) -> list[Message]:
        """All messages that have been persisted, flattened."""
        return [m for batch in self._persist_calls for m in batch]

    def reset(self) -> None:
        """Clear all recorded calls and reset cost. Does not clear the queue."""
        self._invoke_calls.clear()
        self._persist_calls.clear()
        self._file_url_calls.clear()
        self._total_cost = 0.0
