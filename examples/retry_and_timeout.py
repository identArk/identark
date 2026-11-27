"""
Bounded retries with exponential backoff and per-call timeouts.

Demonstrates:
- Wrapping ``gateway.invoke_llm()`` with ``asyncio.wait_for`` for a hard timeout.
- Retrying transient failures (``NetworkError``, ``RateLimitError``, ``ProviderError``)
  with bounded exponential backoff using only the Python standard library.
- Failing fast on non-retryable errors (``AuthenticationError``, ``CostCapExceededError``,
  ``ContentPolicyError``, ``ConfigurationError``).

Run:
    python examples/retry_and_timeout.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from identark import DirectGateway, Message, Role
from identark.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ContentPolicyError,
    CostCapExceededError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from identark.models import LLMResponse

RETRYABLE_ERRORS = (NetworkError, RateLimitError, ProviderError, TimeoutError)
NON_RETRYABLE_ERRORS = (
    AuthenticationError,
    CostCapExceededError,
    ContentPolicyError,
    ConfigurationError,
)


class _FlakyOpenAIClient:
    """Minimal async client simulating transient failures before succeeding."""

    def __init__(self, fail_times: int = 2, delay_seconds: float = 0.0) -> None:
        self.attempts = 0
        self.fail_times = fail_times
        self.delay_seconds = delay_seconds
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    async def _create(self, **_kwargs: object) -> object:
        self.attempts += 1
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if self.attempts <= self.fail_times:
            raise RateLimitError(
                "rate limit exceeded (simulated)",
                retry_after_seconds=0,
                provider="mock",
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Recovered after transient failures.",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6),
            model="gpt-4o-mini",
        )


async def invoke_with_retry_and_timeout(
    gateway: DirectGateway,
    messages: list[Message],
    *,
    timeout_seconds: float = 2.0,
    max_attempts: int = 4,
    base_delay_seconds: float = 0.05,
    max_delay_seconds: float = 0.2,
) -> LLMResponse:
    """Call ``gateway.invoke_llm`` with a timeout and bounded exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.wait_for(
                gateway.invoke_llm(new_messages=messages),
                timeout=timeout_seconds,
            )
        except NON_RETRYABLE_ERRORS:
            # Auth, budget cap, content policy, and config errors will not succeed on retry.
            raise
        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            backoff = base_delay_seconds * (2 ** (attempt - 1))
            if isinstance(exc, RateLimitError) and exc.retry_after_seconds > 0:
                backoff = max(backoff, float(exc.retry_after_seconds))
            backoff = min(backoff, max_delay_seconds)
            print(
                f"[attempt {attempt}/{max_attempts}] transient error "
                f"({type(exc).__name__}); retrying in {backoff:.2f}s"
            )
            await asyncio.sleep(backoff)

    assert last_error is not None
    raise last_error


async def main() -> None:
    messages = [Message(role=Role.USER, content="Ping")]

    # 1. Transient failure recovered via bounded exponential backoff
    flaky_client = _FlakyOpenAIClient(fail_times=2)
    gateway = DirectGateway(llm_client=flaky_client, model="gpt-4o-mini")
    response = await invoke_with_retry_and_timeout(gateway, messages)
    print(f"Success on attempt {flaky_client.attempts}: {response.message.content}")

    # 2. Hard timeout stops a hanging provider call
    slow_gateway = DirectGateway(
        llm_client=_FlakyOpenAIClient(fail_times=0, delay_seconds=1.0),
        model="gpt-4o-mini",
    )
    try:
        await invoke_with_retry_and_timeout(
            slow_gateway,
            messages,
            timeout_seconds=0.05,
            max_attempts=2,
            base_delay_seconds=0.01,
        )
    except TimeoutError:
        print("Timed out as expected after bounded retries.")


if __name__ == "__main__":
    asyncio.run(main())
